# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the Ollama backend intrinsic generation path. No server required.

Mocks the Ollama async client to verify that `_generate_from_intrinsic` correctly:
- appends the io.yaml instruction (the adapter's activation text) as the last message
- passes the io.yaml response schema as `format` and requests logprobs
- routes the call to the Ollama model tag registered for the adapter function
- applies the `IntrinsicsResultProcessor` to the raw response
- user-provided model options override io.yaml parameter defaults
- raises when no adapter is registered, no model tag is configured for it, or
  streaming is requested
"""

import json
import threading
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch

import ollama
import pytest

from mellea.backends import ModelOption
from mellea.backends.adapters import (
    Adapter,
    AdapterType,
    Identity,
    ServerMediatedBinding,
    get_io_contract,
)
from mellea.backends.ollama import OllamaModelBackend, _to_chat_completion_dict
from mellea.core import ModelOutputThunk
from mellea.stdlib import functional as mfuncs
from mellea.stdlib.components import Intrinsic, Message
from mellea.stdlib.context import ChatContext
from mellea.stdlib.requirements import ALoraRequirement, LLMaJRequirement, Requirement

# ---------------------------------------------------------------------------
# Configs
# ---------------------------------------------------------------------------

_SCORE_SCHEMA = {
    "type": "object",
    "properties": {"score": {"type": "string", "enum": [str(i) for i in range(10)]}},
    "required": ["score"],
    "additionalProperties": False,
}

# Minimal config: no transformations, no logprobs.  Good enough for tests that
# only inspect the API call.
_SIMPLE_CONFIG = {
    "model": None,
    "response_format": _SCORE_SCHEMA,
    "transformations": None,
    "instruction": "<certainty>",
    "parameters": {"max_completion_tokens": 64, "temperature": 1.0},
    "sentence_boundaries": None,
}

# Mirrors the real uncertainty io.yaml: likelihood + project transformations.
_UNCERTAINTY_CONFIG = {
    "model": None,
    "response_format": _SCORE_SCHEMA,
    "transformations": [
        {
            "type": "likelihood",
            "categories_to_values": {str(i): 0.1 * i + 0.05 for i in range(10)},
            "input_path": ["score"],
        },
        {
            "type": "project",
            "input_path": [],
            "retained_fields": {"score": "certainty"},
        },
    ],
    "instruction": "<certainty>",
    "parameters": {"max_completion_tokens": 15, "temperature": 0.0},
    "sentence_boundaries": None,
}

_ADAPTER_TAG = "mellea-test/uncertainty-alora:latest"

# ---------------------------------------------------------------------------
# Canned responses
# ---------------------------------------------------------------------------


def _simple_chat_response(content: str = '{"score": "9"}') -> ollama.ChatResponse:
    """Build a minimal ChatResponse with no logprobs."""
    return ollama.ChatResponse.model_validate(
        {
            "model": _ADAPTER_TAG,
            "message": {"role": "assistant", "content": content},
            "done": True,
            "done_reason": "stop",
        }
    )


def _uncertainty_chat_response() -> ollama.ChatResponse:
    """Build a ChatResponse that the uncertainty result processor can parse.

    The likelihood transformation reads top_logprobs to compute an expected value.
    """
    return ollama.ChatResponse.model_validate(
        {
            "model": _ADAPTER_TAG,
            "message": {"role": "assistant", "content": '{"score": "9"}'},
            "done": True,
            "done_reason": "stop",
            "logprobs": [
                {
                    "token": '{"',
                    "logprob": 0.0,
                    "top_logprobs": [{"token": '{"', "logprob": 0.0}],
                },
                {
                    "token": "score",
                    "logprob": 0.0,
                    "top_logprobs": [{"token": "score", "logprob": 0.0}],
                },
                {
                    "token": '":',
                    "logprob": 0.0,
                    "top_logprobs": [{"token": '":', "logprob": 0.0}],
                },
                {
                    "token": ' "',
                    "logprob": 0.0,
                    "top_logprobs": [{"token": ' "', "logprob": 0.0}],
                },
                {
                    "token": "9",
                    "logprob": -0.05,
                    "top_logprobs": [
                        {"token": "9", "logprob": -0.05},
                        {"token": "4", "logprob": -3.0},
                    ],
                },
                {
                    "token": '"}',
                    "logprob": 0.0,
                    "top_logprobs": [{"token": '"}', "logprob": 0.0}],
                },
            ],
        }
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_backend(
    *, model_options: dict | None = None, adapter_models: dict | None = None
) -> OllamaModelBackend:
    """Return an OllamaModelBackend with all network calls patched out."""
    with (
        patch.object(OllamaModelBackend, "_check_ollama_server", return_value=True),
        patch.object(OllamaModelBackend, "_pull_ollama_model", return_value=True),
        patch("mellea.backends.ollama.ollama.Client", return_value=MagicMock()),
        patch("mellea.backends.ollama.ollama.AsyncClient", return_value=MagicMock()),
    ):
        return OllamaModelBackend(
            model_id="granite4.1:3b",
            model_options=model_options,
            adapter_models=adapter_models,
        )


def _make_backend_with_adapter(
    config: dict,
    *,
    model_options: dict | None = None,
    adapter_models: dict | None = None,
) -> OllamaModelBackend:
    """Return an OllamaModelBackend with a registered uncertainty adapter.

    Defaults `adapter_models` to a tag for "uncertainty" so tests that don't
    care about model-tag routing aren't affected by it: generating against an
    adapter with no configured tag now raises (it would otherwise silently
    run against the plain base model with none of the adapter's weights).
    Pass `adapter_models={}` explicitly to opt into that unconfigured case.
    """
    if adapter_models is None:
        adapter_models = {"uncertainty": _ADAPTER_TAG}
    backend = _make_backend(model_options=model_options, adapter_models=adapter_models)
    adapter = Adapter(
        identity=Identity(name="uncertainty", adapter_type="alora"),
        io_contract=get_io_contract("uncertainty"),
        weights=ServerMediatedBinding(),
    )
    backend.add_adapter(adapter, config=config)
    return backend


def _make_context() -> ChatContext:
    """Return a simple two-turn chat context."""
    return (
        ChatContext()
        .add(Message("user", "What is the square root of 4?"))
        .add(Message("assistant", "The square root of 4 is 2."))
    )


async def _run_intrinsic(
    backend: OllamaModelBackend, response: ollama.ChatResponse, **kwargs
):
    """Run the uncertainty intrinsic against a mocked client; return (mot, mock_chat)."""
    mock_chat = AsyncMock(return_value=response)
    mock_client = MagicMock()
    mock_client.chat = mock_chat

    with patch.object(
        OllamaModelBackend,
        "_async_client",
        new_callable=PropertyMock,
        return_value=mock_client,
    ):
        mot, _ = await mfuncs.aact(
            Intrinsic("uncertainty"), _make_context(), backend, strategy=None, **kwargs
        )
        await mot.avalue()
    return mot, mock_chat


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


async def test_documents_folded_into_message_when_not_docs_as_message():
    """Documents extra_body has no Ollama transport, so they must land in a message.

    `_SIMPLE_CONFIG` sets no `docs_as_message`, mirroring the shipped
    answerability/citations/etc. io.yaml configs. Without folding, Ollama
    receives zero documents and still returns a schema-valid, meaningless score.
    """
    backend = _make_backend_with_adapter(_SIMPLE_CONFIG)
    context = ChatContext().add(
        Message("user", "What is the square root of 4?", documents=["4 is 2 squared."])
    )
    mock_chat = AsyncMock(return_value=_simple_chat_response())
    mock_client = MagicMock()
    mock_client.chat = mock_chat

    with patch.object(
        OllamaModelBackend,
        "_async_client",
        new_callable=PropertyMock,
        return_value=mock_client,
    ):
        mot, _ = await mfuncs.aact(
            Intrinsic("uncertainty"), context, backend, strategy=None
        )
        await mot.avalue()

    messages = mock_chat.call_args.kwargs["messages"]
    assert any("4 is 2 squared." in m["content"] for m in messages)


async def test_instruction_appended_as_last_message():
    """The io.yaml instruction (the adapter activation text) is the final user message."""
    backend = _make_backend_with_adapter(_SIMPLE_CONFIG)
    _, mock_chat = await _run_intrinsic(backend, _simple_chat_response())

    mock_chat.assert_called_once()
    messages = mock_chat.call_args.kwargs["messages"]
    assert messages[-1]["role"] == "user"
    assert messages[-1]["content"] == "<certainty>"
    assert messages[0]["content"] == "What is the square root of 4?"


async def test_format_and_logprobs_requested():
    """The io.yaml response schema is passed as `format`; likelihood rules request logprobs."""
    backend = _make_backend_with_adapter(_UNCERTAINTY_CONFIG)
    _, mock_chat = await _run_intrinsic(backend, _uncertainty_chat_response())

    call_kwargs = mock_chat.call_args.kwargs
    assert call_kwargs["format"] == _SCORE_SCHEMA
    assert call_kwargs["logprobs"] is True
    assert call_kwargs["top_logprobs"] == 10
    assert call_kwargs["stream"] is False


async def test_adapter_model_tag_used():
    """The call goes to the Ollama model registered for the adapter function."""
    backend = _make_backend_with_adapter(
        _SIMPLE_CONFIG, adapter_models={"uncertainty": _ADAPTER_TAG}
    )
    _, mock_chat = await _run_intrinsic(backend, _simple_chat_response())

    assert mock_chat.call_args.kwargs["model"] == _ADAPTER_TAG


async def test_generation_without_configured_tag_raises():
    """An adapter with no `adapter_models` entry raises rather than silently
    running against the plain base model.

    The rewriter still builds the adapter's activation prompt and enforces
    its response schema either way, so a silent fallback would return a
    schema-valid, meaningless answer from a model that never saw the
    adapter's weights — the same failure class already guarded against in
    `resolve_adapter()`.
    """
    backend = _make_backend_with_adapter(_SIMPLE_CONFIG, adapter_models={})

    with pytest.raises(ValueError, match="No Ollama model tag configured"):
        await _run_intrinsic(backend, _simple_chat_response())


async def test_alora_requirement_resolves_mapped_adapter(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    """A configured catalogue adapter is resolved before requirement routing."""
    config_path = tmp_path / "io.yaml"
    config_path.write_text(json.dumps(_SIMPLE_CONFIG), encoding="utf-8")
    monkeypatch.setattr(
        "mellea.backends.ollama.granite_formatters.intrinsics.obtain_io_yaml",
        lambda *_args, **_kwargs: config_path,
    )
    backend = _make_backend(adapter_models={"requirement-check": _ADAPTER_TAG})
    mock_chat = AsyncMock(return_value=_simple_chat_response())
    mock_client = MagicMock()
    mock_client.chat = mock_chat

    with patch.object(
        OllamaModelBackend,
        "_async_client",
        new_callable=PropertyMock,
        return_value=mock_client,
    ):
        mot, _ = await mfuncs.aact(
            ALoraRequirement("The response is correct."),
            _make_context(),
            backend,
            strategy=None,
        )
        await mot.avalue()

    assert mock_chat.call_args.kwargs["model"] == _ADAPTER_TAG
    assert backend.list_adapters() == ["requirement-check_alora"]


async def test_alora_requirement_with_explicit_alora_type_still_resolves(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    """An explicit `adapter_types=(ALORA,)` override doesn't block the resolve.

    `resolve_adapter()` has no way to request a specific type — it always
    prefers aLoRA when the catalog publishes it. An override that asks for
    exactly that is satisfied by a cold resolve, so it shouldn't be treated
    like an override the resolve can't satisfy.
    """
    config_path = tmp_path / "io.yaml"
    config_path.write_text(json.dumps(_SIMPLE_CONFIG), encoding="utf-8")
    monkeypatch.setattr(
        "mellea.backends.ollama.granite_formatters.intrinsics.obtain_io_yaml",
        lambda *_args, **_kwargs: config_path,
    )
    backend = _make_backend(adapter_models={"requirement-check": _ADAPTER_TAG})
    mock_chat = AsyncMock(return_value=_simple_chat_response())
    mock_client = MagicMock()
    mock_client.chat = mock_chat

    with patch.object(
        OllamaModelBackend,
        "_async_client",
        new_callable=PropertyMock,
        return_value=mock_client,
    ):
        mot, _ = await mfuncs.aact(
            ALoraRequirement(
                "The response is correct.", adapter_types=(AdapterType.ALORA,)
            ),
            _make_context(),
            backend,
            strategy=None,
        )
        await mot.avalue()

    assert mock_chat.call_args.kwargs["model"] == _ADAPTER_TAG


async def test_alora_requirement_with_explicit_lora_type_skips_resolve(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    """An explicit override that excludes aLoRA does not trigger a cold resolve.

    `resolve_adapter()` can't be steered to a specific type, so resolving
    here could register the wrong one; falling back to regular generation is
    correct instead.
    """
    config_path = tmp_path / "io.yaml"
    config_path.write_text(json.dumps(_SIMPLE_CONFIG), encoding="utf-8")
    resolve_calls: list[str] = []

    def _fake_obtain_io_yaml(name, *_args, **_kwargs) -> Path:
        resolve_calls.append(name)
        return config_path

    monkeypatch.setattr(
        "mellea.backends.ollama.granite_formatters.intrinsics.obtain_io_yaml",
        _fake_obtain_io_yaml,
    )
    backend = _make_backend(adapter_models={"requirement-check": _ADAPTER_TAG})
    action = ALoraRequirement(
        "The response is correct.", adapter_types=(AdapterType.LORA,)
    )
    ctx = _make_context()

    with patch.object(
        OllamaModelBackend, "generate_from_chat_context", new_callable=AsyncMock
    ) as mock_standard:
        mock_standard.return_value = MagicMock()
        await backend._generate_from_context(action, ctx, model_options={})

    assert resolve_calls == []
    assert backend.list_adapters() == []
    mock_standard.assert_awaited_once()


async def test_llmaj_requirement_never_attempts_adapter_resolve():
    """LLMaJRequirement always falls back to LLM-as-judge, so it must never

    pay for a resolve_adapter() download that its result would just discard.
    """
    backend = _make_backend(adapter_models={"requirement-check": _ADAPTER_TAG})
    action = LLMaJRequirement("The response is correct.")
    ctx = _make_context()

    with (
        patch.object(
            backend, "resolve_adapter", side_effect=AssertionError("should not resolve")
        ) as mock_resolve,
        patch.object(
            OllamaModelBackend, "generate_from_chat_context", new_callable=AsyncMock
        ) as mock_standard,
    ):
        mock_standard.return_value = MagicMock()
        await backend._generate_from_context(action, ctx, model_options={})

    mock_resolve.assert_not_called()
    mock_standard.assert_awaited_once()


async def test_requirement_reroute_falls_back_when_resolve_fails():
    """A resolve_adapter() failure during automatic rerouting must not fail the call."""
    backend = _make_backend(adapter_models={"requirement-check": _ADAPTER_TAG})
    action = Requirement("Be polite.")
    ctx = _make_context()

    with (
        patch.object(
            backend, "resolve_adapter", side_effect=RuntimeError("network error")
        ),
        patch.object(
            OllamaModelBackend, "generate_from_chat_context", new_callable=AsyncMock
        ) as mock_standard,
    ):
        mock_standard.return_value = MagicMock()
        await backend._generate_from_context(action, ctx, model_options={})

    # Fell back to regular generation instead of propagating the failure.
    mock_standard.assert_awaited_once()


async def test_requirement_reroute_falls_back_when_warm_adapter_has_no_tag():
    """A warm-added adapter with no adapter_models entry degrades to LLMaJ.

    The adapter is already registered via add_adapter() (so the cold-resolve
    gate never runs), but no Ollama model tag is configured for it. Without
    this check, reroute_to_alora would stay True, _generate_from_intrinsic
    would hit its own ValueError for the missing tag, and that exception
    would propagate out of validate() instead of degrading like the
    not-yet-added case does.
    """
    backend = _make_backend_with_adapter(_SIMPLE_CONFIG, adapter_models={})
    action = ALoraRequirement("The response is correct.", "uncertainty")
    ctx = _make_context()

    with patch.object(
        OllamaModelBackend, "generate_from_chat_context", new_callable=AsyncMock
    ) as mock_standard:
        mock_standard.return_value = MagicMock()
        await backend._generate_from_context(action, ctx, model_options={})

    mock_standard.assert_awaited_once()


async def test_result_processor_applied():
    """Full uncertainty config: likelihood + project transforms produce the expected JSON."""
    backend = _make_backend_with_adapter(_UNCERTAINTY_CONFIG)
    mot, _ = await _run_intrinsic(backend, _uncertainty_chat_response())

    parsed = json.loads(mot.value)
    assert list(parsed.keys()) == ["certainty"]
    score = parsed["certainty"]
    assert isinstance(score, float)
    # Expected value over {9: 0.95, 4: 0.45} weighted by exp(logprob); 9 dominates.
    assert 0.9 < score < 0.95


async def test_io_yaml_parameters_forwarded():
    """io.yaml max_completion_tokens and temperature reach Ollama's options."""
    backend = _make_backend_with_adapter(_SIMPLE_CONFIG)
    _, mock_chat = await _run_intrinsic(backend, _simple_chat_response())

    options = mock_chat.call_args.kwargs["options"]
    assert options["num_predict"] == 64
    assert options["temperature"] == 1.0


async def test_model_options_override_io_yaml_defaults():
    """User-provided temperature overrides the io.yaml default; other defaults remain."""
    backend = _make_backend_with_adapter(_SIMPLE_CONFIG)
    _, mock_chat = await _run_intrinsic(
        backend,
        _simple_chat_response(),
        model_options={ModelOption.TEMPERATURE: 0.5, ModelOption.SEED: 42},
    )

    options = mock_chat.call_args.kwargs["options"]
    assert options["temperature"] == 0.5
    assert options["seed"] == 42
    assert options["num_predict"] == 64


async def test_no_adapter_raises_valueerror():
    """Calling an intrinsic with no registered adapter raises ValueError."""
    backend = _make_backend()

    with pytest.raises(ValueError, match="has no adapter"):
        await mfuncs.aact(
            Intrinsic("uncertainty"), _make_context(), backend, strategy=None
        )


async def test_streaming_raises():
    """Intrinsics do not support streaming, so this raises NotImplementedError."""
    backend = _make_backend_with_adapter(_SIMPLE_CONFIG)

    with pytest.raises(NotImplementedError, match="do not support streaming"):
        await mfuncs.aact(
            Intrinsic("uncertainty"),
            _make_context(),
            backend,
            strategy=None,
            model_options={ModelOption.STREAM: True},
        )


async def test_tools_passed_to_api():
    """Tools are forwarded to the chat call when tool_calls=True."""
    from mellea.backends.tools import MelleaTool

    def get_temperature(location: str) -> int:
        """Returns the temperature of a city.

        Args:
            location: A city name.
        """
        return 21

    backend = _make_backend_with_adapter(_SIMPLE_CONFIG)
    _, mock_chat = await _run_intrinsic(
        backend,
        _simple_chat_response(),
        tool_calls=True,
        model_options={ModelOption.TOOLS: [MelleaTool.from_callable(get_temperature)]},
    )

    tools = mock_chat.call_args.kwargs["tools"]
    assert len(tools) == 1
    assert tools[0]["function"]["name"] == "get_temperature"


# ---------------------------------------------------------------------------
# Adapter registration and response conversion
# ---------------------------------------------------------------------------


def test_add_adapter_registers_server_mediated_adapter():
    backend = _make_backend_with_adapter(_SIMPLE_CONFIG)
    assert backend.list_adapters() == ["uncertainty_alora"]


def test_add_adapter_rejects_other_adapter_types():
    backend = _make_backend()
    with pytest.raises(TypeError, match="ServerMediatedBinding"):
        backend.add_adapter(object())  # type: ignore[arg-type]


def test_add_adapter_requires_io_yaml_config():
    backend = _make_backend()
    adapter = Adapter(
        identity=Identity(name="uncertainty", adapter_type="alora"),
        io_contract=get_io_contract("uncertainty"),
        weights=ServerMediatedBinding(),
    )

    with pytest.raises(ValueError, match=r"No io\.yaml config"):
        backend.add_adapter(adapter)


def test_resolve_adapter_requires_configured_model_tag():
    """Resolving without a configured tag raises instead of silently succeeding.

    A cold resolve with no `adapter_models` entry used to register the io.yaml
    anyway; generation would then fall back to the base `model_id`, producing a
    schema-valid score from a model with no adapter weights.
    """
    backend = _make_backend()

    with pytest.raises(ValueError, match=r"No Ollama model tag configured"):
        backend.resolve_adapter("uncertainty")


def test_resolve_adapter_falls_back_to_lora_when_alora_unavailable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    """citations/hallucination_detection/context-attribution ship LoRA only."""
    config_path = tmp_path / "io.yaml"
    config_path.write_text(json.dumps(_SIMPLE_CONFIG), encoding="utf-8")
    backend = _make_backend(adapter_models={"citations": _ADAPTER_TAG})
    recorded_alora: list[bool] = []

    def _fake_obtain_io_yaml(*_args, alora: bool, **_kwargs) -> Path:
        recorded_alora.append(alora)
        return config_path

    monkeypatch.setattr(
        "mellea.backends.ollama.granite_formatters.intrinsics.obtain_io_yaml",
        _fake_obtain_io_yaml,
    )

    adapter = backend.resolve_adapter("citations")

    assert recorded_alora == [False]
    assert adapter.identity.adapter_type == "lora"


def test_resolve_adapter_registers_server_mediated_adapter(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    """Resolving a capability creates the modern adapter shape without a server call."""
    config_path = tmp_path / "io.yaml"
    config_path.write_text(json.dumps(_SIMPLE_CONFIG), encoding="utf-8")
    backend = _make_backend(adapter_models={"uncertainty": _ADAPTER_TAG})
    monkeypatch.setattr(
        "mellea.backends.ollama.granite_formatters.intrinsics.obtain_io_yaml",
        lambda *_args, **_kwargs: config_path,
    )

    adapter = backend.resolve_adapter("uncertainty")

    assert adapter.identity.name == "uncertainty"
    assert adapter.identity.adapter_type == "alora"
    assert isinstance(adapter.weights, ServerMediatedBinding)
    assert backend.list_adapters() == ["uncertainty_alora"]


def test_resolve_adapter_coalesces_concurrent_cold_resolves(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    """Concurrent resolve_adapter() calls for the same name share one cold resolve.

    Without coalescing, each concurrent caller would independently
    re-download and re-parse the same `io.yaml`.
    """
    config_path = tmp_path / "io.yaml"
    config_path.write_text(json.dumps(_SIMPLE_CONFIG), encoding="utf-8")
    backend = _make_backend(adapter_models={"uncertainty": _ADAPTER_TAG})

    call_count = 0
    in_flight = 0
    max_in_flight = 0
    counters_lock = threading.Lock()

    def _fake_obtain_io_yaml(*_args, **_kwargs) -> Path:
        nonlocal call_count, in_flight, max_in_flight
        with counters_lock:
            call_count += 1
            in_flight += 1
            max_in_flight = max(max_in_flight, in_flight)
        time.sleep(0.05)  # give a second, uncoalesced caller a chance to overlap
        with counters_lock:
            in_flight -= 1
        return config_path

    monkeypatch.setattr(
        "mellea.backends.ollama.granite_formatters.intrinsics.obtain_io_yaml",
        _fake_obtain_io_yaml,
    )

    results: list = []
    errors: list[BaseException] = []

    def _resolve():
        try:
            results.append(backend.resolve_adapter("uncertainty"))
        except BaseException as e:
            errors.append(e)

    threads = [threading.Thread(target=_resolve) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5)

    assert not errors, errors
    # Coalesced onto a single cold resolve: only one download, and never two
    # threads inside the download body at the same time.
    assert call_count == 1
    assert max_in_flight == 1
    assert len(results) == 2
    assert results[0] is results[1]


def test_resolve_adapter_explains_missing_huggingface_extra(
    monkeypatch: pytest.MonkeyPatch,
):
    """The optional dependency failure names the extra users need to install."""
    backend = _make_backend(adapter_models={"uncertainty": _ADAPTER_TAG})
    monkeypatch.setattr(
        "mellea.backends.ollama.granite_formatters.intrinsics.obtain_io_yaml",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            ModuleNotFoundError(name="huggingface_hub")
        ),
    )

    with pytest.raises(ImportError, match=r"uv sync --extra switch"):
        backend.resolve_adapter("uncertainty")


def test_base_model_name_maps_ollama_tag_to_hf_name():
    backend = _make_backend()
    assert backend.base_model_name == "granite-4.1-3b"


def test_base_model_name_unknown_tag_unchanged():
    with (
        patch.object(OllamaModelBackend, "_check_ollama_server", return_value=True),
        patch.object(OllamaModelBackend, "_pull_ollama_model", return_value=True),
        patch("mellea.backends.ollama.ollama.Client", return_value=MagicMock()),
        patch("mellea.backends.ollama.ollama.AsyncClient", return_value=MagicMock()),
    ):
        backend = OllamaModelBackend(model_id="someone/custom:3b")
    assert backend.base_model_name == "someone/custom:3b"


def test_to_chat_completion_dict_with_logprobs():
    result = _to_chat_completion_dict(_uncertainty_chat_response())

    choice = result["choices"][0]
    assert choice["message"]["content"] == '{"score": "9"}'
    assert choice["finish_reason"] == "stop"
    digit = choice["logprobs"]["content"][4]
    assert digit["token"] == "9"
    assert digit["top_logprobs"][1] == {"token": "4", "logprob": -3.0}


def test_to_chat_completion_dict_without_logprobs():
    result = _to_chat_completion_dict(_simple_chat_response())

    assert result["choices"][0]["logprobs"] is None
