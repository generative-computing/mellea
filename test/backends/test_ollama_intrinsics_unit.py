# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the Ollama backend intrinsic generation path. No server required.

Mocks the Ollama async client to verify that `_generate_from_intrinsic` correctly:
- appends the io.yaml instruction (the adapter's activation text) as the last message
- passes the io.yaml response schema as `format` and requests logprobs
- routes the call to the Ollama model tag registered for the adapter function
- applies the `IntrinsicsResultProcessor` to the raw response
- user-provided model options override io.yaml parameter defaults
- raises when no adapter is registered or streaming is requested
"""

import json
from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch

import ollama
import pytest

from mellea.backends import ModelOption
from mellea.backends.adapters.adapter import AdapterType, IntrinsicAdapter
from mellea.backends.ollama import OllamaModelBackend, _to_chat_completion_dict
from mellea.stdlib import functional as mfuncs
from mellea.stdlib.components import Intrinsic, Message
from mellea.stdlib.context import ChatContext

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

_ADAPTER_TAG = "gabegoodhart/granite4.1-uncertainty:3b"

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
    """Return an OllamaModelBackend with a registered uncertainty adapter."""
    backend = _make_backend(model_options=model_options, adapter_models=adapter_models)
    adapter = IntrinsicAdapter(
        "uncertainty", adapter_type=AdapterType.LORA, config_dict=config
    )
    backend.add_adapter(adapter)
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


async def test_adapter_model_tag_defaults_to_model_id():
    """Adapter functions without a registered tag run against the backend's model."""
    backend = _make_backend_with_adapter(_SIMPLE_CONFIG)
    _, mock_chat = await _run_intrinsic(backend, _simple_chat_response())

    assert mock_chat.call_args.kwargs["model"] == "granite4.1:3b"


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


def test_add_adapter_registers_intrinsic_adapter():
    backend = _make_backend_with_adapter(_SIMPLE_CONFIG)
    assert backend.list_adapters() == ["uncertainty_lora"]


def test_add_adapter_rejects_other_adapter_types():
    backend = _make_backend()
    with pytest.raises(TypeError, match="only supports IntrinsicAdapter"):
        backend.add_adapter(object())  # type: ignore[arg-type]


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
