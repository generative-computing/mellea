# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for OpenAI backend pure-logic helpers — no API calls required.

Covers filter_openai_client_kwargs, filter_chat_completions_kwargs,
_simplify_and_merge, and _make_backend_specific_and_remove.
"""

import os
from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch

import pytest
from openai.types import Completion
from openai.types.chat import ChatCompletion, ChatCompletionChunk, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice
from openai.types.completion_choice import CompletionChoice

from mellea.backends import ModelOption, model_ids
from mellea.backends.openai import OpenAIBackend
from mellea.core.base import ModelOutputThunk


def _make_backend(model_options: dict | None = None) -> OpenAIBackend:
    """Return an OpenAIBackend with a fake API key."""
    return OpenAIBackend(
        model_id="gpt-4o",
        api_key="fake-key",
        base_url="http://localhost:9999/v1",
        model_options=model_options,
    )


@pytest.fixture
def backend():
    """Return an OpenAIBackend with no pre-set model options."""
    return _make_backend()


# --- __repr__ / __str__ ---


def test_repr_masks_api_key():
    backend = _make_backend()
    r = repr(backend)
    assert "fake-key" not in r
    assert "***" in r


def test_repr_no_key_shows_none(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("OPENAI_API_KEY", "env-key")
    backend = OpenAIBackend(
        model_id="gpt-4o", api_key=None, base_url="http://localhost:9999/v1"
    )
    r = repr(backend)
    assert "env-key" not in r
    assert "***" not in r
    assert "_api_key=None" in r


def test_str_masks_api_key():
    backend = _make_backend()
    assert "fake-key" not in str(backend)
    assert "***" in str(backend)


# --- model_id resolution ---


def test_model_identifier_resolves_to_openai_name():
    backend = OpenAIBackend(
        model_id=model_ids.NVIDIA_NEMOTRON_3_SUPER_120B_A12B,
        api_key="fake-key",
        base_url="http://localhost:9999/v1",
    )
    assert backend._model_id == model_ids.NVIDIA_NEMOTRON_3_SUPER_120B_A12B.openai_name


def test_model_identifier_without_openai_name_raises():
    # Local-inference-only constants have no name for a hosted endpoint.
    assert model_ids.NVIDIA_NEMOTRON_NANO_12B_V2.openai_name is None
    with pytest.raises(ValueError, match="openai_name"):
        OpenAIBackend(
            model_id=model_ids.NVIDIA_NEMOTRON_NANO_12B_V2,
            api_key="fake-key",
            base_url="http://localhost:9999/v1",
        )


def test_model_id_none_raises_type_error():
    # Forwarding an unset ModelIdentifier field (e.g. `.hf_model_name`) yields None;
    # it must fail here rather than as a missing _model_id at generation time.
    with pytest.raises(TypeError, match="must be a str or ModelIdentifier"):
        OpenAIBackend(
            model_id=None,  # type: ignore[arg-type]
            api_key="fake-key",
            base_url="http://localhost:9999/v1",
        )


def test_model_id_empty_string_raises_value_error():
    with pytest.raises(ValueError, match="empty string"):
        OpenAIBackend(
            model_id="  ", api_key="fake-key", base_url="http://localhost:9999/v1"
        )


# --- filter_openai_client_kwargs ---


def test_filter_openai_client_kwargs_removes_unknown():
    result = OpenAIBackend.filter_openai_client_kwargs(
        api_key="sk-test", unknown_param="x"
    )
    assert "api_key" in result
    assert "unknown_param" not in result


def test_filter_openai_client_kwargs_known_params():
    result = OpenAIBackend.filter_openai_client_kwargs(
        api_key="sk-test", base_url="http://localhost", timeout=30
    )
    assert "api_key" in result
    assert "base_url" in result


def test_filter_openai_client_kwargs_empty():
    result = OpenAIBackend.filter_openai_client_kwargs()
    assert result == {}


# --- filter_chat_completions_kwargs ---


def test_filter_chat_completions_keeps_valid_params(backend):
    result = backend.filter_chat_completions_kwargs(
        {"model": "gpt-4o", "temperature": 0.7, "unknown_option": True}
    )
    assert "model" in result
    assert "temperature" in result
    assert "unknown_option" not in result


def test_filter_chat_completions_empty(backend):
    result = backend.filter_chat_completions_kwargs({})
    assert result == {}


def test_filter_chat_completions_max_tokens(backend):
    result = backend.filter_chat_completions_kwargs({"max_completion_tokens": 100})
    assert "max_completion_tokens" in result


# --- Map consistency ---


@pytest.mark.parametrize("context", ["chats", "completions"])
def test_from_mellea_keys_are_subset_of_to_mellea_values(backend, context):
    """Every key in from_mellea must appear as a value in to_mellea (maps agree)."""
    to_map = getattr(backend, f"to_mellea_model_opts_map_{context}")
    from_map = getattr(backend, f"from_mellea_model_opts_map_{context}")
    to_values = set(to_map.values())
    from_keys = set(from_map.keys())
    assert from_keys <= to_values, (
        f"from_mellea_{context} has keys absent from to_mellea values: {from_keys - to_values}"
    )


# --- _simplify_and_merge ---


def test_simplify_and_merge_none_returns_empty_dict(backend):
    result = backend._simplify_and_merge(None, is_chat_context=True)
    assert result == {}


@pytest.mark.parametrize("context", ["chats", "completions"])
def test_simplify_and_merge_all_to_mellea_entries(backend, context):
    """Every to_mellea entry remaps to its ModelOption via _simplify_and_merge."""
    is_chat = context == "chats"
    to_map = getattr(backend, f"to_mellea_model_opts_map_{context}")
    for backend_key, mellea_key in to_map.items():
        # STOP_SEQUENCES is validated as list[str]; other sentinels accept anything.
        value = ["STOP"] if mellea_key == ModelOption.STOP_SEQUENCES else 42
        result = backend._simplify_and_merge(
            {backend_key: value}, is_chat_context=is_chat
        )
        assert mellea_key in result, f"{backend_key!r} did not produce {mellea_key!r}"
        assert result[mellea_key] == value


def test_simplify_and_merge_remaps_max_completion_tokens(backend):
    """Hardcoded anchor: the critical chat API mapping for generation length."""
    result = backend._simplify_and_merge(
        {"max_completion_tokens": 256}, is_chat_context=True
    )
    assert ModelOption.MAX_NEW_TOKENS in result
    assert result[ModelOption.MAX_NEW_TOKENS] == 256


def test_simplify_and_merge_completions_remaps_max_tokens(backend):
    """Hardcoded anchor: completions API uses a different key for the same sentinel."""
    result = backend._simplify_and_merge({"max_tokens": 100}, is_chat_context=False)
    assert ModelOption.MAX_NEW_TOKENS in result
    assert result[ModelOption.MAX_NEW_TOKENS] == 100


def test_simplify_and_merge_per_call_overrides_backend():
    # Backend sets max_completion_tokens=128; per-call value of 512 must win.
    b = _make_backend(model_options={"max_completion_tokens": 128})
    result = b._simplify_and_merge({"max_completion_tokens": 512}, is_chat_context=True)
    assert result[ModelOption.MAX_NEW_TOKENS] == 512


@pytest.mark.parametrize("is_chat_context", [True, False])
def test_simplify_and_merge_rejects_model_option(
    backend: OpenAIBackend, is_chat_context: bool
) -> None:
    """Regression (#1575): model selection is rejected for both OpenAI APIs.

    Without this check, the raw `model` option reaches the API call alongside
    OpenAIBackend's fixed model argument and raises a duplicate-keyword error.
    """
    with pytest.raises(ValueError, match="model cannot be set via model_options"):
        backend._simplify_and_merge(
            {"model": "other-model"}, is_chat_context=is_chat_context
        )


async def test_openai_backend_rejects_model_option_on_standard_chat_path():
    """Regression (#1575): standard chat rejects per-call model selection.

    Without this check, the user-supplied `model` collides with the backend's
    fixed `model` keyword argument at the OpenAI client call site.
    """
    from mellea.core.base import CBlock
    from mellea.stdlib.context import ChatContext

    backend = _make_backend()

    with pytest.raises(ValueError, match="model cannot be set via model_options"):
        await backend.generate_from_chat_context(
            CBlock(value="hello"), ChatContext(), model_options={"model": "other-model"}
        )


# --- _make_backend_specific_and_remove ---


@pytest.mark.parametrize("context", ["chats", "completions"])
def test_make_backend_specific_all_from_mellea_entries(backend, context):
    """Every from_mellea entry remaps to its backend key via _make_backend_specific_and_remove."""
    is_chat = context == "chats"
    from_map = getattr(backend, f"from_mellea_model_opts_map_{context}")
    for mellea_key, backend_key in from_map.items():
        result = backend._make_backend_specific_and_remove(
            {mellea_key: 42}, is_chat_context=is_chat
        )
        assert backend_key in result, f"{mellea_key!r} did not produce {backend_key!r}"
        assert result[backend_key] == 42


def test_make_backend_specific_chat_remaps_max_new_tokens(backend):
    """Hardcoded anchor: chat API maps MAX_NEW_TOKENS → max_completion_tokens."""
    opts = {ModelOption.MAX_NEW_TOKENS: 200}
    result = backend._make_backend_specific_and_remove(opts, is_chat_context=True)
    assert "max_completion_tokens" in result
    assert result["max_completion_tokens"] == 200


def test_make_backend_specific_completions_remaps_max_new_tokens(backend):
    """Hardcoded anchor: completions API maps MAX_NEW_TOKENS → max_tokens."""
    opts = {ModelOption.MAX_NEW_TOKENS: 100}
    result = backend._make_backend_specific_and_remove(opts, is_chat_context=False)
    assert "max_tokens" in result
    assert result["max_tokens"] == 100


def test_make_backend_specific_unknown_mellea_keys_removed(backend):
    opts = {ModelOption.TOOLS: ["tool1"], ModelOption.SYSTEM_PROMPT: "sys"}
    result = backend._make_backend_specific_and_remove(opts, is_chat_context=True)
    # SYSTEM_PROMPT has no from_mellea mapping — should be removed
    assert ModelOption.SYSTEM_PROMPT not in result


# --- processing(): reasoning / thinking trace extraction ---


def _vllm_chat_completion(reasoning: str, content: str | None) -> ChatCompletion:
    """Build a ChatCompletion that matches vLLM's thinking-model response shape."""
    message = ChatCompletionMessage.model_validate(
        {"role": "assistant", "content": content, "reasoning": reasoning}
    )
    return ChatCompletion(
        id="vllm-test",
        created=0,
        model="qwen3",
        object="chat.completion",
        choices=[Choice(index=0, finish_reason="stop", message=message)],
    )


async def test_processing_captures_vllm_reasoning_field(backend):
    """Non-streaming: mot.thinking captures the raw `reasoning` key from vLLM."""
    mot: ModelOutputThunk = ModelOutputThunk(value=None)
    chunk = _vllm_chat_completion(reasoning="2 + 2 equals 4.", content="4")
    # Sanity check: the SDK object does not expose reasoning_content
    assert not hasattr(chunk.choices[0].message, "reasoning_content")

    await backend.processing(mot, chunk)

    assert mot.thinking == "2 + 2 equals 4."
    assert mot._underlying_value == "4"


async def test_processing_vllm_reasoning_with_null_content(backend):
    """Non-streaming: reasoning is captured even when `content` is null."""
    mot: ModelOutputThunk = ModelOutputThunk(value=None)
    chunk = _vllm_chat_completion(reasoning="some thinking", content=None)

    await backend.processing(mot, chunk)

    assert mot.thinking == "some thinking"
    assert mot._underlying_value == ""


async def test_processing_streaming_captures_vllm_reasoning_field(backend):
    """Streaming: per-chunk `reasoning` deltas accumulate into mot.thinking."""
    mot: ModelOutputThunk = ModelOutputThunk(value=None)
    chunk_a = ChatCompletionChunk.model_validate(
        {
            "id": "vllm-stream",
            "created": 0,
            "model": "qwen3",
            "object": "chat.completion.chunk",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "role": "assistant",
                        "content": None,
                        "reasoning": "first ",
                    },
                    "finish_reason": None,
                }
            ],
        }
    )
    chunk_b = ChatCompletionChunk.model_validate(
        {
            "id": "vllm-stream",
            "created": 0,
            "model": "qwen3",
            "object": "chat.completion.chunk",
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": "ans", "reasoning": "second"},
                    "finish_reason": None,
                }
            ],
        }
    )

    await backend.processing(mot, chunk_a)
    await backend.processing(mot, chunk_b)

    assert mot.thinking == "first second"
    assert mot._underlying_value == "ans"


async def test_processing_reasoning_content_still_used(backend):
    """Regression guard: the pre-existing `reasoning_content` path is preserved.

    Some providers surface the trace as `reasoning_content` on the message
    object itself. The fix must not regress that path in favour of the raw-dict
    fallback.
    """
    message = ChatCompletionMessage.model_validate(
        {
            "role": "assistant",
            "content": "answer",
            "reasoning_content": "attribute-style trace",
        }
    )
    chunk = ChatCompletion(
        id="rc-test",
        created=0,
        model="fake",
        object="chat.completion",
        choices=[Choice(index=0, finish_reason="stop", message=message)],
    )
    assert hasattr(chunk.choices[0].message, "reasoning_content")

    mot: ModelOutputThunk = ModelOutputThunk(value=None)
    await backend.processing(mot, chunk)

    assert mot.thinking == "attribute-style trace"
    assert mot._underlying_value == "answer"


async def test_processing_reasoning_content_takes_precedence_over_reasoning(backend):
    """reasoning_content attribute wins when both it and raw `reasoning` are present."""
    message = ChatCompletionMessage.model_validate(
        {
            "role": "assistant",
            "content": "answer",
            "reasoning_content": "attr-trace",
            "reasoning": "raw-trace",
        }
    )
    chunk = ChatCompletion(
        id="prec-test",
        created=0,
        model="fake",
        object="chat.completion",
        choices=[Choice(index=0, finish_reason="stop", message=message)],
    )
    mot: ModelOutputThunk = ModelOutputThunk(value=None)
    await backend.processing(mot, chunk)

    assert mot.thinking == "attr-trace"
    assert mot._underlying_value == "answer"


# --- _merge_user_extra_body ---


def test_merge_user_extra_body_none_returns_base(backend):
    """A missing user extra_body returns a dict equal to base (a copy, not same object)."""
    base = {"documents": ["d"]}
    result = backend._merge_user_extra_body(base, None)
    assert result == {"documents": ["d"]}
    assert (
        result is not base
    )  # always returns a new dict so default_extra_body can be layered


def test_merge_user_extra_body_user_keys_win(backend):
    """User keys overlay the base, and unrelated base keys survive."""
    merged = backend._merge_user_extra_body(
        {"documents": ["d"], "guided_json": {"type": "integer"}},
        {"guided_json": {"type": "string"}},
    )
    assert merged == {"documents": ["d"], "guided_json": {"type": "string"}}


def test_merge_user_extra_body_deep_merges_chat_template_kwargs(backend):
    """chat_template_kwargs merges key-wise rather than being replaced wholesale."""
    merged = backend._merge_user_extra_body(
        {"chat_template_kwargs": {"adapter_name": "answerability"}},
        {"chat_template_kwargs": {"caller_key": "caller-value"}},
    )
    assert merged["chat_template_kwargs"] == {
        "adapter_name": "answerability",
        "caller_key": "caller-value",
    }


def test_merge_user_extra_body_does_not_mutate_inputs(backend):
    """Neither argument is modified; .pop() operates on a copy."""
    base = {"chat_template_kwargs": {"adapter_name": "answerability"}}
    user = {"chat_template_kwargs": {"caller_key": "caller-value"}}
    backend._merge_user_extra_body(base, user)
    assert base == {"chat_template_kwargs": {"adapter_name": "answerability"}}
    assert user == {"chat_template_kwargs": {"caller_key": "caller-value"}}


async def test_generate_from_raw_merges_user_extra_body(backend):
    """The completions path passes one extra_body, not two spreads (#1241)."""
    import pydantic

    from mellea.core.base import CBlock
    from mellea.stdlib.context import ChatContext

    class Answer(pydantic.BaseModel):
        value: int

    mock_create = AsyncMock(
        return_value=Completion(
            id="raw-test",
            created=0,
            model="fake",
            object="text_completion",
            choices=[CompletionChoice(index=0, finish_reason="stop", text="ok")],
        )
    )
    mock_client = MagicMock()
    mock_client.completions.create = mock_create

    with patch.object(
        OpenAIBackend,
        "_async_client",
        new_callable=PropertyMock,
        return_value=mock_client,
    ):
        await backend._generate_from_raw(
            [CBlock(value="what is 1+1?")],
            ChatContext(),
            format=Answer,
            model_options={"extra_body": {"caller_key": "caller-value"}},
        )

    call_kwargs = mock_create.call_args.kwargs
    extra_body = call_kwargs["extra_body"]
    assert extra_body["caller_key"] == "caller-value"
    assert "guided_json" in extra_body or "structured_outputs" in extra_body


# --- #1502: non-OpenAI format= warning only at init ---

_FORMAT_ASSUMPTION = "NOT using the OpenAI platform"


def _info_msgs(mock_logger) -> list[str]:
    return [str(c.args[0]) for c in mock_logger.info.call_args_list if c.args]


def test_non_openai_format_assumption_logged_once_at_init():
    mock_logger = MagicMock()
    with (
        patch(
            "mellea.backends.openai.MelleaLogger.get_logger", return_value=mock_logger
        ),
        patch(
            "mellea.backends.openai.is_vllm_server_with_structured_output",
            return_value=False,
        ),
    ):
        OpenAIBackend(
            model_id="gpt-4o", api_key="fake-key", base_url="http://localhost:9999/v1"
        )
        OpenAIBackend(
            model_id="gpt-4o", api_key="fake-key", base_url="http://localhost:9999/v1"
        )

    matches = [m for m in _info_msgs(mock_logger) if _FORMAT_ASSUMPTION in m]
    assert len(matches) == 2  # once per backend instance


def test_openai_platform_skips_format_assumption_log():
    mock_logger = MagicMock()
    # Unset OPENAI_BASE_URL for this test: after resolving env into _base_url,
    # a leftover non-OpenAI env would make the no-base_url construction log.
    # These backends point at api.openai.com, so mock the vLLM version probe:
    # __init__ calls is_vllm_server_with_structured_output unconditionally,
    # which would otherwise make a real GET to api.openai.com/version.
    with (
        patch(
            "mellea.backends.openai.MelleaLogger.get_logger", return_value=mock_logger
        ),
        patch(
            "mellea.backends.openai.is_vllm_server_with_structured_output",
            return_value=False,
        ),
        patch.dict(os.environ),
    ):
        os.environ.pop("OPENAI_BASE_URL", None)
        OpenAIBackend(
            model_id="gpt-4o", api_key="fake-key", base_url="https://api.openai.com/v1"
        )
        OpenAIBackend(model_id="gpt-4o", api_key="fake-key")

    matches = [m for m in _info_msgs(mock_logger) if _FORMAT_ASSUMPTION in m]
    assert matches == []


def test_format_assumption_log_honors_openai_base_url_env():
    """Env-only non-OpenAI base_url must still classify as non-OpenAI at init."""
    mock_logger = MagicMock()
    with (
        patch(
            "mellea.backends.openai.MelleaLogger.get_logger", return_value=mock_logger
        ),
        patch(
            "mellea.backends.openai.is_vllm_server_with_structured_output",
            return_value=False,
        ),
        patch.dict(
            os.environ, {"OPENAI_BASE_URL": "http://localhost:9999/v1"}, clear=False
        ),
    ):
        OpenAIBackend(model_id="gpt-4o", api_key="fake-key")

    matches = [m for m in _info_msgs(mock_logger) if _FORMAT_ASSUMPTION in m]
    assert len(matches) == 1


async def test_format_assumption_not_relogged_per_generation():
    """#1502: the notice must not repeat on every format= generation."""
    import pydantic

    from mellea.core.base import CBlock
    from mellea.stdlib.context import ChatContext

    class Answer(pydantic.BaseModel):
        value: int

    backend = OpenAIBackend(
        model_id="gpt-4o", api_key="fake-key", base_url="http://localhost:9999/v1"
    )
    ctx = ChatContext().add(CBlock(value="q"))
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = "{}"
    resp.choices[0].message.role = "assistant"

    mock_logger = MagicMock()
    with (
        patch(
            "mellea.backends.openai.MelleaLogger.get_logger", return_value=mock_logger
        ),
        patch.object(
            backend._async_client.chat.completions, "create", new_callable=AsyncMock
        ) as create,
    ):
        create.return_value = resp
        for _ in range(3):
            await backend.generate_from_chat_context(
                CBlock(value="q"), ctx, _format=Answer, model_options={}
            )

    msgs = [str(c.args[0]) for c in mock_logger.info.call_args_list if c.args]
    assert [m for m in msgs if _FORMAT_ASSUMPTION in m] == []


def test_default_extra_body_applied_when_no_per_call_override():
    """Construction-time default_extra_body is present in every merged result."""
    backend = OpenAIBackend(
        model_id="gpt-4o",
        base_url="http://localhost:9999/v1",
        api_key="test-key",
        default_extra_body={"enable_thinking": False},
    )
    merged = backend._merge_user_extra_body({}, None)
    assert merged["enable_thinking"] is False


def test_default_extra_body_overridden_by_per_call():
    """Per-call model_options take priority over construction-time defaults."""
    backend = OpenAIBackend(
        model_id="gpt-4o",
        base_url="http://localhost:9999/v1",
        api_key="test-key",
        default_extra_body={"enable_thinking": False},
    )
    merged = backend._merge_user_extra_body({}, {"enable_thinking": True})
    assert merged["enable_thinking"] is True


def test_default_extra_body_chat_template_kwargs_deep_merged():
    """chat_template_kwargs from default and per-call are merged key-by-key."""
    backend = OpenAIBackend(
        model_id="gpt-4o",
        base_url="http://localhost:9999/v1",
        api_key="test-key",
        default_extra_body={"chat_template_kwargs": {"enable_thinking": True}},
    )
    merged = backend._merge_user_extra_body(
        {"chat_template_kwargs": {"adapter_name": "foo"}}, None
    )
    assert merged["chat_template_kwargs"] == {
        "enable_thinking": True,
        "adapter_name": "foo",
    }


def test_default_extra_body_does_not_mutate_constructor_arg():
    """The dict passed as default_extra_body is not mutated by merge operations."""
    defaults = {"chat_template_kwargs": {"enable_thinking": True}}
    backend = OpenAIBackend(
        model_id="gpt-4o",
        base_url="http://localhost:9999/v1",
        api_key="test-key",
        default_extra_body=defaults,
    )
    backend._merge_user_extra_body({"other_key": "val"}, {"another": "val2"})
    assert defaults == {"chat_template_kwargs": {"enable_thinking": True}}


def test_default_extra_body_chat_template_kwargs_three_way_merge():
    """default_extra_body, base, and per-call user all contribute simultaneously.

    Each layer sets a distinct chat_template_kwargs key (mirroring a real call:
    a construction-time thinking default, Mellea's own adapter-activation write,
    and a per-call override for something else) — all three must survive.
    """
    backend = OpenAIBackend(
        model_id="gpt-4o",
        base_url="http://localhost:9999/v1",
        api_key="test-key",
        default_extra_body={"chat_template_kwargs": {"enable_thinking": True}},
    )
    merged = backend._merge_user_extra_body(
        {"chat_template_kwargs": {"adapter_name": "answerability"}},
        {"chat_template_kwargs": {"thinking_budget": 5000}},
    )
    assert merged["chat_template_kwargs"] == {
        "enable_thinking": True,
        "adapter_name": "answerability",
        "thinking_budget": 5000,
    }


async def test_standard_chat_path_applies_default_extra_body_without_per_call_override():
    """Regression test for #1453: the standard chat path must not silently
    drop `default_extra_body` when the caller passes no per-call `extra_body`.

    `_generate_from_chat_context_standard` used to call `_merge_user_extra_body`
    only when `model_options` carried a per-call `extra_body` override, so a
    construction-time `default_extra_body` never reached the wire on an
    ordinary call — defeating the "set once at construction" point of the
    feature.
    """
    from mellea.core.base import CBlock
    from mellea.stdlib.context import ChatContext

    backend = OpenAIBackend(
        model_id="gpt-4o",
        base_url="http://localhost:9999/v1",
        api_key="test-key",
        default_extra_body={"chat_template_kwargs": {"enable_thinking": True}},
    )

    with patch.object(
        backend._async_client.chat.completions, "create", new_callable=AsyncMock
    ) as mock_create:
        mock_create.return_value = ChatCompletion(
            id="test",
            choices=[
                Choice(
                    finish_reason="stop",
                    index=0,
                    message=ChatCompletionMessage(role="assistant", content="ok"),
                )
            ],
            created=0,
            model="gpt-4o",
            object="chat.completion",
        )
        mot, _ = await backend.generate_from_chat_context(
            CBlock(value="hello"),
            ChatContext(),
            model_options={ModelOption.STREAM: False},
        )
        await mot.avalue()

    call_kwargs = mock_create.call_args.kwargs
    assert call_kwargs["model"] == "gpt-4o"
    assert call_kwargs["extra_body"]["chat_template_kwargs"]["enable_thinking"] is True


async def test_thinking_true_sends_reasoning_effort_medium_and_enable_thinking():
    """THINKING=True sends reasoning_effort="medium" AND
    extra_body.chat_template_kwargs.enable_thinking=True, so each server type
    (OpenAI-style vs vLLM-style) picks up the mechanism it understands."""
    from mellea.core.base import CBlock
    from mellea.stdlib.context import ChatContext

    backend = OpenAIBackend(
        model_id="gpt-4o", base_url="http://localhost:9999/v1", api_key="test-key"
    )

    with patch.object(
        backend._async_client.chat.completions, "create", new_callable=AsyncMock
    ) as mock_create:
        mock_create.return_value = ChatCompletion(
            id="test",
            choices=[
                Choice(
                    finish_reason="stop",
                    index=0,
                    message=ChatCompletionMessage(role="assistant", content="ok"),
                )
            ],
            created=0,
            model="gpt-4o",
            object="chat.completion",
        )
        mot, _ = await backend.generate_from_chat_context(
            CBlock(value="hello"),
            ChatContext(),
            model_options={ModelOption.THINKING: True},
        )
        await mot.avalue()

    call_kwargs = mock_create.call_args.kwargs
    assert call_kwargs["reasoning_effort"] == "medium"
    assert call_kwargs["extra_body"]["chat_template_kwargs"]["enable_thinking"] is True


async def test_thinking_false_sends_reasoning_effort_none_and_disable():
    """THINKING=False sends reasoning_effort="none" AND
    extra_body.chat_template_kwargs.enable_thinking=False.

    Ollama's /v1 endpoint (>= 0.33.1) maps reasoning_effort="none" to
    think=false; absent the param, thinking-capable models (e.g. granite4.2)
    default to thinking on. vLLM picks up the chat_template_kwargs mechanism
    instead. Regression guard for the granite4.2 CI wedge: the /v1 tests
    must be able to turn thinking off (runs 33093698028, 33104419835).
    """
    from mellea.core.base import CBlock
    from mellea.stdlib.context import ChatContext

    backend = OpenAIBackend(
        model_id="granite4.2:3b",
        base_url="http://localhost:9999/v1",
        api_key="test-key",
    )

    with patch.object(
        backend._async_client.chat.completions, "create", new_callable=AsyncMock
    ) as mock_create:
        mock_create.return_value = ChatCompletion(
            id="test",
            choices=[
                Choice(
                    finish_reason="stop",
                    index=0,
                    message=ChatCompletionMessage(role="assistant", content="ok"),
                )
            ],
            created=0,
            model="granite4.2:3b",
            object="chat.completion",
        )
        mot, _ = await backend.generate_from_chat_context(
            CBlock(value="hello"),
            ChatContext(),
            model_options={ModelOption.THINKING: False},
        )
        await mot.avalue()

    call_kwargs = mock_create.call_args.kwargs
    assert call_kwargs["reasoning_effort"] == "none"
    assert call_kwargs["extra_body"]["chat_template_kwargs"]["enable_thinking"] is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
