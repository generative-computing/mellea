"""Unit tests for OpenAI backend pure-logic helpers — no API calls required.

Covers filter_openai_client_kwargs, filter_chat_completions_kwargs,
_simplify_and_merge, and _make_backend_specific_and_remove.
"""

import pytest
from openai.types.chat import ChatCompletion, ChatCompletionChunk, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice

from mellea.backends import ModelOption
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
    """Non-streaming: mot.thinking captures the raw ``reasoning`` key from vLLM."""
    mot: ModelOutputThunk = ModelOutputThunk(value=None)
    chunk = _vllm_chat_completion(reasoning="2 + 2 equals 4.", content="4")
    # Sanity check: the SDK object does not expose reasoning_content
    assert not hasattr(chunk.choices[0].message, "reasoning_content")

    await backend.processing(mot, chunk)

    assert mot.thinking == "2 + 2 equals 4."
    assert mot._underlying_value == "4"


async def test_processing_vllm_reasoning_with_null_content(backend):
    """Non-streaming: reasoning is captured even when ``content`` is null."""
    mot: ModelOutputThunk = ModelOutputThunk(value=None)
    chunk = _vllm_chat_completion(reasoning="some thinking", content=None)

    await backend.processing(mot, chunk)

    assert mot.thinking == "some thinking"
    assert mot._underlying_value == ""


async def test_processing_streaming_captures_vllm_reasoning_field(backend):
    """Streaming: per-chunk ``reasoning`` deltas accumulate into mot.thinking."""
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
    """Regression guard: the pre-existing ``reasoning_content`` path is preserved.

    Some providers surface the trace as ``reasoning_content`` on the message
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
    """reasoning_content attribute wins when both it and raw ``reasoning`` are present."""
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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
