"""Unit tests for OpenAI backend pure-logic helpers — no API calls required.

Covers filter_openai_client_kwargs, filter_chat_completions_kwargs,
_simplify_and_merge, and _make_backend_specific_and_remove.
"""

import pytest

from mellea.backends import ModelOption
from mellea.backends.openai import OpenAIBackend


@pytest.fixture
def backend():
    """Return an OpenAIBackend with a fake API key."""
    return OpenAIBackend(
        model_id="gpt-4o", api_key="fake-key", base_url="http://localhost:9999/v1"
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


# --- _simplify_and_merge ---


def test_simplify_and_merge_none_returns_backend_opts(backend):
    # No per-call options — returns backend's model_options (remapped)
    result = backend._simplify_and_merge(None, is_chat_context=True)
    assert isinstance(result, dict)


def test_simplify_and_merge_remaps_max_tokens(backend):
    result = backend._simplify_and_merge(
        {"max_completion_tokens": 256}, is_chat_context=True
    )
    assert ModelOption.MAX_NEW_TOKENS in result
    assert result[ModelOption.MAX_NEW_TOKENS] == 256


def test_simplify_and_merge_call_opts_override_backend(backend):
    # Per-call max_tokens should override any backend default for max_new_tokens
    result = backend._simplify_and_merge(
        {"max_completion_tokens": 512}, is_chat_context=True
    )
    assert result[ModelOption.MAX_NEW_TOKENS] == 512


def test_simplify_and_merge_completions_api_uses_different_map(backend):
    result = backend._simplify_and_merge({"max_tokens": 100}, is_chat_context=False)
    assert ModelOption.MAX_NEW_TOKENS in result


# --- _make_backend_specific_and_remove ---


def test_make_backend_specific_removes_mellea_keys(backend):
    opts = {ModelOption.MAX_NEW_TOKENS: 200, ModelOption.SEED: 42}
    result = backend._make_backend_specific_and_remove(opts, is_chat_context=True)
    # Mellea sentinel keys should be gone; mapped keys present
    assert ModelOption.MAX_NEW_TOKENS not in result
    assert "max_completion_tokens" in result
    assert "seed" in result


def test_make_backend_specific_completions_uses_max_tokens(backend):
    opts = {ModelOption.MAX_NEW_TOKENS: 100}
    result = backend._make_backend_specific_and_remove(opts, is_chat_context=False)
    assert "max_tokens" in result
    assert result["max_tokens"] == 100


def test_make_backend_specific_unknown_mellea_keys_removed(backend):
    # Keys without a mapping should be stripped by filter_chat_completions_kwargs
    opts = {ModelOption.TOOLS: ["tool1"], ModelOption.SYSTEM_PROMPT: "sys"}
    result = backend._make_backend_specific_and_remove(opts, is_chat_context=True)
    # SYSTEM_PROMPT has no from_mellea mapping for chats — should be removed
    assert ModelOption.SYSTEM_PROMPT not in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
