"""Unit tests for Ollama backend pure-logic helpers — no Ollama server required.

Covers _simplify_and_merge, _make_backend_specific_and_remove, and
chat_response_delta_merge.
"""

from unittest.mock import MagicMock, patch

import ollama
import pytest

from mellea.backends import ModelOption
from mellea.backends.ollama import OllamaModelBackend, chat_response_delta_merge
from mellea.core import ModelOutputThunk


@pytest.fixture
def backend():
    """Return an OllamaModelBackend with all network calls patched."""
    with (
        patch.object(OllamaModelBackend, "_check_ollama_server", return_value=True),
        patch.object(OllamaModelBackend, "_pull_ollama_model", return_value=True),
        patch("mellea.backends.ollama.ollama.Client", return_value=MagicMock()),
        patch("mellea.backends.ollama.ollama.AsyncClient", return_value=MagicMock()),
    ):
        b = OllamaModelBackend(model_id="granite3.3:8b")
    return b


# --- _simplify_and_merge ---


def test_simplify_and_merge_none_returns_empty_dict(backend):
    result = backend._simplify_and_merge(None)
    assert isinstance(result, dict)


def test_simplify_and_merge_remaps_num_predict(backend):
    result = backend._simplify_and_merge({"num_predict": 128})
    assert ModelOption.MAX_NEW_TOKENS in result
    assert result[ModelOption.MAX_NEW_TOKENS] == 128


def test_simplify_and_merge_remaps_num_ctx(backend):
    result = backend._simplify_and_merge({"num_ctx": 4096})
    assert ModelOption.CONTEXT_WINDOW in result
    assert result[ModelOption.CONTEXT_WINDOW] == 4096


def test_simplify_and_merge_per_call_overrides_backend(backend):
    # Any per-call value should take precedence
    result = backend._simplify_and_merge({"num_predict": 256})
    assert result[ModelOption.MAX_NEW_TOKENS] == 256


# --- _make_backend_specific_and_remove ---


def test_make_backend_specific_remaps_max_new_tokens(backend):
    opts = {ModelOption.MAX_NEW_TOKENS: 64}
    result = backend._make_backend_specific_and_remove(opts)
    assert "num_predict" in result
    assert result["num_predict"] == 64


def test_make_backend_specific_remaps_context_window(backend):
    opts = {ModelOption.CONTEXT_WINDOW: 8192}
    result = backend._make_backend_specific_and_remove(opts)
    assert "num_ctx" in result
    assert result["num_ctx"] == 8192


def test_make_backend_specific_removes_sentinel_keys(backend):
    opts = {ModelOption.MAX_NEW_TOKENS: 32, ModelOption.SYSTEM_PROMPT: "sys"}
    result = backend._make_backend_specific_and_remove(opts)
    # Sentinel keys not in from_mellea_model_opts_map should be removed
    assert ModelOption.SYSTEM_PROMPT not in result


def test_make_backend_specific_seed_preserved(backend):
    opts = {ModelOption.SEED: 42}
    result = backend._make_backend_specific_and_remove(opts)
    assert "seed" in result
    assert result["seed"] == 42


# --- chat_response_delta_merge ---


def _make_delta(
    content: str,
    role: str = "assistant",
    done: bool = False,
    thinking: str | None = None,
) -> ollama.ChatResponse:
    msg = ollama.Message(role=role, content=content, thinking=thinking)
    return ollama.ChatResponse(model="test", created_at=None, message=msg, done=done)


def test_delta_merge_first_sets_chat_response():
    mot = ModelOutputThunk(value=None)
    delta = _make_delta("Hello")
    chat_response_delta_merge(mot, delta)
    assert mot._meta["chat_response"] is delta


def test_delta_merge_second_appends_content():
    mot = ModelOutputThunk(value=None)
    chat_response_delta_merge(mot, _make_delta("Hello"))
    chat_response_delta_merge(mot, _make_delta(" world"))
    assert mot._meta["chat_response"].message.content == "Hello world"


def test_delta_merge_done_propagated():
    mot = ModelOutputThunk(value=None)
    chat_response_delta_merge(mot, _make_delta("partial", done=False))
    chat_response_delta_merge(mot, _make_delta("", done=True))
    assert mot._meta["chat_response"].done is True


def test_delta_merge_role_set_from_first_delta():
    mot = ModelOutputThunk(value=None)
    chat_response_delta_merge(mot, _make_delta("hi", role="assistant"))
    chat_response_delta_merge(mot, _make_delta(" there", role=""))
    assert mot._meta["chat_response"].message.role == "assistant"


def test_delta_merge_thinking_concatenated():
    mot = ModelOutputThunk(value=None)
    chat_response_delta_merge(mot, _make_delta("reply", thinking="step 1"))
    chat_response_delta_merge(mot, _make_delta("", thinking=" step 2"))
    assert mot._meta["chat_response"].message.thinking == "step 1 step 2"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
