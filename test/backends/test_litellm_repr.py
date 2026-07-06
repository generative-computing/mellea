"""Unit tests for LiteLLMBackend.__repr__ and __str__."""

import pytest

pytest.importorskip("litellm", reason="litellm not installed — install mellea[litellm]")

from mellea.backends.litellm import LiteLLMBackend


def test_repr_shows_model_id_and_base_url():
    """Repr includes class name, model_id, and an explicit base_url."""
    backend = LiteLLMBackend(
        model_id="ollama_chat/test-model", base_url="http://localhost:11434/v1"
    )
    r = repr(backend)
    assert "LiteLLMBackend" in r
    assert "ollama_chat/test-model" in r
    assert "http://localhost:11434/v1" in r
    assert "object at 0x" not in r


def test_str_shows_model_id_and_base_url():
    """str() delegates to repr and includes model_id."""
    backend = LiteLLMBackend(
        model_id="ollama_chat/test-model", base_url="http://localhost:11434/v1"
    )
    assert "ollama_chat/test-model" in str(backend)
    assert "object at 0x" not in str(backend)


def test_repr_no_base_url_shows_none():
    """Repr shows base_url=None when no base_url is passed, hiding the default fallback."""
    backend = LiteLLMBackend(model_id="ollama_chat/test-model")
    r = repr(backend)
    assert "LiteLLMBackend" in r
    assert "ollama_chat/test-model" in r
    assert "base_url=None" in r
    assert "localhost:11434" not in r
