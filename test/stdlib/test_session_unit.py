"""Unit tests for session.py pure-logic — no Ollama server required.

Covers backend_name_to_class factory resolution and get_session error path.
"""

import pytest

from mellea.backends.ollama import OllamaModelBackend
from mellea.backends.openai import OpenAIBackend
from mellea.stdlib.session import backend_name_to_class, get_session

# --- backend_name_to_class ---


def test_ollama_resolves_to_ollama_backend():
    cls = backend_name_to_class("ollama")
    assert cls is OllamaModelBackend


def test_openai_resolves_to_openai_backend():
    cls = backend_name_to_class("openai")
    assert cls is OpenAIBackend


def test_unknown_name_returns_none():
    cls = backend_name_to_class("does_not_exist")
    assert cls is None


def test_hf_resolves_or_raises_import_error():
    # Either resolves (if mellea[hf] is installed) or raises ImportError with helpful message
    try:
        cls = backend_name_to_class("hf")
        assert cls is not None
    except ImportError as e:
        assert "mellea[hf]" in str(e)


def test_huggingface_alias_same_as_hf():
    # "hf" and "huggingface" should resolve to the same class
    try:
        cls_hf = backend_name_to_class("hf")
        cls_hf_full = backend_name_to_class("huggingface")
        assert cls_hf is cls_hf_full
    except ImportError:
        pass  # OK if mellea[hf] is not installed


def test_litellm_resolves_or_raises_import_error():
    try:
        cls = backend_name_to_class("litellm")
        assert cls is not None
    except ImportError as e:
        assert "mellea[litellm]" in str(e)


# --- get_session ---


def test_get_session_raises_when_no_active_session():
    with pytest.raises(RuntimeError, match="No active session found"):
        get_session()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
