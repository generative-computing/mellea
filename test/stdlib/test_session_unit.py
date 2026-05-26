"""Unit tests for session.py pure-logic — no Ollama server required.

Covers backend_name_to_class factory resolution, get_session error path,
_resolve_context, context_type on start_session, and start_backend().
"""

from unittest.mock import MagicMock, patch

import pytest

from mellea import start_backend
from mellea.backends.ollama import OllamaModelBackend
from mellea.backends.openai import OpenAIBackend
from mellea.stdlib.context import ChatContext, SimpleContext
from mellea.stdlib.session import (
    _resolve_context,
    backend_name_to_class,
    get_session,
    start_session,
)

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


# --- _resolve_context ---


def test_resolve_context_default_is_simple():
    ctx = _resolve_context(None, None)
    assert isinstance(ctx, SimpleContext)


def test_resolve_context_type_simple():
    ctx = _resolve_context(None, "simple")
    assert isinstance(ctx, SimpleContext)


def test_resolve_context_type_chat():
    ctx = _resolve_context(None, "chat")
    assert isinstance(ctx, ChatContext)


def test_resolve_context_explicit_ctx_passed_through():
    explicit = ChatContext()
    ctx = _resolve_context(explicit, None)
    assert ctx is explicit


def test_resolve_context_both_raises():
    with pytest.raises(ValueError, match="Cannot specify both"):
        _resolve_context(ChatContext(), "simple")


# --- start_backend ---


@patch("mellea.stdlib.start_backend.backend_name_to_class")
def test_start_backend_returns_context_and_backend(mock_bn2c: MagicMock):
    mock_backend_cls = MagicMock()
    mock_bn2c.return_value = mock_backend_cls

    ctx, backend = start_backend("ollama", "some-model")

    assert isinstance(ctx, SimpleContext)
    mock_backend_cls.assert_called_once()
    assert backend is mock_backend_cls.return_value


@patch("mellea.stdlib.start_backend.backend_name_to_class")
def test_start_backend_context_type_chat(mock_bn2c: MagicMock):
    mock_bn2c.return_value = MagicMock()

    ctx, _backend = start_backend("ollama", "some-model", context_type="chat")

    assert isinstance(ctx, ChatContext)


@patch("mellea.stdlib.start_backend.backend_name_to_class")
def test_start_backend_explicit_ctx(mock_bn2c: MagicMock):
    mock_bn2c.return_value = MagicMock()
    explicit = ChatContext()

    ctx, _backend = start_backend("ollama", "some-model", ctx=explicit)

    assert ctx is explicit


def test_start_backend_ctx_and_context_type_raises():
    with pytest.raises(ValueError, match="Cannot specify both"):
        start_backend("ollama", "some-model", ctx=ChatContext(), context_type="simple")


# --- start_session with context_type ---


@patch("mellea.stdlib.session.backend_name_to_class")
def test_start_session_context_type_chat(mock_bn2c: MagicMock):
    mock_bn2c.return_value = MagicMock()

    session = start_session("ollama", "some-model", context_type="chat")

    assert isinstance(session.ctx, ChatContext)


@patch("mellea.stdlib.session.backend_name_to_class")
def test_start_session_context_type_simple(mock_bn2c: MagicMock):
    mock_bn2c.return_value = MagicMock()

    session = start_session("ollama", "some-model", context_type="simple")

    assert isinstance(session.ctx, SimpleContext)


@patch("mellea.stdlib.session.backend_name_to_class")
def test_start_session_context_type_none_default(mock_bn2c: MagicMock):
    mock_bn2c.return_value = MagicMock()

    session = start_session("ollama", "some-model")

    assert isinstance(session.ctx, SimpleContext)


def test_start_session_ctx_and_context_type_raises():
    with pytest.raises(ValueError, match="Cannot specify both"):
        start_session("ollama", "some-model", ctx=ChatContext(), context_type="simple")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
