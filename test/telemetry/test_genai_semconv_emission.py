"""Unit tests for OTel GenAI semantic convention attribute emission.

Covers: gen_ai.provider.name (gap 1), gen_ai.conversation.id (gap 2),
error.type + ERROR status (gap 4), and the MELLEA_TRACE_CONTENT flag.

All tests use a fake span and do not require a live backend or OTel SDK.
"""

from unittest.mock import MagicMock, patch

from mellea.telemetry.backend_instrumentation import (
    finalize_backend_span,
    get_provider_name,
    get_system_name,
    start_generate_span,
)
from mellea.telemetry.context import with_context
from mellea.telemetry.tracing import add_span_event, is_content_tracing_enabled

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_span() -> MagicMock:
    return MagicMock()


def _fake_backend(class_name: str) -> object:
    return type(class_name, (), {})()


def _span_attrs(span: MagicMock) -> dict:
    """Collect all set_attribute calls into a flat dict."""
    return {call.args[0]: call.args[1] for call in span.set_attribute.call_args_list}


# ---------------------------------------------------------------------------
# Gap 1: gen_ai.provider.name alongside gen_ai.system
# ---------------------------------------------------------------------------


def test_provider_name_equals_system_name():
    backend = _fake_backend("OpenAIBackend")
    assert get_provider_name(backend) == get_system_name(backend) == "openai"


def test_provider_name_emitted_in_start_generate_span():
    """Both gen_ai.system and gen_ai.provider.name should be set on the span."""
    backend = _fake_backend("OpenAIBackend")
    backend.model_id = "gpt-4"  # type: ignore[attr-defined]
    action = MagicMock()

    with patch("mellea.telemetry.tracing.start_backend_span") as mock_start:
        mock_start.return_value = _mock_span()
        start_generate_span(backend, action, ctx=[], format=None, tool_calls=False)

    call_kwargs = mock_start.call_args[1]
    assert call_kwargs.get("gen_ai.system") == "openai"
    assert call_kwargs.get("gen_ai.provider.name") == "openai"


# ---------------------------------------------------------------------------
# Gap 2: gen_ai.conversation.id from session_id ContextVar
# ---------------------------------------------------------------------------


def test_conversation_id_emitted_from_session_id():
    backend = _fake_backend("OpenAIBackend")
    backend.model_id = "gpt-4"  # type: ignore[attr-defined]
    action = MagicMock()

    with with_context(session_id="sess-abc"):
        with patch("mellea.telemetry.tracing.start_backend_span") as mock_start:
            mock_start.return_value = _mock_span()
            start_generate_span(backend, action, ctx=[], format=None, tool_calls=False)

    call_kwargs = mock_start.call_args[1]
    assert call_kwargs.get("gen_ai.conversation.id") == "sess-abc"
    assert call_kwargs.get("mellea.session_id") == "sess-abc"


def test_conversation_id_absent_when_no_session():
    backend = _fake_backend("OpenAIBackend")
    backend.model_id = "gpt-4"  # type: ignore[attr-defined]
    action = MagicMock()

    with patch("mellea.telemetry.tracing.start_backend_span") as mock_start:
        mock_start.return_value = _mock_span()
        start_generate_span(backend, action, ctx=[], format=None, tool_calls=False)

    call_kwargs = mock_start.call_args[1]
    assert "gen_ai.conversation.id" not in call_kwargs


# ---------------------------------------------------------------------------
# Gap 4: ERROR span status + error.type
# ---------------------------------------------------------------------------


def test_error_sets_status_and_error_type():
    span = _mock_span()
    exc = RuntimeError("model rejected")

    with (
        patch(
            "mellea.telemetry.backend_instrumentation.set_span_error"
        ) as mock_set_err,
        patch("mellea.telemetry.backend_instrumentation.end_backend_span") as mock_end,
    ):
        finalize_backend_span(span, error=exc)

    mock_set_err.assert_called_once_with(span, exc)
    assert _span_attrs(span).get("error.type") == "RuntimeError"
    mock_end.assert_called_once_with(span)


def test_error_path_always_closes_span():
    span = _mock_span()
    with patch("mellea.telemetry.backend_instrumentation.set_span_error"):
        with patch(
            "mellea.telemetry.backend_instrumentation.end_backend_span"
        ) as mock_end:
            finalize_backend_span(span, error=ValueError("x"))
    mock_end.assert_called_once()


def test_finalize_never_raises_on_span_error():
    """finalize_backend_span must not propagate exceptions from helpers."""
    span = _mock_span()
    span.set_attribute.side_effect = RuntimeError("span broke")

    with patch("mellea.telemetry.backend_instrumentation.end_backend_span"):
        with patch("mellea.telemetry.backend_instrumentation.set_span_error"):
            finalize_backend_span(span, error=ValueError("test"))


def test_finalize_never_raises_if_end_span_raises():
    """end_backend_span exceptions must not propagate on the error path."""
    span = _mock_span()
    with patch(
        "mellea.telemetry.backend_instrumentation.end_backend_span",
        side_effect=RuntimeError("sdk shutdown"),
    ):
        with patch("mellea.telemetry.backend_instrumentation.set_span_error"):
            finalize_backend_span(span, error=ValueError("original error"))


def test_finalize_none_span_is_noop():
    finalize_backend_span(None, error=RuntimeError("x"))


# ---------------------------------------------------------------------------
# Content tracing default (infrastructure for deferred gap 5)
# ---------------------------------------------------------------------------


def test_content_tracing_disabled_by_default():
    assert not is_content_tracing_enabled()


# ---------------------------------------------------------------------------
# add_span_event helper
# ---------------------------------------------------------------------------


def test_add_span_event_calls_span_add_event():
    span = _mock_span()
    with patch("mellea.telemetry.tracing._OTEL_AVAILABLE", True):
        add_span_event(span, "gen_ai.content.prompt", {"gen_ai.prompt": "hello"})
    span.add_event.assert_called_once_with(
        "gen_ai.content.prompt", attributes={"gen_ai.prompt": "hello"}
    )


def test_add_span_event_none_span_is_noop():
    with patch("mellea.telemetry.tracing._OTEL_AVAILABLE", True):
        add_span_event(None, "gen_ai.content.prompt")


def test_add_span_event_defaults_to_empty_attributes():
    span = _mock_span()
    with patch("mellea.telemetry.tracing._OTEL_AVAILABLE", True):
        add_span_event(span, "gen_ai.content.completion")
    span.add_event.assert_called_once_with("gen_ai.content.completion", attributes={})
