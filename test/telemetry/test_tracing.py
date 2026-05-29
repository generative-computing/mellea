"""Tests for OpenTelemetry tracing public API."""

import pytest

pytest.importorskip(
    "opentelemetry", reason="opentelemetry not installed — install mellea[telemetry]"
)

from mellea.telemetry import (
    is_content_tracing_enabled,
    is_tracing_enabled,
    set_span_attribute,
    set_span_error,
    trace_application,
    tracing,
)
from mellea.telemetry.tracing import get_backend_tracer


def _reset_tracing_state() -> None:
    """Reset module state and re-run setup so env-var changes take effect."""
    tracing._tracer_provider = None
    tracing._application_tracer = None
    tracing._backend_tracer = None
    tracing._setup_tracing()


@pytest.fixture
def enable_tracing(monkeypatch):
    """Enable tracing for the duration of a test."""
    monkeypatch.setenv("MELLEA_TRACES_ENABLED", "true")
    _reset_tracing_state()
    yield
    _reset_tracing_state()


@pytest.fixture
def disable_tracing(monkeypatch):
    """Ensure tracing is disabled for the duration of a test."""
    monkeypatch.delenv("MELLEA_TRACES_ENABLED", raising=False)
    monkeypatch.delenv("MELLEA_TRACES_OTLP", raising=False)
    monkeypatch.delenv("MELLEA_TRACES_CONSOLE", raising=False)
    _reset_tracing_state()
    yield
    _reset_tracing_state()


def test_telemetry_disabled_by_default(disable_tracing):
    """Test that telemetry is disabled by default."""
    assert not is_tracing_enabled()


def test_tracing_enabled(enable_tracing):
    """Test that tracing can be enabled."""
    assert is_tracing_enabled()


@pytest.mark.parametrize(
    "env, expected",
    [
        ({}, False),
        ({"MELLEA_TRACES_CONTENT": "true"}, True),
        ({"OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT": "true"}, True),
    ],
    ids=["disabled-by-default", "mellea-var", "otel-standard-var"],
)
def test_content_tracing(monkeypatch, env, expected):
    """Content tracing honors both MELLEA_TRACES_CONTENT and the OTel standard var."""
    monkeypatch.delenv("MELLEA_TRACES_CONTENT", raising=False)
    monkeypatch.delenv(
        "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", raising=False
    )
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    _reset_tracing_state()

    assert is_content_tracing_enabled() is expected


def test_otlp_traces_endpoint_honored(monkeypatch):
    """OTEL_EXPORTER_OTLP_TRACES_ENDPOINT should activate the OTLP exporter."""
    monkeypatch.setenv("MELLEA_TRACES_ENABLED", "true")
    monkeypatch.setenv("MELLEA_TRACES_OTLP", "true")
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", "http://localhost:4317")
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
    _reset_tracing_state()

    assert get_backend_tracer() is not None
    _reset_tracing_state()


def test_trace_application_context_manager():
    """Test that trace_application works as a context manager."""
    _reset_tracing_state()

    # Should not raise even when tracing is disabled
    with trace_application("test_span", test_attr="value") as span:
        # Span will be None when tracing is disabled
        assert span is None or hasattr(span, "set_attribute")


def test_set_span_attribute_with_none_span():
    """Test that set_span_attribute handles None span gracefully."""
    # Should not raise when span is None
    set_span_attribute(None, "key", "value")


def test_set_span_error_with_none_span():
    """Test that set_span_error handles None span gracefully."""
    # Should not raise when span is None
    exception = ValueError("test error")
    set_span_error(None, exception)
