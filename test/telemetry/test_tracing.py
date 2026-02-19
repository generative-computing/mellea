"""Tests for OpenTelemetry instrumentation."""

import os

import pytest


@pytest.fixture
def enable_app_tracing(monkeypatch):
    """Enable application tracing for tests."""
    monkeypatch.setenv("MELLEA_TRACE_APPLICATION", "true")
    monkeypatch.setenv("MELLEA_TRACE_BACKEND", "false")
    # Force reload of tracing module to pick up env vars
    import importlib

    import mellea.telemetry.tracing

    importlib.reload(mellea.telemetry.tracing)
    yield
    # Reset after test
    monkeypatch.setenv("MELLEA_TRACE_APPLICATION", "false")
    importlib.reload(mellea.telemetry.tracing)


@pytest.fixture
def enable_backend_tracing(monkeypatch):
    """Enable backend tracing for tests."""
    monkeypatch.setenv("MELLEA_TRACE_APPLICATION", "false")
    monkeypatch.setenv("MELLEA_TRACE_BACKEND", "true")
    # Force reload of tracing module to pick up env vars
    import importlib

    import mellea.telemetry.tracing

    importlib.reload(mellea.telemetry.tracing)
    yield
    # Reset after test
    monkeypatch.setenv("MELLEA_TRACE_BACKEND", "false")
    importlib.reload(mellea.telemetry.tracing)


def test_telemetry_disabled_by_default():
    """Test that telemetry is disabled by default."""
    from mellea.telemetry import (
        is_application_tracing_enabled,
        is_backend_tracing_enabled,
    )

    assert not is_application_tracing_enabled()
    assert not is_backend_tracing_enabled()


def test_application_tracing_enabled(enable_app_tracing):
    """Test that application tracing can be enabled."""
    from mellea.telemetry import (
        is_application_tracing_enabled,
        is_backend_tracing_enabled,
    )

    assert is_application_tracing_enabled()
    assert not is_backend_tracing_enabled()


def test_backend_tracing_enabled(enable_backend_tracing):
    """Test that backend tracing can be enabled."""
    from mellea.telemetry import (
        is_application_tracing_enabled,
        is_backend_tracing_enabled,
    )

    assert not is_application_tracing_enabled()
    assert is_backend_tracing_enabled()


def test_trace_application_context_manager():
    """Test that trace_application works as a context manager."""
    from mellea.telemetry import trace_application

    # Should not raise even when tracing is disabled
    with trace_application("test_span", test_attr="value") as span:
        # Span will be None when tracing is disabled
        assert span is None or hasattr(span, "set_attribute")


def test_trace_backend_context_manager():
    """Test that trace_backend works as a context manager."""
    from mellea.telemetry import trace_backend

    # Should not raise even when tracing is disabled
    with trace_backend("test_span", test_attr="value") as span:
        # Span will be None when tracing is disabled
        assert span is None or hasattr(span, "set_attribute")


def test_set_span_attribute_with_none_span():
    """Test that set_span_attribute handles None span gracefully."""
    from mellea.telemetry import set_span_attribute

    # Should not raise when span is None
    set_span_attribute(None, "key", "value")


def test_set_span_error_with_none_span():
    """Test that set_span_error handles None span gracefully."""
    from mellea.telemetry import set_span_error

    # Should not raise when span is None
    exception = ValueError("test error")
    set_span_error(None, exception)


@pytest.mark.ollama
def test_session_with_tracing_disabled():
    """Test that session works normally when tracing is disabled."""
    from mellea import start_session

    with start_session() as m:
        result = m.instruct("Say hello")
        assert result is not None


@pytest.mark.ollama
def test_session_with_application_tracing(enable_app_tracing):
    """Test that session works with application tracing enabled."""
    from mellea import start_session

    # This should create application trace spans
    with start_session() as m:
        result = m.instruct("Say hello")
        assert result is not None


@pytest.mark.ollama
def test_session_with_backend_tracing(enable_backend_tracing):
    """Test that session works with backend tracing enabled."""
    from mellea import start_session

    # This should create backend trace spans
    with start_session() as m:
        result = m.instruct("Say hello")
        assert result is not None


@pytest.mark.ollama
def test_generative_function_with_tracing(enable_app_tracing):
    """Test that @generative functions work with tracing enabled."""
    from mellea import generative, start_session

    @generative
    def classify(text: str) -> str:
        """Classify the text."""

    with start_session() as m:
        result = classify(m, text="test")
        assert result is not None


def test_backend_instrumentation_helpers():
    """Test backend instrumentation helper functions."""
    from mellea.telemetry.backend_instrumentation import (
        get_context_size,
        get_model_id_str,
    )

    # Test with mock objects
    class MockBackend:
        def __init__(self):
            self.model_id = "test-model"

    class MockContext:
        def __init__(self):
            self.turns = [1, 2, 3]

    backend = MockBackend()
    ctx = MockContext()

    assert get_model_id_str(backend) == "test-model"
    assert get_context_size(ctx) == 3


def test_instrument_generate_from_context():
    """Test instrument_generate_from_context helper."""
    from mellea.telemetry.backend_instrumentation import (
        instrument_generate_from_context,
    )

    class MockBackend:
        model_id = "test-model"

    class MockAction:
        pass

    class MockContext:
        turns = []

    backend = MockBackend()
    action = MockAction()
    ctx = MockContext()

    # Should return a context manager
    with instrument_generate_from_context(backend, action, ctx) as span:
        # Span will be None when tracing is disabled
        assert span is None or hasattr(span, "set_attribute")


def test_instrument_generate_from_raw():
    """Test instrument_generate_from_raw helper."""
    from mellea.telemetry.backend_instrumentation import instrument_generate_from_raw

    class MockBackend:
        model_id = "test-model"

    backend = MockBackend()

    # Should return a context manager
    with instrument_generate_from_raw(backend, num_actions=5) as span:
        # Span will be None when tracing is disabled
        assert span is None or hasattr(span, "set_attribute")
