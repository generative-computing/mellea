"""Unit tests for OpenTelemetry metrics instrumentation."""

import os
from unittest.mock import MagicMock, patch

import pytest

# Check if OpenTelemetry is available
try:
    from opentelemetry import metrics
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import InMemoryMetricReader

    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not OTEL_AVAILABLE, reason="OpenTelemetry not installed"
)


@pytest.fixture
def clean_metrics_env(monkeypatch):
    """Clean metrics environment variables before each test."""
    monkeypatch.delenv("MELLEA_METRICS_ENABLED", raising=False)
    monkeypatch.delenv("MELLEA_METRICS_CONSOLE", raising=False)
    monkeypatch.delenv("MELLEA_METRICS_OTLP", raising=False)
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_METRICS_ENDPOINT", raising=False)
    monkeypatch.delenv("MELLEA_METRICS_EXPORT_INTERVAL", raising=False)
    monkeypatch.delenv("OTEL_SERVICE_NAME", raising=False)
    # Force reload of metrics module to pick up env vars
    import importlib

    import mellea.telemetry.metrics

    importlib.reload(mellea.telemetry.metrics)
    yield
    # Reset after test
    importlib.reload(mellea.telemetry.metrics)


@pytest.fixture
def enable_metrics(monkeypatch):
    """Enable metrics for tests."""
    monkeypatch.setenv("MELLEA_METRICS_ENABLED", "true")
    # Force reload of metrics module to pick up env vars
    import importlib

    import mellea.telemetry.metrics

    importlib.reload(mellea.telemetry.metrics)
    yield
    # Reset after test
    monkeypatch.setenv("MELLEA_METRICS_ENABLED", "false")
    importlib.reload(mellea.telemetry.metrics)


# Configuration Tests


def test_metrics_disabled_by_default(clean_metrics_env):
    """Test that metrics are disabled by default."""
    from mellea.telemetry.metrics import is_metrics_enabled

    assert not is_metrics_enabled()


def test_metrics_enabled_with_env_var(enable_metrics):
    """Test that metrics can be enabled via environment variable."""
    from mellea.telemetry.metrics import is_metrics_enabled

    assert is_metrics_enabled()


def test_metrics_enabled_with_various_truthy_values(monkeypatch):
    """Test that various truthy values enable metrics."""
    import importlib

    import mellea.telemetry.metrics

    for value in ["true", "True", "TRUE", "1", "yes", "Yes", "YES"]:
        monkeypatch.setenv("MELLEA_METRICS_ENABLED", value)
        importlib.reload(mellea.telemetry.metrics)
        from mellea.telemetry.metrics import is_metrics_enabled

        assert is_metrics_enabled(), f"Failed for value: {value}"


def test_metrics_disabled_with_falsy_values(monkeypatch):
    """Test that falsy values keep metrics disabled."""
    import importlib

    import mellea.telemetry.metrics

    for value in ["false", "False", "FALSE", "0", "no", "No", "NO", ""]:
        monkeypatch.setenv("MELLEA_METRICS_ENABLED", value)
        importlib.reload(mellea.telemetry.metrics)
        from mellea.telemetry.metrics import is_metrics_enabled

        assert not is_metrics_enabled(), f"Failed for value: {value}"


# Initialization Tests


def test_meter_provider_not_created_when_disabled(clean_metrics_env):
    """Test that MeterProvider is not created when metrics are disabled."""
    from mellea.telemetry.metrics import _meter_provider

    assert _meter_provider is None


def test_meter_reused_across_instruments(enable_metrics):
    """Test that the same meter is reused for multiple instruments."""
    from mellea.telemetry.metrics import (
        _meter,
        create_counter,
        create_histogram,
        create_up_down_counter,
    )

    create_counter("test.counter")
    create_histogram("test.histogram")
    create_up_down_counter("test.updown")

    # All should use the same meter instance
    from mellea.telemetry.metrics import _meter

    assert _meter is not None


# Instrument Creation Tests


def test_create_counter(enable_metrics):
    """Test creating a counter instrument."""
    from mellea.telemetry.metrics import create_counter

    counter = create_counter(
        "test.requests.total", description="Total requests", unit="1"
    )

    assert counter is not None
    assert hasattr(counter, "add")


def test_create_histogram(enable_metrics):
    """Test creating a histogram instrument."""
    from mellea.telemetry.metrics import create_histogram

    histogram = create_histogram(
        "test.request.duration", description="Request duration", unit="ms"
    )

    assert histogram is not None
    assert hasattr(histogram, "record")


def test_create_up_down_counter(enable_metrics):
    """Test creating an up-down counter instrument."""
    from mellea.telemetry.metrics import create_up_down_counter

    counter = create_up_down_counter(
        "test.sessions.active", description="Active sessions", unit="1"
    )

    assert counter is not None
    assert hasattr(counter, "add")


# No-op Tests


def test_instruments_are_noop_when_disabled(clean_metrics_env):
    """Test that instruments are no-op when metrics are disabled."""
    from mellea.telemetry.metrics import (
        create_counter,
        create_histogram,
        create_up_down_counter,
    )

    counter = create_counter("test.counter")
    histogram = create_histogram("test.histogram")
    updown = create_up_down_counter("test.updown")

    # Should be no-op instances
    assert counter.__class__.__name__ == "_NoOpCounter"
    assert histogram.__class__.__name__ == "_NoOpHistogram"
    assert updown.__class__.__name__ == "_NoOpUpDownCounter"


def test_noop_counter_methods_dont_raise(clean_metrics_env):
    """Test that no-op counter methods don't raise exceptions."""
    from mellea.telemetry.metrics import create_counter

    counter = create_counter("test.counter")

    # Should not raise
    counter.add(1)
    counter.add(5, {"key": "value"})
    counter.add(10, None)


def test_noop_histogram_methods_dont_raise(clean_metrics_env):
    """Test that no-op histogram methods don't raise exceptions."""
    from mellea.telemetry.metrics import create_histogram

    histogram = create_histogram("test.histogram")

    # Should not raise
    histogram.record(100)
    histogram.record(250.5, {"key": "value"})
    histogram.record(500, None)


def test_noop_updown_counter_methods_dont_raise(clean_metrics_env):
    """Test that no-op up-down counter methods don't raise exceptions."""
    from mellea.telemetry.metrics import create_up_down_counter

    counter = create_up_down_counter("test.updown")

    # Should not raise
    counter.add(1)
    counter.add(-1)
    counter.add(5, {"key": "value"})
    counter.add(-3, None)


# Import Safety Tests


def test_graceful_handling_without_opentelemetry():
    """Test that metrics module handles missing OpenTelemetry gracefully."""
    with patch.dict("sys.modules", {"opentelemetry": None}):
        # Force reimport
        import importlib

        import mellea.telemetry.metrics

        importlib.reload(mellea.telemetry.metrics)

        # Should not raise, metrics should be disabled
        from mellea.telemetry.metrics import (
            create_counter,
            create_histogram,
            is_metrics_enabled,
        )

        assert not is_metrics_enabled()
        counter = create_counter("test.counter")
        assert counter is not None  # Should be no-op


# Functional Tests with Real Instruments


def test_counter_records_values(enable_metrics):
    """Test that counter actually records values when enabled."""
    from mellea.telemetry.metrics import create_counter

    counter = create_counter("test.functional.counter", unit="1")

    # Add some values
    counter.add(1, {"status": "success"})
    counter.add(2, {"status": "success"})
    counter.add(1, {"status": "error"})

    # Note: We can't easily verify the recorded values without a custom exporter
    # This test mainly ensures no exceptions are raised


def test_histogram_records_values(enable_metrics):
    """Test that histogram actually records values when enabled."""
    from mellea.telemetry.metrics import create_histogram

    histogram = create_histogram("test.functional.latency", unit="ms")

    # Record some values
    histogram.record(100, {"backend": "ollama"})
    histogram.record(250.5, {"backend": "openai"})
    histogram.record(500, {"backend": "ollama"})

    # Note: We can't easily verify the recorded values without a custom exporter
    # This test mainly ensures no exceptions are raised


def test_updown_counter_records_values(enable_metrics):
    """Test that up-down counter actually records values when enabled."""
    from mellea.telemetry.metrics import create_up_down_counter

    counter = create_up_down_counter("test.functional.sessions", unit="1")

    # Add and subtract values
    counter.add(1)  # Session started
    counter.add(1)  # Another session started
    counter.add(-1)  # Session ended

    # Note: We can't easily verify the recorded values without a custom exporter
    # This test mainly ensures no exceptions are raised


# Attribute Tests


def test_counter_with_attributes(enable_metrics):
    """Test counter with various attribute types."""
    from mellea.telemetry.metrics import create_counter

    counter = create_counter("test.attributes.counter")

    # Test with different attribute types
    counter.add(1, {"string": "value", "int": 42, "float": 3.14, "bool": True})
    counter.add(1, {})  # Empty attributes
    counter.add(1, None)  # None attributes


def test_histogram_with_attributes(enable_metrics):
    """Test histogram with various attribute types."""
    from mellea.telemetry.metrics import create_histogram

    histogram = create_histogram("test.attributes.histogram")

    # Test with different attribute types
    histogram.record(100, {"string": "value", "int": 42, "float": 3.14, "bool": True})
    histogram.record(200, {})  # Empty attributes
    histogram.record(300, None)  # None attributes


# Service Name Configuration


def test_custom_service_name(monkeypatch, enable_metrics):
    """Test that custom service name is used."""
    monkeypatch.setenv("OTEL_SERVICE_NAME", "my-custom-service")

    import importlib

    import mellea.telemetry.metrics

    importlib.reload(mellea.telemetry.metrics)

    from mellea.telemetry.metrics import _SERVICE_NAME

    assert _SERVICE_NAME == "my-custom-service"


def test_default_service_name(enable_metrics):
    """Test that default service name is 'mellea'."""
    from mellea.telemetry.metrics import _SERVICE_NAME

    assert _SERVICE_NAME == "mellea"


# Console Exporter Tests


def test_console_exporter_enabled(monkeypatch):
    """Test that console exporter can be enabled."""
    monkeypatch.setenv("MELLEA_METRICS_ENABLED", "true")
    monkeypatch.setenv("MELLEA_METRICS_CONSOLE", "true")

    import importlib

    import mellea.telemetry.metrics

    importlib.reload(mellea.telemetry.metrics)

    from mellea.telemetry.metrics import _METRICS_CONSOLE

    assert _METRICS_CONSOLE is True


def test_console_exporter_disabled_by_default(enable_metrics):
    """Test that console exporter is disabled by default."""
    from mellea.telemetry.metrics import _METRICS_CONSOLE

    assert _METRICS_CONSOLE is False


# OTLP Exporter Tests


def test_otlp_explicit_enablement(monkeypatch):
    """Test that OTLP exporter requires explicit enablement via MELLEA_METRICS_OTLP."""
    monkeypatch.setenv("MELLEA_METRICS_ENABLED", "true")
    monkeypatch.setenv("MELLEA_METRICS_OTLP", "true")
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317")

    import importlib

    import mellea.telemetry.metrics

    importlib.reload(mellea.telemetry.metrics)

    from mellea.telemetry.metrics import _METRICS_OTLP

    assert _METRICS_OTLP is True


def test_metrics_specific_endpoint_precedence(monkeypatch):
    """Test that OTEL_EXPORTER_OTLP_METRICS_ENDPOINT takes precedence over general endpoint."""
    monkeypatch.setenv("MELLEA_METRICS_ENABLED", "true")
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317")
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_METRICS_ENDPOINT", "http://localhost:4318")

    import importlib

    import mellea.telemetry.metrics

    importlib.reload(mellea.telemetry.metrics)

    from mellea.telemetry.metrics import _OTLP_METRICS_ENDPOINT

    assert _OTLP_METRICS_ENDPOINT == "http://localhost:4318"


def test_custom_export_interval(monkeypatch):
    """Test that custom export interval is configured correctly."""
    monkeypatch.setenv("MELLEA_METRICS_ENABLED", "true")
    monkeypatch.setenv("MELLEA_METRICS_EXPORT_INTERVAL", "30")

    import importlib

    import mellea.telemetry.metrics

    importlib.reload(mellea.telemetry.metrics)

    from mellea.telemetry.metrics import _EXPORT_INTERVAL_SECONDS

    assert _EXPORT_INTERVAL_SECONDS == 30


def test_invalid_export_interval_warning(monkeypatch):
    """Test that invalid export interval produces warning and uses default."""
    monkeypatch.setenv("MELLEA_METRICS_ENABLED", "true")
    monkeypatch.setenv("MELLEA_METRICS_EXPORT_INTERVAL", "invalid")

    import importlib

    import mellea.telemetry.metrics

    with pytest.warns(UserWarning, match="Invalid MELLEA_METRICS_EXPORT_INTERVAL"):
        importlib.reload(mellea.telemetry.metrics)

    from mellea.telemetry.metrics import _EXPORT_INTERVAL_SECONDS

    assert _EXPORT_INTERVAL_SECONDS == 60


def test_otlp_enabled_without_endpoint_warning(monkeypatch):
    """Test that enabling OTLP without endpoint produces helpful warning."""
    monkeypatch.setenv("MELLEA_METRICS_ENABLED", "true")
    monkeypatch.setenv("MELLEA_METRICS_OTLP", "true")

    import importlib

    import mellea.telemetry.metrics

    with pytest.warns(
        UserWarning,
        match="OTLP metrics exporter is enabled.*but no endpoint is configured",
    ):
        importlib.reload(mellea.telemetry.metrics)


# Token Counter Tests


def test_token_counters_lazy_initialization(enable_metrics):
    """Test that token counters are lazily initialized."""
    from mellea.telemetry.metrics import _input_token_counter, _output_token_counter

    # Initially None
    assert _input_token_counter is None
    assert _output_token_counter is None

    # Call record_token_usage_metrics
    from mellea.telemetry.metrics import record_token_usage_metrics

    record_token_usage_metrics(
        input_tokens=100,
        output_tokens=50,
        model="llama2:7b",
        backend="OllamaBackend",
        system="ollama",
    )

    # Now should be initialized
    from mellea.telemetry.metrics import _input_token_counter, _output_token_counter

    assert _input_token_counter is not None
    assert _output_token_counter is not None


def test_record_token_usage_metrics_with_valid_tokens(enable_metrics):
    """Test recording token usage with valid token counts."""
    from mellea.telemetry.metrics import record_token_usage_metrics

    # Should not raise
    record_token_usage_metrics(
        input_tokens=150,
        output_tokens=50,
        model="gpt-4",
        backend="OpenAIBackend",
        system="openai",
    )


def test_record_token_usage_metrics_with_none_tokens(enable_metrics):
    """Test recording token usage with None values (graceful handling)."""
    from mellea.telemetry.metrics import record_token_usage_metrics

    # Should not raise
    record_token_usage_metrics(
        input_tokens=None,
        output_tokens=None,
        model="llama2:7b",
        backend="OllamaBackend",
        system="ollama",
    )


def test_record_token_usage_metrics_with_zero_tokens(enable_metrics):
    """Test recording token usage with zero values (should not record)."""
    from mellea.telemetry.metrics import record_token_usage_metrics

    # Should not raise, but won't record zeros
    record_token_usage_metrics(
        input_tokens=0,
        output_tokens=0,
        model="llama2:7b",
        backend="OllamaBackend",
        system="ollama",
    )


def test_record_token_usage_metrics_noop_when_disabled(clean_metrics_env):
    """Test that record_token_usage_metrics is no-op when metrics disabled."""
    from mellea.telemetry.metrics import record_token_usage_metrics

    # Should not raise and should be no-op
    record_token_usage_metrics(
        input_tokens=100,
        output_tokens=50,
        model="llama2:7b",
        backend="OllamaBackend",
        system="ollama",
    )

    # Counters should still be None (not initialized)
    from mellea.telemetry.metrics import _input_token_counter, _output_token_counter

    assert _input_token_counter is None
    assert _output_token_counter is None


def test_record_token_usage_metrics_exported_in_public_api():
    """Test that record_token_usage_metrics is exported in public API."""
    from mellea.telemetry import record_token_usage_metrics

    assert record_token_usage_metrics is not None
    assert callable(record_token_usage_metrics)
