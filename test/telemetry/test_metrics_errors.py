"""Integration tests for error counter metrics recording.

These tests verify that record_error() correctly records counter metrics with
proper attributes and values using OpenTelemetry.
"""

import pytest

# Check if OpenTelemetry is available
try:
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import InMemoryMetricReader

    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False

pytestmark = [
    pytest.mark.skipif(not OTEL_AVAILABLE, reason="OpenTelemetry not installed"),
    pytest.mark.integration,
]


@pytest.fixture
def clean_metrics_env(monkeypatch):
    """Enable metrics and reset module state for integration tests."""
    monkeypatch.setenv("MELLEA_METRICS_ENABLED", "true")
    monkeypatch.delenv("MELLEA_METRICS_CONSOLE", raising=False)

    import importlib

    import mellea.telemetry.metrics

    importlib.reload(mellea.telemetry.metrics)
    yield
    importlib.reload(mellea.telemetry.metrics)


def _setup_in_memory_provider(metrics_module):
    """Wire an InMemoryMetricReader into the metrics module globals."""
    reader = InMemoryMetricReader()
    provider = MeterProvider(metric_readers=[reader])
    metrics_module._meter_provider = provider
    metrics_module._meter = provider.get_meter("mellea")
    metrics_module._error_counter = None
    return reader, provider


def _find_error_data_points(metrics_data):
    """Return all data points for mellea.llm.errors."""
    if metrics_data is None:
        return []
    data_points = []
    for rm in metrics_data.resource_metrics:
        for sm in rm.scope_metrics:
            for metric in sm.metrics:
                if metric.name == "mellea.llm.errors":
                    data_points.extend(metric.data.data_points)
    return data_points


def test_record_error_basic(clean_metrics_env):
    """Error counter is populated with correct value and attributes."""
    from mellea.telemetry import metrics as metrics_module

    reader, provider = _setup_in_memory_provider(metrics_module)

    from mellea.telemetry.metrics import record_error

    record_error(
        error_type="rate_limit",
        model="gpt-4",
        provider="openai",
        exception_class="RateLimitError",
    )

    provider.force_flush()
    data_points = _find_error_data_points(reader.get_metrics_data())

    assert len(data_points) == 1
    attrs = dict(data_points[0].attributes)
    assert attrs["error_type"] == "rate_limit"
    assert attrs["gen_ai.request.model"] == "gpt-4"
    assert attrs["gen_ai.provider.name"] == "openai"
    assert attrs["error.type"] == "RateLimitError"
    assert data_points[0].value == 1


def test_record_error_accumulation(clean_metrics_env):
    """Multiple errors with the same attributes accumulate correctly."""
    from mellea.telemetry import metrics as metrics_module

    reader, provider = _setup_in_memory_provider(metrics_module)

    from mellea.telemetry.metrics import record_error

    record_error("timeout", "llama2:7b", "ollama", "TimeoutError")
    record_error("timeout", "llama2:7b", "ollama", "TimeoutError")
    record_error("timeout", "llama2:7b", "ollama", "TimeoutError")

    provider.force_flush()
    data_points = _find_error_data_points(reader.get_metrics_data())

    assert len(data_points) == 1
    assert data_points[0].value == 3


def test_record_error_multiple_types(clean_metrics_env):
    """Different error types are tracked as separate attribute sets."""
    from mellea.telemetry import metrics as metrics_module

    reader, provider = _setup_in_memory_provider(metrics_module)

    from mellea.telemetry.metrics import record_error

    record_error("rate_limit", "gpt-4", "openai", "RateLimitError")
    record_error("timeout", "gpt-4", "openai", "APITimeoutError")
    record_error("auth", "gpt-4", "openai", "AuthenticationError")

    provider.force_flush()
    data_points = _find_error_data_points(reader.get_metrics_data())

    assert len(data_points) == 3
    error_types = {dict(dp.attributes)["error_type"] for dp in data_points}
    assert error_types == {"rate_limit", "timeout", "auth"}
