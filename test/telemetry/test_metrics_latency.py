"""Integration tests for latency metrics recording.

These tests verify that record_request_duration() and record_ttfb() correctly
record histogram metrics with proper attributes and values using OpenTelemetry.
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
    metrics_module._duration_histogram = None
    metrics_module._ttfb_histogram = None
    return reader, provider


def _find_histogram(metrics_data, metric_name):
    """Return all data points for the named histogram metric."""
    if metrics_data is None:
        return []
    data_points = []
    for rm in metrics_data.resource_metrics:
        for sm in rm.scope_metrics:
            for metric in sm.metrics:
                if metric.name == metric_name:
                    data_points.extend(metric.data.data_points)
    return data_points


def test_record_request_duration_non_streaming(clean_metrics_env):
    """Duration histogram is populated with correct attributes for non-streaming."""
    from mellea.telemetry import metrics as metrics_module

    reader, provider = _setup_in_memory_provider(metrics_module)

    from mellea.telemetry.metrics import record_request_duration

    record_request_duration(
        duration_s=1.5, model="llama2:7b", provider="ollama", streaming=False
    )

    provider.force_flush()
    data_points = _find_histogram(
        reader.get_metrics_data(), "mellea.llm.request.duration"
    )

    assert len(data_points) == 1
    attrs = dict(data_points[0].attributes)
    assert attrs["gen_ai.request.model"] == "llama2:7b"
    assert attrs["gen_ai.provider.name"] == "ollama"
    assert attrs["streaming"] is False
    assert data_points[0].count == 1


def test_record_request_duration_streaming(clean_metrics_env):
    """Duration histogram distinguishes streaming vs non-streaming via attribute."""
    from mellea.telemetry import metrics as metrics_module

    reader, provider = _setup_in_memory_provider(metrics_module)

    from mellea.telemetry.metrics import record_request_duration

    record_request_duration(
        duration_s=2.0, model="gpt-4", provider="openai", streaming=True
    )
    record_request_duration(
        duration_s=1.0, model="gpt-4", provider="openai", streaming=False
    )

    provider.force_flush()
    data_points = _find_histogram(
        reader.get_metrics_data(), "mellea.llm.request.duration"
    )

    assert len(data_points) == 2
    streaming_flags = {dict(dp.attributes)["streaming"] for dp in data_points}
    assert streaming_flags == {True, False}


def test_record_ttfb(clean_metrics_env):
    """TTFB histogram is populated with correct attributes."""
    from mellea.telemetry import metrics as metrics_module

    reader, provider = _setup_in_memory_provider(metrics_module)

    from mellea.telemetry.metrics import record_ttfb

    record_ttfb(ttfb_s=0.18, model="llama2:7b", provider="ollama")

    provider.force_flush()
    data_points = _find_histogram(reader.get_metrics_data(), "mellea.llm.ttfb")

    assert len(data_points) == 1
    attrs = dict(data_points[0].attributes)
    assert attrs["gen_ai.request.model"] == "llama2:7b"
    assert attrs["gen_ai.provider.name"] == "ollama"
    assert data_points[0].count == 1
