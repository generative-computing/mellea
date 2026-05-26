"""Integration tests for cost counter metrics recording.

These tests verify that record_cost() correctly records counter metrics with
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
    metrics_module._cost_counter = None
    return reader, provider


def _find_cost_data_points(metrics_data):
    """Return all data points for mellea.llm.cost.usd."""
    if metrics_data is None:
        return []
    data_points = []
    for rm in metrics_data.resource_metrics:
        for sm in rm.scope_metrics:
            for metric in sm.metrics:
                if metric.name == "mellea.llm.cost.usd":
                    data_points.extend(metric.data.data_points)
    return data_points


def test_record_cost_basic(clean_metrics_env):
    """Cost counter is populated with correct value and attributes."""
    from mellea.telemetry import metrics as metrics_module

    reader, provider = _setup_in_memory_provider(metrics_module)

    from mellea.telemetry.metrics import record_cost

    record_cost(cost=0.0042, model="gpt-4o", provider="openai")

    provider.force_flush()
    data_points = _find_cost_data_points(reader.get_metrics_data())

    assert len(data_points) == 1
    attrs = dict(data_points[0].attributes)
    assert attrs["gen_ai.request.model"] == "gpt-4o"
    assert attrs["gen_ai.provider.name"] == "openai"
    assert abs(data_points[0].value - 0.0042) < 1e-9


def test_record_cost_accumulation(clean_metrics_env):
    """Multiple calls with the same attributes accumulate correctly."""
    from mellea.telemetry import metrics as metrics_module

    reader, provider = _setup_in_memory_provider(metrics_module)

    from mellea.telemetry.metrics import record_cost

    record_cost(0.001, "claude-sonnet-4-6", "anthropic")
    record_cost(0.002, "claude-sonnet-4-6", "anthropic")
    record_cost(0.003, "claude-sonnet-4-6", "anthropic")

    provider.force_flush()
    data_points = _find_cost_data_points(reader.get_metrics_data())

    assert len(data_points) == 1
    assert abs(data_points[0].value - 0.006) < 1e-9


def test_record_cost_multiple_models(clean_metrics_env):
    """Different models are tracked as separate attribute sets."""
    from mellea.telemetry import metrics as metrics_module

    reader, provider = _setup_in_memory_provider(metrics_module)

    from mellea.telemetry.metrics import record_cost

    record_cost(0.001, "gpt-4o", "openai")
    record_cost(0.002, "gpt-4o-mini", "openai")
    record_cost(0.003, "claude-sonnet-4-6", "anthropic")

    provider.force_flush()
    data_points = _find_cost_data_points(reader.get_metrics_data())

    assert len(data_points) == 3
    models = {dict(dp.attributes)["gen_ai.request.model"] for dp in data_points}
    assert models == {"gpt-4o", "gpt-4o-mini", "claude-sonnet-4-6"}
