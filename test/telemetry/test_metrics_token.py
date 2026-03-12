"""Integration tests for token usage metrics recording.

These tests verify that the record_token_usage_metrics() function correctly
records token metrics with proper attributes and values using OpenTelemetry.
"""

import pytest

# Check if OpenTelemetry is available
try:
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
    """Clean metrics environment variables and enable metrics for tests."""
    # Enable metrics for integration tests
    monkeypatch.setenv("MELLEA_METRICS_ENABLED", "true")

    # Clean other metrics env vars
    monkeypatch.delenv("MELLEA_METRICS_CONSOLE", raising=False)

    # Force reload of metrics module to pick up env vars
    import importlib

    import mellea.telemetry.metrics

    importlib.reload(mellea.telemetry.metrics)
    yield
    # Reset after test
    importlib.reload(mellea.telemetry.metrics)


def test_record_token_metrics_basic(clean_metrics_env):
    """Test that token metrics are recorded with correct values and attributes."""
    from mellea.telemetry import metrics as metrics_module

    # Create InMemoryMetricReader to capture metrics
    reader = InMemoryMetricReader()

    # Create MeterProvider with our reader
    provider = MeterProvider(metric_readers=[reader])
    metrics_module._meter_provider = provider
    metrics_module._meter = provider.get_meter("mellea")
    metrics_module._input_token_counter = None
    metrics_module._output_token_counter = None

    from mellea.telemetry.metrics import record_token_usage_metrics

    # Record some token usage
    record_token_usage_metrics(
        input_tokens=150,
        output_tokens=50,
        model="llama2:7b",
        backend="OllamaBackend",
        system="ollama",
    )

    # Force metrics collection
    provider.force_flush()
    metrics_data = reader.get_metrics_data()

    # Verify metrics were recorded
    assert metrics_data is not None
    resource_metrics = metrics_data.resource_metrics
    assert len(resource_metrics) > 0

    # Find our token metrics
    found_input = False
    found_output = False

    for rm in resource_metrics:
        for sm in rm.scope_metrics:
            for metric in sm.metrics:
                if metric.name == "mellea.llm.tokens.input":
                    found_input = True
                    # Verify attributes
                    for data_point in metric.data.data_points:
                        attrs = dict(data_point.attributes)
                        assert attrs["gen_ai.system"] == "ollama"
                        assert attrs["gen_ai.request.model"] == "llama2:7b"
                        assert attrs["mellea.backend"] == "OllamaBackend"
                        assert data_point.value == 150

                if metric.name == "mellea.llm.tokens.output":
                    found_output = True
                    # Verify attributes
                    for data_point in metric.data.data_points:
                        attrs = dict(data_point.attributes)
                        assert attrs["gen_ai.system"] == "ollama"
                        assert attrs["gen_ai.request.model"] == "llama2:7b"
                        assert attrs["mellea.backend"] == "OllamaBackend"
                        assert data_point.value == 50

    assert found_input, "Input token metric not found"
    assert found_output, "Output token metric not found"


def test_record_token_metrics_accumulation(clean_metrics_env):
    """Test that multiple token recordings accumulate correctly."""
    from mellea.telemetry import metrics as metrics_module

    reader = InMemoryMetricReader()
    provider = MeterProvider(metric_readers=[reader])
    metrics_module._meter_provider = provider
    metrics_module._meter = provider.get_meter("mellea")
    metrics_module._input_token_counter = None
    metrics_module._output_token_counter = None

    from mellea.telemetry.metrics import record_token_usage_metrics

    # Record multiple token usages with same attributes
    record_token_usage_metrics(
        input_tokens=100,
        output_tokens=30,
        model="gpt-4",
        backend="OpenAIBackend",
        system="openai",
    )
    record_token_usage_metrics(
        input_tokens=200,
        output_tokens=70,
        model="gpt-4",
        backend="OpenAIBackend",
        system="openai",
    )

    # Force metrics collection
    provider.force_flush()
    metrics_data = reader.get_metrics_data()

    # Verify cumulative values (counters should sum)
    for rm in metrics_data.resource_metrics:
        for sm in rm.scope_metrics:
            for metric in sm.metrics:
                if metric.name == "mellea.llm.tokens.input":
                    for data_point in metric.data.data_points:
                        # Should be sum of both recordings
                        assert data_point.value == 300
                if metric.name == "mellea.llm.tokens.output":
                    for data_point in metric.data.data_points:
                        assert data_point.value == 100


def test_record_token_metrics_none_handling(clean_metrics_env):
    """Test that None token values are handled gracefully."""
    from mellea.telemetry import metrics as metrics_module

    reader = InMemoryMetricReader()
    provider = MeterProvider(metric_readers=[reader])
    metrics_module._meter_provider = provider
    metrics_module._meter = provider.get_meter("mellea")
    metrics_module._input_token_counter = None
    metrics_module._output_token_counter = None

    from mellea.telemetry.metrics import record_token_usage_metrics

    # Record with None values (should not crash)
    record_token_usage_metrics(
        input_tokens=None,
        output_tokens=None,
        model="llama2:7b",
        backend="OllamaBackend",
        system="ollama",
    )

    # Should not raise, and no metrics should be recorded for None values
    provider.force_flush()
    metrics_data = reader.get_metrics_data()

    # Verify no token metrics were recorded
    if metrics_data:
        for rm in metrics_data.resource_metrics:
            for sm in rm.scope_metrics:
                for metric in sm.metrics:
                    assert metric.name not in [
                        "mellea.llm.tokens.input",
                        "mellea.llm.tokens.output",
                    ], "Metrics should not be recorded for None token values"


def test_record_token_metrics_multiple_backends(clean_metrics_env):
    """Test token metrics from different backends are tracked separately."""
    from mellea.telemetry import metrics as metrics_module

    reader = InMemoryMetricReader()
    provider = MeterProvider(metric_readers=[reader])
    metrics_module._meter_provider = provider
    metrics_module._meter = provider.get_meter("mellea")
    metrics_module._input_token_counter = None
    metrics_module._output_token_counter = None

    from mellea.telemetry.metrics import record_token_usage_metrics

    # Record from different backends
    record_token_usage_metrics(
        input_tokens=100,
        output_tokens=50,
        model="llama2:7b",
        backend="OllamaBackend",
        system="ollama",
    )
    record_token_usage_metrics(
        input_tokens=200,
        output_tokens=80,
        model="gpt-4",
        backend="OpenAIBackend",
        system="openai",
    )
    record_token_usage_metrics(
        input_tokens=150,
        output_tokens=60,
        model="granite-3-8b",
        backend="WatsonxBackend",
        system="watsonx",
    )

    # Force metrics collection
    provider.force_flush()
    metrics_data = reader.get_metrics_data()
    assert metrics_data is not None

    # Count unique attribute combinations
    input_attrs = set()
    output_attrs = set()

    for rm in metrics_data.resource_metrics:
        for sm in rm.scope_metrics:
            for metric in sm.metrics:
                if metric.name == "mellea.llm.tokens.input":
                    for dp in metric.data.data_points:
                        attrs = dict(dp.attributes)
                        key = (
                            attrs["gen_ai.system"],
                            attrs["gen_ai.request.model"],
                            attrs["mellea.backend"],
                        )
                        input_attrs.add(key)
                if metric.name == "mellea.llm.tokens.output":
                    for dp in metric.data.data_points:
                        attrs = dict(dp.attributes)
                        key = (
                            attrs["gen_ai.system"],
                            attrs["gen_ai.request.model"],
                            attrs["mellea.backend"],
                        )
                        output_attrs.add(key)

    # Should have 3 different backend combinations
    assert len(input_attrs) == 3, (
        f"Expected 3 unique input metric attribute sets, got {len(input_attrs)}"
    )
    assert len(output_attrs) == 3, (
        f"Expected 3 unique output metric attribute sets, got {len(output_attrs)}"
    )
