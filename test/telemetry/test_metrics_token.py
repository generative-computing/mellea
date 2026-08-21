# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for token usage metrics recording.

These tests verify that the record_token_usage_metrics() function correctly
records token metrics with proper attributes and values using OpenTelemetry.
"""

import pytest

from test.telemetry.conftest import reset_metrics_state

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
    """Clean metrics environment variables and enable metrics for tests."""
    monkeypatch.setenv("MELLEA_METRICS_ENABLED", "true")
    monkeypatch.delenv("MELLEA_METRICS_CONSOLE", raising=False)
    reset_metrics_state()
    yield
    reset_metrics_state()


def _setup_in_memory_provider(metrics_module):
    """Wire an InMemoryMetricReader into the metrics module globals."""
    reader = InMemoryMetricReader()
    provider = MeterProvider(metric_readers=[reader])
    metrics_module._meter_provider = provider
    metrics_module._meter = provider.get_meter("mellea")
    metrics_module._token_usage_histogram = None
    return reader, provider


def test_record_token_metrics_basic(clean_metrics_env):
    """Test that token metrics are recorded with correct values and attributes."""
    from mellea.telemetry import metrics as metrics_module

    reader, provider = _setup_in_memory_provider(metrics_module)

    from mellea.telemetry.metrics import record_token_usage_metrics

    # Record some token usage
    record_token_usage_metrics(
        input_tokens=150,
        output_tokens=50,
        model="llama2:7b",
        provider="ollama",
        operation="chat",
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
                if metric.name != "gen_ai.client.token.usage":
                    continue
                for data_point in metric.data.data_points:
                    attrs = dict(data_point.attributes)
                    assert attrs["gen_ai.provider.name"] == "ollama"
                    assert attrs["gen_ai.request.model"] == "llama2:7b"
                    assert attrs["gen_ai.operation.name"] == "chat"
                    if attrs["gen_ai.token.type"] == "input":
                        found_input = True
                        assert data_point.sum == 150
                    if attrs["gen_ai.token.type"] == "output":
                        found_output = True
                        assert data_point.sum == 50

    assert found_input, "Input token data point not found"
    assert found_output, "Output token data point not found"


def test_record_token_metrics_accumulation(clean_metrics_env):
    """Test that multiple token recordings aggregate into the histogram."""
    from mellea.telemetry import metrics as metrics_module

    reader, provider = _setup_in_memory_provider(metrics_module)

    from mellea.telemetry.metrics import record_token_usage_metrics

    # Record multiple token usages with same attributes
    record_token_usage_metrics(
        input_tokens=100,
        output_tokens=30,
        model="gpt-4",
        provider="openai",
        operation="chat",
    )
    record_token_usage_metrics(
        input_tokens=200,
        output_tokens=70,
        model="gpt-4",
        provider="openai",
        operation="chat",
    )

    # Force metrics collection
    provider.force_flush()
    metrics_data = reader.get_metrics_data()

    # Verify aggregated values per token type
    for rm in metrics_data.resource_metrics:
        for sm in rm.scope_metrics:
            for metric in sm.metrics:
                if metric.name != "gen_ai.client.token.usage":
                    continue
                for data_point in metric.data.data_points:
                    token_type = dict(data_point.attributes)["gen_ai.token.type"]
                    if token_type == "input":
                        assert data_point.sum == 300
                        assert data_point.count == 2
                    if token_type == "output":
                        assert data_point.sum == 100
                        assert data_point.count == 2


def test_record_token_metrics_none_handling(clean_metrics_env):
    """Test that None token values are handled gracefully."""
    from mellea.telemetry import metrics as metrics_module

    reader, provider = _setup_in_memory_provider(metrics_module)

    from mellea.telemetry.metrics import record_token_usage_metrics

    # Record with None values (should not crash)
    record_token_usage_metrics(
        input_tokens=None,
        output_tokens=None,
        model="llama2:7b",
        provider="ollama",
        operation="chat",
    )

    # Should not raise, and no metrics should be recorded for None values
    provider.force_flush()
    metrics_data = reader.get_metrics_data()

    # Verify no token metrics were recorded
    if metrics_data:
        for rm in metrics_data.resource_metrics:
            for sm in rm.scope_metrics:
                for metric in sm.metrics:
                    assert metric.name != "gen_ai.client.token.usage", (
                        "Metrics should not be recorded for None token values"
                    )


def test_record_token_metrics_multiple_backends(clean_metrics_env):
    """Test token metrics from different backends are tracked separately."""
    from mellea.telemetry import metrics as metrics_module

    reader, provider = _setup_in_memory_provider(metrics_module)

    from mellea.telemetry.metrics import record_token_usage_metrics

    # Record from different backends
    record_token_usage_metrics(
        input_tokens=100,
        output_tokens=50,
        model="llama2:7b",
        provider="ollama",
        operation="chat",
    )
    record_token_usage_metrics(
        input_tokens=200,
        output_tokens=80,
        model="gpt-4",
        provider="openai",
        operation="chat",
    )
    record_token_usage_metrics(
        input_tokens=150,
        output_tokens=60,
        model="granite-3-8b",
        provider="watsonx",
        operation="chat",
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
                if metric.name != "gen_ai.client.token.usage":
                    continue
                for dp in metric.data.data_points:
                    attrs = dict(dp.attributes)
                    key = (attrs["gen_ai.provider.name"], attrs["gen_ai.request.model"])
                    if attrs["gen_ai.token.type"] == "input":
                        input_attrs.add(key)
                    elif attrs["gen_ai.token.type"] == "output":
                        output_attrs.add(key)

    # Should have 3 different backend combinations
    assert len(input_attrs) == 3, (
        f"Expected 3 unique input metric attribute sets, got {len(input_attrs)}"
    )
    assert len(output_attrs) == 3, (
        f"Expected 3 unique output metric attribute sets, got {len(output_attrs)}"
    )
    assert ("ibm.watsonx.ai", "granite-3-8b") in input_attrs
