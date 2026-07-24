# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for adapter-function (intrinsic) metrics recording.

These tests verify that record_intrinsic_invocation(), record_intrinsic_phase_duration(),
and record_intrinsic_parse_failure() correctly record counter/histogram metrics with
proper attributes and values using OpenTelemetry. No production call site fires these
hooks yet — this exercises the skeleton ahead of the LocalFileBinding/EmbeddedBinding
lifecycle wiring (#1141/#1142).
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
    """Enable metrics and reset module state for integration tests."""
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
    metrics_module._intrinsic_invocations_counter = None
    metrics_module._intrinsic_phase_duration_histogram = None
    metrics_module._intrinsic_parse_failures_counter = None
    return reader, provider


def _find_data_points(metrics_data, name):
    """Return all data points for the named metric."""
    if metrics_data is None:
        return []
    data_points = []
    for rm in metrics_data.resource_metrics:
        for sm in rm.scope_metrics:
            for metric in sm.metrics:
                if metric.name == name:
                    data_points.extend(metric.data.data_points)
    return data_points


def test_record_intrinsic_invocation_basic(clean_metrics_env):
    """Invocations counter is populated with correct labels."""
    from mellea.telemetry import metrics as metrics_module

    reader, provider = _setup_in_memory_provider(metrics_module)

    from mellea.telemetry.metrics import record_intrinsic_invocation

    record_intrinsic_invocation(
        name="answerability",
        revision="abc123",
        binding_type="embedded",
        adapter_type="alora",
        outcome="schema_error",
    )

    provider.force_flush()
    data_points = _find_data_points(
        reader.get_metrics_data(), "mellea.adapter_function.invocations"
    )

    assert len(data_points) == 1
    attrs = dict(data_points[0].attributes)
    assert attrs["name"] == "answerability"
    assert attrs["revision"] == "abc123"
    assert attrs["binding_type"] == "embedded"
    assert attrs["adapter_type"] == "alora"
    assert attrs["outcome"] == "schema_error"


def test_record_intrinsic_invocation_schema_error_also_increments_parse_failures(
    clean_metrics_env,
):
    """A schema_error outcome drives the parse-failures counter too."""
    from mellea.telemetry import metrics as metrics_module

    reader, provider = _setup_in_memory_provider(metrics_module)

    from mellea.telemetry.metrics import (
        record_intrinsic_invocation,
        record_intrinsic_parse_failure,
    )

    record_intrinsic_invocation(
        name="answerability",
        revision="abc123",
        binding_type="embedded",
        adapter_type="alora",
        outcome="schema_error",
    )
    record_intrinsic_parse_failure("answerability", "abc123")

    provider.force_flush()
    data_points = _find_data_points(
        reader.get_metrics_data(), "mellea.adapter_function.parse_failures"
    )

    assert len(data_points) == 1
    assert data_points[0].value == 1


def test_record_intrinsic_phase_duration_basic(clean_metrics_env):
    """Phase-duration histogram is populated with the phase label and seconds value."""
    from mellea.telemetry import metrics as metrics_module

    reader, provider = _setup_in_memory_provider(metrics_module)

    from mellea.telemetry.metrics import record_intrinsic_phase_duration

    record_intrinsic_phase_duration("answerability", "generate", 0.0125)

    provider.force_flush()
    data_points = _find_data_points(
        reader.get_metrics_data(), "mellea.adapter_function.phase_duration"
    )

    assert len(data_points) == 1
    attrs = dict(data_points[0].attributes)
    assert attrs["name"] == "answerability"
    assert attrs["phase"] == "generate"
    assert data_points[0].sum == 0.0125
