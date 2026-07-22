# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for `IntrinsicMetricsPlugin` (Epic #929 Phase 2, issue #1140).

Drives `record_*` methods directly with synthetic hook payloads, matching how
every other metrics plugin in `mellea/telemetry/metrics_plugins.py` is tested.
No production call site fires these hooks yet — this exercises the skeleton
in isolation, ahead of the LocalFileBinding/EmbeddedBinding lifecycle wiring.
"""

from unittest.mock import patch

import pytest

pytest.importorskip("cpex", reason="cpex not installed — install mellea[hooks]")

from mellea.plugins.hooks.intrinsic import (
    IntrinsicInvocationCompletePayload,
    IntrinsicPhaseCompletePayload,
)
from mellea.telemetry.metrics_plugins import IntrinsicMetricsPlugin
from test.telemetry.conftest import reset_metrics_state

try:
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import InMemoryMetricReader

    _OTEL_AVAILABLE = True
except ImportError:
    _OTEL_AVAILABLE = False


@pytest.fixture
def intrinsic_plugin():
    return IntrinsicMetricsPlugin()


@pytest.mark.asyncio
async def test_record_intrinsic_invocation_success(intrinsic_plugin):
    """A successful invocation records the invocations counter, not parse_failures."""
    payload = IntrinsicInvocationCompletePayload(
        name="answerability",
        revision="r1",
        binding_type="local_file",
        adapter_type="lora",
        outcome="success",
    )

    with (
        patch(
            "mellea.telemetry.metrics.record_intrinsic_invocation"
        ) as mock_invocation,
        patch(
            "mellea.telemetry.metrics.record_intrinsic_parse_failure"
        ) as mock_parse_failure,
    ):
        await intrinsic_plugin.record_intrinsic_invocation(payload, {})

    mock_invocation.assert_called_once_with(
        name="answerability",
        revision="r1",
        binding_type="local_file",
        adapter_type="lora",
        outcome="success",
    )
    mock_parse_failure.assert_not_called()


@pytest.mark.asyncio
async def test_record_intrinsic_invocation_schema_error_also_records_parse_failure(
    intrinsic_plugin,
):
    """A schema_error outcome records both the invocations counter and parse_failures."""
    payload = IntrinsicInvocationCompletePayload(
        name="answerability",
        revision="r1",
        binding_type="local_file",
        adapter_type="lora",
        outcome="schema_error",
    )

    with (
        patch(
            "mellea.telemetry.metrics.record_intrinsic_invocation"
        ) as mock_invocation,
        patch(
            "mellea.telemetry.metrics.record_intrinsic_parse_failure"
        ) as mock_parse_failure,
    ):
        await intrinsic_plugin.record_intrinsic_invocation(payload, {})

    mock_invocation.assert_called_once_with(
        name="answerability",
        revision="r1",
        binding_type="local_file",
        adapter_type="lora",
        outcome="schema_error",
    )
    mock_parse_failure.assert_called_once_with("answerability", "r1")


@pytest.mark.asyncio
async def test_record_intrinsic_invocation_missing_revision_defaults_to_unknown(
    intrinsic_plugin,
):
    """A None revision is normalized to 'unknown' before being recorded."""
    payload = IntrinsicInvocationCompletePayload(
        name="answerability",
        revision=None,
        binding_type="embedded",
        adapter_type="alora",
        outcome="error",
    )

    with patch(
        "mellea.telemetry.metrics.record_intrinsic_invocation"
    ) as mock_invocation:
        await intrinsic_plugin.record_intrinsic_invocation(payload, {})

    mock_invocation.assert_called_once_with(
        name="answerability",
        revision="unknown",
        binding_type="embedded",
        adapter_type="alora",
        outcome="error",
    )


@pytest.mark.asyncio
async def test_record_intrinsic_phase_duration(intrinsic_plugin):
    """Phase-complete events record the phase-duration histogram in seconds."""
    payload = IntrinsicPhaseCompletePayload(
        name="answerability", phase="prepare", duration_ms=12.5
    )

    with patch(
        "mellea.telemetry.metrics.record_intrinsic_phase_duration"
    ) as mock_phase:
        await intrinsic_plugin.record_intrinsic_phase(payload, {})

    # payload ms is converted to seconds before recording
    mock_phase.assert_called_once_with("answerability", "prepare", 0.0125)


# --- Emission tests -----------------------------------------------------------
# The tests above drive the record_* dispatch directly with mocks. The issue's
# test plan also calls for a synthetic OTel exporter; these assert the metrics
# actually reach a reader with the right labels/values, mirroring the
# InMemoryMetricReader pattern used by the sibling metrics-plugin tests.


def _setup_in_memory_provider():
    """Wire an InMemoryMetricReader into the metrics module globals."""
    from mellea.telemetry import metrics as m

    reader = InMemoryMetricReader()
    provider = MeterProvider(metric_readers=[reader])
    m._meter_provider = provider
    m._meter = provider.get_meter("mellea")
    m._intrinsic_invocations_counter = None
    m._intrinsic_phase_duration_histogram = None
    m._intrinsic_parse_failures_counter = None
    return reader, provider


def _points(metrics_data, name):
    """Return all data points for the named metric."""
    points = []
    if metrics_data is None:
        return points
    for rm in metrics_data.resource_metrics:
        for sm in rm.scope_metrics:
            for metric in sm.metrics:
                if metric.name == name:
                    points.extend(metric.data.data_points)
    return points


@pytest.mark.integration
@pytest.mark.skipif(not _OTEL_AVAILABLE, reason="OpenTelemetry not installed")
@pytest.mark.asyncio
async def test_intrinsic_metrics_emit_through_exporter(intrinsic_plugin, monkeypatch):
    """Firing the plugin emits real counter + histogram points via an in-memory
    OTel exporter — not just calls to the record_* functions."""
    monkeypatch.setenv("MELLEA_METRICS_ENABLED", "true")
    reset_metrics_state()
    reader, provider = _setup_in_memory_provider()
    try:
        await intrinsic_plugin.record_intrinsic_invocation(
            IntrinsicInvocationCompletePayload(
                name="answerability",
                revision="abc123",
                binding_type="embedded",
                adapter_type="alora",
                outcome="schema_error",
            ),
            {},
        )
        await intrinsic_plugin.record_intrinsic_phase(
            IntrinsicPhaseCompletePayload(
                name="answerability", phase="generate", duration_ms=12.5
            ),
            {},
        )
        provider.force_flush()
        data = reader.get_metrics_data()

        invocations = _points(data, "mellea.intrinsic.invocations")
        assert len(invocations) == 1
        assert dict(invocations[0].attributes)["outcome"] == "schema_error"

        # a schema_error outcome also drives the parse-failures counter
        assert len(_points(data, "mellea.intrinsic.parse_failures")) == 1

        phase = _points(data, "mellea.intrinsic.phase_duration")
        assert len(phase) == 1
        assert dict(phase[0].attributes)["phase"] == "generate"
        # payload milliseconds are recorded as seconds
        assert phase[0].sum == 0.0125
    finally:
        reset_metrics_state()
