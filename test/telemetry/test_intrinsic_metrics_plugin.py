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
    """Phase-complete events record the phase_duration_ms histogram."""
    payload = IntrinsicPhaseCompletePayload(
        name="answerability", phase="prepare", duration_ms=12.5
    )

    with patch(
        "mellea.telemetry.metrics.record_intrinsic_phase_duration"
    ) as mock_phase:
        await intrinsic_plugin.record_intrinsic_phase(payload, {})

    mock_phase.assert_called_once_with("answerability", "prepare", 12.5)
