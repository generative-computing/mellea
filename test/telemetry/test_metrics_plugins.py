"""Unit tests for TokenMetricsPlugin and LatencyMetricsPlugin."""

from unittest.mock import patch

import pytest

pytest.importorskip("cpex", reason="cpex not installed — install mellea[hooks]")

from mellea.core.base import ModelOutputThunk
from mellea.plugins.hooks.generation import GenerationPostCallPayload
from mellea.telemetry.metrics_plugins import LatencyMetricsPlugin, TokenMetricsPlugin


@pytest.fixture
def token_plugin():
    return TokenMetricsPlugin()


def _make_token_payload(usage=None, model="test-model", provider="test-provider"):
    """Create a GenerationPostCallPayload with a ModelOutputThunk."""
    mot = ModelOutputThunk(value="hello")
    mot.usage = usage
    mot.model = model
    mot.provider = provider
    return GenerationPostCallPayload(model_output=mot)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "usage,expected_input,expected_output",
    [
        ({"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}, 10, 5),
        ({"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}, 0, 0),
    ],
    ids=["normal-usage", "zero-tokens"],
)
async def test_record_token_metrics_with_usage(
    token_plugin, usage, expected_input, expected_output
):
    """Test that metrics are recorded when usage is present."""
    payload = _make_token_payload(usage=usage)

    with patch("mellea.telemetry.metrics.record_token_usage_metrics") as mock_record:
        await token_plugin.record_token_metrics(payload, {})

        mock_record.assert_called_once_with(
            input_tokens=expected_input,
            output_tokens=expected_output,
            model="test-model",
            provider="test-provider",
        )


@pytest.mark.asyncio
async def test_record_token_metrics_no_usage(token_plugin):
    """Test that metrics are not recorded when usage is None."""
    payload = _make_token_payload(usage=None)

    with patch("mellea.telemetry.metrics.record_token_usage_metrics") as mock_record:
        await token_plugin.record_token_metrics(payload, {})

        mock_record.assert_not_called()


@pytest.mark.asyncio
async def test_record_token_metrics_missing_model_provider(token_plugin):
    """Test fallback to 'unknown' when model/provider are None."""
    payload = _make_token_payload(
        usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        model=None,
        provider=None,
    )

    with patch("mellea.telemetry.metrics.record_token_usage_metrics") as mock_record:
        await token_plugin.record_token_metrics(payload, {})

        mock_record.assert_called_once_with(
            input_tokens=10, output_tokens=5, model="unknown", provider="unknown"
        )


# LatencyMetricsPlugin tests


@pytest.fixture
def latency_plugin():
    return LatencyMetricsPlugin()


def _make_latency_payload(
    latency_ms=500.0,
    ttfb_ms=None,
    streaming=False,
    model="test-model",
    provider="test-provider",
):
    """Create a GenerationPostCallPayload for latency tests."""
    mot = ModelOutputThunk(value="hello")
    mot.model = model
    mot.provider = provider
    mot.ttfb_ms = ttfb_ms
    mot.streaming = streaming
    return GenerationPostCallPayload(model_output=mot, latency_ms=latency_ms)


@pytest.mark.asyncio
async def test_latency_non_streaming_records_duration_only(latency_plugin):
    """Non-streaming requests record duration but not TTFB."""
    payload = _make_latency_payload(latency_ms=1200.0, streaming=False)

    with (
        patch("mellea.telemetry.metrics.record_request_duration") as mock_dur,
        patch("mellea.telemetry.metrics.record_ttfb") as mock_ttfb,
    ):
        await latency_plugin.record_latency_metrics(payload, {})

        mock_dur.assert_called_once_with(
            duration_s=1.2,
            model="test-model",
            provider="test-provider",
            streaming=False,
        )
        mock_ttfb.assert_not_called()


@pytest.mark.asyncio
async def test_latency_streaming_with_ttfb_records_both(latency_plugin):
    """Streaming requests with a measured TTFB record both duration and TTFB."""
    payload = _make_latency_payload(latency_ms=2000.0, ttfb_ms=180.0, streaming=True)

    with (
        patch("mellea.telemetry.metrics.record_request_duration") as mock_dur,
        patch("mellea.telemetry.metrics.record_ttfb") as mock_ttfb,
    ):
        await latency_plugin.record_latency_metrics(payload, {})

        mock_dur.assert_called_once_with(
            duration_s=2.0, model="test-model", provider="test-provider", streaming=True
        )
        mock_ttfb.assert_called_once_with(
            ttfb_s=0.18, model="test-model", provider="test-provider"
        )


@pytest.mark.asyncio
async def test_latency_streaming_without_ttfb_records_duration_only(latency_plugin):
    """Streaming requests with ttfb_ms=None record only duration."""
    payload = _make_latency_payload(latency_ms=800.0, ttfb_ms=None, streaming=True)

    with (
        patch("mellea.telemetry.metrics.record_request_duration") as mock_dur,
        patch("mellea.telemetry.metrics.record_ttfb") as mock_ttfb,
    ):
        await latency_plugin.record_latency_metrics(payload, {})

        mock_dur.assert_called_once()
        mock_ttfb.assert_not_called()


@pytest.mark.asyncio
async def test_latency_missing_model_provider(latency_plugin):
    """model/provider default to 'unknown' when None."""
    payload = _make_latency_payload(
        latency_ms=500.0, streaming=False, model=None, provider=None
    )

    with (
        patch("mellea.telemetry.metrics.record_request_duration") as mock_dur,
        patch("mellea.telemetry.metrics.record_ttfb"),
    ):
        await latency_plugin.record_latency_metrics(payload, {})

        mock_dur.assert_called_once_with(
            duration_s=0.5, model="unknown", provider="unknown", streaming=False
        )
