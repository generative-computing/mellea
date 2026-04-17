"""Unit tests for TokenMetricsPlugin, LatencyMetricsPlugin, and ErrorMetricsPlugin."""

from unittest.mock import patch

import pytest

pytest.importorskip("cpex", reason="cpex not installed — install mellea[hooks]")

from mellea.core.base import ModelOutputThunk
from mellea.plugins.hooks.generation import (
    GenerationErrorPayload,
    GenerationPostCallPayload,
)
from mellea.telemetry.metrics import (
    ERROR_TYPE_TIMEOUT,
    ERROR_TYPE_TRANSPORT_ERROR,
    ERROR_TYPE_UNKNOWN,
)
from mellea.telemetry.metrics_plugins import (
    CostMetricsPlugin,
    ErrorMetricsPlugin,
    LatencyMetricsPlugin,
    TokenMetricsPlugin,
)


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


# ErrorMetricsPlugin tests


@pytest.fixture
def error_plugin():
    return ErrorMetricsPlugin()


def _make_error_payload(exc, model="test-model", provider="test-provider"):
    """Create a GenerationErrorPayload wrapping the given exception."""
    mot = ModelOutputThunk(value="")
    mot.model = model
    mot.provider = provider
    return GenerationErrorPayload(exception=exc, model_output=mot)


@pytest.mark.asyncio
async def test_error_plugin_records_correct_type(error_plugin):
    """Plugin classifies the exception and calls record_error with the right type."""
    payload = _make_error_payload(TimeoutError("timed out"))

    with patch("mellea.telemetry.metrics.record_error") as mock_record:
        await error_plugin.record_error_metrics(payload, {})

        mock_record.assert_called_once_with(
            error_type=ERROR_TYPE_TIMEOUT,
            model="test-model",
            provider="test-provider",
            exception_class="TimeoutError",
        )


@pytest.mark.asyncio
async def test_error_plugin_unknown_exception(error_plugin):
    """Unrecognized exceptions are classified as unknown."""
    payload = _make_error_payload(ValueError("something broke"))

    with patch("mellea.telemetry.metrics.record_error") as mock_record:
        await error_plugin.record_error_metrics(payload, {})

        mock_record.assert_called_once_with(
            error_type=ERROR_TYPE_UNKNOWN,
            model="test-model",
            provider="test-provider",
            exception_class="ValueError",
        )


@pytest.mark.asyncio
async def test_error_plugin_falls_back_to_unknown_when_model_none(error_plugin):
    """model/provider fall back to 'unknown' when None on the MOT."""
    payload = _make_error_payload(ConnectionError("refused"), model=None, provider=None)

    with patch("mellea.telemetry.metrics.record_error") as mock_record:
        await error_plugin.record_error_metrics(payload, {})

        mock_record.assert_called_once_with(
            error_type=ERROR_TYPE_TRANSPORT_ERROR,
            model="unknown",
            provider="unknown",
            exception_class="ConnectionError",
        )


@pytest.mark.asyncio
async def test_error_plugin_handles_none_model_output(error_plugin):
    """Plugin handles a None model_output gracefully."""
    payload = GenerationErrorPayload(
        exception=RuntimeError("queue error"), model_output=None
    )

    with patch("mellea.telemetry.metrics.record_error") as mock_record:
        await error_plugin.record_error_metrics(payload, {})

        mock_record.assert_called_once_with(
            error_type=ERROR_TYPE_UNKNOWN,
            model="unknown",
            provider="unknown",
            exception_class="RuntimeError",
        )


# CostMetricsPlugin tests


@pytest.fixture
def cost_plugin():
    return CostMetricsPlugin()


def _make_cost_payload(usage=None, model="test-model", provider="test-provider"):
    """Create a GenerationPostCallPayload for cost tests."""
    mot = ModelOutputThunk(value="hello")
    mot.usage = usage
    mot.model = model
    mot.provider = provider
    return GenerationPostCallPayload(model_output=mot)


@pytest.mark.asyncio
async def test_cost_plugin_records_cost_for_known_model(cost_plugin):
    """Plugin calls record_cost when compute_cost returns a value."""
    payload = _make_cost_payload(
        usage={"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}
    )

    with (
        patch(
            "mellea.telemetry.pricing.compute_cost", return_value=0.0042
        ) as mock_cost,
        patch("mellea.telemetry.metrics.record_cost") as mock_record,
    ):
        await cost_plugin.record_cost_metrics(payload, {})

        mock_cost.assert_called_once_with(
            model="test-model", input_tokens=100, output_tokens=50
        )
        mock_record.assert_called_once_with(
            cost=0.0042, model="test-model", provider="test-provider"
        )


@pytest.mark.asyncio
async def test_cost_plugin_skips_unknown_model(cost_plugin):
    """Plugin does not call record_cost when compute_cost returns None."""
    payload = _make_cost_payload(
        usage={"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}
    )

    with (
        patch("mellea.telemetry.pricing.compute_cost", return_value=None),
        patch("mellea.telemetry.metrics.record_cost") as mock_record,
    ):
        await cost_plugin.record_cost_metrics(payload, {})

        mock_record.assert_not_called()


@pytest.mark.asyncio
async def test_cost_plugin_skips_none_usage(cost_plugin):
    """Plugin does not call record_cost when mot.usage is None."""
    payload = _make_cost_payload(usage=None)

    with (
        patch("mellea.telemetry.pricing.compute_cost") as mock_cost,
        patch("mellea.telemetry.metrics.record_cost") as mock_record,
    ):
        await cost_plugin.record_cost_metrics(payload, {})

        mock_cost.assert_not_called()
        mock_record.assert_not_called()


@pytest.mark.asyncio
async def test_cost_plugin_unknown_model_provider_fallback(cost_plugin):
    """model/provider fall back to 'unknown' when None on the MOT."""
    payload = _make_cost_payload(
        usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        model=None,
        provider=None,
    )

    with (
        patch("mellea.telemetry.pricing.compute_cost", return_value=0.001) as mock_cost,
        patch("mellea.telemetry.metrics.record_cost") as mock_record,
    ):
        await cost_plugin.record_cost_metrics(payload, {})

        mock_cost.assert_called_once_with(
            model="unknown", input_tokens=10, output_tokens=5
        )
        mock_record.assert_called_once_with(
            cost=0.001, model="unknown", provider="unknown"
        )
