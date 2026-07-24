# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for metrics plugins."""

from unittest.mock import patch

import pytest

pytest.importorskip("cpex", reason="cpex not installed — install mellea[hooks]")

from mellea.core.base import GenerationMetadata, ModelOutputThunk
from mellea.core.requirement import PartialValidationResult
from mellea.plugins.hooks.generation import (
    GenerationBatchErrorPayload,
    GenerationBatchPostCallPayload,
    GenerationErrorPayload,
    GenerationPostCallPayload,
)
from mellea.plugins.hooks.intrinsic import (
    IntrinsicInvocationCompletePayload,
    IntrinsicPhaseCompletePayload,
)
from mellea.plugins.hooks.sampling import (
    SamplingIterationPayload,
    SamplingLoopEndPayload,
)
from mellea.plugins.hooks.streaming import StreamingEndPayload, StreamingEventPayload
from mellea.plugins.hooks.tool import ToolPostInvokePayload
from mellea.plugins.hooks.validation import ValidationPostCheckPayload
from mellea.stdlib.streaming import ChunkEvent, QuickCheckEvent
from mellea.telemetry.metrics import (
    ERROR_TYPE_TIMEOUT,
    ERROR_TYPE_TRANSPORT_ERROR,
    ERROR_TYPE_UNKNOWN,
)
from mellea.telemetry.metrics_plugins import (
    CostMetricsPlugin,
    ErrorMetricsPlugin,
    IntrinsicMetricsPlugin,
    LatencyMetricsPlugin,
    RequirementMetricsPlugin,
    SamplingMetricsPlugin,
    TokenMetricsPlugin,
    ToolMetricsPlugin,
)


@pytest.fixture
def token_plugin():
    return TokenMetricsPlugin()


def _make_token_payload(usage=None, model="test-model", provider="test-provider"):
    """Create a GenerationPostCallPayload with a ModelOutputThunk."""
    mot = ModelOutputThunk(value="hello")
    mot.generation = GenerationMetadata(usage=usage, model=model, provider=provider)
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


# TokenMetricsPlugin batch tests


def _make_batch_post_payload(
    usage=None, model="batch-model", provider="batch-provider", latency_ms=300.0
):
    """Create a GenerationBatchPostCallPayload."""
    return GenerationBatchPostCallPayload(
        generation_id="batch-1",
        model_outputs=[],
        usage=usage,
        model=model,
        provider=provider,
        latency_ms=latency_ms,
    )


@pytest.mark.asyncio
async def test_record_batch_token_metrics_with_usage(token_plugin):
    """Batch token metrics are recorded from payload.usage."""
    payload = _make_batch_post_payload(
        usage={"prompt_tokens": 50, "completion_tokens": 25, "total_tokens": 75}
    )

    with patch("mellea.telemetry.metrics.record_token_usage_metrics") as mock_record:
        await token_plugin.record_batch_token_metrics(payload, {})

        mock_record.assert_called_once_with(
            input_tokens=50,
            output_tokens=25,
            model="batch-model",
            provider="batch-provider",
        )


@pytest.mark.asyncio
async def test_record_batch_token_metrics_no_usage(token_plugin):
    """Batch token metrics are not recorded when usage is None."""
    payload = _make_batch_post_payload(usage=None)

    with patch("mellea.telemetry.metrics.record_token_usage_metrics") as mock_record:
        await token_plugin.record_batch_token_metrics(payload, {})

        mock_record.assert_not_called()


@pytest.mark.asyncio
async def test_record_batch_token_metrics_missing_model_provider(token_plugin):
    """Batch fallback to 'unknown' when model/provider are None."""
    payload = _make_batch_post_payload(
        usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        model=None,
        provider=None,
    )

    with patch("mellea.telemetry.metrics.record_token_usage_metrics") as mock_record:
        await token_plugin.record_batch_token_metrics(payload, {})

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
    mot.generation = GenerationMetadata(
        model=model, provider=provider, ttfb_ms=ttfb_ms, streaming=streaming
    )
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


# LatencyMetricsPlugin batch tests


@pytest.mark.asyncio
async def test_batch_latency_records_duration_non_streaming(latency_plugin):
    """Batch generations record duration with streaming=False."""
    payload = _make_batch_post_payload(latency_ms=1500.0)

    with (
        patch("mellea.telemetry.metrics.record_request_duration") as mock_dur,
        patch("mellea.telemetry.metrics.record_ttfb") as mock_ttfb,
    ):
        await latency_plugin.record_batch_latency_metrics(payload, {})

        mock_dur.assert_called_once_with(
            duration_s=1.5,
            model="batch-model",
            provider="batch-provider",
            streaming=False,
        )
        mock_ttfb.assert_not_called()


@pytest.mark.asyncio
async def test_batch_latency_missing_model_provider(latency_plugin):
    """Batch model/provider default to 'unknown' when None."""
    payload = _make_batch_post_payload(latency_ms=400.0, model=None, provider=None)

    with patch("mellea.telemetry.metrics.record_request_duration") as mock_dur:
        await latency_plugin.record_batch_latency_metrics(payload, {})

        mock_dur.assert_called_once_with(
            duration_s=0.4, model="unknown", provider="unknown", streaming=False
        )


# ErrorMetricsPlugin tests


@pytest.fixture
def error_plugin():
    return ErrorMetricsPlugin()


def _make_error_payload(exc, model="test-model", provider="test-provider"):
    """Create a GenerationErrorPayload wrapping the given exception."""
    mot = ModelOutputThunk(value="")
    mot.generation = GenerationMetadata(model=model, provider=provider)
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


@pytest.mark.asyncio
async def test_error_plugin_records_streaming_exception(error_plugin):
    """streaming_end carrying an exception records an error with its model/provider."""
    payload = StreamingEndPayload(
        streaming_id="sid",
        success=False,
        exception=ValueError("boom"),
        model="m",
        provider="p",
    )

    with patch("mellea.telemetry.metrics.record_error") as mock_record:
        await error_plugin.record_streaming_error_metrics(payload, {})

        mock_record.assert_called_once()
        kwargs = mock_record.call_args.kwargs
        assert kwargs["model"] == "m"
        assert kwargs["provider"] == "p"
        assert kwargs["exception_class"] == "ValueError"


@pytest.mark.asyncio
async def test_error_plugin_skips_streaming_end_without_exception(error_plugin):
    """A non-exception streaming_end (success or validation-fail) records no error."""
    payload = StreamingEndPayload(streaming_id="sid", success=False, exception=None)

    with patch("mellea.telemetry.metrics.record_error") as mock_record:
        await error_plugin.record_streaming_error_metrics(payload, {})

        mock_record.assert_not_called()


# ErrorMetricsPlugin batch tests


def _make_batch_error_payload(exc, model="batch-model", provider="batch-provider"):
    """Create a GenerationBatchErrorPayload wrapping the given exception."""
    return GenerationBatchErrorPayload(
        generation_id="batch-err",
        exception=exc,
        model=model,
        provider=provider,
        latency_ms=10.0,
    )


@pytest.mark.asyncio
async def test_batch_error_plugin_records_correct_type(error_plugin):
    """Batch plugin classifies the exception and calls record_error correctly."""
    payload = _make_batch_error_payload(TimeoutError("timed out"))

    with patch("mellea.telemetry.metrics.record_error") as mock_record:
        await error_plugin.record_batch_error_metrics(payload, {})

        mock_record.assert_called_once_with(
            error_type=ERROR_TYPE_TIMEOUT,
            model="batch-model",
            provider="batch-provider",
            exception_class="TimeoutError",
        )


@pytest.mark.asyncio
async def test_batch_error_plugin_unknown_exception(error_plugin):
    """Batch plugin classifies unknown exceptions correctly."""
    payload = _make_batch_error_payload(ValueError("nope"))

    with patch("mellea.telemetry.metrics.record_error") as mock_record:
        await error_plugin.record_batch_error_metrics(payload, {})

        mock_record.assert_called_once_with(
            error_type=ERROR_TYPE_UNKNOWN,
            model="batch-model",
            provider="batch-provider",
            exception_class="ValueError",
        )


@pytest.mark.asyncio
async def test_batch_error_plugin_unknown_model_provider_fallback(error_plugin):
    """Batch model/provider fall back to 'unknown' when None."""
    payload = _make_batch_error_payload(
        ConnectionError("refused"), model=None, provider=None
    )

    with patch("mellea.telemetry.metrics.record_error") as mock_record:
        await error_plugin.record_batch_error_metrics(payload, {})

        mock_record.assert_called_once_with(
            error_type=ERROR_TYPE_TRANSPORT_ERROR,
            model="unknown",
            provider="unknown",
            exception_class="ConnectionError",
        )


# CostMetricsPlugin tests


@pytest.fixture
def cost_plugin():
    return CostMetricsPlugin()


def _make_cost_payload(usage=None, model="test-model", provider="test-provider"):
    """Create a GenerationPostCallPayload for cost tests."""
    mot = ModelOutputThunk(value="hello")
    mot.generation = GenerationMetadata(usage=usage, model=model, provider=provider)
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
            model="test-model",
            provider="test-provider",
            prompt_tokens=100,
            completion_tokens=50,
            cached_tokens=0,
            cache_creation_tokens=0,
        )
        mock_record.assert_called_once_with(
            cost=0.0042, model="test-model", provider="test-provider"
        )


@pytest.mark.asyncio
async def test_cost_plugin_cache_tokens_forwarded(cost_plugin):
    """Cache token fields are extracted and forwarded to compute_cost correctly.

    Simulates LiteLLM-normalised Anthropic usage where prompt_tokens already
    includes cache_creation and cache_read tokens (40 base + 50 read + 10 write = 100).
    """
    payload = _make_cost_payload(
        usage={
            "prompt_tokens": 100,
            "completion_tokens": 20,
            "total_tokens": 120,
            "prompt_tokens_details": {"cached_tokens": 50},
            "cache_creation_input_tokens": 10,
        }
    )

    with (
        patch("mellea.telemetry.pricing.compute_cost", return_value=0.005) as mock_cost,
        patch("mellea.telemetry.metrics.record_cost"),
    ):
        await cost_plugin.record_cost_metrics(payload, {})

        mock_cost.assert_called_once_with(
            model="test-model",
            provider="test-provider",
            prompt_tokens=100,  # full amount; litellm handles cached-token deduction internally
            completion_tokens=20,
            cached_tokens=50,
            cache_creation_tokens=10,
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
    """Plugin does not call record_cost when mot.generation.usage is None."""
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
            model="unknown",
            provider=None,  # gen.provider is None; "unknown" fallback only used for record_cost
            prompt_tokens=10,
            completion_tokens=5,
            cached_tokens=0,
            cache_creation_tokens=0,
        )
        mock_record.assert_called_once_with(
            cost=0.001, model="unknown", provider="unknown"
        )


# CostMetricsPlugin batch tests


@pytest.mark.asyncio
async def test_batch_cost_plugin_records_cost_for_known_model(cost_plugin):
    """Batch plugin calls record_cost when compute_cost returns a value."""
    payload = _make_batch_post_payload(
        usage={"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}
    )

    with (
        patch(
            "mellea.telemetry.pricing.compute_cost", return_value=0.0042
        ) as mock_cost,
        patch("mellea.telemetry.metrics.record_cost") as mock_record,
    ):
        await cost_plugin.record_batch_cost_metrics(payload, {})

        mock_cost.assert_called_once_with(
            model="batch-model",
            provider="batch-provider",
            prompt_tokens=100,
            completion_tokens=50,
            cached_tokens=0,
            cache_creation_tokens=0,
        )
        mock_record.assert_called_once_with(
            cost=0.0042, model="batch-model", provider="batch-provider"
        )


@pytest.mark.asyncio
async def test_batch_cost_plugin_cache_tokens_forwarded(cost_plugin):
    """Batch plugin forwards cache token fields to compute_cost correctly."""
    payload = _make_batch_post_payload(
        usage={
            "prompt_tokens": 100,
            "completion_tokens": 20,
            "total_tokens": 120,
            "prompt_tokens_details": {"cached_tokens": 50},
            "cache_creation_input_tokens": 10,
        }
    )

    with (
        patch("mellea.telemetry.pricing.compute_cost", return_value=0.005) as mock_cost,
        patch("mellea.telemetry.metrics.record_cost"),
    ):
        await cost_plugin.record_batch_cost_metrics(payload, {})

        mock_cost.assert_called_once_with(
            model="batch-model",
            provider="batch-provider",
            prompt_tokens=100,
            completion_tokens=20,
            cached_tokens=50,
            cache_creation_tokens=10,
        )


@pytest.mark.asyncio
async def test_batch_cost_plugin_skips_unknown_model(cost_plugin):
    """Batch plugin does not call record_cost when compute_cost returns None."""
    payload = _make_batch_post_payload(
        usage={"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}
    )

    with (
        patch("mellea.telemetry.pricing.compute_cost", return_value=None),
        patch("mellea.telemetry.metrics.record_cost") as mock_record,
    ):
        await cost_plugin.record_batch_cost_metrics(payload, {})

        mock_record.assert_not_called()


@pytest.mark.asyncio
async def test_batch_cost_plugin_skips_none_usage(cost_plugin):
    """Batch plugin does not call record_cost when payload.usage is None."""
    payload = _make_batch_post_payload(usage=None)

    with (
        patch("mellea.telemetry.pricing.compute_cost") as mock_cost,
        patch("mellea.telemetry.metrics.record_cost") as mock_record,
    ):
        await cost_plugin.record_batch_cost_metrics(payload, {})

        mock_cost.assert_not_called()
        mock_record.assert_not_called()


@pytest.mark.asyncio
async def test_batch_cost_plugin_unknown_model_provider_fallback(cost_plugin):
    """Batch model/provider fall back to 'unknown' when None on the payload."""
    payload = _make_batch_post_payload(
        usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        model=None,
        provider=None,
    )

    with (
        patch("mellea.telemetry.pricing.compute_cost", return_value=0.001) as mock_cost,
        patch("mellea.telemetry.metrics.record_cost") as mock_record,
    ):
        await cost_plugin.record_batch_cost_metrics(payload, {})

        mock_cost.assert_called_once_with(
            model="unknown",
            provider=None,
            prompt_tokens=10,
            completion_tokens=5,
            cached_tokens=0,
            cache_creation_tokens=0,
        )
        mock_record.assert_called_once_with(
            cost=0.001, model="unknown", provider="unknown"
        )


# SamplingMetricsPlugin tests


class _PassResult:
    reason = None

    def __bool__(self) -> bool:
        return True


class _FailResult:
    def __init__(self, reason: str = "constraint not met") -> None:
        self.reason = reason

    def __bool__(self) -> bool:
        return False


class _FakeReq:
    def validation_fn(self, _ctx):
        return None


@pytest.fixture
def sampling_plugin():
    return SamplingMetricsPlugin()


@pytest.mark.asyncio
async def test_sampling_plugin_records_attempt(sampling_plugin):
    """Plugin calls record_sampling_attempt with the strategy name on each iteration."""
    payload = SamplingIterationPayload(strategy_name="RejectionSamplingStrategy")

    with patch("mellea.telemetry.metrics.record_sampling_attempt") as mock_record:
        await sampling_plugin.record_sampling_attempt(payload, {})

        mock_record.assert_called_once_with("RejectionSamplingStrategy")


@pytest.mark.asyncio
async def test_sampling_plugin_attempt_empty_name_falls_back_to_unknown(
    sampling_plugin,
):
    """Empty strategy_name falls back to 'unknown'."""
    payload = SamplingIterationPayload(strategy_name="")

    with patch("mellea.telemetry.metrics.record_sampling_attempt") as mock_record:
        await sampling_plugin.record_sampling_attempt(payload, {})

        mock_record.assert_called_once_with("unknown")


@pytest.mark.asyncio
async def test_sampling_plugin_records_success_outcome(sampling_plugin):
    """Plugin calls record_sampling_outcome(success=True) on a successful loop end."""
    payload = SamplingLoopEndPayload(
        strategy_name="RejectionSamplingStrategy", success=True
    )

    with patch("mellea.telemetry.metrics.record_sampling_outcome") as mock_record:
        await sampling_plugin.record_sampling_outcome(payload, {})

        mock_record.assert_called_once_with("RejectionSamplingStrategy", True)


@pytest.mark.asyncio
async def test_sampling_plugin_records_failure_outcome(sampling_plugin):
    """Plugin calls record_sampling_outcome(success=False) on a failed loop end."""
    payload = SamplingLoopEndPayload(strategy_name="MultiTurnStrategy", success=False)

    with patch("mellea.telemetry.metrics.record_sampling_outcome") as mock_record:
        await sampling_plugin.record_sampling_outcome(payload, {})

        mock_record.assert_called_once_with("MultiTurnStrategy", False)


@pytest.mark.asyncio
async def test_sampling_plugin_records_streaming_success_outcome(sampling_plugin):
    """streaming_end with success=True records a `stream_with_chunking` success."""
    payload = StreamingEndPayload(streaming_id="sid", success=True)

    with patch("mellea.telemetry.metrics.record_sampling_outcome") as mock_record:
        await sampling_plugin.record_streaming_outcome(payload, {})

        mock_record.assert_called_once_with("stream_with_chunking", True)


@pytest.mark.asyncio
async def test_sampling_plugin_records_streaming_failure_outcome(sampling_plugin):
    """streaming_end with success=False records a `stream_with_chunking` failure."""
    payload = StreamingEndPayload(streaming_id="sid", success=False)

    with patch("mellea.telemetry.metrics.record_sampling_outcome") as mock_record:
        await sampling_plugin.record_streaming_outcome(payload, {})

        mock_record.assert_called_once_with("stream_with_chunking", False)


# RequirementMetricsPlugin tests


@pytest.fixture
def requirement_plugin():
    return RequirementMetricsPlugin()


@pytest.mark.asyncio
async def test_requirement_plugin_records_checks_and_no_failures_when_all_pass(
    requirement_plugin,
):
    """When all requirements pass, only checks are recorded."""
    req = _FakeReq()
    payload = ValidationPostCheckPayload(
        requirements=[req],
        results=[_PassResult()],
        all_validations_passed=True,
        passed_count=1,
        failed_count=0,
    )

    with (
        patch("mellea.telemetry.metrics.record_requirement_check") as mock_check,
        patch("mellea.telemetry.metrics.record_requirement_failure") as mock_fail,
    ):
        await requirement_plugin.record_requirement_metrics(payload, {})

        mock_check.assert_called_once_with("_FakeReq")
        mock_fail.assert_not_called()


@pytest.mark.asyncio
async def test_requirement_plugin_records_failure_with_reason(requirement_plugin):
    """Failed requirements record both a check and a failure with the reason."""
    req = _FakeReq()
    payload = ValidationPostCheckPayload(
        requirements=[req],
        results=[_FailResult("output too short")],
        all_validations_passed=False,
        passed_count=0,
        failed_count=1,
    )

    with (
        patch("mellea.telemetry.metrics.record_requirement_check") as mock_check,
        patch("mellea.telemetry.metrics.record_requirement_failure") as mock_fail,
    ):
        await requirement_plugin.record_requirement_metrics(payload, {})

        mock_check.assert_called_once_with("_FakeReq")
        mock_fail.assert_called_once_with("_FakeReq", "output too short")


@pytest.mark.asyncio
async def test_requirement_plugin_mixed_pass_fail(requirement_plugin):
    """Mixed results record a check for each and a failure only for failing ones."""
    req_a = _FakeReq()
    req_b = _FakeReq()
    payload = ValidationPostCheckPayload(
        requirements=[req_a, req_b],
        results=[_PassResult(), _FailResult("constraint not met")],
        all_validations_passed=False,
        passed_count=1,
        failed_count=1,
    )

    with (
        patch("mellea.telemetry.metrics.record_requirement_check") as mock_check,
        patch("mellea.telemetry.metrics.record_requirement_failure") as mock_fail,
    ):
        await requirement_plugin.record_requirement_metrics(payload, {})

        assert mock_check.call_count == 2
        mock_fail.assert_called_once_with("_FakeReq", "constraint not met")


@pytest.mark.asyncio
async def test_requirement_plugin_failure_with_no_reason_uses_default(
    requirement_plugin,
):
    """A None reason falls back to the default reason."""
    req = _FakeReq()
    payload = ValidationPostCheckPayload(
        requirements=[req],
        results=[_FailResult(reason=None)],  # type: ignore[arg-type]
        all_validations_passed=False,
        passed_count=0,
        failed_count=1,
    )

    with (
        patch("mellea.telemetry.metrics.record_requirement_check"),
        patch("mellea.telemetry.metrics.record_requirement_failure") as mock_fail,
    ):
        await requirement_plugin.record_requirement_metrics(payload, {})

        mock_fail.assert_called_once_with("_FakeReq", "LLM judgment")


class _LengthRequirement:
    """Stand-in whose class name is the metric label, mirroring real requirements."""


class _TopicRequirement:
    """Stand-in whose class name is the metric label, mirroring real requirements."""


@pytest.mark.asyncio
async def test_requirement_metrics_records_per_requirement_type_on_quick_check(
    requirement_plugin,
):
    qc = QuickCheckEvent(
        chunk_index=0,
        attempt=1,
        passed=False,
        results=[
            PartialValidationResult("pass"),
            PartialValidationResult("fail", reason="too short"),
        ],
    )
    payload = StreamingEventPayload(
        streaming_id="sid",
        event=qc,
        requirements=[_LengthRequirement(), _TopicRequirement()],
    )

    with (
        patch("mellea.telemetry.metrics.record_requirement_check") as mock_check,
        patch("mellea.telemetry.metrics.record_requirement_failure") as mock_fail,
    ):
        await requirement_plugin.record_streaming_requirement_metrics(payload, {})

    assert mock_check.call_count == 2
    mock_check.assert_any_call("_LengthRequirement")
    mock_check.assert_any_call("_TopicRequirement")
    mock_fail.assert_called_once_with("_TopicRequirement", "too short")


@pytest.mark.asyncio
async def test_requirement_metrics_ignores_non_quick_check_events(requirement_plugin):
    payload = StreamingEventPayload(
        streaming_id="sid", event=ChunkEvent(text="hi", chunk_index=0, attempt=1)
    )

    with (
        patch("mellea.telemetry.metrics.record_requirement_check") as mock_check,
        patch("mellea.telemetry.metrics.record_requirement_failure") as mock_fail,
    ):
        await requirement_plugin.record_streaming_requirement_metrics(payload, {})

    mock_check.assert_not_called()
    mock_fail.assert_not_called()


# ToolMetricsPlugin tests


class _MockToolCall:
    def __init__(self, name: str) -> None:
        self.name = name


@pytest.fixture
def tool_plugin():
    return ToolMetricsPlugin()


@pytest.mark.asyncio
async def test_tool_plugin_records_success(tool_plugin):
    """Successful tool calls are recorded with status='success'."""
    payload = ToolPostInvokePayload(
        model_tool_call=_MockToolCall("search"), success=True
    )

    with patch("mellea.telemetry.metrics.record_tool_call") as mock_record:
        await tool_plugin.record_tool_call(payload, {})

        mock_record.assert_called_once_with("search", "success")


@pytest.mark.asyncio
async def test_tool_plugin_records_failure(tool_plugin):
    """Failed tool calls are recorded with status='failure'."""
    payload = ToolPostInvokePayload(
        model_tool_call=_MockToolCall("calculator"), success=False
    )

    with patch("mellea.telemetry.metrics.record_tool_call") as mock_record:
        await tool_plugin.record_tool_call(payload, {})

        mock_record.assert_called_once_with("calculator", "failure")


@pytest.mark.asyncio
async def test_tool_plugin_none_tool_call_falls_back_to_unknown(tool_plugin):
    """A None model_tool_call falls back to tool name 'unknown'."""
    payload = ToolPostInvokePayload(model_tool_call=None, success=True)

    with patch("mellea.telemetry.metrics.record_tool_call") as mock_record:
        await tool_plugin.record_tool_call(payload, {})

        mock_record.assert_called_once_with("unknown", "success")


# IntrinsicMetricsPlugin tests


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
async def test_record_intrinsic_invocation_missing_revision_defaults_to_unpinned(
    intrinsic_plugin,
):
    """A None revision is normalized to 'unpinned' before being recorded."""
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
        revision="unpinned",
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
