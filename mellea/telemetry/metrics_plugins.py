# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Metrics plugins for recording telemetry data via hooks.

This module contains plugins that hook into the generation pipeline to
automatically record metrics when enabled. Currently includes:

- TokenMetricsPlugin: Records token usage statistics from generation usage data
- LatencyMetricsPlugin: Records request duration and TTFB latency histograms
- ErrorMetricsPlugin: Records LLM error counts categorized by semantic error type
- CostMetricsPlugin: Records estimated request cost in USD from pricing registry
- SamplingMetricsPlugin: Records sampling attempt/success/failure counts per strategy
- RequirementMetricsPlugin: Records requirement validation check and failure counts
- ToolMetricsPlugin: Records tool invocation counts by name and status
- IntrinsicMetricsPlugin: Records adapter function (intrinsic) invocation and
  phase-duration metrics
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from mellea.plugins.base import Plugin
from mellea.plugins.decorators import hook
from mellea.plugins.types import PluginMode

if TYPE_CHECKING:
    from mellea.core.base import GenerationMetadata
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
    from mellea.plugins.hooks.streaming import (
        StreamingEndPayload,
        StreamingEventPayload,
    )
    from mellea.plugins.hooks.tool import ToolPostInvokePayload
    from mellea.plugins.hooks.validation import ValidationPostCheckPayload


class TokenMetricsPlugin(Plugin, name="token_metrics", priority=50):
    """Records token usage metrics from generation outputs.

    This plugin hooks into the generation_post_call and
    generation_batch_post_call events to automatically record token usage
    metrics when usage data is present.

    The plugin reads the standardized usage field (OpenAI-compatible format)
    and records metrics following OpenTelemetry Gen-AI semantic conventions.
    """

    @hook("generation_post_call", mode=PluginMode.FIRE_AND_FORGET)
    async def record_token_metrics(
        self, payload: GenerationPostCallPayload, context: dict[str, Any]
    ) -> None:
        """Record token metrics after generation completes.

        Args:
            payload: Contains the model_output (ModelOutputThunk) with usage data
            context: Plugin context (unused)
        """
        from mellea.telemetry.metrics import record_token_usage_metrics

        gen = payload.model_output.generation
        if gen.usage is None:
            return

        # Record metrics (no-op if metrics disabled)
        record_token_usage_metrics(
            input_tokens=gen.usage.get("prompt_tokens"),
            output_tokens=gen.usage.get("completion_tokens"),
            model=gen.model or "unknown",
            provider=gen.provider or "unknown",
        )

    @hook("generation_batch_post_call", mode=PluginMode.FIRE_AND_FORGET)
    async def record_batch_token_metrics(
        self, payload: GenerationBatchPostCallPayload, context: dict[str, Any]
    ) -> None:
        """Record token metrics after a batch generation completes.

        Args:
            payload: Contains the batch-level usage dict, model, and provider.
            context: Plugin context (unused).
        """
        from mellea.telemetry.metrics import record_token_usage_metrics

        if payload.usage is None:
            return

        record_token_usage_metrics(
            input_tokens=payload.usage.get("prompt_tokens"),
            output_tokens=payload.usage.get("completion_tokens"),
            model=payload.model or "unknown",
            provider=payload.provider or "unknown",
        )


class LatencyMetricsPlugin(Plugin, name="latency_metrics", priority=51):
    """Records request duration and TTFB latency metrics from generation outputs.

    This plugin hooks into the generation_post_call and
    generation_batch_post_call events to automatically record latency
    metrics. It records total request duration for every request and
    time-to-first-token (TTFB) for streaming requests.
    """

    @hook("generation_post_call", mode=PluginMode.FIRE_AND_FORGET)
    async def record_latency_metrics(
        self, payload: GenerationPostCallPayload, context: dict[str, Any]
    ) -> None:
        """Record latency metrics after generation completes.

        Args:
            payload: Contains latency_ms and model_output
            context: Plugin context (unused)
        """
        from mellea.telemetry.metrics import record_request_duration, record_ttfb

        gen = payload.model_output.generation
        model = gen.model or "unknown"
        provider = gen.provider or "unknown"

        # Record total request duration (convert ms → seconds)
        record_request_duration(
            duration_s=payload.latency_ms / 1000.0,
            model=model,
            provider=provider,
            streaming=gen.streaming,
        )

        # Record TTFB only for streaming requests with a measured value
        if gen.streaming and gen.ttfb_ms is not None:
            record_ttfb(ttfb_s=gen.ttfb_ms / 1000.0, model=model, provider=provider)

    @hook("generation_batch_post_call", mode=PluginMode.FIRE_AND_FORGET)
    async def record_batch_latency_metrics(
        self, payload: GenerationBatchPostCallPayload, context: dict[str, Any]
    ) -> None:
        """Record request duration after a batch generation completes.

        Batch generations (`generate_from_raw`) are non-streaming, so only the
        total request duration is recorded; TTFB does not apply.

        Args:
            payload: Contains latency_ms, model, and provider for the batch.
            context: Plugin context (unused).
        """
        from mellea.telemetry.metrics import record_request_duration

        record_request_duration(
            duration_s=payload.latency_ms / 1000.0,
            model=payload.model or "unknown",
            provider=payload.provider or "unknown",
            streaming=False,
        )


class ErrorMetricsPlugin(Plugin, name="error_metrics", priority=52):
    """Records LLM error counts from generation errors.

    This plugin hooks into the generation_error and generation_batch_error
    events to classify exceptions by semantic error type and increment the
    `mellea.llm.errors` counter.
    """

    @hook("generation_error", mode=PluginMode.FIRE_AND_FORGET)
    async def record_error_metrics(
        self, payload: GenerationErrorPayload, context: dict[str, Any]
    ) -> None:
        """Record error metrics when a generation error occurs.

        Args:
            payload: Contains the exception and the ModelOutputThunk at the time of the error.
            context: Plugin context (unused).
        """
        from mellea.core.base import GenerationMetadata
        from mellea.telemetry.metrics import classify_error, record_error

        gen = (
            payload.model_output.generation
            if payload.model_output is not None
            else GenerationMetadata()
        )
        error_type = classify_error(payload.exception)
        record_error(
            error_type=error_type,
            model=gen.model or "unknown",
            provider=gen.provider or "unknown",
            exception_class=type(payload.exception).__name__,
        )

    @hook("generation_batch_error", mode=PluginMode.FIRE_AND_FORGET)
    async def record_batch_error_metrics(
        self, payload: GenerationBatchErrorPayload, context: dict[str, Any]
    ) -> None:
        """Record error metrics when a batch generation fails.

        Args:
            payload: Contains the exception, model, and provider for the batch.
            context: Plugin context (unused).
        """
        from mellea.telemetry.metrics import classify_error, record_error

        error_type = classify_error(payload.exception)
        record_error(
            error_type=error_type,
            model=payload.model or "unknown",
            provider=payload.provider or "unknown",
            exception_class=type(payload.exception).__name__,
        )

    @hook("streaming_end", mode=PluginMode.FIRE_AND_FORGET)
    async def record_streaming_error_metrics(
        self, payload: StreamingEndPayload, context: dict[str, Any]
    ) -> None:
        """Record error metrics when `stream_with_chunking` ends with an exception.

        Args:
            payload: Contains the exception plus the model and provider from
                the underlying generation.
            context: Plugin context (unused).
        """
        from mellea.telemetry.metrics import classify_error, record_error

        if payload.exception is None:
            return
        record_error(
            error_type=classify_error(payload.exception),
            model=payload.model or "unknown",
            provider=payload.provider or "unknown",
            exception_class=type(payload.exception).__name__,
        )


class CostMetricsPlugin(Plugin, name="cost_metrics", priority=53):
    """Records estimated request cost metrics from generation outputs.

    This plugin hooks into the generation_post_call and
    generation_batch_post_call events to automatically record cost metrics
    when token usage and model pricing data are available. Cost is skipped
    and a warning is logged for models not in the pricing registry.
    """

    @hook("generation_post_call", mode=PluginMode.FIRE_AND_FORGET)
    async def record_cost_metrics(
        self, payload: GenerationPostCallPayload, context: dict[str, Any]
    ) -> None:
        """Record cost metrics after generation completes.

        Args:
            payload: Contains the model_output (ModelOutputThunk) with usage data.
            context: Plugin context (unused).
        """
        from mellea.telemetry.metrics import record_cost
        from mellea.telemetry.pricing import compute_cost

        gen = payload.model_output.generation
        if gen.usage is None:
            return

        model = gen.model or "unknown"
        provider = gen.provider or "unknown"
        details = gen.usage.get("prompt_tokens_details")
        cached_tokens = (
            details.get("cached_tokens") if isinstance(details, dict) else 0
        ) or 0
        cache_creation = gen.usage.get("cache_creation_input_tokens") or 0
        prompt_tokens = gen.usage.get("prompt_tokens") or 0
        cost = compute_cost(
            model=model,
            provider=gen.provider,
            prompt_tokens=prompt_tokens,
            completion_tokens=gen.usage.get("completion_tokens"),
            cached_tokens=cached_tokens,
            cache_creation_tokens=cache_creation,
        )
        if cost is not None:
            record_cost(cost=cost, model=model, provider=provider)

    @hook("generation_batch_post_call", mode=PluginMode.FIRE_AND_FORGET)
    async def record_batch_cost_metrics(
        self, payload: GenerationBatchPostCallPayload, context: dict[str, Any]
    ) -> None:
        """Record cost metrics after a batch generation completes.

        Args:
            payload: Contains the batch-level usage dict, model, and provider.
            context: Plugin context (unused).
        """
        from mellea.telemetry.metrics import record_cost
        from mellea.telemetry.pricing import compute_cost

        if payload.usage is None:
            return

        model = payload.model or "unknown"
        provider = payload.provider or "unknown"
        details = payload.usage.get("prompt_tokens_details")
        cached_tokens = (
            details.get("cached_tokens") if isinstance(details, dict) else 0
        ) or 0
        cache_creation = payload.usage.get("cache_creation_input_tokens") or 0
        prompt_tokens = payload.usage.get("prompt_tokens") or 0
        cost = compute_cost(
            model=model,
            provider=payload.provider,
            prompt_tokens=prompt_tokens,
            completion_tokens=payload.usage.get("completion_tokens"),
            cached_tokens=cached_tokens,
            cache_creation_tokens=cache_creation,
        )
        if cost is not None:
            record_cost(cost=cost, model=model, provider=provider)


class SamplingMetricsPlugin(Plugin, name="sampling_metrics", priority=54):
    """Records sampling loop attempt and outcome metrics.

    Hooks into `sampling_iteration` to count attempts per strategy and
    `sampling_loop_end` to count successes and failures.
    """

    @hook("sampling_iteration", mode=PluginMode.FIRE_AND_FORGET)
    async def record_sampling_attempt(
        self, payload: SamplingIterationPayload, context: dict[str, Any]
    ) -> None:
        """Record one sampling attempt after each iteration.

        Args:
            payload: Contains strategy_name and iteration metadata.
            context: Plugin context (unused).
        """
        from mellea.telemetry.metrics import record_sampling_attempt

        record_sampling_attempt(payload.strategy_name or "unknown")

    @hook("sampling_loop_end", mode=PluginMode.FIRE_AND_FORGET)
    async def record_sampling_outcome(
        self, payload: SamplingLoopEndPayload, context: dict[str, Any]
    ) -> None:
        """Record success or failure when the sampling loop ends.

        Args:
            payload: Contains strategy_name and success flag.
            context: Plugin context (unused).
        """
        from mellea.telemetry.metrics import record_sampling_outcome

        record_sampling_outcome(payload.strategy_name or "unknown", payload.success)

    @hook("streaming_end", mode=PluginMode.FIRE_AND_FORGET)
    async def record_streaming_outcome(
        self, payload: StreamingEndPayload, context: dict[str, Any]
    ) -> None:
        """Record the `stream_with_chunking` outcome when the orchestrator finishes.

        Args:
            payload: Contains the orchestrator's success flag.
            context: Plugin context (unused).
        """
        from mellea.telemetry.metrics import record_sampling_outcome

        record_sampling_outcome("stream_with_chunking", payload.success)


class RequirementMetricsPlugin(Plugin, name="requirement_metrics", priority=55):
    """Records requirement validation check and failure metrics.

    Hooks into `validation_post_check` to count checks and failures per
    requirement type after each validation batch.
    """

    @hook("validation_post_check", mode=PluginMode.FIRE_AND_FORGET)
    async def record_requirement_metrics(
        self, payload: ValidationPostCheckPayload, context: dict[str, Any]
    ) -> None:
        """Record validation checks and failures for each requirement.

        Args:
            payload: Contains requirements list and corresponding results.
            context: Plugin context (unused).
        """
        from mellea.telemetry.metrics import (
            record_requirement_check,
            record_requirement_failure,
        )

        for req, result in zip(payload.requirements, payload.results):
            req_name = type(req).__name__
            record_requirement_check(req_name)
            if not bool(result):
                reason = (
                    getattr(result, "reason", None)
                    if req.validation_fn is not None
                    else None
                ) or "LLM judgment"
                record_requirement_failure(req_name, reason)

    @hook("streaming_event", mode=PluginMode.FIRE_AND_FORGET)
    async def record_streaming_requirement_metrics(
        self, payload: StreamingEventPayload, context: dict[str, Any]
    ) -> None:
        """Record per-chunk requirement metrics for `QuickCheckEvent`s.

        Args:
            payload: Contains the streaming `StreamEvent` and, for a
                `QuickCheckEvent`, the active `requirements` in result order.
            context: Plugin context (unused).
        """
        from mellea.stdlib.streaming import QuickCheckEvent
        from mellea.telemetry.metrics import (
            record_requirement_check,
            record_requirement_failure,
        )

        ev = payload.event
        if not isinstance(ev, QuickCheckEvent):
            return
        for req, pvr in zip(payload.requirements, ev.results):
            req_name = type(req).__name__
            record_requirement_check(req_name)
            if pvr.success == "fail":
                record_requirement_failure(req_name, pvr.reason or "")


class ToolMetricsPlugin(Plugin, name="tool_metrics", priority=56):
    """Records tool invocation metrics.

    Hooks into `tool_post_invoke` to count tool calls by name and success/failure status.
    """

    @hook("tool_post_invoke", mode=PluginMode.FIRE_AND_FORGET)
    async def record_tool_call(
        self, payload: ToolPostInvokePayload, context: dict[str, Any]
    ) -> None:
        """Record one tool invocation after it completes.

        Args:
            payload: Contains model_tool_call (with name) and success flag.
            context: Plugin context (unused).
        """
        from mellea.telemetry.metrics import record_tool_call

        tool_name = (
            payload.model_tool_call.name
            if payload.model_tool_call is not None
            else "unknown"
        )
        status = "success" if payload.success else "failure"
        record_tool_call(tool_name, status)


class IntrinsicMetricsPlugin(Plugin, name="intrinsic_metrics", priority=57):
    """Records adapter function (intrinsic) invocation and phase-duration metrics.

    Hooks into `intrinsic_invocation_complete` and `intrinsic_phase_complete`.
    No production call site fires these hooks yet — real `prepare`/`activate`/
    `generate`/`parse`/`deactivate` wiring lands with the LocalFileBinding and
    EmbeddedBinding lifecycle work (Epic #929 Phase 2 follow-ups).
    """

    @hook("intrinsic_invocation_complete", mode=PluginMode.FIRE_AND_FORGET)
    async def record_intrinsic_invocation(
        self, payload: IntrinsicInvocationCompletePayload, context: dict[str, Any]
    ) -> None:
        """Record one adapter function invocation after it completes.

        Args:
            payload: Contains name, revision, binding_type, adapter_type, and outcome.
            context: Plugin context (unused).
        """
        from mellea.telemetry.metrics import (
            record_intrinsic_invocation,
            record_intrinsic_parse_failure,
        )

        # revision is an optional catalog pin; None means the adapter is unpinned
        # (the actually-served version can't be determined here). Normalise to a
        # string label — "unpinned", not "unknown", since None is a known state.
        revision = payload.revision or "unpinned"
        record_intrinsic_invocation(
            name=payload.name,
            revision=revision,
            binding_type=payload.binding_type,
            adapter_type=payload.adapter_type,
            outcome=payload.outcome,
        )
        if payload.outcome == "schema_error":
            record_intrinsic_parse_failure(payload.name, revision)

    @hook("intrinsic_phase_complete", mode=PluginMode.FIRE_AND_FORGET)
    async def record_intrinsic_phase(
        self, payload: IntrinsicPhaseCompletePayload, context: dict[str, Any]
    ) -> None:
        """Record one adapter function lifecycle phase after it completes.

        Args:
            payload: Contains name, phase, and duration_ms.
            context: Plugin context (unused).
        """
        from mellea.telemetry.metrics import record_intrinsic_phase_duration

        # payload carries milliseconds; the metric is in seconds, matching
        # LatencyMetricsPlugin and the OTel base-unit convention for durations.
        record_intrinsic_phase_duration(
            payload.name, payload.phase, payload.duration_ms / 1000.0
        )


# All metrics plugins to auto-register when metrics are enabled
_METRICS_PLUGIN_CLASSES = (
    TokenMetricsPlugin,
    LatencyMetricsPlugin,
    ErrorMetricsPlugin,
    CostMetricsPlugin,
    SamplingMetricsPlugin,
    RequirementMetricsPlugin,
    ToolMetricsPlugin,
    IntrinsicMetricsPlugin,
)
