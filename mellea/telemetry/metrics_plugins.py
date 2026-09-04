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
- AdapterFunctionMetricsPlugin: Records adapter function invocation and
  phase-duration metrics
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from mellea.plugins.base import Plugin
from mellea.plugins.decorators import hook
from mellea.plugins.types import PluginMode

if TYPE_CHECKING:
    from mellea.core.base import GenerationMetadata
    from mellea.plugins.hooks.adapter_function import (
        AdapterFunctionInvocationCompletePayload,
        AdapterFunctionPhaseCompletePayload,
    )
    from mellea.plugins.hooks.generation import (
        GenerationBatchErrorPayload,
        GenerationBatchPostCallPayload,
        GenerationErrorPayload,
        GenerationEventPayload,
        GenerationPostCallPayload,
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


class TokenMetricsPlugin(Plugin, name="token_metrics", priority=1050):
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
            operation="chat",
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
            operation="text_completion",
        )


class LatencyMetricsPlugin(Plugin, name="latency_metrics", priority=1051):
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
            operation="chat",
            streaming=gen.streaming,
        )

        # Record TTFB only for streaming requests with a measured value
        if gen.streaming and gen.ttfb_ms is not None:
            record_ttfb(
                ttfb_s=gen.ttfb_ms / 1000.0,
                model=model,
                provider=provider,
                operation="chat",
            )

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
            operation="text_completion",
            streaming=False,
        )

    @hook("generation_event", mode=PluginMode.FIRE_AND_FORGET)
    async def record_chunk_interval_metrics(
        self, payload: GenerationEventPayload, context: dict[str, Any]
    ) -> None:
        """Record inter-chunk timing from `chunk_processed` events.

        Each streamed chunk after the first carries an interval; the first has
        none and is skipped. `chunk_processed` events are opt-in via
        `MELLEA_GENERATION_CHUNK_EVENTS`, so the
        `gen_ai.client.operation.time_per_output_chunk` histogram stays empty
        unless that flag is set.

        Args:
            payload: A `generation_event`; the `chunk_processed` variant carries
                `time_since_last_chunk_ms` in `data`.
            context: Plugin context (unused).
        """
        if payload.event_name != "chunk_processed":
            return
        time_since_last_chunk_ms = payload.data.get("time_since_last_chunk_ms")
        if time_since_last_chunk_ms is None:
            return

        from mellea.telemetry.metrics import record_time_per_output_chunk

        record_time_per_output_chunk(
            time_s=time_since_last_chunk_ms / 1000.0,
            model=payload.model or "unknown",
            provider=payload.provider or "unknown",
            operation="chat",
        )


class ErrorMetricsPlugin(Plugin, name="error_metrics", priority=1052):
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
        from mellea.telemetry.metrics import (
            classify_error,
            record_error,
            record_request_duration,
        )

        gen = (
            payload.model_output.generation
            if payload.model_output is not None
            else GenerationMetadata()
        )
        exception_class = type(payload.exception).__name__
        record_error(
            error_type=classify_error(payload.exception),
            model=gen.model or "unknown",
            provider=gen.provider or "unknown",
            exception_class=exception_class,
            operation="chat",
        )

        # Record duration for failures only when a call actually ran (a MOT
        # exists); pre-dispatch setup failures have no operation to time.
        if payload.model_output is not None:
            record_request_duration(
                duration_s=payload.latency_ms / 1000.0,
                model=gen.model or "unknown",
                provider=gen.provider or "unknown",
                operation="chat",
                streaming=gen.streaming,
                exception_class=exception_class,
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
        from mellea.telemetry.metrics import (
            classify_error,
            record_error,
            record_request_duration,
        )

        exception_class = type(payload.exception).__name__
        record_error(
            error_type=classify_error(payload.exception),
            model=payload.model or "unknown",
            provider=payload.provider or "unknown",
            exception_class=exception_class,
            operation="text_completion",
        )
        record_request_duration(
            duration_s=payload.latency_ms / 1000.0,
            model=payload.model or "unknown",
            provider=payload.provider or "unknown",
            operation="text_completion",
            streaming=False,
            exception_class=exception_class,
        )

    @hook("streaming_end", mode=PluginMode.FIRE_AND_FORGET)
    async def record_streaming_error_metrics(
        self, payload: StreamingEndPayload, context: dict[str, Any]
    ) -> None:
        """Record error metrics when `stream` ends with an exception.

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
            operation="chat",
        )


class CostMetricsPlugin(Plugin, name="cost_metrics", priority=1053):
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
            record_cost(cost=cost, model=model, provider=provider, operation="chat")

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
            record_cost(
                cost=cost, model=model, provider=provider, operation="text_completion"
            )


class SamplingMetricsPlugin(Plugin, name="sampling_metrics", priority=1054):
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
        """Record success or failure when the sampling loop ends, unless it raised.

        A raised loop is not a sampling outcome, so it is skipped.

        Args:
            payload: Contains strategy_name, success flag, and exception.
            context: Plugin context (unused).
        """
        from mellea.telemetry.metrics import record_sampling_outcome

        if payload.exception is not None:
            return
        record_sampling_outcome(payload.strategy_name or "unknown", payload.success)

    @hook("streaming_end", mode=PluginMode.FIRE_AND_FORGET)
    async def record_streaming_outcome(
        self, payload: StreamingEndPayload, context: dict[str, Any]
    ) -> None:
        """Record the `stream` outcome when the stream finishes.

        Args:
            payload: Contains the stream's success flag.
            context: Plugin context (unused).
        """
        from mellea.telemetry.metrics import record_sampling_outcome

        record_sampling_outcome("stream", payload.success)


class RequirementMetricsPlugin(Plugin, name="requirement_metrics", priority=1055):
    """Records requirement validation check and failure metrics.

    Hooks into `validation_post_check` to count checks and failures per
    requirement type after each validation batch.
    """

    @hook("validation_post_check", mode=PluginMode.FIRE_AND_FORGET)
    async def record_requirement_metrics(
        self, payload: ValidationPostCheckPayload, context: dict[str, Any]
    ) -> None:
        """Record validation checks and failures for each requirement, unless it raised.

        A raised validation has no results to count, so it is skipped.

        Args:
            payload: Contains requirements list, corresponding results, and exception.
            context: Plugin context (unused).
        """
        from mellea.telemetry.metrics import (
            record_requirement_check,
            record_requirement_failure,
        )

        if payload.exception is not None:
            return
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


class ToolMetricsPlugin(Plugin, name="tool_metrics", priority=1056):
    """Records tool invocation metrics.

    Hooks into `tool_post_invoke` to count tool calls by name and success/failure status.
    """

    @hook("tool_post_invoke", mode=PluginMode.FIRE_AND_FORGET)
    async def record_tool_call(
        self, payload: ToolPostInvokePayload, context: dict[str, Any]
    ) -> None:
        """Record one tool invocation after it completes.

        Args:
            payload: Contains model_tool_call (with func.name) and success flag.
            context: Plugin context (unused).
        """
        from mellea.telemetry.metrics import record_tool_call

        tool_name = (
            payload.model_tool_call.func.name
            if payload.model_tool_call is not None
            else "unknown"
        )
        status = "success" if payload.success else "failure"
        record_tool_call(tool_name, status)


class AdapterFunctionMetricsPlugin(
    Plugin, name="adapter_function_metrics", priority=1057
):
    """Records adapter function invocation and phase-duration metrics.

    Hooks into `adapter_function_invocation_complete` and
    `adapter_function_phase_complete`. `phase` is one of the values in
    `AdapterFunctionPhaseCompletePayload`'s `phase` field.
    """

    @hook("adapter_function_invocation_complete", mode=PluginMode.FIRE_AND_FORGET)
    async def record_adapter_function_invocation(
        self, payload: AdapterFunctionInvocationCompletePayload, context: dict[str, Any]
    ) -> None:
        """Record one adapter function invocation after it completes.

        Args:
            payload: Contains name, revision, binding_type, adapter_type, and outcome.
            context: Plugin context (unused).
        """
        from mellea.telemetry.metrics import (
            record_adapter_function_invocation,
            record_adapter_function_parse_failure,
        )

        record_adapter_function_invocation(
            name=payload.name,
            revision=payload.revision,
            binding_type=payload.binding_type,
            adapter_type=payload.adapter_type,
            outcome=payload.outcome,
        )
        if payload.outcome == "schema_error":
            record_adapter_function_parse_failure(payload.name, payload.revision)

    @hook("adapter_function_phase_complete", mode=PluginMode.FIRE_AND_FORGET)
    async def record_adapter_function_phase(
        self, payload: AdapterFunctionPhaseCompletePayload, context: dict[str, Any]
    ) -> None:
        """Record one adapter function lifecycle phase after it completes.

        Args:
            payload: Contains name, phase, and duration_ms.
            context: Plugin context (unused).
        """
        from mellea.telemetry.metrics import record_adapter_function_phase_duration

        # payload carries milliseconds; the metric is in seconds, matching
        # LatencyMetricsPlugin and the OTel base-unit convention for durations.
        record_adapter_function_phase_duration(
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
    AdapterFunctionMetricsPlugin,
)
