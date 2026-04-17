"""Metrics plugins for recording telemetry data via hooks.

This module contains plugins that hook into the generation pipeline to
automatically record metrics when enabled. Currently includes:

- TokenMetricsPlugin: Records token usage statistics from ModelOutputThunk.usage
- LatencyMetricsPlugin: Records request duration and TTFB latency histograms
- ErrorMetricsPlugin: Records LLM error counts categorized by semantic error type
- SamplingMetricsPlugin: Records sampling attempt/success/failure counts per strategy
- RequirementMetricsPlugin: Records requirement validation check and failure counts
- ToolMetricsPlugin: Records tool invocation counts by name and status
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from mellea.plugins.base import Plugin
from mellea.plugins.decorators import hook
from mellea.plugins.types import PluginMode

if TYPE_CHECKING:
    from mellea.plugins.hooks.generation import (
        GenerationErrorPayload,
        GenerationPostCallPayload,
    )
    from mellea.plugins.hooks.sampling import (
        SamplingIterationPayload,
        SamplingLoopEndPayload,
    )
    from mellea.plugins.hooks.tool import ToolPostInvokePayload
    from mellea.plugins.hooks.validation import ValidationPostCheckPayload


class TokenMetricsPlugin(Plugin, name="token_metrics", priority=50):
    """Records token usage metrics from generation outputs.

    This plugin hooks into the generation_post_call event to automatically
    record token usage metrics when the usage field is populated on
    ModelOutputThunk instances.

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

        mot = payload.model_output
        if mot.usage is None:
            return

        # Record metrics (no-op if metrics disabled)
        record_token_usage_metrics(
            input_tokens=mot.usage.get("prompt_tokens"),
            output_tokens=mot.usage.get("completion_tokens"),
            model=mot.model or "unknown",
            provider=mot.provider or "unknown",
        )


class LatencyMetricsPlugin(Plugin, name="latency_metrics", priority=51):
    """Records request duration and TTFB latency metrics from generation outputs.

    This plugin hooks into the generation_post_call event to automatically
    record latency metrics. It records total request duration for every request
    and time-to-first-token (TTFB) for streaming requests.
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

        mot = payload.model_output
        model = mot.model or "unknown"
        provider = mot.provider or "unknown"

        # Record total request duration (convert ms → seconds)
        record_request_duration(
            duration_s=payload.latency_ms / 1000.0,
            model=model,
            provider=provider,
            streaming=mot.streaming,
        )

        # Record TTFB only for streaming requests with a measured value
        if mot.streaming and mot.ttfb_ms is not None:
            record_ttfb(ttfb_s=mot.ttfb_ms / 1000.0, model=model, provider=provider)


class ErrorMetricsPlugin(Plugin, name="error_metrics", priority=52):
    """Records LLM error counts from generation errors.

    This plugin hooks into the generation_error event to classify exceptions
    by semantic error type and increment the ``mellea.llm.errors`` counter.
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
        from mellea.telemetry.metrics import classify_error, record_error

        mot = payload.model_output
        error_type = classify_error(payload.exception)
        record_error(
            error_type=error_type,
            model=mot.model if mot is not None and mot.model is not None else "unknown",
            provider=mot.provider
            if mot is not None and mot.provider is not None
            else "unknown",
            exception_class=type(payload.exception).__name__,
        )


class SamplingMetricsPlugin(Plugin, name="sampling_metrics", priority=54):
    """Records sampling loop attempt and outcome metrics.

    Hooks into ``sampling_iteration`` to count attempts per strategy and
    ``sampling_loop_end`` to count successes and failures.
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


class RequirementMetricsPlugin(Plugin, name="requirement_metrics", priority=55):
    """Records requirement validation check and failure metrics.

    Hooks into ``validation_post_check`` to count checks and failures per
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
                reason = getattr(result, "reason", None) or "unknown"
                record_requirement_failure(req_name, reason)


class ToolMetricsPlugin(Plugin, name="tool_metrics", priority=56):
    """Records tool invocation metrics.

    Hooks into ``tool_post_invoke`` to count tool calls by name and success/failure status.
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


# All metrics plugins to auto-register when metrics are enabled
_METRICS_PLUGIN_CLASSES = (
    TokenMetricsPlugin,
    LatencyMetricsPlugin,
    ErrorMetricsPlugin,
    SamplingMetricsPlugin,
    RequirementMetricsPlugin,
    ToolMetricsPlugin,
)
