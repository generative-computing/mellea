# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tracing plugins for emitting OpenTelemetry spans via hooks.

This module contains plugins that hook into the generation and component
pipelines to automatically emit spans when tracing is enabled:

- BackendTracingPlugin: Emits Gen-AI semconv backend spans for every LLM
  generation, on both chat and raw (batch) paths, plus mid-generation span
  events from the generation_event hook.
- ComponentTracingPlugin: Emits application-level spans tracking component
  execution.
- StreamingTracingPlugin: Emits an application-level span and per-chunk span
  events for `stream` runs.
- ToolTracingPlugin: Emits an `execute_tool` span for every tool invocation.
- SamplingTracingPlugin: Emits a `sampling` span per sampling loop, with a span
  event per iteration and repair.
- ValidationTracingPlugin: Emits a `validation` span per requirement-check batch.
- AdapterFunctionTracingPlugin: Emits the `adapter_function` span tree (one
  parent span per invocation, one `adapter_function.<phase>` child per
  lifecycle phase) for the adapter-function lifecycle. Covers
  prepare/activate/deactivate only as of #1466; generate/parse land with #1465.
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Any

from mellea.plugins.base import Plugin
from mellea.plugins.decorators import hook

# Whether an OTel context attach performed inside a hook survives back to the
# caller. cpex runs every hook through `asyncio.wait_for(...)`, which on Python
# <=3.11 wraps it in a child Task with a copied contextvars Context — so the
# attach is lost when the hook returns. Python 3.12 runs it in the caller task,
# so the mutation survives. When False, hook spans are emitted flat (no nesting).
_CONTEXT_ATTACH_SUPPORTED: bool = sys.version_info >= (3, 12)

if TYPE_CHECKING:
    from mellea.plugins.hooks.adapter_function import (
        AdapterFunctionInvocationCompletePayload,
        AdapterFunctionInvocationStartPayload,
        AdapterFunctionPhaseCompletePayload,
        AdapterFunctionPhaseStartPayload,
    )
    from mellea.plugins.hooks.component import (
        ComponentPostErrorPayload,
        ComponentPostSuccessPayload,
        ComponentPreExecutePayload,
    )
    from mellea.plugins.hooks.generation import (
        GenerationBatchErrorPayload,
        GenerationBatchPostCallPayload,
        GenerationBatchPreCallPayload,
        GenerationErrorPayload,
        GenerationEventPayload,
        GenerationPostCallPayload,
        GenerationPreCallPayload,
    )
    from mellea.plugins.hooks.sampling import (
        SamplingIterationPayload,
        SamplingLoopEndPayload,
        SamplingLoopStartPayload,
        SamplingRepairPayload,
    )
    from mellea.plugins.hooks.streaming import (
        StreamingEndPayload,
        StreamingEventPayload,
        StreamingStartPayload,
    )
    from mellea.plugins.hooks.tool import ToolPostInvokePayload, ToolPreInvokePayload
    from mellea.plugins.hooks.validation import (
        ValidationPostCheckPayload,
        ValidationPreCheckPayload,
    )


class BackendTracingPlugin(Plugin, name="backend_tracing", priority=1040):
    """Emits Gen-AI semconv backend spans for every LLM generation.

    This plugin hooks into the generation pre-call, post-call, and error
    events on both the chat and raw (batch) paths to automatically emit one
    span per LLM call. Spans are started on pre-call and ended on post-call
    or error, correlated across hooks via generation_id. It also records
    mid-generation span events from the `generation_event` hook onto the
    in-flight span.

    All hooks run SEQUENTIAL so the OTel context token attached in pre-call
    can be detached on the same task in post-call / error, and so
    `generation_event` appends to the span before post-call ends it.
    """

    # --- Chat hooks ---

    @hook("generation_pre_call")
    async def on_pre_call(
        self, payload: GenerationPreCallPayload, context: dict[str, Any]
    ) -> None:
        """Start a backend chat span for this generation."""
        if not payload.generation_id:
            return
        from mellea.backends.model_options import ModelOption
        from mellea.telemetry.tracing import start_backend_span

        action = payload.action
        fmt = payload.format
        start_backend_span(
            "chat",
            payload.generation_id,
            model=payload.model,
            provider=payload.provider,
            action_class_name=action.__class__.__name__ if action is not None else None,
            has_format=fmt is not None,
            format_type=fmt.__name__ if fmt is not None else None,
            tool_calls_enabled=payload.tool_calls,
            streaming=bool(payload.model_options.get(ModelOption.STREAM, False)),
            attach_context=_CONTEXT_ATTACH_SUPPORTED,
        )

    @hook("generation_post_call")
    async def on_post_call(
        self, payload: GenerationPostCallPayload, context: dict[str, Any]
    ) -> None:
        """Add usage / mellea attrs and end the chat span."""
        if not payload.generation_id:
            return
        from mellea.telemetry.tracing import finish_backend_span_success

        mot = payload.model_output
        gen = mot.generation
        finish_backend_span_success(
            payload.generation_id, operation="chat", usage=gen.usage, mot=mot, gen=gen
        )

    @hook("generation_error")
    async def on_error(
        self, payload: GenerationErrorPayload, context: dict[str, Any]
    ) -> None:
        """Set ERROR status and end the chat span."""
        if not payload.generation_id:
            return
        from mellea.telemetry.tracing import finish_backend_span_error

        mot = payload.model_output
        gen = mot.generation if mot is not None else None
        finish_backend_span_error(
            payload.generation_id,
            operation="chat",
            exception=payload.exception,
            gen=gen,
        )

    @hook("generation_event")
    async def on_generation_event(
        self, payload: GenerationEventPayload, context: dict[str, Any]
    ) -> None:
        """Record a span event on the in-flight backend span for one `generation_event`."""
        if not payload.generation_id:
            return
        from mellea.telemetry.tracing import add_span_event

        if payload.event_name == "chunk_processed":
            add_span_event(
                payload.generation_id,
                event_name="chunk_processed",
                attributes={
                    "mellea.generation.chunk_index": payload.data.get("chunk_index"),
                    "mellea.generation.chunk_text_length": payload.data.get(
                        "chunk_text_length"
                    ),
                    "mellea.generation.time_since_last_chunk_ms": payload.data.get(
                        "time_since_last_chunk_ms"
                    ),
                },
            )

    # --- Batch hooks ---

    @hook("generation_batch_pre_call")
    async def on_batch_pre_call(
        self, payload: GenerationBatchPreCallPayload, context: dict[str, Any]
    ) -> None:
        """Start a backend text_completion span for the whole batch."""
        if not payload.generation_id:
            return
        from mellea.telemetry.tracing import start_backend_span

        fmt = payload.format
        start_backend_span(
            "text_completion",
            payload.generation_id,
            model=payload.model,
            provider=payload.provider,
            num_actions=payload.num_actions,
            has_format=fmt is not None,
            format_type=fmt.__name__ if fmt is not None else None,
            tool_calls_enabled=payload.tool_calls,
            attach_context=_CONTEXT_ATTACH_SUPPORTED,
        )

    @hook("generation_batch_post_call")
    async def on_batch_post_call(
        self, payload: GenerationBatchPostCallPayload, context: dict[str, Any]
    ) -> None:
        """Add aggregate usage attrs and end the batch span."""
        if not payload.generation_id:
            return
        from mellea.telemetry.tracing import finish_backend_span_success

        finish_backend_span_success(
            payload.generation_id,
            operation="text_completion",
            usage=payload.usage,
            mot=None,
            gen=None,
        )

    @hook("generation_batch_error")
    async def on_batch_error(
        self, payload: GenerationBatchErrorPayload, context: dict[str, Any]
    ) -> None:
        """Set ERROR status and end the batch span."""
        if not payload.generation_id:
            return
        from mellea.telemetry.tracing import finish_backend_span_error

        finish_backend_span_error(
            payload.generation_id,
            operation="text_completion",
            exception=payload.exception,
        )


class ComponentTracingPlugin(Plugin, name="component_tracing", priority=1041):
    """Emits application-level spans tracking component execution.

    This plugin hooks into component pre-execute, post-success, and
    post-error events to emit one span per component execution. Spans are
    correlated across hooks via action_id.

    All hooks run SEQUENTIAL so the OTel context token attached on each open
    hook can be detached on the same task on the corresponding close hook.
    """

    @hook("component_pre_execute")
    async def on_component_pre_execute(
        self, payload: ComponentPreExecutePayload, context: dict[str, Any]
    ) -> None:
        """Open the action span for this component execution."""
        if not payload.action_id:
            return
        from mellea.telemetry.tracing import start_action_span

        action = payload.action
        strategy = payload.strategy
        start_action_span(
            payload.action_id,
            action_class_name=action.__class__.__name__ if action is not None else None,
            has_requirements=bool(payload.requirements),
            has_strategy=strategy is not None,
            strategy_type=strategy.__class__.__name__ if strategy is not None else None,
            has_format=payload.format is not None,
            tool_calls=payload.tool_calls_enabled,
            attach_context=_CONTEXT_ATTACH_SUPPORTED,
        )

    @hook("component_post_success")
    async def on_component_post_success(
        self, payload: ComponentPostSuccessPayload, context: dict[str, Any]
    ) -> None:
        """End the action span with response-side attributes."""
        if not payload.action_id:
            return
        from mellea.telemetry.tracing import finish_action_span_success

        result = payload.result
        sampling = payload.sampling_results

        response_text: str | None = None
        response_length: int | None = None
        if result is not None:
            try:
                response_text = (
                    str(result.value)
                    if hasattr(result, "value") and result.value
                    else str(result)
                )
                response_length = len(response_text)
            except Exception:
                # Never let attribute capture fail the post hook.
                pass

        sampling_success = payload.sampling_success

        num_logs = 1 if payload.generate_log is not None else 0
        if sampling is not None:
            num_logs = len(sampling)

        finish_action_span_success(
            payload.action_id,
            num_generate_logs=num_logs,
            sampling_success=sampling_success,
            response_text=response_text,
            response_length=response_length,
        )

    @hook("component_post_error")
    async def on_component_post_error(
        self, payload: ComponentPostErrorPayload, context: dict[str, Any]
    ) -> None:
        """End the action span with ERROR status."""
        if not payload.action_id:
            return
        from mellea.telemetry.tracing import finish_action_span_error

        finish_action_span_error(payload.action_id, exception=payload.error)


class StreamingTracingPlugin(Plugin, name="streaming_tracing", priority=1042):
    """Emits the `stream` application span.

    `streaming_start` opens the span; `streaming_event` records a span event for
    each `StreamEvent`; `streaming_end` closes the span.

    All hooks run SEQUENTIAL so the OTel context Token attached in start is
    detached on the same task in end.
    """

    @hook("streaming_start")
    async def on_streaming_start(
        self, payload: StreamingStartPayload, context: dict[str, Any]
    ) -> None:
        """Open the stream span for this streaming run."""
        if not payload.streaming_id:
            return
        from mellea.telemetry.tracing import start_streaming_span

        start_streaming_span(
            payload.streaming_id,
            has_requirements=payload.has_requirements,
            requirement_count=payload.requirement_count,
            chunking_strategy=payload.chunking_strategy,
            attach_context=_CONTEXT_ATTACH_SUPPORTED,
        )

    @hook("streaming_event")
    async def on_streaming_event(
        self, payload: StreamingEventPayload, context: dict[str, Any]
    ) -> None:
        """Record a span event for one `StreamEvent`."""
        if not payload.streaming_id or payload.event is None:
            return
        from mellea.stdlib.streaming import (
            ChunkEvent,
            CompletedEvent,
            ErrorEvent,
            FullValidationEvent,
            QuickCheckEvent,
            StreamingDoneEvent,
        )
        from mellea.telemetry.tracing import add_span_event

        ev = payload.event
        if isinstance(ev, QuickCheckEvent):
            add_span_event(
                payload.streaming_id,
                event_name="quick_check",
                attributes={
                    "mellea.streaming.chunk_index": ev.chunk_index,
                    "mellea.validation.passed": ev.passed,
                    "mellea.validation.requirement_count": len(ev.results),
                },
            )
        elif isinstance(ev, ChunkEvent):
            add_span_event(
                payload.streaming_id,
                event_name="chunk",
                attributes={
                    "mellea.streaming.chunk_index": ev.chunk_index,
                    "mellea.streaming.chunk_text_length": len(ev.text),
                },
            )
        elif isinstance(ev, StreamingDoneEvent):
            add_span_event(
                payload.streaming_id,
                event_name="streaming_done",
                attributes={"mellea.streaming.full_text_length": len(ev.full_text)},
            )
        elif isinstance(ev, FullValidationEvent):
            add_span_event(
                payload.streaming_id,
                event_name="full_validation",
                attributes={
                    "mellea.validation.passed": ev.passed,
                    "mellea.validation.requirement_count": len(ev.results),
                },
            )
        elif isinstance(ev, ErrorEvent):
            # Not OTel's reserved `exception.*`: `detail` isn't always the plain message.
            add_span_event(
                payload.streaming_id,
                event_name="error",
                attributes={
                    "mellea.error.type": ev.exception_type,
                    "mellea.error.detail": ev.detail,
                },
            )
        elif isinstance(ev, CompletedEvent):
            add_span_event(
                payload.streaming_id,
                event_name="completed",
                attributes={
                    "mellea.streaming.success": ev.success,
                    "mellea.streaming.full_text_length": len(ev.full_text),
                },
            )

    @hook("streaming_end")
    async def on_streaming_end(
        self, payload: StreamingEndPayload, context: dict[str, Any]
    ) -> None:
        """Close the stream span."""
        if not payload.streaming_id:
            return
        from mellea.telemetry.tracing import finish_streaming_span

        finish_streaming_span(
            payload.streaming_id,
            success=payload.success,
            failure_reason=payload.failure_reason,
            exception=payload.exception,
            model=payload.model,
            provider=payload.provider,
            full_text_length=payload.full_text_length,
        )


class ToolTracingPlugin(Plugin, name="tool_tracing", priority=1043):
    """Emits an `execute_tool` span per tool invocation (pre/post lifecycle).

    `tool_pre_invoke` opens the span; `tool_post_invoke` closes it with success
    or error status, correlated via `tool_invocation_id`.

    All hooks run SEQUENTIAL so the OTel context token attached in the pre hook
    can be detached on the same task in the post hook.
    """

    @hook("tool_pre_invoke")
    async def on_tool_pre_invoke(
        self, payload: ToolPreInvokePayload, context: dict[str, Any]
    ) -> None:
        """Open the `execute_tool` span for this tool invocation."""
        if not payload.tool_invocation_id:
            return
        from mellea.telemetry.tracing import start_tool_span

        start_tool_span(
            payload.tool_invocation_id,
            payload.model_tool_call,
            is_control_flow=payload.is_control_flow,
            attach_context=_CONTEXT_ATTACH_SUPPORTED,
        )

    @hook("tool_post_invoke")
    async def on_tool_post_invoke(
        self, payload: ToolPostInvokePayload, context: dict[str, Any]
    ) -> None:
        """Close the `execute_tool` span with success or error status."""
        if not payload.tool_invocation_id:
            return
        from mellea.telemetry.tracing import (
            finish_tool_span_error,
            finish_tool_span_success,
        )

        if payload.success:
            finish_tool_span_success(
                payload.tool_invocation_id,
                execution_time_ms=payload.execution_time_ms,
                result=payload.tool_output,
            )
        else:
            finish_tool_span_error(
                payload.tool_invocation_id,
                execution_time_ms=payload.execution_time_ms,
                exception=payload.error,
            )


class SamplingTracingPlugin(Plugin, name="sampling_tracing", priority=1044):
    """Emits a `sampling` span per sampling loop.

    `sampling_loop_start` opens the span; `sampling_iteration` and
    `sampling_repair` record span events on it; `sampling_loop_end` closes it,
    correlated across hooks via `sampling_id`.

    Iterations and repairs are recorded as span events, not child spans.

    All hooks run SEQUENTIAL so the OTel context token attached in loop_start
    can be detached on the same task in loop_end.
    """

    @hook("sampling_loop_start")
    async def on_loop_start(
        self, payload: SamplingLoopStartPayload, context: dict[str, Any]
    ) -> None:
        """Open the sampling span for this loop."""
        if not payload.sampling_id:
            return
        from mellea.telemetry.tracing import start_sampling_span

        start_sampling_span(
            payload.sampling_id,
            strategy_type=payload.strategy_name or None,
            loop_budget=payload.loop_budget,
            requirement_count=len(payload.requirements),
            attach_context=_CONTEXT_ATTACH_SUPPORTED,
        )

    @hook("sampling_iteration")
    async def on_iteration(
        self, payload: SamplingIterationPayload, context: dict[str, Any]
    ) -> None:
        """Record a span event for one sampling attempt."""
        if not payload.sampling_id:
            return
        from mellea.telemetry.tracing import add_span_event

        add_span_event(
            payload.sampling_id,
            event_name="iteration",
            attributes={
                "mellea.sampling.iteration": payload.iteration,
                "mellea.sampling.all_validations_passed": payload.all_validations_passed,
                "mellea.validation.valid_count": payload.valid_count,
                "mellea.validation.requirement_count": payload.total_count,
            },
        )

    @hook("sampling_repair")
    async def on_repair(
        self, payload: SamplingRepairPayload, context: dict[str, Any]
    ) -> None:
        """Record a span event for one repair."""
        if not payload.sampling_id:
            return
        from mellea.telemetry.tracing import add_span_event

        add_span_event(
            payload.sampling_id,
            event_name="repair",
            attributes={
                "mellea.sampling.repair_iteration": payload.repair_iteration,
                "mellea.sampling.repair_type": payload.repair_type,
                "mellea.validation.failed_count": len(payload.failed_validations),
            },
        )

    @hook("sampling_loop_end")
    async def on_loop_end(
        self, payload: SamplingLoopEndPayload, context: dict[str, Any]
    ) -> None:
        """Close the sampling span, ERROR only when the loop raised."""
        if not payload.sampling_id:
            return
        from mellea.telemetry.tracing import finish_sampling_span

        finish_sampling_span(
            payload.sampling_id,
            success=payload.success,
            iterations_used=payload.iterations_used,
            failure_reason=payload.failure_reason,
            exception=payload.exception,
        )


class ValidationTracingPlugin(Plugin, name="validation_tracing", priority=1045):
    """Emits a `validation` span per requirement-validation batch.

    `validation_pre_check` opens the span; `validation_post_check` closes it,
    correlated via `validation_id`.

    All hooks run SEQUENTIAL so the OTel context token attached in pre_check
    can be detached on the same task in post_check.
    """

    @hook("validation_pre_check")
    async def on_pre_check(
        self, payload: ValidationPreCheckPayload, context: dict[str, Any]
    ) -> None:
        """Open the validation span for this check."""
        if not payload.validation_id:
            return
        from mellea.telemetry.tracing import start_validation_span

        start_validation_span(
            payload.validation_id,
            requirement_count=len(payload.requirements),
            attach_context=_CONTEXT_ATTACH_SUPPORTED,
        )

    @hook("validation_post_check")
    async def on_post_check(
        self, payload: ValidationPostCheckPayload, context: dict[str, Any]
    ) -> None:
        """Close the validation span, ERROR only when validation raised."""
        if not payload.validation_id:
            return
        from mellea.telemetry.tracing import finish_validation_span

        # One reason per failing requirement (results with no reason are skipped).
        reasons = [r.reason for r in payload.results if not bool(r) and r.reason]
        finish_validation_span(
            payload.validation_id,
            all_validations_passed=payload.all_validations_passed,
            passed_count=payload.passed_count,
            failed_count=payload.failed_count,
            failure_reasons=reasons,
            exception=payload.exception,
        )


class AdapterFunctionTracingPlugin(
    Plugin, name="adapter_function_tracing", priority=1046
):
    """Emits the `adapter_function` span tree for the adapter-function lifecycle.

    `adapter_function_invocation_start` opens the `adapter_function` parent
    span; `adapter_function_invocation_complete` closes it, recording the
    outcome and defensively closing any `adapter_function.<phase>` child span
    left open by a phase that raised (see `finish_adapter_function_span`).
    `adapter_function_phase_start`/`adapter_function_phase_complete` open/close
    one child span per lifecycle phase, correlated with the parent via
    `adapter_function_invocation_id`.

    On the `mellea.backend` tracer (adapter/model lifecycle work, not a
    user-facing operation — see `docs/docs/observability/tracing.md`).

    Covers `prepare`/`activate`/`deactivate` only as of #1466. `generate`/
    `parse` fire no hooks yet (blocked on #1465 wiring generation through
    `AdapterMixin.adapter_scope`), so this plugin does not yet emit
    `adapter_function.generate`/`adapter_function.parse` — it will, once those
    hooks fire, with no changes needed here. `release` fires no hooks at all
    (see `AdapterFunctionPhaseCompletePayload`'s `phase` field for why) and so
    never gets a span either. Content capture (`MELLEA_TRACES_CONTENT`) is not
    wired here: no phase in scope carries adapter input/output content —
    `generate`/`parse` are where that will apply.

    Unlike every other plugin in this module, these hooks don't rely on
    `_CONTEXT_ATTACH_SUPPORTED` ambient-context attach/detach at all — they
    fire from sync code (`AdapterMixin.adapter_scope`, `LocalFileBinding.prepare`)
    via `_run_async_in_thread`, under which ambient attach can't establish a
    parent/child edge (see `start_adapter_function_span`'s docstring).
    `start_adapter_function_phase_span` parents each child explicitly instead,
    so nesting works identically on every Python version.

    Exemplar linkage (see `docs/docs/observability/tracing.md`) is **not**
    established for these spans, and can't be with the current architecture:
    an OTel histogram exemplar samples whatever span is *ambiently* current at
    the moment the metric is recorded, but no span in this family is ever
    attached as ambient context (see above) — `AdapterFunctionMetricsPlugin`
    (`mellea/telemetry/metrics_plugins.py`, a separate plugin subscribed to the
    same hooks) at best samples whatever enclosing application span happens to
    be ambient, never the adapter-function span the metric is actually about,
    regardless of firing order between the two plugins. This is a known,
    structural gap, not an oversight: fixing it would need the metric recorded
    from inside this plugin (so it can pass the span's context explicitly)
    rather than from a same-hook sibling plugin, which is a larger change than
    adding spans and is left for a follow-up.
    """

    @hook("adapter_function_invocation_start")
    async def on_invocation_start(
        self, payload: AdapterFunctionInvocationStartPayload, context: dict[str, Any]
    ) -> None:
        """Open the `adapter_function` parent span for this invocation."""
        from mellea.telemetry.tracing import start_adapter_function_span

        start_adapter_function_span(
            payload.adapter_function_invocation_id,
            name=payload.name,
            revision=payload.revision,
            binding_type=payload.binding_type,
            adapter_type=payload.adapter_type,
        )

    @hook("adapter_function_invocation_complete")
    async def on_invocation_complete(
        self, payload: AdapterFunctionInvocationCompletePayload, context: dict[str, Any]
    ) -> None:
        """Close the `adapter_function` span with its outcome."""
        from mellea.telemetry.tracing import finish_adapter_function_span

        finish_adapter_function_span(
            payload.adapter_function_invocation_id,
            outcome=payload.outcome,
            exception=payload.error,
        )

    @hook("adapter_function_phase_start")
    async def on_phase_start(
        self, payload: AdapterFunctionPhaseStartPayload, context: dict[str, Any]
    ) -> None:
        """Open the `adapter_function.<phase>` child span."""
        from mellea.telemetry.tracing import start_adapter_function_phase_span

        start_adapter_function_phase_span(
            payload.adapter_function_invocation_id,
            payload.phase,
            revision=payload.revision,
        )

    @hook("adapter_function_phase_complete")
    async def on_phase_complete(
        self, payload: AdapterFunctionPhaseCompletePayload, context: dict[str, Any]
    ) -> None:
        """Close the `adapter_function.<phase>` child span."""
        from mellea.telemetry.tracing import finish_adapter_function_phase_span

        finish_adapter_function_phase_span(
            payload.adapter_function_invocation_id, payload.phase
        )


# All tracing plugins to auto-register when tracing is enabled.
_TRACING_PLUGIN_CLASSES = (
    BackendTracingPlugin,
    ComponentTracingPlugin,
    StreamingTracingPlugin,
    ToolTracingPlugin,
    SamplingTracingPlugin,
    ValidationTracingPlugin,
    AdapterFunctionTracingPlugin,
)
