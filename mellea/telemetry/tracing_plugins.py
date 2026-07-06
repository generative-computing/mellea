"""Tracing plugins for emitting OpenTelemetry spans via hooks.

This module contains plugins that hook into the generation and component
pipelines to automatically emit spans when tracing is enabled:

- BackendTracingPlugin: Emits Gen-AI semconv backend spans for every LLM
  generation, on both chat and raw (batch) paths.
- ComponentTracingPlugin: Emits application-level spans tracking component
  execution.
- StreamingTracingPlugin: Emits an application-level orchestration span and
  per-chunk span events for `stream_with_chunking` runs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from mellea.plugins.base import Plugin
from mellea.plugins.decorators import hook

if TYPE_CHECKING:
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
        GenerationPostCallPayload,
        GenerationPreCallPayload,
    )
    from mellea.plugins.hooks.streaming import (
        StreamingEndPayload,
        StreamingEventPayload,
        StreamingOrchestrationEndPayload,
        StreamingOrchestrationStartPayload,
        StreamingStartPayload,
    )


class BackendTracingPlugin(Plugin, name="backend_tracing", priority=40):
    """Emits Gen-AI semconv backend spans for every LLM generation.

    This plugin hooks into the generation pre-call, post-call, and error
    events on both the chat and raw (batch) paths to automatically emit one
    span per LLM call. Spans are started on pre-call and ended on post-call
    or error, correlated across hooks via generation_id.

    All hooks run SEQUENTIAL so the OTel context token attached in pre-call
    can be detached on the same task in post-call / error.
    """

    # --- Chat hooks ---

    @hook("generation_pre_call")
    async def on_pre_call(
        self, payload: GenerationPreCallPayload, context: dict[str, Any]
    ) -> None:
        """Start a backend chat span for this generation."""
        if not payload.generation_id:
            return
        from mellea.telemetry.tracing import start_backend_span

        action = payload.action
        fmt = payload.format
        start_backend_span(
            "chat",
            payload.generation_id,
            model=None,
            provider=None,
            action_class_name=action.__class__.__name__ if action is not None else None,
            has_format=fmt is not None,
            format_type=fmt.__name__ if fmt is not None else None,
            tool_calls_enabled=payload.tool_calls,
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


class ComponentTracingPlugin(Plugin, name="component_tracing", priority=41):
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

        exc = payload.error if payload.error is not None else Exception("unknown error")
        finish_action_span_error(payload.action_id, exception=exc)


class StreamingTracingPlugin(Plugin, name="streaming_tracing", priority=42):
    """Emits the `stream_with_chunking` application span.

    `streaming_start` opens the span; `streaming_event` records a span event for
    each mid-stream `StreamEvent`; `streaming_end` records the `completed` span
    event and closes the span. `streaming_orchestration_start` /
    `streaming_orchestration_end` re-attach the span on the orchestration task
    so mid-stream spans parent under it (see `reattach_span`).

    All hooks run SEQUENTIAL: the OTel context Token attached in start is
    detached on the originating task in end, and `streaming_orchestration_end`
    releases the reattached span before `streaming_end` closes it
    (FIRE_AND_FORGET would reorder these and break span nesting).
    """

    @hook("streaming_start")
    async def on_streaming_start(
        self, payload: StreamingStartPayload, context: dict[str, Any]
    ) -> None:
        """Open the stream_with_chunking span for this orchestrator invocation."""
        if not payload.streaming_id:
            return
        from mellea.telemetry.tracing import start_streaming_span

        start_streaming_span(
            payload.streaming_id,
            has_requirements=payload.has_requirements,
            requirement_count=payload.requirement_count,
            chunking_strategy=payload.chunking_strategy,
        )

    @hook("streaming_orchestration_start")
    async def on_streaming_orchestration_start(
        self, payload: StreamingOrchestrationStartPayload, context: dict[str, Any]
    ) -> None:
        """Re-attach the streaming span as the orchestration task's ambient context."""
        if not payload.streaming_id:
            return
        from mellea.telemetry.tracing import reattach_span

        reattach_span(payload.streaming_id)

    @hook("streaming_orchestration_end")
    async def on_streaming_orchestration_end(
        self, payload: StreamingOrchestrationEndPayload, context: dict[str, Any]
    ) -> None:
        """Detach the streaming span re-attached on the orchestration task."""
        if not payload.streaming_id:
            return
        from mellea.telemetry.tracing import release_reattached_span

        release_reattached_span(payload.streaming_id)

    @hook("streaming_event")
    async def on_streaming_event(
        self, payload: StreamingEventPayload, context: dict[str, Any]
    ) -> None:
        """Record a span event for one `StreamEvent`."""
        if not payload.streaming_id or payload.event is None:
            return
        from mellea.stdlib.streaming import (
            ChunkEvent,
            ErrorEvent,
            FullValidationEvent,
            QuickCheckEvent,
            StreamingDoneEvent,
        )
        from mellea.telemetry.tracing import add_streaming_event

        ev = payload.event
        if isinstance(ev, QuickCheckEvent):
            add_streaming_event(
                payload.streaming_id,
                event_name="quick_check",
                attributes={
                    "chunk_index": ev.chunk_index,
                    "passed": ev.passed,
                    "requirement_count": len(ev.results),
                },
            )
        elif isinstance(ev, ChunkEvent):
            add_streaming_event(
                payload.streaming_id,
                event_name="chunk",
                attributes={"chunk_index": ev.chunk_index, "text_length": len(ev.text)},
            )
        elif isinstance(ev, StreamingDoneEvent):
            add_streaming_event(
                payload.streaming_id,
                event_name="streaming_done",
                attributes={"full_text_length": len(ev.full_text)},
            )
        elif isinstance(ev, FullValidationEvent):
            add_streaming_event(
                payload.streaming_id,
                event_name="full_validation",
                attributes={"passed": ev.passed, "requirement_count": len(ev.results)},
            )
        elif isinstance(ev, ErrorEvent):
            add_streaming_event(
                payload.streaming_id,
                event_name="error",
                attributes={"exception_type": ev.exception_type, "detail": ev.detail},
            )

    @hook("streaming_end")
    async def on_streaming_end(
        self, payload: StreamingEndPayload, context: dict[str, Any]
    ) -> None:
        """Record the `completed` span event and close the stream_with_chunking span."""
        if not payload.streaming_id:
            return
        from mellea.telemetry.tracing import add_streaming_event, finish_streaming_span

        add_streaming_event(
            payload.streaming_id,
            event_name="completed",
            attributes={
                "success": payload.success,
                "full_text_length": payload.full_text_length,
            },
        )
        finish_streaming_span(
            payload.streaming_id,
            success=payload.success,
            failure_reason=payload.failure_reason,
            exception=payload.exception,
            model=payload.model,
            provider=payload.provider,
            full_text_length=payload.full_text_length,
        )


# All tracing plugins to auto-register when tracing is enabled.
_TRACING_PLUGIN_CLASSES = (
    BackendTracingPlugin,
    ComponentTracingPlugin,
    StreamingTracingPlugin,
)
