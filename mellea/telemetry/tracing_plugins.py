"""Tracing plugins for emitting OpenTelemetry spans via hooks.

This module contains plugins that hook into the generation pipeline to
automatically emit spans when tracing is enabled:

- BackendTracingPlugin: Emits Gen-AI semconv backend spans for every LLM
  generation, on both chat and raw (batch) paths
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from mellea.plugins.base import Plugin
from mellea.plugins.decorators import hook
from mellea.plugins.types import PluginMode
from mellea.telemetry.tracing import (
    finish_backend_span_error,
    finish_backend_span_success,
    start_backend_span,
)

if TYPE_CHECKING:
    from mellea.plugins.hooks.generation import (
        GenerationBatchErrorPayload,
        GenerationBatchPostCallPayload,
        GenerationBatchPreCallPayload,
        GenerationErrorPayload,
        GenerationPostCallPayload,
        GenerationPreCallPayload,
    )


class BackendTracingPlugin(Plugin, name="backend_tracing", priority=50):
    """Emits Gen-AI semconv backend spans for every LLM generation.

    This plugin hooks into the generation pre-call, post-call, and error
    events on both the chat and raw (batch) paths to automatically emit one
    span per LLM call. Spans are started on pre-call and ended on post-call
    or error, correlated across hooks via generation_id.
    """

    # --- Chat hooks ---

    @hook("generation_pre_call")
    async def on_pre_call(
        self, payload: GenerationPreCallPayload, context: dict[str, Any]
    ) -> None:
        """Start a backend chat span for this generation."""
        if not payload.generation_id:
            return
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

    @hook("generation_post_call", mode=PluginMode.FIRE_AND_FORGET)
    async def on_post_call(
        self, payload: GenerationPostCallPayload, context: dict[str, Any]
    ) -> None:
        """Add usage / mellea attrs and end the chat span."""
        if not payload.generation_id:
            return
        mot = payload.model_output
        gen = mot.generation
        finish_backend_span_success(
            payload.generation_id, operation="chat", usage=gen.usage, mot=mot, gen=gen
        )

    @hook("generation_error", mode=PluginMode.FIRE_AND_FORGET)
    async def on_error(
        self, payload: GenerationErrorPayload, context: dict[str, Any]
    ) -> None:
        """Set ERROR status and end the chat span."""
        if not payload.generation_id:
            return
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

    @hook("generation_batch_post_call", mode=PluginMode.FIRE_AND_FORGET)
    async def on_batch_post_call(
        self, payload: GenerationBatchPostCallPayload, context: dict[str, Any]
    ) -> None:
        """Add aggregate usage attrs and end the batch span."""
        if not payload.generation_id:
            return
        finish_backend_span_success(
            payload.generation_id,
            operation="text_completion",
            usage=payload.usage,
            mot=None,
            gen=None,
        )

    @hook("generation_batch_error", mode=PluginMode.FIRE_AND_FORGET)
    async def on_batch_error(
        self, payload: GenerationBatchErrorPayload, context: dict[str, Any]
    ) -> None:
        """Set ERROR status and end the batch span."""
        if not payload.generation_id:
            return
        finish_backend_span_error(
            payload.generation_id,
            operation="text_completion",
            exception=payload.exception,
        )


# All tracing plugins to auto-register when tracing is enabled.
_TRACING_PLUGIN_CLASSES = (BackendTracingPlugin,)
