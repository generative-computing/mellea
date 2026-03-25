"""Metrics plugins for recording telemetry data via hooks.

This module contains plugins that hook into the generation pipeline to
automatically record metrics when enabled. Currently includes:

- TokenMetricsPlugin: Records token usage statistics from ModelOutputThunk.usage
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from mellea.plugins.base import Plugin
from mellea.plugins.decorators import hook
from mellea.plugins.types import PluginMode

if TYPE_CHECKING:
    from mellea.plugins.hooks.generation import GenerationPostCallPayload


class TokenMetricsPlugin(Plugin, name="token_metrics", priority=50):
    """Records token usage metrics from generation outputs.

    This plugin hooks into the generation_post_call event to automatically
    record token usage metrics when the usage field is populated on
    ModelOutputThunk instances.

    The plugin reads the standardized usage field (OpenAI-compatible format)
    and records metrics following OpenTelemetry Gen-AI semantic conventions.

    Example:
        >>> from mellea.telemetry.metrics_plugins import TokenMetricsPlugin
        >>> from mellea.telemetry.metrics import enable_metrics
        >>>
        >>> enable_metrics()
        >>> with TokenMetricsPlugin():
        ...     result = session.instruct("Hello, world!")
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
