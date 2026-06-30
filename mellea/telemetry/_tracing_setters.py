"""Span attribute setters used by tracing plugins."""

from __future__ import annotations

from typing import Any

from mellea.core.base import GenerationMetadata


def set_request_attrs(span: Any, gen: GenerationMetadata, operation: str) -> None:
    """Emit request-side `gen_ai.*` attributes."""
    if gen.provider:
        span.set_attribute("gen_ai.provider.name", gen.provider)
    if gen.model:
        span.set_attribute("gen_ai.request.model", gen.model)
    span.set_attribute("gen_ai.operation.name", operation)


def set_usage_attrs(span: Any, usage: dict[str, Any] | None) -> None:
    """Emit `gen_ai.usage.*` attributes from an OpenAI-style usage dict.

    Handles top-level token counts plus nested `prompt_tokens_details` /
    `completion_tokens_details` for cache and reasoning attribution.
    """
    if usage is None:
        return

    prompt_tokens = usage.get("prompt_tokens")
    if prompt_tokens is not None:
        span.set_attribute("gen_ai.usage.input_tokens", prompt_tokens)

    completion_tokens = usage.get("completion_tokens")
    if completion_tokens is not None:
        span.set_attribute("gen_ai.usage.output_tokens", completion_tokens)

    total_tokens = usage.get("total_tokens")
    if total_tokens is not None:
        span.set_attribute("gen_ai.usage.total_tokens", total_tokens)

    # Cache-read tokens: prefer top-level, fall back to prompt_tokens_details.cached_tokens
    cache_read = usage.get("cache_read_input_tokens")
    if cache_read is None:
        details = usage.get("prompt_tokens_details")
        if isinstance(details, dict):
            cache_read = details.get("cached_tokens")
    if cache_read is not None:
        span.set_attribute("gen_ai.usage.cache_read.input_tokens", cache_read)

    cache_creation = usage.get("cache_creation_input_tokens")
    if cache_creation is not None:
        span.set_attribute("gen_ai.usage.cache_creation.input_tokens", cache_creation)

    # Reasoning tokens: prefer top-level, fall back to completion_tokens_details
    reasoning = usage.get("reasoning_tokens")
    if reasoning is None:
        details = usage.get("completion_tokens_details")
        if isinstance(details, dict):
            reasoning = details.get("reasoning_tokens")
    if reasoning is not None:
        span.set_attribute("gen_ai.usage.reasoning.output_tokens", reasoning)


def set_response_attrs(span: Any, gen: GenerationMetadata) -> None:
    """Emit response-side `gen_ai.response.*` attributes."""
    if gen.response_model:
        span.set_attribute("gen_ai.response.model", gen.response_model)
    if gen.response_id:
        span.set_attribute("gen_ai.response.id", gen.response_id)
    if gen.finish_reasons:
        span.set_attribute("gen_ai.response.finish_reasons", list(gen.finish_reasons))


def set_mellea_attrs(span: Any, mot: Any, gen: GenerationMetadata) -> None:
    """Emit `mellea.*` attributes derivable from the `ModelOutputThunk`."""
    call = getattr(mot, "_call", None)
    action = getattr(call, "action", None)
    if action is not None:
        span.set_attribute("mellea.action_type", action.__class__.__name__)

    ctx = getattr(call, "context", None)
    span.set_attribute("mellea.context_size", len(ctx) if ctx else 0)

    if gen.streaming:
        span.set_attribute("mellea.streaming", True)


def set_conversation_id(span: Any) -> None:
    """Emit `gen_ai.conversation.id` from the current telemetry context, if set."""
    from mellea.telemetry.context import get_session_id

    session_id = get_session_id()
    if session_id is not None:
        span.set_attribute("gen_ai.conversation.id", session_id)
