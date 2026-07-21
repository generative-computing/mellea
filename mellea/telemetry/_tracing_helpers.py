# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Helpers for the tracing instrumentation."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from typing import Any

from mellea.core.base import GenerationMetadata

# Max characters of captured content recorded on a span.
_MAX_CONTENT_LEN = 500


def _env_true(name: str) -> bool:
    """Return True if `name` is set to a truthy value (1/true/yes)."""
    return os.getenv(name, "false").lower() in ("true", "1", "yes")


def content_capture_enabled() -> bool:
    """Check if content capture is opted into via environment variable.

    Internal env-var check behind `is_content_tracing_enabled`.

    Returns:
        True if `MELLEA_TRACES_CONTENT` or
        `OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT` is truthy.
    """
    return _env_true("MELLEA_TRACES_CONTENT") or _env_true(
        "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT"
    )


def get_capture_content_value(value: Any) -> str | None:
    """Stringify and truncate a content value for a span attribute.

    Args:
        value: The content value to record.

    Returns:
        The value truncated to `_MAX_CONTENT_LEN` characters, or `None` when
        content capture is disabled or `value` is `None`.
    """
    if value is None or not content_capture_enabled():
        return None
    text = str(value)
    return text[:_MAX_CONTENT_LEN] + "..." if len(text) > _MAX_CONTENT_LEN else text


def set_attribute_safe(span: Any, key: str, value: Any) -> None:
    """Set an attribute on a span, coercing to an OTel-compatible type.

    Args:
        span: The span object.
        key: Attribute key.
        value: Attribute value; `None` is skipped, lists/tuples are stringified
            element-wise, and other non-primitive values fall back to `str()`.
    """
    if value is None:
        return

    if isinstance(value, bool):
        span.set_attribute(key, value)
    elif isinstance(value, int | float):
        span.set_attribute(key, value)
    elif isinstance(value, str):
        span.set_attribute(key, value)
    elif isinstance(value, list | tuple):
        span.set_attribute(key, [str(v) for v in value])
    else:
        span.set_attribute(key, str(value))


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


def _serialize_arguments(arguments: Mapping[str, Any] | None) -> str | None:
    """Return a stable, key-sorted JSON string of tool arguments, or `None`.

    Falls back to `str()` for values JSON cannot serialize.
    """
    if not arguments:
        return None
    try:
        return json.dumps(arguments, sort_keys=True, default=str)
    except Exception:
        return str(arguments)


def _tool_schema_attrs(tool_call: Any) -> tuple[str | None, str | None]:
    """Return `(tool_type, tool_description)` from a `ModelToolCall`'s tool schema.

    Reads `tool_call.func.as_json_tool` defensively; returns `(None, None)` on any
    failure so attribute capture never breaks the caller.
    """
    func = getattr(tool_call, "func", None)
    try:
        schema = func.as_json_tool if func is not None else None
    except Exception:
        return None, None
    if not isinstance(schema, dict):
        return None, None
    tool_type = schema.get("type")
    function_schema = schema.get("function")
    description = None
    if isinstance(function_schema, Mapping):
        description = function_schema.get("description") or None
    return tool_type, description


def get_tool_call_attrs(tool_call: Any) -> dict[str, Any]:
    """Unpack a `ModelToolCall` into `execute_tool` span attributes.

    `gen_ai.tool.call.arguments` (semconv Opt-In, may contain PII) is included
    only when content capture is enabled; `mellea.tool.arguments_hash` (a
    non-content stable hash) is recorded independent of content capture,
    whenever the call has arguments. Attributes the tool does not provide (type,
    description, call id) are left as `None`.

    Args:
        tool_call: The `ModelToolCall` being executed.

    Returns:
        Span attributes keyed by attribute name.
    """
    tool_type, tool_description = _tool_schema_attrs(tool_call)
    serialized = _serialize_arguments(getattr(tool_call, "args", None))
    arguments_hash = (
        hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:16]
        if serialized is not None
        else None
    )
    return {
        "gen_ai.operation.name": "execute_tool",
        "gen_ai.tool.name": getattr(tool_call, "name", None) or "unknown",
        "gen_ai.tool.type": tool_type,
        "gen_ai.tool.description": tool_description,
        "gen_ai.tool.call.id": getattr(tool_call, "tool_call_id", None),
        "mellea.tool.arguments_hash": arguments_hash,
        "gen_ai.tool.call.arguments": get_capture_content_value(serialized),
    }
