"""Backend instrumentation helpers for OpenTelemetry tracing.

Follows OpenTelemetry Gen-AI semantic conventions:
https://opentelemetry.io/docs/specs/semconv/gen-ai/

Content capture (``gen_ai.input.messages``, ``gen_ai.output.messages``,
``gen_ai.system_instructions``) is opt-in and gated by ``is_content_tracing_enabled()``.
These attributes may contain PII — enable only in controlled environments.
"""

import json
from typing import Any

from ..backends.utils import get_value
from .tracing import (
    add_span_event,
    end_backend_span,
    is_content_tracing_enabled,
    set_span_attribute,
    set_span_error,
    trace_backend,
)


def get_model_id_str(backend: Any) -> str:
    """Extract model_id string from a backend instance.

    Args:
        backend: Backend instance

    Returns:
        String representation of the model_id
    """
    if hasattr(backend, "model_id"):
        model_id = backend.model_id
        if hasattr(model_id, "hf_model_name"):
            return str(model_id.hf_model_name)
        return str(model_id)
    return backend.__class__.__name__


def get_system_name(backend: Any) -> str:
    """Get the Gen-AI system name from backend.

    Kept for back-compatibility with existing dashboards keyed on ``gen_ai.system``.
    New code should prefer ``get_provider_name()``.

    Args:
        backend: Backend instance

    Returns:
        System name (e.g., 'openai', 'ollama', 'huggingface')
    """
    backend_class = backend.__class__.__name__.lower()
    if "openai" in backend_class:
        return "openai"
    elif "ollama" in backend_class:
        return "ollama"
    elif "huggingface" in backend_class or "hf" in backend_class:
        return "huggingface"
    elif "watsonx" in backend_class:
        return "watsonx"
    elif "litellm" in backend_class:
        return "litellm"
    else:
        return backend.__class__.__name__


def get_provider_name(backend: Any) -> str:
    """Get the Gen-AI provider name from backend.

    Returns the value for ``gen_ai.provider.name`` (semconv v1.37.0+), which
    supersedes the deprecated ``gen_ai.system`` attribute.

    Args:
        backend: Backend instance

    Returns:
        Provider name (e.g., 'openai', 'ollama', 'huggingface')
    """
    return get_system_name(backend)


def get_context_size(ctx: Any) -> int:
    """Get the size of a context.

    Args:
        ctx: Context object

    Returns:
        Number of items in context, or 0 if cannot be determined
    """
    try:
        if hasattr(ctx, "__len__"):
            return len(ctx)
        if hasattr(ctx, "turns") and hasattr(ctx.turns, "__len__"):
            return len(ctx.turns)
    except Exception:
        pass
    return 0


def instrument_generate_from_context(
    backend: Any, action: Any, ctx: Any, format: Any = None, tool_calls: bool = False
):
    """Create a backend trace span for generate_from_context.

    Follows Gen-AI semantic conventions for chat operations.

    Args:
        backend: Backend instance
        action: Action component
        ctx: Context
        format: Response format (BaseModel subclass or None)
        tool_calls: Whether tool calling is enabled

    Returns:
        Context manager for the trace span
    """
    model_id = get_model_id_str(backend)
    system_name = get_system_name(backend)

    return trace_backend(
        "chat",  # Gen-AI convention: use 'chat' for chat completions
        **{
            # Gen-AI semantic convention attributes
            "gen_ai.system": system_name,
            "gen_ai.provider.name": system_name,
            "gen_ai.request.model": model_id,
            "gen_ai.operation.name": "chat",
            # Mellea-specific attributes
            "mellea.backend": backend.__class__.__name__,
            "mellea.action_type": action.__class__.__name__,
            "mellea.context_size": get_context_size(ctx),
            "mellea.has_format": format is not None,
            "mellea.format_type": format.__name__ if format else None,
            "mellea.tool_calls_enabled": tool_calls,
        },
    )


def start_generate_span(
    backend: Any,
    action: Any,
    ctx: Any,
    format: Any = None,
    tool_calls: bool = False,
    *,
    model_options: dict | None = None,
):
    """Start a backend trace span for generate_from_context (without auto-closing).

    Use this for async operations where the span should remain open until
    post-processing completes.

    Args:
        backend: Backend instance
        action: Action component
        ctx: Context
        format: Response format (BaseModel subclass or None)
        tool_calls: Whether tool calling is enabled
        model_options: Raw model options dict for request-parameter attributes

    Returns:
        Span object or None if tracing is disabled
    """
    from .tracing import start_backend_span

    model_id = get_model_id_str(backend)
    system_name = get_system_name(backend)

    from .context import get_current_context

    telemetry_ctx = get_current_context()
    span_attrs: dict = {
        # Gen-AI semantic convention attributes
        "gen_ai.system": system_name,
        "gen_ai.provider.name": system_name,
        "gen_ai.request.model": model_id,
        "gen_ai.operation.name": "chat",
        # Mellea-specific attributes
        "mellea.backend": backend.__class__.__name__,
        "mellea.action_type": action.__class__.__name__,
        "mellea.context_size": get_context_size(ctx),
        "mellea.has_format": format is not None,
        "mellea.format_type": format.__name__ if format else None,
        "mellea.tool_calls_enabled": tool_calls,
    }

    # Propagate telemetry context to span
    for key, value in telemetry_ctx.items():
        span_attrs[f"mellea.{key}"] = value

    # gen_ai.conversation.id maps from the existing session_id ContextVar
    session_id = telemetry_ctx.get("session_id")
    if session_id is not None:
        span_attrs["gen_ai.conversation.id"] = session_id

    # Request parameters from model_options (plain-string keys only)
    if model_options:
        for mellea_key, otel_key in _REQUEST_PARAM_MAP.items():
            val = model_options.get(mellea_key)
            if val is not None:
                span_attrs[otel_key] = val

    # Prompt template attributes (duck-typed; works for Instruction and GenerativeStub)
    tmpl = getattr(action, "prompt_template_metadata", None)
    if callable(tmpl):
        metadata: Any = tmpl()
        if metadata is not None:
            template_text, template_vars, template_version = metadata
            if template_text:
                span_attrs["llm.prompt_template.template"] = template_text
            if template_version:
                span_attrs["llm.prompt_template.version"] = template_version
            # Variables contain user-provided values — only emit with content gate
            if template_vars and is_content_tracing_enabled():
                span_attrs["llm.prompt_template.variables"] = _serialize_json(
                    template_vars
                )

    return start_backend_span("chat", **span_attrs)


def instrument_generate_from_raw(
    backend: Any, num_actions: int, format: Any = None, tool_calls: bool = False
):
    """Create a backend trace span for generate_from_raw.

    Follows Gen-AI semantic conventions for text generation operations.

    Args:
        backend: Backend instance
        num_actions: Number of actions in the batch
        format: Response format (BaseModel subclass or None)
        tool_calls: Whether tool calling is enabled

    Returns:
        Context manager for the trace span
    """
    model_id = get_model_id_str(backend)
    system_name = get_system_name(backend)

    return trace_backend(
        "text_completion",  # Gen-AI convention: use 'text_completion' for completions
        **{
            # Gen-AI semantic convention attributes
            "gen_ai.system": system_name,
            "gen_ai.provider.name": system_name,
            "gen_ai.request.model": model_id,
            "gen_ai.operation.name": "text_completion",
            # Mellea-specific attributes
            "mellea.backend": backend.__class__.__name__,
            "mellea.num_actions": num_actions,
            "mellea.has_format": format is not None,
            "mellea.format_type": format.__name__ if format else None,
            "mellea.tool_calls_enabled": tool_calls,
        },
    )


def record_token_usage(span: Any, usage: Any) -> None:
    """Record token usage metrics following Gen-AI semantic conventions.

    Args:
        span: The span object (may be None if tracing is disabled)
        usage: Usage object or dict from the LLM response (e.g., OpenAI usage object)
    """
    if span is None or usage is None:
        return

    try:
        # Gen-AI semantic convention attributes for token usage
        # Handle both objects and dicts
        prompt_tokens = get_value(usage, "prompt_tokens")
        if prompt_tokens is not None:
            set_span_attribute(span, "gen_ai.usage.input_tokens", prompt_tokens)

        completion_tokens = get_value(usage, "completion_tokens")
        if completion_tokens is not None:
            set_span_attribute(span, "gen_ai.usage.output_tokens", completion_tokens)

        total_tokens = get_value(usage, "total_tokens")
        if total_tokens is not None:
            set_span_attribute(span, "gen_ai.usage.total_tokens", total_tokens)

        cache_read = get_value(usage, "cache_read_input_tokens")
        if cache_read is not None:
            set_span_attribute(span, "gen_ai.usage.cache_read.input_tokens", cache_read)

        cache_creation = get_value(usage, "cache_creation_input_tokens")
        if cache_creation is not None:
            set_span_attribute(
                span, "gen_ai.usage.cache_creation.input_tokens", cache_creation
            )

        reasoning_tokens = get_value(usage, "reasoning_tokens")
        if reasoning_tokens is not None:
            set_span_attribute(
                span, "gen_ai.usage.reasoning.output_tokens", reasoning_tokens
            )
    except Exception:
        # Don't fail if we can't extract token usage
        pass


def record_response_metadata(
    span: Any, response: Any, model_id: str | None = None
) -> None:
    """Record response metadata following Gen-AI semantic conventions.

    Args:
        span: The span object (may be None if tracing is disabled)
        response: Response object or dict from the LLM
        model_id: Model ID used for the response (if different from request)
    """
    if span is None or response is None:
        return

    try:
        # Record the actual model used in the response (may differ from request)
        if model_id:
            set_span_attribute(span, "gen_ai.response.model", model_id)
        else:
            model = get_value(response, "model")
            if model:
                set_span_attribute(span, "gen_ai.response.model", model)

        # Record finish reason
        choices = get_value(response, "choices")
        if choices and len(choices) > 0:
            choice = choices[0] if isinstance(choices, list) else choices
            finish_reason = get_value(choice, "finish_reason")
            if finish_reason:
                set_span_attribute(
                    span, "gen_ai.response.finish_reasons", [finish_reason]
                )

        # Record response ID if available
        response_id = get_value(response, "id")
        if response_id:
            set_span_attribute(span, "gen_ai.response.id", response_id)
    except Exception:
        # Don't fail if we can't extract response metadata
        pass


def finalize_backend_span(
    span: Any,
    *,
    response: Any = None,
    usage: Any = None,
    model_id: str | None = None,
    error: Exception | None = None,
    conversation: list[dict] | None = None,
    output_text: str | None = None,
    finish_reason: str | None = None,
) -> None:
    """Close a backend span, recording telemetry on both success and error paths.

    On the error path, records the exception, sets ``error.type``, and marks
    the span with ERROR status before closing.  On the success path, records
    token usage, response metadata, and (when content capture is enabled)
    structured input/output message attributes.

    This replaces the three-line ``record_token_usage`` + ``record_response_metadata``
    + ``end_backend_span`` pattern used in each backend's ``post_processing``.

    Args:
        span: The span to finalise (no-op when ``None``).
        response: Raw backend response (for model id, finish reason, response id).
        usage: Token usage object or dict.
        model_id: Explicit model id override.
        error: Exception to record on the error path.
        conversation: The prompt conversation (``list[dict]`` with ``role``/``content``
            keys).  Used for ``gen_ai.input.messages`` and
            ``gen_ai.system_instructions`` when content capture is enabled.
        output_text: The assistant's reply text.  Used for
            ``gen_ai.output.messages`` when content capture is enabled.
        finish_reason: Finish reason string (defaults to ``"stop"`` when omitted).
    """
    if span is None:
        return

    try:
        try:
            if error is not None:
                set_span_error(span, error)
                # error.type is a Stable OTel cross-signal attribute
                set_span_attribute(span, "error.type", type(error).__name__)
            else:
                record_token_usage(span, usage)
                record_response_metadata(span, response, model_id=model_id)

                if is_content_tracing_enabled() and conversation is not None:
                    _emit_content_attributes(
                        span,
                        conversation=conversation,
                        output_text=output_text,
                        finish_reason=finish_reason,
                        response=response,
                    )
        except Exception:
            # Telemetry helpers must never break application code.
            pass
    finally:
        end_backend_span(span)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

# Mapping from Mellea/OpenAI plain-string model_options keys to OTel request attrs.
_REQUEST_PARAM_MAP: dict[str, str] = {
    "temperature": "gen_ai.request.temperature",
    "top_p": "gen_ai.request.top_p",
    "top_k": "gen_ai.request.top_k",
    "frequency_penalty": "gen_ai.request.frequency_penalty",
    "presence_penalty": "gen_ai.request.presence_penalty",
}


def _serialize_json(obj: Any) -> str:
    """Serialise *obj* to a JSON string, coercing non-serialisable values to str."""
    return json.dumps(obj, default=str, ensure_ascii=False)


def _conversation_to_parts(conversation: list[dict]) -> tuple[list[dict], list[dict]]:
    """Split a conversation into system instructions and input messages.

    Args:
        conversation: List of ``{"role": ..., "content": ...}`` dicts.

    Returns:
        Tuple of ``(system_parts, input_messages)`` in the spec JSON shape.
        ``system_parts`` is a list of ``{"type": "text", "content": ...}`` items.
        ``input_messages`` is a list of
        ``{"role": ..., "parts": [{"type": "text", "content": ...}]}`` items.
    """
    system_parts: list[dict] = []
    input_messages: list[dict] = []
    for msg in conversation:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role == "system":
            system_parts.append({"type": "text", "content": str(content)})
        else:
            input_messages.append(
                {"role": role, "parts": [{"type": "text", "content": str(content)}]}
            )
    return system_parts, input_messages


def _emit_content_attributes(
    span: Any,
    *,
    conversation: list[dict],
    output_text: str | None,
    finish_reason: str | None,
    response: Any = None,
) -> None:
    """Set structured content attributes on the span (content gate must be checked by caller)."""
    try:
        system_parts, input_messages = _conversation_to_parts(conversation)

        if system_parts:
            set_span_attribute(
                span, "gen_ai.system_instructions", _serialize_json(system_parts)
            )
        if input_messages:
            set_span_attribute(
                span, "gen_ai.input.messages", _serialize_json(input_messages)
            )

        # Attempt to derive output text from an OpenAI-format response if not provided
        if output_text is None and response is not None:
            try:
                choices = get_value(response, "choices")
                if choices:
                    first = choices[0] if isinstance(choices, list) else choices
                    msg = get_value(first, "message")
                    if msg is not None:
                        output_text = str(get_value(msg, "content") or "")
            except Exception:
                pass

        if output_text is not None:
            output_msg = [
                {
                    "role": "assistant",
                    "parts": [{"type": "text", "content": output_text}],
                    "finish_reason": finish_reason or "stop",
                }
            ]
            set_span_attribute(
                span, "gen_ai.output.messages", _serialize_json(output_msg)
            )

        # Emit a span event so log-oriented receivers also see the content payload.
        add_span_event(span, "gen_ai.client.inference.operation.details")
    except Exception:
        # Content capture is best-effort — never fail the span close
        pass


__all__ = [
    "finalize_backend_span",
    "get_context_size",
    "get_model_id_str",
    "get_provider_name",
    "get_system_name",
    "instrument_generate_from_context",
    "instrument_generate_from_raw",
    "record_response_metadata",
    "record_token_usage",
    "start_generate_span",
]
