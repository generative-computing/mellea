"""Backend instrumentation helpers for OpenTelemetry tracing.

Follows OpenTelemetry Gen-AI semantic conventions:
https://opentelemetry.io/docs/specs/semconv/gen-ai/
"""

from typing import Any

from ..telemetry import set_span_attribute, trace_backend


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
    elif "vllm" in backend_class:
        return "vllm"
    else:
        return backend.__class__.__name__


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
    backend: Any, action: Any, ctx: Any, format: Any = None, tool_calls: bool = False
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

    Returns:
        Span object or None if tracing is disabled
    """
    from . import start_backend_span

    model_id = get_model_id_str(backend)
    system_name = get_system_name(backend)

    return start_backend_span(
        "chat",  # Gen-AI convention: use 'chat' for chat completions
        **{
            # Gen-AI semantic convention attributes
            "gen_ai.system": system_name,
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
        def get_value(obj, key):
            if isinstance(obj, dict):
                return obj.get(key)
            return getattr(obj, key, None)

        prompt_tokens = get_value(usage, "prompt_tokens")
        if prompt_tokens is not None:
            set_span_attribute(span, "gen_ai.usage.input_tokens", prompt_tokens)

        completion_tokens = get_value(usage, "completion_tokens")
        if completion_tokens is not None:
            set_span_attribute(span, "gen_ai.usage.output_tokens", completion_tokens)

        total_tokens = get_value(usage, "total_tokens")
        if total_tokens is not None:
            set_span_attribute(span, "gen_ai.usage.total_tokens", total_tokens)
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
        # Helper to get values from both objects and dicts
        def get_value(obj, key):
            if isinstance(obj, dict):
                return obj.get(key)
            return getattr(obj, key, None)

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


__all__ = [
    "get_context_size",
    "get_model_id_str",
    "get_system_name",
    "instrument_generate_from_context",
    "instrument_generate_from_raw",
    "record_response_metadata",
    "record_token_usage",
]
