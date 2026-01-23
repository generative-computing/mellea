"""Backend instrumentation helpers for OpenTelemetry tracing."""

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
    backend: Any,
    action: Any,
    ctx: Any,
    format: Any = None,
    tool_calls: bool = False,
):
    """Create a backend trace span for generate_from_context.

    Args:
        backend: Backend instance
        action: Action component
        ctx: Context
        format: Response format (BaseModel subclass or None)
        tool_calls: Whether tool calling is enabled

    Returns:
        Context manager for the trace span
    """
    return trace_backend(
        "generate_from_context",
        backend=backend.__class__.__name__,
        model_id=get_model_id_str(backend),
        action_type=action.__class__.__name__,
        context_size=get_context_size(ctx),
        has_format=format is not None,
        format_type=format.__name__ if format else None,
        tool_calls=tool_calls,
    )


def instrument_generate_from_raw(
    backend: Any,
    num_actions: int,
    format: Any = None,
    tool_calls: bool = False,
):
    """Create a backend trace span for generate_from_raw.

    Args:
        backend: Backend instance
        num_actions: Number of actions in the batch
        format: Response format (BaseModel subclass or None)
        tool_calls: Whether tool calling is enabled

    Returns:
        Context manager for the trace span
    """
    return trace_backend(
        "generate_from_raw",
        backend=backend.__class__.__name__,
        model_id=get_model_id_str(backend),
        num_actions=num_actions,
        has_format=format is not None,
        format_type=format.__name__ if format else None,
        tool_calls=tool_calls,
    )


__all__ = [
    "get_model_id_str",
    "get_context_size",
    "instrument_generate_from_context",
    "instrument_generate_from_raw",
]

