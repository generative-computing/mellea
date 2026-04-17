"""Async-safe context propagation for Mellea telemetry.

Carries ``session_id``, ``request_id``, ``model_id``, and ``sampling_iteration``
through async call chains via :mod:`contextvars`.  Values automatically appear
in log records (via :class:`MelleaContextFilter`) and can be attached to
OpenTelemetry spans.

Example::

    from mellea.telemetry.context import with_context, async_with_context, get_current_context

    # Synchronous (also works inside async functions):
    with with_context(session_id="s-1", model_id="granite"):
        result = backend.generate(...)

    # Async-with syntax:
    async with async_with_context(session_id="s-1", model_id="granite"):
        result = await backend.generate(...)
        # logs emitted here carry session_id and model_id automatically
"""

import contextlib
import contextvars
import logging
import uuid
from collections.abc import AsyncGenerator, Generator
from typing import Any

# ---------------------------------------------------------------------------
# ContextVar declarations
# ---------------------------------------------------------------------------

_session_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "mellea_session_id", default=None
)
_request_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "mellea_request_id", default=None
)
_model_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "mellea_model_id", default=None
)
_sampling_iteration: contextvars.ContextVar[int | None] = contextvars.ContextVar(
    "mellea_sampling_iteration", default=None
)

# Internal registry mapping public kwarg names to their ContextVar objects.
_CONTEXT_VARS: dict[str, contextvars.ContextVar[Any]] = {
    "session_id": _session_id,
    "request_id": _request_id,
    "model_id": _model_id,
    "sampling_iteration": _sampling_iteration,
}


# ---------------------------------------------------------------------------
# Helper getters
# ---------------------------------------------------------------------------


def get_session_id() -> str | None:
    """Return the session_id for the current async context, or ``None``."""
    return _session_id.get()


def get_request_id() -> str | None:
    """Return the request_id for the current async context, or ``None``."""
    return _request_id.get()


def get_model_id() -> str | None:
    """Return the model_id for the current async context, or ``None``."""
    return _model_id.get()


def get_sampling_iteration() -> int | None:
    """Return the sampling_iteration for the current async context, or ``None``."""
    return _sampling_iteration.get()


def get_current_context() -> dict[str, Any]:
    """Return a snapshot of all non-``None`` context values.

    Returns:
        Mapping of field names to their current values, omitting keys whose
        value is ``None``.
    """
    return {k: var.get() for k, var in _CONTEXT_VARS.items() if var.get() is not None}


def generate_request_id() -> str:
    """Generate a new unique request ID (UUID4 hex string).

    Returns:
        A short hex string suitable for use as a ``request_id``.
    """
    return uuid.uuid4().hex


# ---------------------------------------------------------------------------
# Context manager
# ---------------------------------------------------------------------------


def _apply_context(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Validate kwargs and set ContextVars, returning the reset tokens.

    Args:
        kwargs: Field names and values to set.

    Returns:
        Mapping of field name → reset token for later restoration.

    Raises:
        ValueError: If any key is not a recognised context field name.
    """
    unknown = set(kwargs) - set(_CONTEXT_VARS)
    if unknown:
        raise ValueError(
            f"Unknown context fields: {sorted(unknown)}. "
            f"Valid fields: {sorted(_CONTEXT_VARS)}"
        )
    return {name: _CONTEXT_VARS[name].set(value) for name, value in kwargs.items()}


def _restore_context(tokens: dict[str, Any]) -> None:
    """Reset all ContextVars to their pre-call values using saved tokens."""
    for name, token in tokens.items():
        _CONTEXT_VARS[name].reset(token)


@contextlib.contextmanager
def with_context(**kwargs: Any) -> Generator[None, None, None]:
    """Synchronous context manager that sets telemetry context for the block duration.

    On exit the previous values are restored, making this safe for nested
    usage and concurrent asyncio tasks (each ``asyncio.Task`` owns an isolated
    copy of its ``ContextVar`` state).

    Accepted keyword arguments: ``session_id``, ``request_id``, ``model_id``,
    ``sampling_iteration``.  Unknown keys raise :exc:`ValueError`.

    Args:
        **kwargs: Context fields to set within the block.

    Yields:
        None.

    Raises:
        ValueError: If an unknown context field name is passed.

    Example::

        with with_context(session_id="s-1", model_id="granite-4.0"):
            logger.info("generating")   # log record includes both fields
    """
    tokens = _apply_context(kwargs)
    try:
        yield
    finally:
        _restore_context(tokens)


@contextlib.asynccontextmanager
async def async_with_context(**kwargs: Any) -> AsyncGenerator[None, None]:
    """Async-with variant of :func:`with_context`.

    Identical semantics but usable with ``async with`` syntax.  Note that
    :func:`with_context` also works inside ``async`` functions — use this
    only when you specifically need ``async with`` syntax.

    Args:
        **kwargs: Context fields to set within the block.

    Yields:
        None.

    Raises:
        ValueError: If an unknown context field name is passed.
    """
    tokens = _apply_context(kwargs)
    try:
        yield
    finally:
        _restore_context(tokens)


# ---------------------------------------------------------------------------
# Logging filter
# ---------------------------------------------------------------------------


class MelleaContextFilter(logging.Filter):
    """Logging filter that injects telemetry context fields into every log record.

    Fields from :func:`get_current_context` (``session_id``, ``request_id``,
    ``model_id``, ``sampling_iteration``) are copied onto each
    :class:`logging.LogRecord` so they appear automatically in structured
    JSON output produced by :class:`~mellea.core.utils.JsonFormatter`.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        """Attach telemetry context fields to *record*.

        Args:
            record: The log record being processed.

        Returns:
            Always ``True`` — the record is never suppressed.
        """
        for key, value in get_current_context().items():
            if not hasattr(record, key):
                setattr(record, key, value)
        return True
