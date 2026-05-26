"""Generation pipeline hook payloads."""

from __future__ import annotations

from typing import Any

from mellea.plugins.base import MelleaBasePayload


class GenerationPreCallPayload(MelleaBasePayload):
    """Payload for ``generation_pre_call`` — before LLM backend call.

    Attributes:
        action: The ``Component`` or ``CBlock`` about to be sent to the backend.

        context: The ``Context`` being used for this generation call.

        model_options: Dict of model options (writable — plugins may adjust temperature, etc.).
        format: Optional ``BaseModel`` subclass for constrained decoding (writable).
        tool_calls: Whether tool calls are enabled for this generation (writable).
    """

    action: Any = None
    context: Any = None
    model_options: dict[str, Any] = {}
    format: Any = None
    tool_calls: bool = False


class GenerationPostCallPayload(MelleaBasePayload):
    """Payload for ``generation_post_call`` — fires once the model output is fully computed.

    For lazy ``ModelOutputThunk`` objects this hook fires inside
    ``ModelOutputThunk.astream`` after ``post_process`` completes, so
    ``model_output.value`` is guaranteed to be available. For already-computed
    thunks (e.g. cached responses) it fires before ``generate_from_context``
    returns.

    Attributes:
        prompt: The formatted prompt sent to the backend (str or list of message dicts).
        model_output: The fully-computed ``ModelOutputThunk``.
        latency_ms: Elapsed milliseconds from the ``generate_from_context`` call
            to when the value was fully materialized.
    """

    prompt: str | list[dict[str, Any]] = ""
    model_output: Any = None
    latency_ms: float = 0.0


class GenerationErrorPayload(MelleaBasePayload):
    """Payload for ``generation_error`` — fires when the LLM backend raises an exception.

    This hook fires inside ``ModelOutputThunk.astream`` just before the exception
    is re-raised, giving plugins a chance to observe (but not suppress) the error.

    Attributes:
        exception: The exception raised by the backend.
        model_output: The ``ModelOutputThunk`` at the time of the error. ``model``
            and ``provider`` are set when the backend set them early (before the
            async task); otherwise they are ``None``.
    """

    exception: BaseException
    model_output: Any = None
