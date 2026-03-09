"""Component lifecycle hook payloads."""

from __future__ import annotations

from typing import Any

from mellea.plugins.base import MelleaBasePayload, WeakProxy


class ComponentPreExecutePayload(MelleaBasePayload):
    """Payload for ``component_pre_execute`` — before component execution via ``aact()``.

    Attributes:
        component_type: Class name of the component being executed.
        action: The ``Component`` or ``CBlock`` about to be executed (writable).
            Held as a weak reference — do not cache this payload.
        context: The current ``Context`` that will be passed to generation (writable).
            Held as a weak reference — do not cache this payload.
        context_view: Optional snapshot of the context as a list.
        requirements: List of ``Requirement`` instances for validation (writable).
        model_options: Dict of model options passed to the backend (writable).
        format: Optional ``BaseModel`` subclass for structured output / constrained decoding (writable).
        strategy: Optional ``SamplingStrategy`` instance controlling retry logic (writable).
            Held as a weak reference — do not cache this payload.
        tool_calls_enabled: Whether tool calling is enabled for this execution (writable).
    """

    component_type: str = ""
    action: WeakProxy = None
    context: WeakProxy = None
    context_view: list[Any] | None = None
    requirements: list[Any] = []
    model_options: dict[str, Any] = {}
    format: Any = None
    strategy: WeakProxy = None
    tool_calls_enabled: bool = False


class ComponentPostSuccessPayload(MelleaBasePayload):
    """Payload for ``component_post_success`` — after successful component execution.

    Attributes:
        component_type: Class name of the executed component.
        action: The ``Component`` or ``CBlock`` that was executed.
            Held as a weak reference — do not cache this payload.
        result: The ``ModelOutputThunk`` containing the generation result (writable).
        context_before: The ``Context`` before execution.
            Held as a weak reference — do not cache this payload.
        context_after: The ``Context`` after execution (with action + result appended).
            Held as a weak reference — do not cache this payload.
        generate_log: The ``GenerateLog`` from the final generation pass.
        sampling_results: Optional list of ``ModelOutputThunk`` from all sampling attempts.
        latency_ms: Wall-clock time for the full execution in milliseconds.
    """

    component_type: str = ""
    action: WeakProxy = None
    result: Any = None
    context_before: WeakProxy = None
    context_after: WeakProxy = None
    generate_log: Any = None
    sampling_results: list[Any] | None = None
    latency_ms: int = 0


class ComponentPostErrorPayload(MelleaBasePayload):
    """Payload for ``component_post_error`` — after component execution fails.

    Attributes:
        component_type: Class name of the component that failed.
        action: The ``Component`` or ``CBlock`` that was being executed.
            Held as a weak reference — do not cache this payload.
        error: The ``Exception`` that was raised.
        error_type: Class name of the exception (e.g. ``"ValueError"``).
        stack_trace: Formatted traceback string.
        context: The ``Context`` at the time of the error.
            Held as a weak reference — do not cache this payload.
        model_options: Dict of model options that were in effect.
    """

    component_type: str = ""
    action: WeakProxy = None
    error: Any = None
    error_type: str = ""
    stack_trace: str = ""
    context: WeakProxy = None
    model_options: dict[str, Any] = {}
