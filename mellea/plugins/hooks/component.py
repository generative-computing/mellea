"""Component lifecycle hook payloads."""

from __future__ import annotations

from typing import Any

from mellea.plugins.base import MelleaBasePayload


class ComponentPreExecutePayload(MelleaBasePayload):
    """Payload for ``component_pre_execute`` — before component execution via ``aact()``.

    Attributes:
        component_type: Class name of the component being executed.
        action: The ``Component`` or ``CBlock`` about to be executed.
        context: The current ``Context`` that will be passed to generation.
        context_view: Optional snapshot of the context as a list.
        requirements: List of ``Requirement`` instances for validation.
        model_options: Dict of model options passed to the backend.
        format: Optional ``BaseModel`` subclass for structured output / constrained decoding.
        strategy: Optional ``SamplingStrategy`` instance controlling retry logic.
        tool_calls_enabled: Whether tool calling is enabled for this execution.
    """

    component_type: str = ""
    action: Any = None
    context: Any = None
    context_view: list[Any] | None = None
    requirements: list[Any] = []
    model_options: dict[str, Any] = {}
    format: Any = None
    strategy: Any = None
    tool_calls_enabled: bool = False


class ComponentPostSuccessPayload(MelleaBasePayload):
    """Payload for ``component_post_success`` — after successful component execution.

    Attributes:
        component_type: Class name of the executed component.
        action: The ``Component`` or ``CBlock`` that was executed.
        result: The ``ModelOutputThunk`` containing the generation result.
        context_before: The ``Context`` before execution.
        context_after: The ``Context`` after execution (with action + result appended).
        generate_log: The ``GenerateLog`` from the final generation pass.
        sampling_results: Optional list of ``ModelOutputThunk`` from all sampling attempts.
        latency_ms: Wall-clock time for the full execution in milliseconds.
    """

    component_type: str = ""
    action: Any = None
    result: Any = None
    context_before: Any = None
    context_after: Any = None
    generate_log: Any = None
    sampling_results: list[Any] | None = None
    latency_ms: int = 0


class ComponentPostErrorPayload(MelleaBasePayload):
    """Payload for ``component_post_error`` — after component execution fails.

    Attributes:
        component_type: Class name of the component that failed.
        action: The ``Component`` or ``CBlock`` that was being executed.
        error: The ``Exception`` that was raised.
        error_type: Class name of the exception (e.g. ``"ValueError"``).
        stack_trace: Formatted traceback string.
        context: The ``Context`` at the time of the error.
        model_options: Dict of model options that were in effect.
    """

    component_type: str = ""
    action: Any = None
    error: Any = None
    error_type: str = ""
    stack_trace: str = ""
    context: Any = None
    model_options: dict[str, Any] = {}
