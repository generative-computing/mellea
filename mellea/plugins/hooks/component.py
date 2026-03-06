"""Component lifecycle hook payloads."""

from __future__ import annotations

from typing import Any

from mellea.plugins.base import MelleaBasePayload


class ComponentPreCreatePayload(MelleaBasePayload):
    """Payload for ``component_pre_create`` — before component creation.

    Attributes:
        component_type: Class name of the component being created (e.g. ``"Instruction"``, ``"Message"``).
        description: The description / prompt text for the component.
        images: Optional list of ``ImageBlock`` instances attached to the component.
        requirements: List of ``Requirement`` instances for validation.
        icl_examples: List of in-context learning examples (``str`` or ``CBlock``).
        grounding_context: Dict mapping variable names to grounding values (``str``, ``CBlock``, or ``Component``).
        user_variables: Optional dict of user-defined Jinja template variables.
        prefix: Optional prefix (``str`` or ``CBlock``) prepended to the prompt.
        template_id: Optional template identifier for custom formatting.
    """

    component_type: str = ""
    description: str = ""
    images: list[Any] | None = None
    requirements: list[Any] = []
    icl_examples: list[Any] = []
    grounding_context: dict[str, Any] = {}
    user_variables: dict[str, str] | None = None
    prefix: Any = None
    template_id: str | None = None


class ComponentPostCreatePayload(MelleaBasePayload):
    """Payload for ``component_post_create`` — after component created, before execution.

    Attributes:
        component_type: Class name of the created component.
        component: The created ``Component`` instance.
    """

    component_type: str = ""
    component: Any = None


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
