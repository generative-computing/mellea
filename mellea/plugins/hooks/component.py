"""Component lifecycle hook payloads."""

from __future__ import annotations

from typing import Any

from mellea.plugins.base import MelleaBasePayload


class ComponentPreCreatePayload(MelleaBasePayload):
    """Payload for ``component_pre_create`` — before component creation."""

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
    """Payload for ``component_post_create`` — after component created, before execution."""

    component_type: str = ""
    component: Any = None  # Component


class ComponentPreExecutePayload(MelleaBasePayload):
    """Payload for ``component_pre_execute`` — before component execution via ``aact()``."""

    component_type: str = ""
    action: Any = None  # Component | CBlock
    context: Any = None  # Context
    context_view: list[Any] | None = None
    requirements: list[Any] = []
    model_options: dict[str, Any] = {}
    format: Any = None  # type | None
    strategy: Any = None  # SamplingStrategy | None
    tool_calls_enabled: bool = False


class ComponentPostSuccessPayload(MelleaBasePayload):
    """Payload for ``component_post_success`` — after successful component execution."""

    component_type: str = ""
    action: Any = None  # Component | CBlock
    result: Any = None  # ModelOutputThunk
    context_before: Any = None  # Context
    context_after: Any = None  # Context
    generate_log: Any = None  # GenerateLog
    sampling_results: list[Any] | None = None
    latency_ms: int = 0


class ComponentPostErrorPayload(MelleaBasePayload):
    """Payload for ``component_post_error`` — after component execution fails."""

    component_type: str = ""
    action: Any = None  # Component | CBlock
    error: Any = None  # Exception
    error_type: str = ""
    stack_trace: str = ""
    context: Any = None  # Context
    model_options: dict[str, Any] = {}
