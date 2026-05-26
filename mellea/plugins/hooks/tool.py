"""Tool execution hook payloads."""

from __future__ import annotations

from typing import Any

from mellea.plugins.base import MelleaBasePayload


class ToolPreInvokePayload(MelleaBasePayload):
    """Payload for ``tool_pre_invoke`` — before tool/function invocation.

    Attributes:
        model_tool_call: The ``ModelToolCall`` about to be executed (writable —
            plugins may modify arguments or swap the tool entirely).
        is_control_flow: ``True`` when this tool is used for framework control
            flow (e.g. ``final_answer`` in ReAct) rather than data processing.
            Plugins should check this field to decide whether to act.
    """

    model_tool_call: Any = None
    is_control_flow: bool = False


class ToolPostInvokePayload(MelleaBasePayload):
    """Payload for ``tool_post_invoke`` — after tool execution.

    Attributes:
        model_tool_call: The ``ModelToolCall`` that was executed.
        tool_output: The return value of the tool function (writable —
            plugins may transform the output before it is formatted).
        tool_message: The ``ToolMessage`` constructed from the output.
        execution_time_ms: Wall-clock time of the tool execution in milliseconds.
        success: ``True`` if the tool executed without raising an exception.
        error: The ``Exception`` raised during execution, or ``None`` on success.
        is_control_flow: ``True`` when this tool is used for framework control
            flow (e.g. ``final_answer`` in ReAct) rather than data processing.
            Plugins should check this field to decide whether to act.
    """

    model_tool_call: Any = None
    tool_output: Any = None
    tool_message: Any = None
    execution_time_ms: int = 0
    success: bool = True
    error: Any = None
    is_control_flow: bool = False
