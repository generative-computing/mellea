"""Tool execution hook payloads."""

from __future__ import annotations

from typing import Any

from mellea.plugins.base import MelleaBasePayload


class ToolPreInvokePayload(MelleaBasePayload):
    """Payload for ``tool_pre_invoke`` — before tool/function invocation."""

    tool_name: str = ""
    tool_args: dict[str, Any] = {}
    tool_callable: Any = None  # Callable
    model_tool_call: Any = None  # ModelToolCall


class ToolPostInvokePayload(MelleaBasePayload):
    """Payload for ``tool_post_invoke`` — after tool execution."""

    tool_name: str = ""
    tool_args: dict[str, Any] = {}
    tool_output: Any = None
    tool_message: Any = None  # ToolMessage
    execution_time_ms: int = 0
    success: bool = True
    error: Any = None  # Exception | None
