"""Shared utility functions used across formatter-based backend implementations.

Provides ``to_chat``, which converts a ``Context`` and a ``Component`` action into
the list of role/content dicts expected by ``apply_chat_template``; and
``to_tool_calls``, which parses a raw model output string into validated
``ModelToolCall`` objects. These helpers are consumed internally by all
``FormatterBackend`` subclasses.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any

from ..core import CBlock, Component, Context, MelleaLogger, ModelToolCall
from ..core.base import AbstractMelleaTool
from ..formatters import ChatFormatter
from ..stdlib.components import Message
from .tools import parse_tools, validate_tool_arguments

# Chat = dict[Literal["role", "content"], str] # external apply_chat_template type hint is weaker
# Chat = dict[str, str | list[dict[str, Any]] ] # for multi-modal models
Chat = dict[str, str]


def get_value(obj: Any, key: str) -> Any:
    """Get value from dict or object attribute.

    Args:
        obj: Dict or object
        key: Key or attribute name

    Returns:
        Value if found, None otherwise
    """
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)


def to_chat(
    action: Component | CBlock,
    ctx: Context,
    formatter: ChatFormatter,
    system_prompt: str | None,
) -> list[Chat]:
    """Converts a context and an action into a series of dicts to be passed to apply_chat_template.

    This function is used by local inference backends.

    Args:
        action: The next component or CBlock to generate output for.
        ctx: The current conversation context.
        formatter: The chat formatter used to convert context and action to messages.
        system_prompt: Optional system prompt to prepend; overrides any system message in the context.

    Returns:
        List of role/content dicts suitable for ``apply_chat_template``.
    """
    assert ctx.is_chat_context

    linearized_ctx = ctx.view_for_generation()
    assert linearized_ctx is not None, (
        "If ctx.is_chat_context, then the context should be linearizable."
    )
    ctx_as_message_list: list[Message] = formatter.to_chat_messages(linearized_ctx)
    # add action
    ctx_as_message_list.extend(formatter.to_chat_messages([action]))

    # NOTE: `self.formatter.to_chat_messages` explicitly skips `Message` objects. However, we need
    # to print `Message`s to correctly serialize any documents with the message. Do the printing here.
    ctx_as_conversation: list = [
        {"role": m.role, "content": formatter.print(m)} for m in ctx_as_message_list
    ]

    # Check that we ddin't accidentally end up with CBlocks.
    for msg in ctx_as_conversation:
        for v in msg.values():
            if "CBlock" in v:
                MelleaLogger.get_logger().error(
                    f"Found the string `CBlock` in what should've been a stringified context: {ctx_as_conversation}"
                )

    # handle custom system prompts. It's important that we do this before the _parse_and_**clean**_model_options step.
    if system_prompt is not None:
        system_msg: Chat = {"role": "system", "content": system_prompt}
        ctx_as_conversation.insert(0, system_msg)

    return ctx_as_conversation


def to_tool_calls(
    tools: dict[str, AbstractMelleaTool], decoded_result: str
) -> dict[str, ModelToolCall] | None:
    """Parse a tool call string.

    Args:
        tools: Mapping of tool name to the corresponding ``AbstractMelleaTool`` object.
        decoded_result: Raw model output string that may contain tool call markup.

    Returns:
        Dict mapping tool name to validated ``ModelToolCall``, or ``None`` if no tool calls were found.
    """
    model_tool_calls: dict[str, ModelToolCall] = dict()
    for tool_name, tool_args in parse_tools(decoded_result):
        func = tools.get(tool_name)
        if func is None:
            MelleaLogger.get_logger().warning(
                f"model attempted to call a non-existing function: {tool_name}"
            )
            continue

        # Clean up the function args slightly. Some models seem to
        # hallucinate parameters when none are required.
        param_map = func.as_json_tool["function"]["parameters"]["properties"]
        if len(param_map) == 0:
            tool_args = {}

        # Validate and coerce argument types
        validated_args = validate_tool_arguments(func, tool_args, strict=False)
        model_tool_calls[tool_name] = ModelToolCall(tool_name, func, validated_args)

    if len(model_tool_calls) > 0:
        return model_tool_calls
    return None
