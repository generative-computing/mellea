from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any, Literal

from mellea.backends.aloras import Alora
from mellea.backends.formatter import Formatter
from mellea.backends.tools import parse_tools
from mellea.helpers.fancy_logger import FancyLogger
from mellea.stdlib.base import CBlock, Component, Context, ModelToolCall
from mellea.stdlib.chat import Message
from mellea.stdlib.requirement import ALoraRequirement, LLMaJRequirement, Requirement

# Chat = dict[Literal["role", "content"], str] # external apply_chat_template type hint is weaker
# Chat = dict[str, str | list[dict[str, Any]] ] # for multi-modal models
Chat = dict[str, str]


def to_chat(
    action: Component | CBlock,
    ctx: Context,
    formatter: Formatter,
    system_prompt: str | None,
) -> list[Chat]:
    """Converts a context and an action into a series of dicts to be passed to apply_chat_template .

    This function is used by local inference backends.
    """
    assert ctx.is_chat_context

    linearized_ctx = ctx.view_for_generation()
    assert linearized_ctx is not None, (
        "If ctx.is_chat_context, then the context should be linearizable."
    )
    ctx_as_message_list: list[Message] = formatter.to_chat_messages(linearized_ctx)
    # add action
    ctx_as_message_list.extend(formatter.to_chat_messages([action]))

    ctx_as_conversation: list = [
        {"role": m.role, "content": m.content} for m in ctx_as_message_list
    ]

    # Check that we ddin't accidentally end up with CBlocks.
    for msg in ctx_as_conversation:
        for v in msg.values():
            if "CBlock" in v:
                FancyLogger.get_logger().error(
                    f"Found the string `CBlock` in what should've been a stringified context: {ctx_as_conversation}"
                )

    # handle custom system prompts. It's important that we do this before the _parse_and_**clean**_model_options step.
    if system_prompt is not None:
        system_msg: Chat = {"role": "system", "content": system_prompt}
        ctx_as_conversation.insert(0, system_msg)

    return ctx_as_conversation


def use_alora(
    action: Component | CBlock,
    alora: Alora | None,
    default_to_constraint_checking_alora: bool,
) -> bool:
    """Returns True when the condition for using alora is met.

    See `docs/dev/requirement_aLoRA_rerouting.md` for an explanation of the following code block.
    """
    if issubclass(type(action), Requirement):
        # The general rule is that we reroute to the alora if it exists.
        reroute_to_alora = alora is not None
        # However, there are some exceptions:
        if not default_to_constraint_checking_alora:
            reroute_to_alora = False
        if issubclass(type(action), LLMaJRequirement):
            reroute_to_alora = False
        if issubclass(type(action), ALoraRequirement):
            reroute_to_alora = True
        return reroute_to_alora
    else:
        return False


def to_tool_calls(
    tools: dict[str, Callable], decoded_result: str
) -> dict[str, ModelToolCall] | None:
    """Parse a tool call string."""
    model_tool_calls: dict[str, ModelToolCall] = dict()
    for tool_name, tool_args in parse_tools(decoded_result):
        func = tools.get(tool_name)
        if func is None:
            FancyLogger.get_logger().warning(
                f"model attempted to call a non-existing function: {tool_name}"
            )
            continue

        # Clean up the function args slightly. Some models seem to
        # hallucinate parameters when none are required.
        sig = inspect.signature(func)
        if len(sig.parameters) == 0:
            tool_args = {}

        model_tool_calls[tool_name] = ModelToolCall(tool_name, func, tool_args)

    if len(model_tool_calls) > 0:
        return model_tool_calls
    return None
