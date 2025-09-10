from __future__ import annotations

from typing import Literal

from mellea.backends.formatter import Formatter
from mellea.helpers.fancy_logger import FancyLogger
from mellea.stdlib.base import CBlock, Component, Context
from mellea.stdlib.chat import Message

# Chat = dict[Literal["role", "content"], str] # external apply_chat_template type hint is weaker
Chat = dict[str, str]


def to_chat(
    action: Component | CBlock,
    ctx: Context,
    formatter: Formatter,
    system_prompt: str | None,
) -> list[Chat]:
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
        system_msg: dict[str, str] = {"role": "system", "content": system_prompt}
        ctx_as_conversation.insert(0, system_msg)

    return ctx_as_conversation
