"""Message conversion utilities between LangChain and Mellea formats."""

from typing import Any

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.outputs import ChatGeneration, ChatResult

try:
    from mellea.core import ModelToolCall
    from mellea.stdlib.components import Message
except ImportError:
    # Fallback for type hints if mellea is not installed
    Message = Any  # type: ignore
    ModelToolCall = Any  # type: ignore


def langchain_to_mellea_messages(messages: list[BaseMessage]) -> list[Message]:
    """Convert LangChain messages to Mellea format.

    Args:
        messages: List of LangChain BaseMessage objects

    Returns:
        List of Mellea Message objects

    Mapping:
        - SystemMessage -> Message(role="system", content=...)
        - HumanMessage -> Message(role="user", content=...)
        - AIMessage -> Message(role="assistant", content=...)
        - ToolMessage -> Message(role="user", content=...) [Note: tool_call_id is lost]

    Known Limitations:
        - AIMessage tool_calls are not preserved during conversion (pending Mellea API support)
        - ToolMessage tool_call_id association is lost (converted to user message with prefix)
        - These limitations should be addressed when Mellea's tool message API is finalized
    """
    mellea_messages = []

    for msg in messages:
        if isinstance(msg, SystemMessage):
            mellea_messages.append(Message(role="system", content=str(msg.content)))
        elif isinstance(msg, HumanMessage):
            mellea_messages.append(Message(role="user", content=str(msg.content)))
        elif isinstance(msg, AIMessage):
            # TODO: Preserve tool_calls when Mellea's Message API supports it
            # Currently, tool_calls in AIMessage are dropped during conversion.
            # This is a known limitation that should be addressed when:
            # 1. Mellea's Message class supports a tool_calls parameter, OR
            # 2. Mellea provides a separate ToolCallMessage type
            #
            # For now, we only preserve the text content of the message.
            # If the AIMessage has tool_calls but no content, this may result
            # in an empty message being sent to Mellea.
            mellea_messages.append(
                Message(role="assistant", content=str(msg.content) if msg.content else "")
            )
        elif isinstance(msg, ToolMessage):
            # TODO: Preserve tool_call_id when Mellea's Message API supports it
            # Currently, ToolMessages are converted to user messages with a prefix,
            # which loses the tool_call_id association. This is a known limitation.
            #
            # Ideal implementation would be:
            #   Message(role="tool", content=..., tool_call_id=msg.tool_call_id)
            #
            # This should be updated when Mellea supports:
            # 1. A "tool" role for messages, AND
            # 2. A tool_call_id parameter to associate results with calls
            #
            # For now, we use a simple prefix to indicate this is a tool result.
            mellea_messages.append(Message(role="user", content=f"Tool result: {msg.content!s}"))
        else:
            # Fallback for unknown message types - treat as user message
            mellea_messages.append(Message(role="user", content=str(msg.content)))

    return mellea_messages


def mellea_to_langchain_result(response: Any, **kwargs: Any) -> ChatResult:
    """Convert Mellea response to LangChain ChatResult.

    Args:
        response: Mellea ModelOutputThunk
        **kwargs: Additional metadata (generation_info, llm_output)

    Returns:
        LangChain ChatResult with AIMessage
    """
    # Extract content from Mellea response
    content = str(response.content) if hasattr(response, "content") and response.content else ""

    # Extract tool calls if present (for future use)
    # For now, we don't include tool_calls to avoid validation errors
    tool_calls = []
    if hasattr(response, "_tool_calls") and response._tool_calls:
        tool_calls = [
            {"id": tc.id, "name": tc.name, "args": tc.arguments} for tc in response._tool_calls
        ]

    # Create AIMessage
    # Only include tool_calls if we have them (LangChain requires a list, not None)
    if tool_calls:
        message = AIMessage(content=content, tool_calls=tool_calls)
    else:
        message = AIMessage(content=content)

    # Create ChatGeneration
    generation = ChatGeneration(message=message, generation_info=kwargs.get("generation_info", {}))

    # Create ChatResult
    return ChatResult(generations=[generation], llm_output=kwargs.get("llm_output", {}))
