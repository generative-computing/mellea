"""Converters for external library message types to Mellea Message types.

This module provides functions to convert chat messages from external libraries
(like LangChain) into Mellea Message types.

Example:
    >>> from langchain_core.messages import HumanMessage, AIMessage
    >>> from mellea.stdlib.components.chat_converters import langchain_messages_to_mellea
    >>>
    >>> lc_messages = [
    ...     HumanMessage(content="Hello!"),
    ...     AIMessage(content="Hi there!"),
    ... ]
    >>> mellea_messages = langchain_messages_to_mellea(lc_messages)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ...core import ImageBlock
from .chat import Message

if TYPE_CHECKING:
    from langchain_core.messages import BaseMessage


# Role mappings from LangChain message types to Mellea roles
_LANGCHAIN_TYPE_TO_ROLE: dict[str, Message.Role] = {
    "human": "user",
    "user": "user",
    "ai": "assistant",
    "assistant": "assistant",
    "system": "system",
    "tool": "tool",
    "function": "tool",  # LangChain's deprecated FunctionMessage maps to tool
}


def _extract_text_content(content: str | list[dict[str, Any]]) -> str:
    """Extract text content from LangChain message content.

    LangChain messages can have content as either:
    - A simple string
    - A list of content blocks (for multimodal messages)

    Args:
        content: The content field from a LangChain message.

    Returns:
        The extracted text content as a string.
    """
    if isinstance(content, str):
        return content

    # Content is a list of content blocks
    text_parts = []
    for block in content:
        if isinstance(block, dict):
            if block.get("type") == "text":
                text_parts.append(block.get("text", ""))
            elif "text" in block and "type" not in block:
                # Some messages just have {"text": "..."} without type
                text_parts.append(block["text"])
        elif isinstance(block, str):
            text_parts.append(block)

    return "".join(text_parts)


def _extract_images(content: str | list[dict[str, Any]]) -> list[ImageBlock] | None:
    """Extract images from LangChain message content.

    LangChain multimodal messages can contain image blocks in various formats:
    - {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
    - {"type": "image_url", "image_url": {"url": "https://..."}}
    - {"type": "image", "source": {"type": "base64", "data": "..."}}

    Args:
        content: The content field from a LangChain message.

    Returns:
        A list of ImageBlock objects, or None if no images found.
    """
    if isinstance(content, str):
        return None

    images = []
    for block in content:
        if not isinstance(block, dict):
            continue

        block_type = block.get("type")

        # Handle OpenAI-style image_url format
        if block_type == "image_url":
            image_url = block.get("image_url", {})
            url = image_url.get("url", "") if isinstance(image_url, dict) else ""

            if url.startswith("data:image/"):
                # Extract base64 data from data URI
                # Format: data:image/png;base64,<base64_data>
                try:
                    _, base64_data = url.split(",", 1)
                    images.append(ImageBlock(base64_data))
                except (ValueError, Exception):
                    continue
            # Note: We skip external URLs as ImageBlock requires base64 data

        # Handle Anthropic-style image format
        elif block_type == "image":
            source = block.get("source", {})
            if isinstance(source, dict) and source.get("type") == "base64":
                base64_data = source.get("data", "")
                if base64_data:
                    images.append(ImageBlock(base64_data))

    return images if images else None


def langchain_message_to_mellea(message: BaseMessage) -> Message:
    """Convert a single LangChain message to a Mellea Message.

    Supports the following LangChain message types:
    - HumanMessage -> Message(role="user", ...)
    - AIMessage -> Message(role="assistant", ...)
    - SystemMessage -> Message(role="system", ...)
    - ToolMessage -> Message(role="tool", ...)
    - FunctionMessage (deprecated) -> Message(role="tool", ...)

    Multimodal messages with images are also supported.

    Args:
        message: A LangChain BaseMessage or subclass.

    Returns:
        A Mellea Message object.

    Raises:
        ValueError: If the message type is not recognized.

    Example:
        >>> from langchain_core.messages import HumanMessage
        >>> from mellea.stdlib.components.chat_converters import langchain_message_to_mellea
        >>>
        >>> lc_msg = HumanMessage(content="Hello!")
        >>> mellea_msg = langchain_message_to_mellea(lc_msg)
        >>> mellea_msg.role
        'user'
    """
    # Get the message type from LangChain's type property
    msg_type = getattr(message, "type", None)
    if msg_type is None:
        # Fallback to class name based detection
        class_name = message.__class__.__name__.lower()
        if "human" in class_name:
            msg_type = "human"
        elif "ai" in class_name:
            msg_type = "ai"
        elif "system" in class_name:
            msg_type = "system"
        elif "tool" in class_name:
            msg_type = "tool"
        elif "function" in class_name:
            msg_type = "function"
        else:
            raise ValueError(
                f"Unknown LangChain message type: {message.__class__.__name__}"
            )

    role = _LANGCHAIN_TYPE_TO_ROLE.get(msg_type)
    if role is None:
        raise ValueError(f"Unknown LangChain message type: {msg_type}")

    # Extract content
    content = getattr(message, "content", "")
    text_content = _extract_text_content(content)
    images = _extract_images(content)

    # Handle tool messages specially if they have the required metadata
    if role == "tool":
        # Note: LangChain ToolMessage has name and tool_call_id attributes,
        # but Mellea's ToolMessage requires additional context (args, tool callable)
        # that LangChain ToolMessage doesn't provide. So we create a regular
        # Message with role="tool" for now.
        # Full ToolMessage support would require the original tool definition.
        return Message(role="tool", content=text_content, images=images)

    return Message(role=role, content=text_content, images=images)


def langchain_messages_to_mellea(messages: list[BaseMessage]) -> list[Message]:
    """Convert a list of LangChain messages to Mellea Messages.

    This is a convenience function that applies langchain_message_to_mellea
    to each message in the list.

    Args:
        messages: A list of LangChain BaseMessage objects.

    Returns:
        A list of Mellea Message objects.

    Example:
        >>> from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
        >>> from mellea.stdlib.components.chat_converters import langchain_messages_to_mellea
        >>>
        >>> lc_messages = [
        ...     SystemMessage(content="You are a helpful assistant"),
        ...     HumanMessage(content="Hello!"),
        ...     AIMessage(content="Hi there!"),
        ... ]
        >>> mellea_messages = langchain_messages_to_mellea(lc_messages)
        >>> [m.role for m in mellea_messages]
        ['system', 'user', 'assistant']
    """
    return [langchain_message_to_mellea(msg) for msg in messages]
