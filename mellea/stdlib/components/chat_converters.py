"""Converters for external library message types to Mellea Message types.

This module provides functions to convert chat messages between Mellea and external
libraries (like LangChain) using two approaches:

1. Direct conversion: Framework-specific parsing (e.g., `langchain_message_to_mellea`)
2. OpenAI intermediate: Uses OpenAI format as a common intermediate representation
   (e.g., `langchain_messages_to_mellea_via_openai`)

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

from typing import TYPE_CHECKING, Any, TypedDict, cast

from ...core import ImageBlock
from ...helpers.openai_compatible_helpers import message_to_openai_message
from .chat import Message

if TYPE_CHECKING:
    from langchain_core.messages import BaseMessage


# =============================================================================
# OpenAI Format Converters (Core)
# =============================================================================


class OpenAIMessage(TypedDict, total=False):
    """TypedDict for OpenAI message format."""

    role: str
    content: str | list[dict[str, Any]]
    name: str
    tool_call_id: str


def _extract_text_from_openai_content(content: str | list[dict[str, Any]]) -> str:
    """Extract text content from OpenAI message content.

    OpenAI messages can have content as either:
    - A simple string
    - A list of content blocks (for multimodal messages)

    Args:
        content: The content field from an OpenAI message.

    Returns:
        The extracted text content as a string.
    """
    if isinstance(content, str):
        return content

    text_parts = []
    for block in content:
        if isinstance(block, dict):
            if block.get("type") == "text":
                text_parts.append(block.get("text", ""))
        elif isinstance(block, str):
            text_parts.append(block)

    return "".join(text_parts)


def _extract_images_from_openai_content(
    content: str | list[dict[str, Any]],
) -> list[ImageBlock] | None:
    """Extract images from OpenAI message content.

    OpenAI multimodal messages contain image blocks in the format:
    {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}

    Args:
        content: The content field from an OpenAI message.

    Returns:
        A list of ImageBlock objects, or None if no images found.
    """
    if isinstance(content, str):
        return None

    images = []
    for block in content:
        if not isinstance(block, dict):
            continue

        if block.get("type") == "image_url":
            image_url = block.get("image_url", {})
            url = image_url.get("url", "") if isinstance(image_url, dict) else ""

            if url.startswith("data:image/"):
                try:
                    _, base64_data = url.split(",", 1)
                    images.append(ImageBlock(base64_data))
                except (ValueError, Exception):
                    continue

    return images if images else None


def openai_message_to_mellea(message: dict[str, Any]) -> Message:
    """Convert an OpenAI format message dict to a Mellea Message.

    This is the core converter that other framework converters can use
    as an intermediate step.

    Args:
        message: A dict in OpenAI message format with 'role' and 'content' keys.

    Returns:
        A Mellea Message object.

    Raises:
        ValueError: If required fields are missing or role is invalid.

    Example:
        >>> from mellea.stdlib.components.chat_converters import openai_message_to_mellea
        >>>
        >>> oai_msg = {"role": "user", "content": "Hello!"}
        >>> mellea_msg = openai_message_to_mellea(oai_msg)
        >>> mellea_msg.role
        'user'
    """
    role = message.get("role")
    if role is None:
        raise ValueError("OpenAI message missing 'role' field")

    if role not in ("system", "user", "assistant", "tool", "function"):
        raise ValueError(f"Invalid OpenAI message role: {role}")

    # Normalize function role to tool
    if role == "function":
        role = "tool"

    content = message.get("content", "")
    text_content = _extract_text_from_openai_content(content)
    images = _extract_images_from_openai_content(content)

    return Message(role=cast(Message.Role, role), content=text_content, images=images)


def openai_messages_to_mellea(messages: list[dict[str, Any]]) -> list[Message]:
    """Convert a list of OpenAI format messages to Mellea Messages.

    Args:
        messages: A list of dicts in OpenAI message format.

    Returns:
        A list of Mellea Message objects.
    """
    return [openai_message_to_mellea(msg) for msg in messages]


def mellea_message_to_openai(message: Message) -> dict[str, Any]:
    """Convert a Mellea Message to OpenAI format dict.

    This is an alias for the existing helper function, provided here
    for API consistency.

    Args:
        message: A Mellea Message object.

    Returns:
        A dict in OpenAI message format.
    """
    return message_to_openai_message(message)


def mellea_messages_to_openai(messages: list[Message]) -> list[dict[str, Any]]:
    """Convert a list of Mellea Messages to OpenAI format dicts.

    Args:
        messages: A list of Mellea Message objects.

    Returns:
        A list of dicts in OpenAI message format.
    """
    return [mellea_message_to_openai(msg) for msg in messages]


# =============================================================================
# LangChain Converters - Direct Approach
# =============================================================================

# Role mappings from LangChain message types to Mellea roles
_LANGCHAIN_TYPE_TO_ROLE: dict[str, Message.Role] = {
    "human": "user",
    "user": "user",
    "ai": "assistant",
    "assistant": "assistant",
    "system": "system",
    "tool": "tool",
    "function": "tool",
}


def _extract_text_from_langchain_content(content: str | list[dict[str, Any]]) -> str:
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

    text_parts = []
    for block in content:
        if isinstance(block, dict):
            if block.get("type") == "text":
                text_parts.append(block.get("text", ""))
            elif "text" in block and "type" not in block:
                text_parts.append(block["text"])
        elif isinstance(block, str):
            text_parts.append(block)

    return "".join(text_parts)


def _extract_images_from_langchain_content(
    content: str | list[dict[str, Any]],
) -> list[ImageBlock] | None:
    """Extract images from LangChain message content.

    LangChain multimodal messages can contain image blocks in various formats:
    - {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
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
                try:
                    _, base64_data = url.split(",", 1)
                    images.append(ImageBlock(base64_data))
                except (ValueError, Exception):
                    continue

        # Handle Anthropic-style image format
        elif block_type == "image":
            source = block.get("source", {})
            if isinstance(source, dict) and source.get("type") == "base64":
                base64_data = source.get("data", "")
                if base64_data:
                    images.append(ImageBlock(base64_data))

    return images if images else None


def langchain_message_to_mellea(message: BaseMessage) -> Message:
    """Convert a single LangChain message to a Mellea Message (direct approach).

    This function directly parses LangChain message attributes without using
    an intermediate format. For the OpenAI intermediate approach, use
    `langchain_messages_to_mellea_via_openai`.

    Supports the following LangChain message types:
    - HumanMessage -> Message(role="user", ...)
    - AIMessage -> Message(role="assistant", ...)
    - SystemMessage -> Message(role="system", ...)
    - ToolMessage -> Message(role="tool", ...)
    - FunctionMessage (deprecated) -> Message(role="tool", ...)

    Args:
        message: A LangChain BaseMessage or subclass.

    Returns:
        A Mellea Message object.

    Raises:
        ValueError: If the message type is not recognized.
    """
    msg_type = getattr(message, "type", None)
    if msg_type is None:
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

    content = getattr(message, "content", "")
    text_content = _extract_text_from_langchain_content(content)
    images = _extract_images_from_langchain_content(content)

    return Message(role=role, content=text_content, images=images)


def langchain_messages_to_mellea(messages: list[BaseMessage]) -> list[Message]:
    """Convert a list of LangChain messages to Mellea Messages (direct approach).

    This function directly parses LangChain message attributes. For the OpenAI
    intermediate approach, use `langchain_messages_to_mellea_via_openai`.

    Args:
        messages: A list of LangChain BaseMessage objects.

    Returns:
        A list of Mellea Message objects.
    """
    return [langchain_message_to_mellea(msg) for msg in messages]


# =============================================================================
# LangChain Converters - OpenAI Intermediate Approach
# =============================================================================


def langchain_messages_to_mellea_via_openai(
    messages: list[BaseMessage],
) -> list[Message]:
    """Convert LangChain messages to Mellea Messages via OpenAI intermediate format.

    This function uses LangChain's built-in `convert_to_openai_messages` to first
    convert to OpenAI format, then converts to Mellea. This approach leverages
    LangChain's own conversion logic.

    Args:
        messages: A list of LangChain BaseMessage objects.

    Returns:
        A list of Mellea Message objects.

    Raises:
        ImportError: If langchain_core is not installed.
    """
    from langchain_core.messages import convert_to_openai_messages

    openai_messages = convert_to_openai_messages(messages)
    return openai_messages_to_mellea(openai_messages)


def mellea_messages_to_langchain(messages: list[Message]) -> list[BaseMessage]:
    """Convert Mellea Messages to LangChain messages via OpenAI intermediate format.

    This function converts Mellea messages to OpenAI format, then uses LangChain's
    `convert_to_messages` to create LangChain message objects.

    Args:
        messages: A list of Mellea Message objects.

    Returns:
        A list of LangChain BaseMessage objects.

    Raises:
        ImportError: If langchain_core is not installed.
    """
    from langchain_core.messages import convert_to_messages

    openai_messages = mellea_messages_to_openai(messages)
    return convert_to_messages(openai_messages)
