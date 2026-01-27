"""Tests for chat_converters module (LangChain message conversion)."""

import base64

import pytest

# Check if langchain is available
try:
    from langchain_core.messages import (
        AIMessage,
        HumanMessage,
        SystemMessage,
        ToolMessage as LCToolMessage,
    )

    HAS_LANGCHAIN = True
except ImportError:
    HAS_LANGCHAIN = False

from mellea.stdlib.components import Message

# Skip all tests in this module if langchain is not installed
pytestmark = pytest.mark.skipif(
    not HAS_LANGCHAIN, reason="langchain-core not installed"
)


@pytest.fixture
def sample_human_message():
    """Create a sample HumanMessage."""
    return HumanMessage(content="Hello, how are you?")


@pytest.fixture
def sample_ai_message():
    """Create a sample AIMessage."""
    return AIMessage(content="I'm doing well, thank you!")


@pytest.fixture
def sample_system_message():
    """Create a sample SystemMessage."""
    return SystemMessage(content="You are a helpful assistant.")


@pytest.fixture
def sample_tool_message():
    """Create a sample ToolMessage."""
    return LCToolMessage(content="Tool result here", tool_call_id="call_123")


class TestLangchainMessageToMellea:
    """Tests for langchain_message_to_mellea function."""

    def test_human_message(self, sample_human_message):
        """Test converting HumanMessage to Mellea Message."""
        from mellea.stdlib.components.chat_converters import langchain_message_to_mellea

        result = langchain_message_to_mellea(sample_human_message)

        assert isinstance(result, Message)
        assert result.role == "user"
        assert result.content == "Hello, how are you?"

    def test_ai_message(self, sample_ai_message):
        """Test converting AIMessage to Mellea Message."""
        from mellea.stdlib.components.chat_converters import langchain_message_to_mellea

        result = langchain_message_to_mellea(sample_ai_message)

        assert isinstance(result, Message)
        assert result.role == "assistant"
        assert result.content == "I'm doing well, thank you!"

    def test_system_message(self, sample_system_message):
        """Test converting SystemMessage to Mellea Message."""
        from mellea.stdlib.components.chat_converters import langchain_message_to_mellea

        result = langchain_message_to_mellea(sample_system_message)

        assert isinstance(result, Message)
        assert result.role == "system"
        assert result.content == "You are a helpful assistant."

    def test_tool_message(self, sample_tool_message):
        """Test converting ToolMessage to Mellea Message."""
        from mellea.stdlib.components.chat_converters import langchain_message_to_mellea

        result = langchain_message_to_mellea(sample_tool_message)

        assert isinstance(result, Message)
        assert result.role == "tool"
        assert result.content == "Tool result here"

    def test_multimodal_message_with_text_content_blocks(self):
        """Test converting a message with text content blocks."""
        from mellea.stdlib.components.chat_converters import langchain_message_to_mellea

        content = [
            {"type": "text", "text": "First part. "},
            {"type": "text", "text": "Second part."},
        ]
        msg = HumanMessage(content=content)
        result = langchain_message_to_mellea(msg)

        assert result.role == "user"
        assert result.content == "First part. Second part."

    def test_multimodal_message_with_base64_image(self):
        """Test converting a message with a base64 image."""
        from mellea.stdlib.components.chat_converters import langchain_message_to_mellea

        # Create a minimal valid PNG (1x1 transparent pixel)
        # PNG header + IHDR + IDAT + IEND
        png_bytes = (
            b"\x89PNG\r\n\x1a\n"  # PNG signature
            b"\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06"
            b"\x00\x00\x00\x1f\x15\xc4\x89"  # IHDR chunk
            b"\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01"
            b"\r\n-\xb4"  # IDAT chunk
            b"\x00\x00\x00\x00IEND\xaeB`\x82"  # IEND chunk
        )
        base64_png = base64.b64encode(png_bytes).decode("utf-8")
        data_uri = f"data:image/png;base64,{base64_png}"

        content = [
            {"type": "text", "text": "What is in this image?"},
            {"type": "image_url", "image_url": {"url": data_uri}},
        ]
        msg = HumanMessage(content=content)
        result = langchain_message_to_mellea(msg)

        assert result.role == "user"
        assert result.content == "What is in this image?"
        assert result.images is not None
        assert len(result.images) == 1


class TestLangchainMessagesToMellea:
    """Tests for langchain_messages_to_mellea function."""

    def test_convert_message_list(
        self, sample_system_message, sample_human_message, sample_ai_message
    ):
        """Test converting a list of LangChain messages."""
        from mellea.stdlib.components.chat_converters import (
            langchain_messages_to_mellea,
        )

        messages = [sample_system_message, sample_human_message, sample_ai_message]
        results = langchain_messages_to_mellea(messages)

        assert len(results) == 3
        assert results[0].role == "system"
        assert results[1].role == "user"
        assert results[2].role == "assistant"

    def test_empty_list(self):
        """Test converting an empty list."""
        from mellea.stdlib.components.chat_converters import (
            langchain_messages_to_mellea,
        )

        results = langchain_messages_to_mellea([])
        assert results == []

    def test_conversation_flow(self):
        """Test a typical conversation flow."""
        from mellea.stdlib.components.chat_converters import (
            langchain_messages_to_mellea,
        )

        messages = [
            SystemMessage(content="You are a helpful coding assistant."),
            HumanMessage(content="Can you help me write a Python function?"),
            AIMessage(content="Of course! What would you like the function to do?"),
            HumanMessage(content="I need a function to calculate factorial."),
            AIMessage(
                content="Here's a recursive factorial function:\n\ndef factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n - 1)"
            ),
        ]
        results = langchain_messages_to_mellea(messages)

        assert len(results) == 5
        assert [m.role for m in results] == [
            "system",
            "user",
            "assistant",
            "user",
            "assistant",
        ]
        assert "factorial" in results[4].content


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_content(self):
        """Test message with empty content."""
        from mellea.stdlib.components.chat_converters import langchain_message_to_mellea

        msg = HumanMessage(content="")
        result = langchain_message_to_mellea(msg)

        assert result.role == "user"
        assert result.content == ""

    def test_content_with_special_characters(self):
        """Test message with special characters."""
        from mellea.stdlib.components.chat_converters import langchain_message_to_mellea

        content = "Special chars: \n\t\"'\\<>&"
        msg = HumanMessage(content=content)
        result = langchain_message_to_mellea(msg)

        assert result.content == content

    def test_content_with_unicode(self):
        """Test message with unicode characters."""
        from mellea.stdlib.components.chat_converters import langchain_message_to_mellea

        content = "Hello! 你好! مرحبا! 🎉"
        msg = HumanMessage(content=content)
        result = langchain_message_to_mellea(msg)

        assert result.content == content


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
