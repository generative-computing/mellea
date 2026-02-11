"""Tests for message conversion utilities."""

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from mellea_langchain.message_conversion import (
    langchain_to_mellea_messages,
    mellea_to_langchain_result,
)


class TestMessageConversion:
    """Test message conversion between LangChain and Mellea formats."""

    def test_human_message_conversion(self):
        """Test conversion of HumanMessage to Mellea format."""
        lc_messages = [HumanMessage(content="Hello, world!")]
        mellea_messages = langchain_to_mellea_messages(lc_messages)

        assert len(mellea_messages) == 1
        assert mellea_messages[0].role == "user"
        assert mellea_messages[0].content == "Hello, world!"

    def test_system_message_conversion(self):
        """Test conversion of SystemMessage to Mellea format."""
        lc_messages = [SystemMessage(content="You are a helpful assistant.")]
        mellea_messages = langchain_to_mellea_messages(lc_messages)

        assert len(mellea_messages) == 1
        assert mellea_messages[0].role == "system"
        assert mellea_messages[0].content == "You are a helpful assistant."

    def test_ai_message_conversion(self):
        """Test conversion of AIMessage to Mellea format."""
        lc_messages = [AIMessage(content="I'm here to help!")]
        mellea_messages = langchain_to_mellea_messages(lc_messages)

        assert len(mellea_messages) == 1
        assert mellea_messages[0].role == "assistant"
        assert mellea_messages[0].content == "I'm here to help!"

    def test_tool_message_conversion(self):
        """Test conversion of ToolMessage to Mellea format."""
        lc_messages = [ToolMessage(content="Tool result", tool_call_id="call_123")]
        mellea_messages = langchain_to_mellea_messages(lc_messages)

        assert len(mellea_messages) == 1
        # Tool messages are currently converted to user messages with context
        assert mellea_messages[0].role == "user"
        assert "Tool result" in mellea_messages[0].content

    def test_multiple_messages_conversion(self):
        """Test conversion of multiple messages."""
        lc_messages = [
            SystemMessage(content="You are helpful."),
            HumanMessage(content="Hi!"),
            AIMessage(content="Hello!"),
            HumanMessage(content="How are you?"),
        ]
        mellea_messages = langchain_to_mellea_messages(lc_messages)

        assert len(mellea_messages) == 4
        assert mellea_messages[0].role == "system"
        assert mellea_messages[1].role == "user"
        assert mellea_messages[2].role == "assistant"
        assert mellea_messages[3].role == "user"

    def test_empty_messages_list(self):
        """Test conversion of empty messages list."""
        lc_messages = []
        mellea_messages = langchain_to_mellea_messages(lc_messages)

        assert len(mellea_messages) == 0

    def test_ai_message_with_tool_calls(self):
        """Test conversion of AIMessage with tool calls."""
        lc_messages = [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "call_123",
                        "name": "get_weather",
                        "args": {"location": "San Francisco"},
                    }
                ],
            )
        ]
        mellea_messages = langchain_to_mellea_messages(lc_messages)

        assert len(mellea_messages) == 1
        assert mellea_messages[0].role == "assistant"
        # Tool calls are currently not preserved in conversion
        # This can be enhanced when Mellea's tool calling API is clarified
        assert mellea_messages[0].content == ""


class TestMelleaToLangChainConversion:
    """Test conversion from Mellea responses to LangChain format."""

    def test_simple_response_conversion(self):
        """Test conversion of simple Mellea response."""

        # Mock Mellea response
        class MockResponse:
            def __init__(self):
                self.content = "This is a response"
                self._tool_calls = None

        response = MockResponse()
        result = mellea_to_langchain_result(response)

        assert len(result.generations) == 1
        assert result.generations[0].message.content == "This is a response"
        assert isinstance(result.generations[0].message, AIMessage)

    def test_response_with_tool_calls(self):
        """Test conversion of Mellea response with tool calls."""

        # Mock Mellea response with tool calls
        class MockToolCall:
            def __init__(self):
                self.id = "call_123"
                self.name = "get_weather"
                self.arguments = {"location": "Boston"}

        class MockResponse:
            def __init__(self):
                self.content = ""
                self._tool_calls = [MockToolCall()]

        response = MockResponse()
        result = mellea_to_langchain_result(response)

        assert len(result.generations) == 1
        message = result.generations[0].message
        assert message.tool_calls is not None
        assert len(message.tool_calls) == 1
        assert message.tool_calls[0]["name"] == "get_weather"

    def test_response_with_metadata(self):
        """Test conversion with additional metadata."""

        class MockResponse:
            def __init__(self):
                self.content = "Response"
                self._tool_calls = None

        response = MockResponse()
        result = mellea_to_langchain_result(
            response, generation_info={"model": "test"}, llm_output={"token_usage": 100}
        )

        assert result.generations[0].generation_info == {"model": "test"}
        assert result.llm_output == {"token_usage": 100}


class TestMessageConversionEdgeCases:
    """Test edge cases and error handling in message conversion."""

    def test_message_with_none_content(self):
        """Test conversion handles string conversion properly."""
        # LangChain validates content types, so we test with valid string
        lc_messages = [HumanMessage(content="None")]
        mellea_messages = langchain_to_mellea_messages(lc_messages)

        assert len(mellea_messages) == 1
        assert mellea_messages[0].role == "user"
        assert mellea_messages[0].content == "None"

    def test_message_with_numeric_content(self):
        """Test conversion with string representation of numbers."""
        # LangChain requires string content, so we test with string
        lc_messages = [HumanMessage(content="42")]
        mellea_messages = langchain_to_mellea_messages(lc_messages)

        assert len(mellea_messages) == 1
        assert mellea_messages[0].content == "42"

    def test_message_with_unicode_content(self):
        """Test conversion of message with unicode characters."""
        lc_messages = [
            HumanMessage(content="Hello 世界 🌍"),
            AIMessage(content="你好 مرحبا שלום"),
        ]
        mellea_messages = langchain_to_mellea_messages(lc_messages)

        assert len(mellea_messages) == 2
        assert mellea_messages[0].content == "Hello 世界 🌍"
        assert mellea_messages[1].content == "你好 مرحبا שלום"

    def test_message_with_special_characters(self):
        """Test conversion with special characters."""
        special_content = "Test\n\t<>&\"'`\\/"
        lc_messages = [HumanMessage(content=special_content)]
        mellea_messages = langchain_to_mellea_messages(lc_messages)

        assert len(mellea_messages) == 1
        assert mellea_messages[0].content == special_content

    def test_message_with_very_long_content(self):
        """Test conversion of message with very long content."""
        long_content = "A" * 100000
        lc_messages = [HumanMessage(content=long_content)]
        mellea_messages = langchain_to_mellea_messages(lc_messages)

        assert len(mellea_messages) == 1
        assert mellea_messages[0].content == long_content
        assert len(mellea_messages[0].content) == 100000

    def test_message_with_empty_string_content(self):
        """Test conversion of message with empty string."""
        lc_messages = [HumanMessage(content="")]
        mellea_messages = langchain_to_mellea_messages(lc_messages)

        assert len(mellea_messages) == 1
        assert mellea_messages[0].content == ""

    def test_message_with_whitespace_only(self):
        """Test conversion of message with only whitespace."""
        lc_messages = [HumanMessage(content="   \n\t  ")]
        mellea_messages = langchain_to_mellea_messages(lc_messages)

        assert len(mellea_messages) == 1
        assert mellea_messages[0].content == "   \n\t  "

    def test_ai_message_with_empty_tool_calls(self):
        """Test conversion of AIMessage with empty tool_calls list."""
        lc_messages = [AIMessage(content="Response", tool_calls=[])]
        mellea_messages = langchain_to_mellea_messages(lc_messages)

        assert len(mellea_messages) == 1
        assert mellea_messages[0].role == "assistant"
        assert mellea_messages[0].content == "Response"

    def test_ai_message_with_multiple_tool_calls(self):
        """Test conversion of AIMessage with multiple tool calls."""
        lc_messages = [
            AIMessage(
                content="",
                tool_calls=[
                    {"id": "call_1", "name": "search", "args": {"query": "Python"}},
                    {
                        "id": "call_2",
                        "name": "calculator",
                        "args": {"expression": "2+2"},
                    },
                ],
            )
        ]
        mellea_messages = langchain_to_mellea_messages(lc_messages)

        assert len(mellea_messages) == 1
        assert mellea_messages[0].role == "assistant"
        # Tool calls are not preserved in current implementation
        assert mellea_messages[0].content == ""

    def test_tool_message_with_empty_content(self):
        """Test conversion of ToolMessage with empty content."""
        lc_messages = [ToolMessage(content="", tool_call_id="call_123")]
        mellea_messages = langchain_to_mellea_messages(lc_messages)

        assert len(mellea_messages) == 1
        assert mellea_messages[0].role == "user"
        assert "Tool result:" in mellea_messages[0].content

    def test_tool_message_with_json_content(self):
        """Test conversion of ToolMessage with JSON content."""
        json_content = '{"result": "success", "data": [1, 2, 3]}'
        lc_messages = [ToolMessage(content=json_content, tool_call_id="call_123")]
        mellea_messages = langchain_to_mellea_messages(lc_messages)

        assert len(mellea_messages) == 1
        assert json_content in mellea_messages[0].content

    def test_mixed_message_types_with_edge_cases(self):
        """Test conversion of mixed message types with edge cases."""
        lc_messages = [
            SystemMessage(content=""),
            HumanMessage(content="Hello"),
            AIMessage(content="", tool_calls=[{"id": "1", "name": "tool", "args": {}}]),
            ToolMessage(content="Result", tool_call_id="1"),
            HumanMessage(content="Thanks"),
        ]
        mellea_messages = langchain_to_mellea_messages(lc_messages)

        assert len(mellea_messages) == 5
        assert mellea_messages[0].role == "system"
        assert mellea_messages[0].content == ""
        assert mellea_messages[1].role == "user"
        assert mellea_messages[2].role == "assistant"
        assert mellea_messages[3].role == "user"  # ToolMessage converted to user
        assert mellea_messages[4].role == "user"

    def test_response_with_none_content(self):
        """Test conversion of Mellea response with None content."""

        class MockResponse:
            def __init__(self):
                self.content = None
                self._tool_calls = None

        response = MockResponse()
        result = mellea_to_langchain_result(response)

        assert result.generations[0].message.content == ""

    def test_response_without_content_attribute(self):
        """Test conversion of Mellea response without content attribute."""

        class MockResponse:
            pass

        response = MockResponse()
        result = mellea_to_langchain_result(response)

        assert result.generations[0].message.content == ""

    def test_response_with_empty_tool_calls_list(self):
        """Test conversion of response with empty tool_calls list."""

        class MockResponse:
            def __init__(self):
                self.content = "Response"
                self._tool_calls = []

        response = MockResponse()
        result = mellea_to_langchain_result(response)

        # Should not include tool_calls if empty
        assert (
            not hasattr(result.generations[0].message, "tool_calls")
            or not result.generations[0].message.tool_calls
        )

    def test_response_with_malformed_tool_call(self):
        """Test conversion of response with malformed tool call."""

        class MockToolCall:
            def __init__(self):
                self.id = "call_123"
                self.name = "tool"
                # Missing arguments attribute

        class MockResponse:
            def __init__(self):
                self.content = ""
                self._tool_calls = [MockToolCall()]

        response = MockResponse()

        # Should handle gracefully (may raise AttributeError)
        try:
            result = mellea_to_langchain_result(response)
            # If it doesn't raise, check the result
            assert result is not None
        except AttributeError:
            # Expected if tool call is malformed
            pass

    def test_response_with_complex_generation_info(self):
        """Test conversion with complex generation_info."""

        class MockResponse:
            def __init__(self):
                self.content = "Response"
                self._tool_calls = None

        response = MockResponse()
        generation_info = {
            "model": "test-model",
            "tokens": 100,
            "finish_reason": "stop",
            "metadata": {"key": "value"},
        }

        result = mellea_to_langchain_result(response, generation_info=generation_info)

        assert result.generations[0].generation_info == generation_info
        assert result.generations[0].generation_info["model"] == "test-model"

    def test_response_with_complex_llm_output(self):
        """Test conversion with complex llm_output."""

        class MockResponse:
            def __init__(self):
                self.content = "Response"
                self._tool_calls = None

        response = MockResponse()
        llm_output = {
            "token_usage": {"prompt": 10, "completion": 20, "total": 30},
            "model_name": "test",
        }

        result = mellea_to_langchain_result(response, llm_output=llm_output)

        assert result.llm_output == llm_output
        assert result.llm_output["token_usage"]["total"] == 30

    def test_response_with_numeric_content(self):
        """Test conversion of response with numeric content."""

        class MockResponse:
            def __init__(self):
                self.content = 42
                self._tool_calls = None

        response = MockResponse()
        result = mellea_to_langchain_result(response)

        assert result.generations[0].message.content == "42"

    def test_response_with_unicode_content(self):
        """Test conversion of response with unicode content."""

        class MockResponse:
            def __init__(self):
                self.content = "Hello 世界 🌍"
                self._tool_calls = None

        response = MockResponse()
        result = mellea_to_langchain_result(response)

        assert result.generations[0].message.content == "Hello 世界 🌍"

    def test_large_conversation_history(self):
        """Test conversion of large conversation history."""
        messages = []
        for i in range(1000):
            if i % 2 == 0:
                messages.append(HumanMessage(content=f"Message {i}"))
            else:
                messages.append(AIMessage(content=f"Response {i}"))

        mellea_messages = langchain_to_mellea_messages(messages)

        assert len(mellea_messages) == 1000
        assert mellea_messages[0].content == "Message 0"
        assert mellea_messages[-1].content == "Response 999"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
