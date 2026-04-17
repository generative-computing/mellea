"""Tests for tool calling support in m serve OpenAI-compatible API server."""

import json
from typing import Any
from unittest.mock import Mock

import pytest

from cli.serve.app import make_chat_endpoint
from cli.serve.models import (
    ChatCompletion,
    ChatCompletionRequest,
    ChatMessage,
    FunctionDefinition,
    FunctionParameters,
    ToolFunction,
)
from mellea.backends import ModelOption
from mellea.core.base import AbstractMelleaTool, ModelOutputThunk, ModelToolCall


class MockTool(AbstractMelleaTool):
    """Mock tool for testing."""

    name = "get_weather"

    def run(self, location: str) -> str:
        """Mock run method."""
        return f"Weather in {location} is sunny"

    @property
    def as_json_tool(self) -> dict[str, Any]:
        """Return JSON schema for this tool."""
        return {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather",
                "parameters": {
                    "type": "object",
                    "properties": {"location": {"type": "string"}},
                    "required": ["location"],
                },
            },
        }


@pytest.fixture
def mock_module():
    """Create a mock module with a serve function."""
    module = Mock()
    module.__name__ = "test_module"
    return module


@pytest.fixture
def sample_tool_request():
    """Create a sample ChatCompletionRequest with tools."""
    return ChatCompletionRequest(
        model="test-model",
        messages=[ChatMessage(role="user", content="What's the weather in Paris?")],
        tools=[
            ToolFunction(
                type="function",
                function=FunctionDefinition(
                    name="get_weather",
                    description="Get the current weather in a location",
                    parameters=FunctionParameters(
                        RootModel={
                            "type": "object",
                            "properties": {
                                "location": {
                                    "type": "string",
                                    "description": "The city name",
                                }
                            },
                            "required": ["location"],
                        }
                    ),
                ),
            )
        ],
        tool_choice="auto",
    )


class TestToolCalling:
    """Tests for tool calling functionality."""

    @pytest.mark.asyncio
    async def test_tool_calls_in_response(self, mock_module, sample_tool_request):
        """Test that tool calls are properly formatted in the response."""
        # Setup mock output with tool calls
        mock_output = ModelOutputThunk("I'll check the weather for you.")
        mock_tool = MockTool()
        mock_output.tool_calls = {
            "get_weather": ModelToolCall(
                name="get_weather", func=mock_tool, args={"location": "Paris"}
            )
        }
        mock_module.serve.return_value = mock_output

        # Create endpoint and call it
        endpoint = make_chat_endpoint(mock_module)
        response = await endpoint(sample_tool_request)

        # Verify response structure
        assert isinstance(response, ChatCompletion)
        assert response.choices[0].finish_reason == "tool_calls"
        assert response.choices[0].message.tool_calls is not None
        assert len(response.choices[0].message.tool_calls) == 1

        # Verify tool call details
        tool_call = response.choices[0].message.tool_calls[0]
        assert tool_call.type == "function"
        assert tool_call.function.name == "get_weather"

        # Parse and verify arguments
        args = json.loads(tool_call.function.arguments)
        assert args == {"location": "Paris"}

        # Verify tool call ID format
        assert tool_call.id.startswith("call_")
        assert len(tool_call.id) > len("call_")

    @pytest.mark.asyncio
    async def test_multiple_tool_calls(self, mock_module, sample_tool_request):
        """Test handling multiple tool calls in a single response."""
        mock_output = ModelOutputThunk("I'll check multiple locations.")
        mock_tool = MockTool()
        mock_output.tool_calls = {
            "get_weather_paris": ModelToolCall(
                name="get_weather", func=mock_tool, args={"location": "Paris"}
            ),
            "get_weather_london": ModelToolCall(
                name="get_weather", func=mock_tool, args={"location": "London"}
            ),
        }
        mock_module.serve.return_value = mock_output

        endpoint = make_chat_endpoint(mock_module)
        response = await endpoint(sample_tool_request)

        # Verify multiple tool calls
        assert response.choices[0].finish_reason == "tool_calls"
        assert len(response.choices[0].message.tool_calls) == 2

        # Verify each tool call has unique ID
        ids = [tc.id for tc in response.choices[0].message.tool_calls]
        assert len(ids) == len(set(ids)), "Tool call IDs should be unique"

    @pytest.mark.asyncio
    async def test_no_tool_calls_finish_reason_stop(
        self, mock_module, sample_tool_request
    ):
        """Test that finish_reason is 'stop' when no tool calls are made."""
        mock_output = ModelOutputThunk("The weather is sunny.")
        # No tool_calls set
        mock_module.serve.return_value = mock_output

        endpoint = make_chat_endpoint(mock_module)
        response = await endpoint(sample_tool_request)

        assert response.choices[0].finish_reason == "stop"
        assert response.choices[0].message.tool_calls is None

    @pytest.mark.asyncio
    async def test_empty_tool_calls_dict_finish_reason_stop(
        self, mock_module, sample_tool_request
    ):
        """Test that finish_reason is 'stop' when tool_calls is an empty dict.

        Regression test for bug where empty tool_calls dict {} produces
        finish_reason='tool_calls' with an empty array instead of
        finish_reason='stop' with tool_calls=None.
        """
        mock_output = ModelOutputThunk("Hello! How can I help?")
        # Set tool_calls to empty dict (the bug case)
        mock_output.tool_calls = {}
        mock_module.serve.return_value = mock_output

        endpoint = make_chat_endpoint(mock_module)
        response = await endpoint(sample_tool_request)

        # Should behave like no tool calls at all
        assert response.choices[0].finish_reason == "stop"
        assert response.choices[0].message.tool_calls is None

    @pytest.mark.asyncio
    async def test_tools_passed_to_model_options(
        self, mock_module, sample_tool_request
    ):
        """Test that tools are passed to serve function in model_options."""
        from mellea.backends.model_options import ModelOption

        mock_output = ModelOutputThunk("Test response")
        mock_module.serve.return_value = mock_output

        endpoint = make_chat_endpoint(mock_module)
        await endpoint(sample_tool_request)

        # Verify serve was called with tools in model_options
        call_args = mock_module.serve.call_args
        assert call_args is not None
        model_options = call_args.kwargs["model_options"]

        # Tools should be in model_options with the ModelOption.TOOLS key
        assert ModelOption.TOOLS in model_options
        assert model_options[ModelOption.TOOLS] is not None

    @pytest.mark.asyncio
    async def test_tool_choice_passed_to_model_options(
        self, mock_module, sample_tool_request
    ):
        """Test that tool_choice is passed to serve function in model_options."""
        mock_output = ModelOutputThunk("Test response")
        mock_module.serve.return_value = mock_output

        endpoint = make_chat_endpoint(mock_module)
        await endpoint(sample_tool_request)

        # Verify serve was called with tool_choice in model_options
        call_args = mock_module.serve.call_args
        assert call_args is not None
        model_options = call_args.kwargs["model_options"]

        # tool_choice should be passed through using ModelOption.TOOL_CHOICE
        assert ModelOption.TOOL_CHOICE in model_options
        assert model_options[ModelOption.TOOL_CHOICE] == "auto"

    @pytest.mark.asyncio
    async def test_tool_calls_with_complex_arguments(
        self, mock_module, sample_tool_request
    ):
        """Test tool calls with complex nested arguments."""
        mock_output = ModelOutputThunk("Processing complex request.")
        mock_tool = MockTool()
        mock_output.tool_calls = {
            "complex_tool": ModelToolCall(
                name="complex_function",
                func=mock_tool,
                args={
                    "location": "Paris",
                    "options": {
                        "units": "celsius",
                        "include_forecast": True,
                        "days": 5,
                    },
                    "tags": ["weather", "forecast"],
                },
            )
        }
        mock_module.serve.return_value = mock_output

        endpoint = make_chat_endpoint(mock_module)
        response = await endpoint(sample_tool_request)

        # Verify complex arguments are properly serialized
        tool_call = response.choices[0].message.tool_calls[0]
        args = json.loads(tool_call.function.arguments)

        assert args["location"] == "Paris"
        assert args["options"]["units"] == "celsius"
        assert args["options"]["include_forecast"] is True
        assert args["options"]["days"] == 5
        assert args["tags"] == ["weather", "forecast"]

    @pytest.mark.asyncio
    async def test_tool_calls_with_usage_info(self, mock_module, sample_tool_request):
        """Test that usage info is included alongside tool calls."""
        mock_output = ModelOutputThunk("Calling tool.")
        mock_tool = MockTool()
        mock_output.tool_calls = {
            "get_weather": ModelToolCall(
                name="get_weather", func=mock_tool, args={"location": "Paris"}
            )
        }
        mock_output.usage = {
            "prompt_tokens": 50,
            "completion_tokens": 20,
            "total_tokens": 70,
        }
        mock_module.serve.return_value = mock_output

        endpoint = make_chat_endpoint(mock_module)
        response = await endpoint(sample_tool_request)

        # Verify both tool calls and usage are present
        assert response.choices[0].finish_reason == "tool_calls"
        assert response.choices[0].message.tool_calls is not None
        assert response.usage is not None
        assert response.usage.total_tokens == 70

    @pytest.mark.asyncio
    async def test_request_without_tools(self, mock_module):
        """Test that requests without tools still work normally."""
        request = ChatCompletionRequest(
            model="test-model",
            messages=[ChatMessage(role="user", content="Hello")],
            # No tools specified
        )

        mock_output = ModelOutputThunk("Hello! How can I help?")
        mock_module.serve.return_value = mock_output

        endpoint = make_chat_endpoint(mock_module)
        response = await endpoint(request)

        # Should work normally without tool-related fields
        assert isinstance(response, ChatCompletion)
        assert response.choices[0].finish_reason == "stop"
        assert response.choices[0].message.tool_calls is None
        assert response.choices[0].message.content == "Hello! How can I help?"
