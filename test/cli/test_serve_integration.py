"""Integration tests for m serve using FastAPI TestClient.

Tests the full HTTP request/response cycle including:
- Streaming responses (SSE format, headers, chunking)
- Tool calling responses via HTTP
- Error handling at the HTTP layer
"""

import json
from typing import Any
from unittest.mock import Mock

import pytest
from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.testclient import TestClient

from cli.serve.app import make_chat_endpoint, validation_exception_handler
from cli.serve.models import FunctionDefinition, FunctionParameters, ToolFunction
from mellea.core.base import AbstractMelleaTool, ModelOutputThunk, ModelToolCall

# Mark all tests in this module as integration tests
pytestmark = pytest.mark.integration


class MockWeatherTool(AbstractMelleaTool):
    """Mock weather tool for testing."""

    name = "get_weather"

    def run(self, location: str, units: str = "celsius") -> str:
        """Mock run method."""
        return f"Weather in {location} is 22°{units[0].upper()}"

    @property
    def as_json_tool(self) -> dict[str, Any]:
        """Return JSON schema for this tool."""
        return {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather in a location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string", "description": "City name"},
                        "units": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"],
                            "description": "Temperature units",
                        },
                    },
                    "required": ["location"],
                },
            },
        }

    @property
    def as_tool_function(self) -> ToolFunction:
        """Return ToolFunction model for HTTP requests."""
        return ToolFunction(
            type="function",
            function=FunctionDefinition(
                name="get_weather",
                description="Get the current weather in a location",
                parameters=FunctionParameters(
                    RootModel={
                        "type": "object",
                        "properties": {
                            "location": {"type": "string", "description": "City name"},
                            "units": {
                                "type": "string",
                                "enum": ["celsius", "fahrenheit"],
                                "description": "Temperature units",
                            },
                        },
                        "required": ["location"],
                    }
                ),
            ),
        )


@pytest.fixture
def mock_module():
    """Create a mock module with a serve function."""
    module = Mock()
    module.__name__ = "test_integration_module"
    return module


@pytest.fixture
def test_app(mock_module):
    """Create a FastAPI test app with the chat endpoint."""
    app = FastAPI()
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.add_api_route(
        "/v1/chat/completions", make_chat_endpoint(mock_module), methods=["POST"]
    )
    return app


@pytest.fixture
def client(test_app):
    """Create a TestClient for the app."""
    return TestClient(test_app)


class TestStreamingIntegration:
    """Integration tests for streaming responses via HTTP."""

    def test_streaming_response_headers(self, client, mock_module):
        """Test that streaming responses have correct HTTP headers."""
        mock_output = ModelOutputThunk("Hello, streaming world!")
        mock_module.serve.return_value = mock_output

        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "Hello"}],
                "stream": True,
            },
        )

        # Verify streaming headers
        assert response.status_code == 200
        assert response.headers["content-type"] == "text/event-stream; charset=utf-8"

    def test_streaming_sse_format(self, client, mock_module):
        """Test that streaming responses follow SSE format."""
        mock_output = ModelOutputThunk("Test response")
        mock_module.serve.return_value = mock_output

        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "Hello"}],
                "stream": True,
            },
        )

        # Parse SSE chunks
        chunks = []
        for line in response.text.split("\n\n"):
            if line.startswith("data: "):
                data = line[6:].strip()
                if data != "[DONE]":
                    chunks.append(json.loads(data))

        # Verify chunk structure
        assert len(chunks) > 0
        for chunk in chunks:
            assert chunk["object"] == "chat.completion.chunk"
            assert "id" in chunk
            assert "model" in chunk
            assert "created" in chunk
            assert "choices" in chunk
            assert len(chunk["choices"]) == 1

        # Verify final chunk has finish_reason
        assert chunks[-1]["choices"][0]["finish_reason"] == "stop"

    def test_streaming_content_chunks(self, client, mock_module):
        """Test that content is properly chunked in streaming response."""
        mock_output = ModelOutputThunk("Hello world!")
        mock_module.serve.return_value = mock_output

        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "Say hello"}],
                "stream": True,
            },
        )

        # Parse chunks
        chunks = []
        for line in response.text.split("\n\n"):
            if line.startswith("data: "):
                data = line[6:].strip()
                if data != "[DONE]":
                    chunks.append(json.loads(data))

        # First chunk should have role
        assert chunks[0]["choices"][0]["delta"].get("role") == "assistant"

        # Second chunk should have content
        assert chunks[1]["choices"][0]["delta"].get("content") == "Hello world!"

        # Final chunk should have finish_reason
        assert chunks[-1]["choices"][0]["finish_reason"] == "stop"

    def test_streaming_with_usage_field(self, client, mock_module):
        """Test streaming response includes usage when stream_options.include_usage=True."""
        mock_output = ModelOutputThunk("Response")
        mock_output.usage = {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
        }
        mock_module.serve.return_value = mock_output

        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "Hello"}],
                "stream": True,
                "stream_options": {"include_usage": True},
            },
        )

        # Parse chunks
        chunks = []
        for line in response.text.split("\n\n"):
            if line.startswith("data: "):
                data = line[6:].strip()
                if data != "[DONE]":
                    chunks.append(json.loads(data))

        # Final chunk should include usage
        final_chunk = chunks[-1]
        assert "usage" in final_chunk
        assert final_chunk["usage"]["prompt_tokens"] == 10
        assert final_chunk["usage"]["completion_tokens"] == 5
        assert final_chunk["usage"]["total_tokens"] == 15

    def test_streaming_done_marker(self, client, mock_module):
        """Test that streaming response ends with [DONE] marker."""
        mock_output = ModelOutputThunk("Test")
        mock_module.serve.return_value = mock_output

        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "Hello"}],
                "stream": True,
            },
        )

        # Verify [DONE] marker is present
        assert "data: [DONE]" in response.text
        assert response.text.strip().endswith("data: [DONE]")


class TestToolCallingIntegration:
    """Integration tests for tool calling via HTTP."""

    def test_tool_call_response_structure(self, client, mock_module):
        """Test that tool calls are properly formatted in HTTP response."""
        mock_output = ModelOutputThunk("I'll check the weather.")
        mock_tool = MockWeatherTool()
        mock_output.tool_calls = {
            "get_weather": ModelToolCall(
                name="get_weather",
                func=mock_tool,
                args={"location": "Paris", "units": "celsius"},
            )
        }
        mock_module.serve.return_value = mock_output

        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [
                    {"role": "user", "content": "What's the weather in Paris?"}
                ],
                "tools": [mock_tool.as_tool_function.model_dump(mode="json")],
            },
        )

        assert response.status_code == 200
        data = response.json()

        # Verify response structure
        assert data["object"] == "chat.completion"
        assert data["choices"][0]["finish_reason"] == "tool_calls"
        assert data["choices"][0]["message"]["tool_calls"] is not None
        assert len(data["choices"][0]["message"]["tool_calls"]) == 1

        # Verify tool call details
        tool_call = data["choices"][0]["message"]["tool_calls"][0]
        assert tool_call["type"] == "function"
        assert tool_call["function"]["name"] == "get_weather"
        assert tool_call["id"].startswith("call_")

        # Verify arguments
        args = json.loads(tool_call["function"]["arguments"])
        assert args["location"] == "Paris"
        assert args["units"] == "celsius"

    def test_multiple_tool_calls_via_http(self, client, mock_module):
        """Test multiple tool calls in a single HTTP response."""
        mock_output = ModelOutputThunk("Checking multiple locations.")
        mock_tool = MockWeatherTool()
        mock_output.tool_calls = {
            "weather_paris": ModelToolCall(
                name="get_weather", func=mock_tool, args={"location": "Paris"}
            ),
            "weather_london": ModelToolCall(
                name="get_weather", func=mock_tool, args={"location": "London"}
            ),
            "weather_tokyo": ModelToolCall(
                name="get_weather", func=mock_tool, args={"location": "Tokyo"}
            ),
        }
        mock_module.serve.return_value = mock_output

        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "Weather in multiple cities"}],
                "tools": [mock_tool.as_tool_function.model_dump(mode="json")],
            },
        )

        assert response.status_code == 200
        data = response.json()

        # Verify multiple tool calls
        tool_calls = data["choices"][0]["message"]["tool_calls"]
        assert len(tool_calls) == 3

        # Verify each has unique ID
        ids = [tc["id"] for tc in tool_calls]
        assert len(ids) == len(set(ids)), "Tool call IDs should be unique"

        # Verify locations
        locations = [
            json.loads(tc["function"]["arguments"])["location"] for tc in tool_calls
        ]
        assert set(locations) == {"Paris", "London", "Tokyo"}

    def test_tool_calls_with_usage_info(self, client, mock_module):
        """Test that usage info is included with tool calls."""
        mock_output = ModelOutputThunk("Calling tool.")
        mock_tool = MockWeatherTool()
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

        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "Weather?"}],
                "tools": [mock_tool.as_tool_function.model_dump(mode="json")],
            },
        )

        assert response.status_code == 200
        data = response.json()

        # Verify both tool calls and usage
        assert data["choices"][0]["finish_reason"] == "tool_calls"
        assert data["usage"] is not None
        assert data["usage"]["total_tokens"] == 70


class TestStreamingWithToolCalls:
    """Integration tests for streaming responses with tool calls."""

    def test_streaming_tool_call_response(self, client, mock_module):
        """Test streaming response with tool calls via HTTP."""
        mock_output = ModelOutputThunk("I'll check that for you.")
        mock_tool = MockWeatherTool()
        mock_output.tool_calls = {
            "get_weather": ModelToolCall(
                name="get_weather", func=mock_tool, args={"location": "Paris"}
            )
        }
        mock_module.serve.return_value = mock_output

        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "Weather in Paris?"}],
                "tools": [mock_tool.as_tool_function.model_dump(mode="json")],
                "stream": True,
            },
        )

        assert response.status_code == 200
        assert response.headers["content-type"] == "text/event-stream; charset=utf-8"

        # Parse chunks
        chunks = []
        for line in response.text.split("\n\n"):
            if line.startswith("data: "):
                data = line[6:].strip()
                if data != "[DONE]":
                    chunks.append(json.loads(data))

        # Should have: initial (role), content, tool_calls, final
        assert len(chunks) == 4

        # Verify chunk sequence
        assert chunks[0]["choices"][0]["delta"].get("role") == "assistant"
        assert (
            chunks[1]["choices"][0]["delta"].get("content")
            == "I'll check that for you."
        )
        assert "tool_calls" in chunks[2]["choices"][0]["delta"]
        assert chunks[3]["choices"][0]["finish_reason"] == "tool_calls"

        # Verify tool call structure in streaming chunk
        tool_calls = chunks[2]["choices"][0]["delta"]["tool_calls"]
        assert len(tool_calls) == 1
        assert "index" in tool_calls[0], "Streaming tool calls must include index"
        assert tool_calls[0]["index"] == 0
        assert tool_calls[0]["type"] == "function"
        assert tool_calls[0]["function"]["name"] == "get_weather"

    def test_streaming_multiple_tool_calls(self, client, mock_module):
        """Test streaming with multiple tool calls via HTTP."""
        mock_output = ModelOutputThunk("Checking multiple locations.")
        mock_tool = MockWeatherTool()
        mock_output.tool_calls = {
            "weather_1": ModelToolCall(
                name="get_weather", func=mock_tool, args={"location": "Paris"}
            ),
            "weather_2": ModelToolCall(
                name="get_weather", func=mock_tool, args={"location": "London"}
            ),
        }
        mock_module.serve.return_value = mock_output

        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "Weather?"}],
                "tools": [mock_tool.as_tool_function.model_dump(mode="json")],
                "stream": True,
            },
        )

        # Parse chunks
        chunks = []
        for line in response.text.split("\n\n"):
            if line.startswith("data: "):
                data = line[6:].strip()
                if data != "[DONE]":
                    chunks.append(json.loads(data))

        # Find tool call chunk (with non-None tool_calls)
        tool_call_chunk = next(
            c
            for c in chunks
            if "tool_calls" in c["choices"][0]["delta"]
            and c["choices"][0]["delta"]["tool_calls"] is not None
        )
        tool_calls = tool_call_chunk["choices"][0]["delta"]["tool_calls"]

        # Verify multiple tool calls with indices
        assert len(tool_calls) == 2
        indices = [tc["index"] for tc in tool_calls]
        assert indices == [0, 1]

    def test_streaming_tool_calls_with_usage(self, client, mock_module):
        """Test streaming tool calls with usage info via HTTP."""
        mock_output = ModelOutputThunk("Calling tool.")
        mock_tool = MockWeatherTool()
        mock_output.tool_calls = {
            "get_weather": ModelToolCall(
                name="get_weather", func=mock_tool, args={"location": "Paris"}
            )
        }
        mock_output.usage = {
            "prompt_tokens": 30,
            "completion_tokens": 15,
            "total_tokens": 45,
        }
        mock_module.serve.return_value = mock_output

        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "Weather?"}],
                "tools": [mock_tool.as_tool_function.model_dump(mode="json")],
                "stream": True,
                "stream_options": {"include_usage": True},
            },
        )

        # Parse chunks
        chunks = []
        for line in response.text.split("\n\n"):
            if line.startswith("data: "):
                data = line[6:].strip()
                if data != "[DONE]":
                    chunks.append(json.loads(data))

        # Final chunk should have both finish_reason and usage
        final_chunk = chunks[-1]
        assert final_chunk["choices"][0]["finish_reason"] == "tool_calls"
        assert "usage" in final_chunk
        assert final_chunk["usage"]["total_tokens"] == 45


class TestHTTPErrorHandling:
    """Integration tests for error handling at HTTP layer."""

    def test_invalid_request_returns_400(self, client, mock_module):
        """Test that invalid requests return 400 with OpenAI error format."""
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "Hello"}],
                "n": 0,  # Invalid: must be >= 1
            },
        )

        assert response.status_code == 400
        data = response.json()
        assert "error" in data
        assert data["error"]["type"] == "invalid_request_error"
        assert data["error"]["param"] == "n"

    def test_unsupported_n_parameter(self, client, mock_module):
        """Test that n > 1 is rejected with proper error."""
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "Hello"}],
                "n": 2,
            },
        )

        assert response.status_code == 400
        data = response.json()
        assert "error" in data
        assert data["error"]["type"] == "invalid_request_error"
        assert data["error"]["param"] == "n"
        assert "not supported" in data["error"]["message"].lower()

    def test_server_error_returns_500(self, client, mock_module):
        """Test that server errors return 500 with OpenAI error format."""
        # Make serve raise an exception
        mock_module.serve.side_effect = RuntimeError("Internal error")

        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )

        assert response.status_code == 500
        data = response.json()
        assert "error" in data
        assert data["error"]["type"] == "server_error"
        assert "Internal error" in data["error"]["message"]
