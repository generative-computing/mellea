"""Tests for the m serve OpenAI-compatible API server."""

import json
from unittest.mock import Mock

import pytest
from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.testclient import TestClient
from pydantic import BaseModel, ValidationError

from cli.serve.app import make_chat_endpoint, validation_exception_handler
from cli.serve.models import (
    ChatCompletion,
    ChatCompletionRequest,
    ChatMessage,
    CompletionUsage,
    FunctionDefinition,
    FunctionParameters,
    JsonSchemaFormat,
    ResponseFormat,
    ToolFunction,
)
from mellea.backends.model_options import ModelOption
from mellea.core.base import ModelOutputThunk, ModelToolCall


@pytest.fixture
def mock_module():
    """Create a mock module with a serve function."""
    module = Mock()
    module.__name__ = "test_module"
    return module


@pytest.fixture
def sample_request():
    """Create a sample ChatCompletionRequest."""
    return ChatCompletionRequest(
        model="test-model",
        messages=[ChatMessage(role="user", content="Hello, world!")],
        temperature=0.7,
        max_tokens=100,
    )


class TestChatEndpoint:
    """Tests for the chat completion endpoint."""

    @pytest.mark.asyncio
    async def test_basic_completion(self, mock_module, sample_request):
        """Test basic chat completion returns correct structure."""
        # Setup mock output
        mock_output = ModelOutputThunk("Hello! How can I help you?")
        mock_module.serve.return_value = mock_output

        # Create endpoint and call it
        endpoint = make_chat_endpoint(mock_module)
        response = await endpoint(sample_request)

        # Verify response structure
        assert isinstance(response, ChatCompletion)
        assert response.model == "test-model"
        assert len(response.choices) == 1
        assert response.choices[0].message.content == "Hello! How can I help you?"
        assert response.choices[0].message.role == "assistant"
        assert response.choices[0].index == 0

    @pytest.mark.asyncio
    async def test_finish_reason_included(self, mock_module, sample_request):
        """Test that finish_reason is included in the response."""
        mock_output = ModelOutputThunk("Test response")
        mock_module.serve.return_value = mock_output

        endpoint = make_chat_endpoint(mock_module)
        response = await endpoint(sample_request)

        assert response.choices[0].finish_reason == "stop"

    @pytest.mark.asyncio
    async def test_usage_field_populated(self, mock_module, sample_request):
        """Test that usage field is populated when available."""
        mock_output = ModelOutputThunk("Test response")
        mock_output.generation.usage = {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
        }
        mock_module.serve.return_value = mock_output

        endpoint = make_chat_endpoint(mock_module)
        response = await endpoint(sample_request)

        assert response.usage is not None
        assert isinstance(response.usage, CompletionUsage)
        assert response.usage.prompt_tokens == 10
        assert response.usage.completion_tokens == 5
        assert response.usage.total_tokens == 15

    @pytest.mark.asyncio
    async def test_usage_field_none_when_unavailable(self, mock_module, sample_request):
        """Test that usage field is None when not available."""
        mock_output = ModelOutputThunk("Test response")
        # Don't set usage field
        mock_module.serve.return_value = mock_output

        endpoint = make_chat_endpoint(mock_module)
        response = await endpoint(sample_request)

        assert response.usage is None

    @pytest.mark.asyncio
    async def test_system_fingerprint_always_none(self, mock_module, sample_request):
        """Test that system_fingerprint is always None.

        Per OpenAI spec, system_fingerprint represents a hash of backend config,
        not the model name. The model name is in response.model.
        We don't currently track backend config fingerprints.
        """
        mock_output = ModelOutputThunk("Test response")
        mock_module.serve.return_value = mock_output

        endpoint = make_chat_endpoint(mock_module)
        response = await endpoint(sample_request)

        # system_fingerprint should be None, not the model name
        assert response.system_fingerprint is None
        # Model name should be in the model field
        assert response.model == sample_request.model

    @pytest.mark.asyncio
    async def test_model_options_passed_correctly(self, mock_module, sample_request):
        """Test that model options are passed to serve function correctly."""
        mock_output = ModelOutputThunk("Test response")
        mock_module.serve.return_value = mock_output

        endpoint = make_chat_endpoint(mock_module)
        await endpoint(sample_request)

        # Verify serve was called with correct arguments
        call_args = mock_module.serve.call_args
        assert call_args is not None
        assert "model_options" in call_args.kwargs
        model_options = call_args.kwargs["model_options"]

        # Should include ModelOption keys for temperature and max_tokens
        # Note: TEMPERATURE is just "temperature" (not a sentinel), so it stays as-is
        assert ModelOption.TEMPERATURE in model_options
        assert model_options[ModelOption.TEMPERATURE] == 0.7
        assert ModelOption.MAX_NEW_TOKENS in model_options
        assert model_options[ModelOption.MAX_NEW_TOKENS] == 100
        assert "messages" not in model_options
        assert "requirements" not in model_options

    @pytest.mark.asyncio
    async def test_completion_id_format(self, mock_module, sample_request):
        """Test that completion ID follows OpenAI format."""
        mock_output = ModelOutputThunk("Test response")
        mock_module.serve.return_value = mock_output

        endpoint = make_chat_endpoint(mock_module)
        response = await endpoint(sample_request)

        # Should start with "chatcmpl-" and have a non-empty suffix
        assert response.id.startswith("chatcmpl-")
        assert len(response.id) > len("chatcmpl-"), "ID should have a suffix"

    @pytest.mark.asyncio
    async def test_created_timestamp_present(self, mock_module, sample_request):
        """Test that created timestamp is present and reasonable."""
        mock_output = ModelOutputThunk("Test response")
        mock_module.serve.return_value = mock_output

        endpoint = make_chat_endpoint(mock_module)
        response = await endpoint(sample_request)

        # Should be a Unix timestamp (positive integer)
        assert isinstance(response.created, int)
        assert response.created > 0

    @pytest.mark.asyncio
    async def test_object_type_correct(self, mock_module, sample_request):
        """Test that object type is set correctly."""
        mock_output = ModelOutputThunk("Test response")
        mock_module.serve.return_value = mock_output

        endpoint = make_chat_endpoint(mock_module)
        response = await endpoint(sample_request)

        assert response.object == "chat.completion"

    @pytest.mark.asyncio
    async def test_usage_with_partial_data(self, mock_module, sample_request):
        """Test that usage handles missing fields gracefully."""
        mock_output = ModelOutputThunk("Test response")
        # Only provide some fields
        mock_output.generation.usage = {
            "prompt_tokens": 10
            # Missing completion_tokens and total_tokens
        }
        mock_module.serve.return_value = mock_output

        endpoint = make_chat_endpoint(mock_module)
        response = await endpoint(sample_request)

        assert response.usage is not None
        assert response.usage.prompt_tokens == 10
        assert response.usage.completion_tokens == 0  # Should default to 0
        assert (
            response.usage.total_tokens == 10
        )  # Should be prompt_tokens + completion_tokens

    @pytest.mark.asyncio
    async def test_all_fields_together(self, mock_module, sample_request):
        """Test that all new fields work together correctly."""
        mock_output = ModelOutputThunk("Complete response")
        mock_output.generation.usage = {
            "prompt_tokens": 20,
            "completion_tokens": 10,
            "total_tokens": 30,
        }
        mock_module.serve.return_value = mock_output

        endpoint = make_chat_endpoint(mock_module)
        response = await endpoint(sample_request)

        # Verify all fields are present
        assert response.choices[0].finish_reason == "stop"
        assert response.usage is not None
        assert response.usage.total_tokens == 30
        assert response.system_fingerprint is None  # Not tracking backend config
        assert response.object == "chat.completion"
        assert response.id.startswith("chatcmpl-")

    @pytest.mark.asyncio
    async def test_n_greater_than_1_rejected(self, mock_module):
        """Test that requests with n > 1 are rejected with appropriate error."""

        request = ChatCompletionRequest(
            model="test-model",
            messages=[ChatMessage(role="user", content="Hello")],
            n=2,  # Request multiple completions
        )

        endpoint = make_chat_endpoint(mock_module)
        response = await endpoint(request)

        # Should return a JSONResponse error, not a ChatCompletion
        assert isinstance(response, JSONResponse)
        assert response.status_code == 400

        # Decode the response body
        body_bytes = response.body
        if isinstance(body_bytes, memoryview):
            body_bytes = bytes(body_bytes)
        error_data = json.loads(body_bytes.decode("utf-8"))
        assert "error" in error_data
        assert error_data["error"]["type"] == "invalid_request_error"
        assert error_data["error"]["param"] == "n"
        assert "not supported" in error_data["error"]["message"].lower()

        # Verify serve was never called
        mock_module.serve.assert_not_called()

    @pytest.mark.asyncio
    async def test_n_equals_1_accepted(self, mock_module):
        """Test that requests with n=1 are accepted."""
        request = ChatCompletionRequest(
            model="test-model",
            messages=[ChatMessage(role="user", content="Hello")],
            n=1,  # Explicitly set to 1
        )

        mock_output = ModelOutputThunk("Test response")
        mock_module.serve.return_value = mock_output

        endpoint = make_chat_endpoint(mock_module)
        response = await endpoint(request)

        # Should succeed
        assert isinstance(response, ChatCompletion)
        assert response.choices[0].message.content == "Test response"

        # Verify serve was called
        mock_module.serve.assert_called_once()

    @pytest.mark.asyncio
    async def test_n_less_than_1_rejected_by_pydantic(self, mock_module):
        """Test that requests with n < 1 are rejected by Pydantic validation.

        FastAPI automatically validates request models before they reach the endpoint,
        so n=0 or negative values will be caught by the framework, not our code.
        This test documents that behavior.
        """

        # Pydantic validation happens before the endpoint is called
        with pytest.raises(ValidationError) as exc_info:
            ChatCompletionRequest(
                model="test-model",
                messages=[ChatMessage(role="user", content="Hello")],
                n=0,  # Invalid: less than 1
            )

        # Verify the error is about the 'n' field
        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert errors[0]["loc"] == ("n",)
        assert errors[0]["type"] == "greater_than_equal"


class TestHTTPValidation:
    """Tests for HTTP-level validation via FastAPI TestClient."""

    def test_n_zero_rejected_at_http_level(self, mock_module):
        """Test that n=0 is rejected with OpenAI-compatible error format.

        Pydantic validation errors are caught by our custom exception handler
        and converted to OpenAI-compatible 400 errors (not FastAPI's default 422).
        """
        # Setup a test app with the exception handler
        app = FastAPI()
        app.add_exception_handler(RequestValidationError, validation_exception_handler)
        app.add_api_route(
            "/v1/chat/completions", make_chat_endpoint(mock_module), methods=["POST"]
        )
        client = TestClient(app)

        # Send request with n=0
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "Hello"}],
                "n": 0,
            },
        )

        # Our exception handler converts to OpenAI-compatible 400 error
        assert response.status_code == 400
        error_data = response.json()
        assert "error" in error_data
        assert error_data["error"]["type"] == "invalid_request_error"
        assert error_data["error"]["param"] == "n"
        # Pydantic's error message is used as-is
        assert "greater than or equal to 1" in error_data["error"]["message"].lower()

        # Verify serve was never called
        mock_module.serve.assert_not_called()

    def test_n_two_rejected_at_endpoint_level(self, mock_module):
        """Test that n=2 is rejected by our endpoint logic (not Pydantic).

        While n=2 passes Pydantic validation (ge=1), our endpoint explicitly
        rejects it because we don't support multiple completions.
        """
        # Setup a test app
        app = FastAPI()
        app.add_api_route(
            "/v1/chat/completions", make_chat_endpoint(mock_module), methods=["POST"]
        )
        client = TestClient(app)

        # Send request with n=2
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "Hello"}],
                "n": 2,
            },
        )

        # Our endpoint returns 400 for unsupported n > 1
        assert response.status_code == 400
        error_data = response.json()
        assert "error" in error_data
        assert error_data["error"]["type"] == "invalid_request_error"
        assert error_data["error"]["param"] == "n"
        assert "not supported" in error_data["error"]["message"].lower()

        # Verify serve was never called
        mock_module.serve.assert_not_called()

    @pytest.mark.asyncio
    async def test_n_none_accepted(self, mock_module):
        """Test that requests with n=None (default) are accepted."""
        request = ChatCompletionRequest(
            model="test-model",
            messages=[ChatMessage(role="user", content="Hello")],
            # n not specified, defaults to 1
        )

        mock_output = ModelOutputThunk("Test response")
        mock_module.serve.return_value = mock_output

        endpoint = make_chat_endpoint(mock_module)
        response = await endpoint(request)

        # Should succeed
        assert isinstance(response, ChatCompletion)
        assert response.choices[0].message.content == "Test response"

        # Verify serve was called
        mock_module.serve.assert_called_once()

    @pytest.mark.asyncio
    async def test_unsupported_params_excluded_from_model_options(self, mock_module):
        """Test that unsupported OpenAI parameters are excluded from model_options."""
        request = ChatCompletionRequest(
            model="test-model",
            messages=[ChatMessage(role="user", content="Hello")],
            temperature=0.7,
            max_tokens=100,
            # Unsupported parameters that should be excluded
            stream=False,
            stop=["END"],
            top_p=0.9,
            presence_penalty=0.5,
            frequency_penalty=0.3,
            logit_bias={"123": 1.0},
        )

        mock_output = ModelOutputThunk("Test response")
        mock_module.serve.return_value = mock_output

        endpoint = make_chat_endpoint(mock_module)
        response = await endpoint(request)

        # Should succeed
        assert isinstance(response, ChatCompletion)

        # Verify serve was called with correct model_options
        call_args = mock_module.serve.call_args
        assert call_args is not None
        model_options = call_args.kwargs["model_options"]

        # Supported parameters should be present
        assert ModelOption.TEMPERATURE in model_options
        assert model_options[ModelOption.TEMPERATURE] == 0.7
        assert ModelOption.MAX_NEW_TOKENS in model_options
        assert model_options[ModelOption.MAX_NEW_TOKENS] == 100

        # Unsupported parameters should NOT be in model_options
        assert "stream" not in model_options
        assert "stop" not in model_options
        assert "top_p" not in model_options
        assert "presence_penalty" not in model_options
        assert "frequency_penalty" not in model_options
        assert "logit_bias" not in model_options

    @pytest.mark.asyncio
    async def test_tool_params_passed_to_model_options(self, mock_module):
        """Test that tool-related parameters are passed to model_options."""
        request = ChatCompletionRequest(
            model="test-model",
            messages=[ChatMessage(role="user", content="Hello")],
            # Tool-related parameters
            tools=[
                ToolFunction(
                    type="function",
                    function=FunctionDefinition(
                        name="test_func",
                        description="A test function",
                        parameters=FunctionParameters(RootModel={"type": "object"}),
                    ),
                )
            ],
            tool_choice="auto",
            functions=[
                FunctionDefinition(
                    name="legacy_func",
                    description="A legacy function",
                    parameters=FunctionParameters(RootModel={"type": "object"}),
                )
            ],
            function_call="auto",
        )

        mock_output = ModelOutputThunk("Test response")
        mock_module.serve.return_value = mock_output

        endpoint = make_chat_endpoint(mock_module)
        response = await endpoint(request)

        # Should succeed
        assert isinstance(response, ChatCompletion)

        # Verify serve was called
        call_args = mock_module.serve.call_args
        assert call_args is not None
        model_options = call_args.kwargs["model_options"]

        # Tools should be passed with ModelOption.TOOLS key
        assert ModelOption.TOOLS in model_options
        # tool_choice should be passed through using ModelOption.TOOL_CHOICE
        assert ModelOption.TOOL_CHOICE in model_options
        assert model_options[ModelOption.TOOL_CHOICE] == "auto"
        # Legacy function calling parameters should still be excluded
        assert "functions" not in model_options
        assert "function_call" not in model_options

    @pytest.mark.asyncio
    async def test_response_format_excluded_from_model_options(self, mock_module):
        """Test that response_format parameter is excluded from model_options."""

        request = ChatCompletionRequest(
            model="test-model",
            messages=[ChatMessage(role="user", content="Hello")],
            response_format=ResponseFormat(type="json_object"),
        )

        mock_output = ModelOutputThunk("Test response")
        mock_module.serve.return_value = mock_output

        endpoint = make_chat_endpoint(mock_module)
        response = await endpoint(request)

        # Should succeed
        assert isinstance(response, ChatCompletion)

        # Verify serve was called
        call_args = mock_module.serve.call_args
        assert call_args is not None
        model_options = call_args.kwargs["model_options"]

        # response_format should NOT be in model_options
        assert "response_format" not in model_options

    @pytest.mark.asyncio
    async def test_streaming_with_tool_calls(self, mock_module):
        """Test that tool calls are properly emitted in streaming responses."""

        # Create a mock tool
        mock_tool = Mock()
        mock_tool.name = "get_weather"

        # Create a mock output with tool calls
        # Real backends may return content alongside tool calls (e.g., "I'll check that for you")
        mock_output = ModelOutputThunk("I'll check the weather for you.")
        mock_output.tool_calls = {
            "get_weather": ModelToolCall(
                name="get_weather",
                func=mock_tool,
                args={"location": "San Francisco", "units": "celsius"},
            )
        }
        mock_module.serve.return_value = mock_output

        request = ChatCompletionRequest(
            model="test-model",
            messages=[ChatMessage(role="user", content="What's the weather?")],
            stream=True,
        )

        endpoint = make_chat_endpoint(mock_module)
        response = await endpoint(request)

        # Verify it's a streaming response
        assert isinstance(response, StreamingResponse)

        # Collect all chunks
        chunks = []
        async for chunk_data in response.body_iterator:
            # Convert to string for parsing
            if isinstance(chunk_data, (bytes, memoryview)):
                chunk_str = (
                    bytes(chunk_data).decode("utf-8")
                    if isinstance(chunk_data, memoryview)
                    else chunk_data.decode("utf-8")
                )
            else:
                chunk_str = chunk_data

            # Parse SSE format: "data: {json}\n\n"
            if chunk_str.startswith("data: "):
                json_str = chunk_str[6:].strip()
                if json_str and json_str != "[DONE]":
                    chunks.append(json.loads(json_str))

        # Verify we have the expected chunk sequence
        # Expected: initial (role), content, tool_calls, final = 4 chunks
        assert len(chunks) == 4, f"Should have exactly 4 chunks, got {len(chunks)}"

        # Chunk 0: Initial chunk with role
        initial_chunk = chunks[0]
        assert initial_chunk["choices"][0]["delta"].get("role") == "assistant"
        assert initial_chunk["choices"][0]["finish_reason"] is None

        # Chunk 1: Content chunk
        content_chunk = chunks[1]
        assert (
            content_chunk["choices"][0]["delta"].get("content")
            == "I'll check the weather for you."
        )
        assert content_chunk["choices"][0]["finish_reason"] is None

        # Chunk 2: Tool call chunk
        tool_call_chunk = chunks[2]
        tool_calls = tool_call_chunk["choices"][0]["delta"]["tool_calls"]
        assert len(tool_calls) == 1
        # Verify required index field is present (OpenAI streaming spec requirement)
        assert "index" in tool_calls[0], "tool_calls delta must include index field"
        assert tool_calls[0]["index"] == 0
        assert tool_calls[0]["function"]["name"] == "get_weather"
        assert "location" in tool_calls[0]["function"]["arguments"]
        assert tool_call_chunk["choices"][0]["finish_reason"] is None

        # Chunk 3: Final chunk has finish_reason="tool_calls"
        final_chunk = chunks[3]
        assert final_chunk["choices"][0]["delta"].get("content") is None
        assert final_chunk["choices"][0]["finish_reason"] == "tool_calls"


class TestResponseFormat:
    """Tests for response_format parameter handling."""

    @pytest.mark.asyncio
    async def test_json_schema_format_passed_to_serve(self):
        """Test that json_schema response_format is converted to Pydantic model and passed to serve."""

        # Create a mock module with serve that accepts format parameter
        mock_module = Mock()
        mock_module.__name__ = "test_module"

        # Track calls manually
        captured_format = None

        def mock_serve(input, requirements=None, model_options=None, format=None):
            nonlocal captured_format
            captured_format = format
            return ModelOutputThunk('{"name": "Alice", "age": 30}')

        # Assign the real function so signature inspection works
        mock_module.serve = mock_serve

        # Create a request with json_schema response_format
        request = ChatCompletionRequest(
            model="test-model",
            messages=[ChatMessage(role="user", content="Generate a person")],
            response_format=ResponseFormat(
                type="json_schema",
                json_schema=JsonSchemaFormat(
                    name="Person",
                    schema={
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "age": {"type": "integer"},
                        },
                        "required": ["name", "age"],
                    },
                ),
            ),
        )

        endpoint = make_chat_endpoint(mock_module)
        response = await endpoint(request)

        # Verify format was passed
        assert captured_format is not None
        assert issubclass(captured_format, BaseModel)
        assert "name" in captured_format.model_fields
        assert "age" in captured_format.model_fields

        # Verify response is successful
        assert isinstance(response, ChatCompletion)
        assert response.choices[0].message.content == '{"name": "Alice", "age": 30}'

    @pytest.mark.asyncio
    async def test_json_object_format_no_schema(self, mock_module):
        """Test that json_object response_format doesn't pass a format model."""

        request = ChatCompletionRequest(
            model="test-model",
            messages=[ChatMessage(role="user", content="Generate JSON")],
            response_format=ResponseFormat(type="json_object"),
        )

        mock_output = ModelOutputThunk('{"result": "success"}')
        mock_module.serve.return_value = mock_output

        endpoint = make_chat_endpoint(mock_module)
        response = await endpoint(request)

        # Verify serve was called
        call_args = mock_module.serve.call_args
        assert call_args is not None

        # For json_object, format should be None (no specific schema)
        if "format" in call_args.kwargs:
            assert call_args.kwargs["format"] is None

        # Verify response is successful
        assert isinstance(response, ChatCompletion)

    @pytest.mark.asyncio
    async def test_text_format_no_schema(self, mock_module):
        """Test that text response_format doesn't pass a format model."""

        request = ChatCompletionRequest(
            model="test-model",
            messages=[ChatMessage(role="user", content="Hello")],
            response_format=ResponseFormat(type="text"),
        )

        mock_output = ModelOutputThunk("Hello there!")
        mock_module.serve.return_value = mock_output

        endpoint = make_chat_endpoint(mock_module)
        response = await endpoint(request)

        # Verify serve was called
        call_args = mock_module.serve.call_args
        assert call_args is not None

        # For text, format should be None
        if "format" in call_args.kwargs:
            assert call_args.kwargs["format"] is None

        # Verify response is successful
        assert isinstance(response, ChatCompletion)

    @pytest.mark.asyncio
    async def test_json_schema_missing_schema_field(self, mock_module):
        """Test that json_schema without schema field returns error."""

        request = ChatCompletionRequest(
            model="test-model",
            messages=[ChatMessage(role="user", content="Generate")],
            response_format=ResponseFormat(
                type="json_schema",
                json_schema=None,  # Missing schema
            ),
        )

        endpoint = make_chat_endpoint(mock_module)
        response = await endpoint(request)

        # Should return error
        assert isinstance(response, JSONResponse)
        assert response.status_code == 400

        body_bytes = response.body
        if isinstance(body_bytes, memoryview):
            body_bytes = bytes(body_bytes)
        error_data = json.loads(body_bytes.decode("utf-8"))
        assert "error" in error_data
        assert error_data["error"]["type"] == "invalid_request_error"
        assert "json_schema" in error_data["error"]["message"].lower()

    @pytest.mark.asyncio
    async def test_json_schema_invalid_schema(self, mock_module):
        """Test that invalid JSON schema returns error."""

        request = ChatCompletionRequest(
            model="test-model",
            messages=[ChatMessage(role="user", content="Generate")],
            response_format=ResponseFormat(
                type="json_schema",
                json_schema=JsonSchemaFormat(
                    name="Invalid",
                    schema={
                        "type": "array",  # Not supported (only object)
                        "items": {"type": "string"},
                    },
                ),
            ),
        )

        endpoint = make_chat_endpoint(mock_module)
        response = await endpoint(request)

        # Should return error
        assert isinstance(response, JSONResponse)
        assert response.status_code == 400

        body_bytes = response.body
        if isinstance(body_bytes, memoryview):
            body_bytes = bytes(body_bytes)
        error_data = json.loads(body_bytes.decode("utf-8"))
        assert "error" in error_data
        assert error_data["error"]["type"] == "invalid_request_error"
        assert "schema" in error_data["error"]["message"].lower()

    @pytest.mark.asyncio
    async def test_serve_without_format_parameter(self, mock_module):
        """Test that serve functions without format parameter still work."""

        # Create a serve function that doesn't accept format
        def serve_no_format(input, requirements=None, model_options=None):
            return ModelOutputThunk("Response without format")

        mock_module.serve = serve_no_format

        request = ChatCompletionRequest(
            model="test-model",
            messages=[ChatMessage(role="user", content="Hello")],
            response_format=ResponseFormat(
                type="json_schema",
                json_schema=JsonSchemaFormat(
                    name="Test",
                    schema={
                        "type": "object",
                        "properties": {"result": {"type": "string"}},
                        "required": ["result"],
                    },
                ),
            ),
        )

        endpoint = make_chat_endpoint(mock_module)
        response = await endpoint(request)

        # Should succeed even though serve doesn't accept format
        assert isinstance(response, ChatCompletion)
        assert response.choices[0].message.content == "Response without format"

    @pytest.mark.asyncio
    async def test_json_schema_with_optional_fields(self):
        """Test that JSON schema with optional fields is handled correctly."""

        # Create a mock module with serve that accepts format parameter
        mock_module = Mock()
        mock_module.__name__ = "test_module"

        # Track calls manually
        captured_format = None

        def mock_serve(input, requirements=None, model_options=None, format=None):
            nonlocal captured_format
            captured_format = format
            return ModelOutputThunk('{"name": "Widget", "price": 9.99}')

        # Assign the real function so signature inspection works
        mock_module.serve = mock_serve

        request = ChatCompletionRequest(
            model="test-model",
            messages=[ChatMessage(role="user", content="Generate")],
            response_format=ResponseFormat(
                type="json_schema",
                json_schema=JsonSchemaFormat(
                    name="Product",
                    schema={
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "price": {"type": "number"},
                            "description": {"type": "string"},
                        },
                        "required": ["name", "price"],  # description is optional
                    },
                ),
            ),
        )

        endpoint = make_chat_endpoint(mock_module)
        response = await endpoint(request)

        # Verify format model was created correctly
        assert captured_format is not None
        assert issubclass(captured_format, BaseModel)
        assert "name" in captured_format.model_fields
        assert "price" in captured_format.model_fields
        assert "description" in captured_format.model_fields

        # Verify response is successful
        assert isinstance(response, ChatCompletion)

    @pytest.mark.asyncio
    async def test_json_schema_rejects_non_local_ref(self, mock_module):
        """Test that non-local refs still return a request error."""

        request = ChatCompletionRequest(
            model="test-model",
            messages=[ChatMessage(role="user", content="Generate")],
            response_format=ResponseFormat(
                type="json_schema",
                json_schema=JsonSchemaFormat(
                    name="RemoteRefExample",
                    schema={
                        "type": "object",
                        "properties": {
                            "value": {"$ref": "https://example.com/schemas/value.json"}
                        },
                        "required": ["value"],
                    },
                ),
            ),
        )

        endpoint = make_chat_endpoint(mock_module)
        response = await endpoint(request)

        assert isinstance(response, JSONResponse)
        assert response.status_code == 400

        body_bytes = response.body
        if isinstance(body_bytes, memoryview):
            body_bytes = bytes(body_bytes)
        error_data = json.loads(body_bytes.decode("utf-8"))
        assert error_data["error"]["type"] == "invalid_request_error"
        assert "local" in error_data["error"]["message"].lower()
        assert "$ref" in error_data["error"]["message"].lower()


class TestResponseFormatStreaming:
    """Tests for response_format parameter with streaming enabled."""

    @pytest.mark.asyncio
    async def test_json_schema_format_with_streaming(self):
        """Test that json_schema response_format works with stream=True."""

        # Create a mock module with serve that accepts format parameter
        mock_module = Mock()
        mock_module.__name__ = "test_module"

        # Create a mock output that supports streaming
        mock_output = ModelOutputThunk('{"name": "Alice", "age": 30}')
        mock_output._computed = True  # Mark as pre-computed

        def mock_serve(input, requirements=None, model_options=None, format=None):
            return mock_output

        mock_module.serve = mock_serve

        # Create a request with json_schema response_format and streaming
        request = ChatCompletionRequest(
            model="test-model",
            messages=[ChatMessage(role="user", content="Generate a person")],
            stream=True,
            response_format=ResponseFormat(
                type="json_schema",
                json_schema=JsonSchemaFormat(
                    name="Person",
                    schema={
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "age": {"type": "integer"},
                        },
                        "required": ["name", "age"],
                    },
                ),
            ),
        )

        endpoint = make_chat_endpoint(mock_module)
        response = await endpoint(request)

        # Verify it's a streaming response
        assert isinstance(response, StreamingResponse)

        # Consume the stream and verify chunks
        chunks = []
        async for chunk in response.body_iterator:
            chunks.append(chunk)

        # Should have multiple chunks including initial, content, final, and [DONE]
        assert len(chunks) > 0

        # Verify no error chunks (all should start with "data: ")
        for chunk in chunks:
            chunk_str = chunk.decode("utf-8") if isinstance(chunk, bytes) else chunk
            assert chunk_str.startswith("data: ")

    @pytest.mark.asyncio
    async def test_json_object_format_with_streaming(self):
        """Test that json_object response_format works with stream=True."""

        mock_module = Mock()
        mock_module.__name__ = "test_module"

        # Valid JSON output
        mock_output = ModelOutputThunk('{"result": "success"}')
        mock_output._computed = True
        mock_module.serve.return_value = mock_output

        request = ChatCompletionRequest(
            model="test-model",
            messages=[ChatMessage(role="user", content="Generate JSON")],
            stream=True,
            response_format=ResponseFormat(type="json_object"),
        )

        endpoint = make_chat_endpoint(mock_module)
        response = await endpoint(request)

        assert isinstance(response, StreamingResponse)

        # Consume the stream
        chunks = []
        async for chunk in response.body_iterator:
            chunks.append(chunk)

        # Should complete successfully without errors
        assert len(chunks) > 0
        # Verify no error chunks
        for chunk in chunks:
            chunk_str = chunk.decode("utf-8") if isinstance(chunk, bytes) else chunk
            assert "error" not in chunk_str.lower() or chunk_str.startswith(
                "data: [DONE]"
            )

    @pytest.mark.asyncio
    async def test_text_format_with_streaming(self):
        """Test that text response_format works with stream=True."""

        mock_module = Mock()
        mock_module.__name__ = "test_module"

        mock_output = ModelOutputThunk("Plain text response")
        mock_output._computed = True
        mock_module.serve.return_value = mock_output

        request = ChatCompletionRequest(
            model="test-model",
            messages=[ChatMessage(role="user", content="Hello")],
            stream=True,
            response_format=ResponseFormat(type="text"),
        )

        endpoint = make_chat_endpoint(mock_module)
        response = await endpoint(request)

        assert isinstance(response, StreamingResponse)

        # Consume the stream
        chunks = []
        async for chunk in response.body_iterator:
            chunks.append(chunk)

        # Should complete successfully
        assert len(chunks) > 0
