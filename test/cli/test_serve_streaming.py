"""Tests for streaming support in the m serve OpenAI-compatible API server."""

import json
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from cli.serve.app import make_chat_endpoint
from cli.serve.models import ChatCompletionRequest, ChatMessage
from cli.serve.streaming import stream_chat_completion_chunks
from mellea.core.base import ModelOutputThunk
from mellea.helpers.openai_compatible_helpers import build_completion_usage


@pytest.fixture
def mock_module():
    """Create a mock module with an async serve function."""
    module = Mock()
    module.__name__ = "test_streaming_module"
    module.serve = AsyncMock()
    return module


@pytest.fixture
def streaming_request():
    """Create a sample streaming ChatCompletionRequest."""
    return ChatCompletionRequest(
        model="test-model",
        messages=[ChatMessage(role="user", content="Hello, world!")],
        stream=True,
        temperature=0.7,
    )


@pytest.fixture
def non_streaming_request():
    """Create a sample non-streaming ChatCompletionRequest."""
    return ChatCompletionRequest(
        model="test-model",
        messages=[ChatMessage(role="user", content="Hello, world!")],
        stream=False,
        temperature=0.7,
    )


class TestCompletionUsageHelpers:
    """Tests for completion usage normalization helpers."""

    def test_build_completion_usage_with_full_usage(self):
        """Test usage normalization with complete usage data."""
        output = ModelOutputThunk("done")
        output.usage = {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8}

        usage = build_completion_usage(output)

        assert usage is not None
        assert usage.prompt_tokens == 5
        assert usage.completion_tokens == 3
        assert usage.total_tokens == 8

    def test_build_completion_usage_with_partial_usage(self):
        """Test usage normalization fills missing values safely."""
        output = ModelOutputThunk("done")
        output.usage = {"prompt_tokens": 5}

        usage = build_completion_usage(output)

        assert usage is not None
        assert usage.prompt_tokens == 5
        assert usage.completion_tokens == 0
        assert usage.total_tokens == 5

    def test_build_completion_usage_without_usage(self):
        """Test usage normalization returns None when usage is unavailable."""
        output = ModelOutputThunk("done")

        assert build_completion_usage(output) is None


class TestStreamingHelpers:
    """Tests for reusable streaming helper functions."""

    @pytest.mark.asyncio
    async def test_stream_chat_completion_chunks_emits_incremental_content(self):
        """Test helper emits only incremental content fragments."""
        output = ModelOutputThunk(None)
        output._computed = False
        output._generate_type = output._generate_type.ASYNC

        chunks = ["Hello", " there", "!"]

        async def mock_astream():
            if chunks:
                value = chunks.pop(0)
                if not chunks:
                    output._computed = True
                return value
            output._computed = True
            return ""

        output.astream = mock_astream
        output.is_computed = lambda: output._computed
        output.usage = {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8}

        events = []
        async for event in stream_chat_completion_chunks(
            output=output,
            completion_id="chatcmpl-test123",
            model="test-model",
            created=123,
            stream_options={"include_usage": True},
        ):
            events.append(event)

        assert events[0].startswith("data: ")
        assert events[-1] == "data: [DONE]\n\n"

        parsed = []
        for event in events:
            if event.startswith("data: ") and event != "data: [DONE]\n\n":
                parsed.append(json.loads(event[6:].strip()))

        assert parsed[0]["choices"][0]["delta"]["role"] == "assistant"
        content_chunks = [
            chunk["choices"][0]["delta"].get("content")
            for chunk in parsed[1:-1]
            if chunk["choices"][0]["delta"].get("content")
        ]
        assert content_chunks == ["Hello", " there", "!"]
        assert parsed[-1]["choices"][0]["finish_reason"] == "stop"
        assert parsed[-1]["usage"]["total_tokens"] == 8

    @pytest.mark.asyncio
    async def test_stream_chat_completion_chunks_emits_error_event(self):
        """Test helper emits an error payload and [DONE] when streaming fails."""
        output = ModelOutputThunk(None)
        output._computed = False
        output._generate_type = output._generate_type.ASYNC

        async def mock_astream():
            raise RuntimeError("boom")

        output.astream = mock_astream
        output.is_computed = lambda: output._computed

        events = []
        async for event in stream_chat_completion_chunks(
            output=output,
            completion_id="chatcmpl-test123",
            model="test-model",
            created=123,
        ):
            events.append(event)

        # Should emit: initial chunk, error payload, [DONE]
        assert len(events) == 3
        # First event is initial chunk with role
        initial_chunk = json.loads(events[0][6:].strip())
        assert initial_chunk["choices"][0]["delta"]["role"] == "assistant"
        # Second event is error payload
        error_payload = json.loads(events[1][6:].strip())
        assert error_payload["error"]["type"] == "server_error"
        assert "boom" in error_payload["error"]["message"]
        # Third event is [DONE] sentinel
        assert events[2] == "data: [DONE]\n\n"


class TestStreamingEndpoint:
    """Tests for streaming chat completion endpoint."""

    def test_streaming_response_format(self, mock_module, streaming_request):
        """Test that streaming returns SSE format with proper chunks."""
        # Create a mock output that simulates streaming
        mock_output = ModelOutputThunk(None)
        mock_output._computed = False
        mock_output._generate_type = mock_output._generate_type.ASYNC

        # Simulate streaming chunks (deltas, not accumulated)
        chunks = ["Hello", " there", "!"]

        async def mock_astream():
            if chunks:
                delta = chunks.pop(0)
                if not chunks:
                    mock_output._computed = True
                return delta
            mock_output._computed = True
            return ""

        mock_output.astream = mock_astream
        mock_output.is_computed = lambda: mock_output._computed
        mock_output.usage = {
            "prompt_tokens": 5,
            "completion_tokens": 3,
            "total_tokens": 8,
        }

        mock_module.serve.return_value = mock_output

        # Create test app
        app = FastAPI()
        app.add_api_route(
            "/v1/chat/completions", make_chat_endpoint(mock_module), methods=["POST"]
        )
        client = TestClient(app)

        # Make streaming request
        response = client.post(
            "/v1/chat/completions", json=streaming_request.model_dump(mode="json")
        )

        assert response.status_code == 200
        assert response.headers["content-type"] == "text/event-stream; charset=utf-8"

        # Parse SSE events
        events = []
        for line in response.text.strip().split("\n\n"):
            if line.startswith("data: "):
                data = line[6:]  # Remove "data: " prefix
                if data != "[DONE]":
                    events.append(json.loads(data))

        # Verify we got multiple chunks
        assert len(events) > 0

        # First chunk should have role
        first_chunk = events[0]
        assert first_chunk["object"] == "chat.completion.chunk"
        assert first_chunk["choices"][0]["delta"]["role"] == "assistant"

        # Last chunk should have finish_reason but no usage (not requested)
        last_chunk = events[-1]
        assert last_chunk["choices"][0]["finish_reason"] == "stop"
        assert last_chunk["usage"] is None

    def test_non_streaming_still_works(self, mock_module, non_streaming_request):
        """Test that non-streaming requests still work correctly."""
        mock_output = ModelOutputThunk("Complete response")
        mock_output.usage = {
            "prompt_tokens": 5,
            "completion_tokens": 2,
            "total_tokens": 7,
        }
        mock_module.serve.return_value = mock_output

        # Create test app
        app = FastAPI()
        app.add_api_route(
            "/v1/chat/completions", make_chat_endpoint(mock_module), methods=["POST"]
        )
        client = TestClient(app)

        # Make non-streaming request
        response = client.post(
            "/v1/chat/completions", json=non_streaming_request.model_dump(mode="json")
        )

        assert response.status_code == 200
        data = response.json()

        # Should be a regular ChatCompletion, not streaming
        assert data["object"] == "chat.completion"
        assert data["choices"][0]["message"]["content"] == "Complete response"
        assert data["choices"][0]["finish_reason"] == "stop"
        assert data["usage"]["total_tokens"] == 7

    def test_stream_parameter_passed_to_model_options(
        self, mock_module, streaming_request
    ):
        """Test that stream parameter is passed to model_options."""
        from mellea.backends.model_options import ModelOption

        mock_output = ModelOutputThunk(None)
        mock_output._computed = False
        mock_output._generate_type = mock_output._generate_type.ASYNC
        mock_output.astream = AsyncMock(
            side_effect=lambda: setattr(mock_output, "_computed", True) or "done"
        )
        mock_output.is_computed = lambda: mock_output._computed
        mock_module.serve.return_value = mock_output

        # Create test app
        app = FastAPI()
        app.add_api_route(
            "/v1/chat/completions", make_chat_endpoint(mock_module), methods=["POST"]
        )
        client = TestClient(app)

        # Make streaming request
        client.post(
            "/v1/chat/completions", json=streaming_request.model_dump(mode="json")
        )

        # Verify serve was called with stream in model_options
        call_args = mock_module.serve.call_args
        assert call_args is not None
        model_options = call_args.kwargs["model_options"]
        assert ModelOption.STREAM in model_options
        assert model_options[ModelOption.STREAM] is True

    def test_streaming_with_empty_content(self, mock_module, streaming_request):
        """Test streaming handles empty content chunks gracefully."""
        mock_output = ModelOutputThunk(None)
        mock_output._computed = False
        mock_output._generate_type = mock_output._generate_type.ASYNC

        # Simulate streaming with some empty incremental chunks
        chunks = ["", "Hello", "", " world", ""]

        async def mock_astream():
            if chunks:
                chunk = chunks.pop(0)
                if not chunks:
                    mock_output._computed = True
                return chunk
            mock_output._computed = True
            return ""

        mock_output.astream = mock_astream
        mock_output.is_computed = lambda: mock_output._computed
        mock_output.usage = None

        mock_module.serve.return_value = mock_output

        # Create test app
        app = FastAPI()
        app.add_api_route(
            "/v1/chat/completions", make_chat_endpoint(mock_module), methods=["POST"]
        )
        client = TestClient(app)

        # Make streaming request
        response = client.post(
            "/v1/chat/completions", json=streaming_request.model_dump(mode="json")
        )

        assert response.status_code == 200

        # Parse events and verify we only get chunks with actual content
        events = []
        for line in response.text.strip().split("\n\n"):
            if line.startswith("data: "):
                data = line[6:]
                if data != "[DONE]":
                    chunk = json.loads(data)
                    events.append(chunk)

        # Count chunks with actual content (excluding initial role chunk and final finish chunk)
        content_chunks = [e for e in events if e["choices"][0]["delta"].get("content")]
        assert len(content_chunks) == 2  # "Hello" and " world"

    def test_streaming_completion_id_consistent(self, mock_module, streaming_request):
        """Test that completion ID is consistent across all chunks."""
        mock_output = ModelOutputThunk(None)
        mock_output._computed = False
        mock_output._generate_type = mock_output._generate_type.ASYNC

        chunks = ["A", "B"]

        async def mock_astream():
            if chunks:
                delta = chunks.pop(0)
                if not chunks:
                    mock_output._computed = True
                return delta
            mock_output._computed = True
            return ""

        mock_output.astream = mock_astream
        mock_output.is_computed = lambda: mock_output._computed
        mock_module.serve.return_value = mock_output

        # Create test app
        app = FastAPI()
        app.add_api_route(
            "/v1/chat/completions", make_chat_endpoint(mock_module), methods=["POST"]
        )
        client = TestClient(app)

        # Make streaming request
        response = client.post(
            "/v1/chat/completions", json=streaming_request.model_dump(mode="json")
        )

        # Parse events
        events = []
        for line in response.text.strip().split("\n\n"):
            if line.startswith("data: "):
                data = line[6:]
                if data != "[DONE]":
                    events.append(json.loads(data))

        # All chunks should have the same ID
        ids = [e["id"] for e in events]
        assert len(set(ids)) == 1  # All IDs are the same
        assert ids[0].startswith("chatcmpl-")

    def test_streaming_ends_with_done(self, mock_module, streaming_request):
        """Test that streaming response ends with [DONE] marker."""
        mock_output = ModelOutputThunk(None)
        mock_output._computed = False
        mock_output._generate_type = mock_output._generate_type.ASYNC

        async def mock_astream():
            mock_output._computed = True
            return "done"

        mock_output.astream = mock_astream
        mock_output.is_computed = lambda: mock_output._computed
        mock_module.serve.return_value = mock_output

        # Create test app
        app = FastAPI()
        app.add_api_route(
            "/v1/chat/completions", make_chat_endpoint(mock_module), methods=["POST"]
        )
        client = TestClient(app)

        # Make streaming request
        response = client.post(
            "/v1/chat/completions", json=streaming_request.model_dump(mode="json")
        )

        # Verify response ends with [DONE]
        assert response.text.strip().endswith("data: [DONE]")

    def test_streaming_model_field_correct(self, mock_module, streaming_request):
        """Test that model field is correctly set in streaming chunks."""
        mock_output = ModelOutputThunk(None)
        mock_output._computed = False
        mock_output._generate_type = mock_output._generate_type.ASYNC

        async def mock_astream():
            mock_output._computed = True
            return "test"

        mock_output.astream = mock_astream
        mock_output.is_computed = lambda: mock_output._computed
        mock_module.serve.return_value = mock_output

        # Create test app
        app = FastAPI()
        app.add_api_route(
            "/v1/chat/completions", make_chat_endpoint(mock_module), methods=["POST"]
        )
        client = TestClient(app)

        # Make streaming request
        response = client.post(
            "/v1/chat/completions", json=streaming_request.model_dump(mode="json")
        )

        # Parse first chunk
        first_line = response.text.split("\n\n")[0]
        first_chunk = json.loads(first_line[6:])  # Remove "data: "

        # Model should match request
        assert first_chunk["model"] == "test-model"

    def test_stream_options_include_usage_true(self, mock_module):
        """Test that stream_options with include_usage=true includes usage in final chunk."""
        request = ChatCompletionRequest(
            model="test-model",
            messages=[ChatMessage(role="user", content="Hello")],
            stream=True,
            stream_options={"include_usage": True},
        )

        mock_output = ModelOutputThunk(None)
        mock_output._computed = False
        mock_output._generate_type = mock_output._generate_type.ASYNC

        async def mock_astream():
            mock_output._computed = True
            return "response"

        mock_output.astream = mock_astream
        mock_output.is_computed = lambda: mock_output._computed
        mock_output.usage = {
            "prompt_tokens": 5,
            "completion_tokens": 3,
            "total_tokens": 8,
        }
        mock_module.serve.return_value = mock_output

        # Create test app
        app = FastAPI()
        app.add_api_route(
            "/v1/chat/completions", make_chat_endpoint(mock_module), methods=["POST"]
        )
        client = TestClient(app)

        # Make streaming request
        response = client.post(
            "/v1/chat/completions", json=request.model_dump(mode="json")
        )

        assert response.status_code == 200

        # Parse events
        events = []
        for line in response.text.strip().split("\n\n"):
            if line.startswith("data: "):
                data = line[6:]
                if data != "[DONE]":
                    events.append(json.loads(data))

        # Last chunk should have usage
        last_chunk = events[-1]
        assert last_chunk["usage"] is not None
        assert last_chunk["usage"]["total_tokens"] == 8

    def test_stream_options_include_usage_false(self, mock_module):
        """Test that stream_options with include_usage=false excludes usage from final chunk."""
        request = ChatCompletionRequest(
            model="test-model",
            messages=[ChatMessage(role="user", content="Hello")],
            stream=True,
            stream_options={"include_usage": False},
        )

        mock_output = ModelOutputThunk(None)
        mock_output._computed = False
        mock_output._generate_type = mock_output._generate_type.ASYNC

        async def mock_astream():
            mock_output._computed = True
            return "response"

        mock_output.astream = mock_astream
        mock_output.is_computed = lambda: mock_output._computed
        mock_output.usage = {
            "prompt_tokens": 5,
            "completion_tokens": 3,
            "total_tokens": 8,
        }
        mock_module.serve.return_value = mock_output

        # Create test app
        app = FastAPI()
        app.add_api_route(
            "/v1/chat/completions", make_chat_endpoint(mock_module), methods=["POST"]
        )
        client = TestClient(app)

        # Make streaming request
        response = client.post(
            "/v1/chat/completions", json=request.model_dump(mode="json")
        )

        assert response.status_code == 200

        # Parse events
        events = []
        for line in response.text.strip().split("\n\n"):
            if line.startswith("data: "):
                data = line[6:]
                if data != "[DONE]":
                    events.append(json.loads(data))

        # Last chunk should NOT have usage
        last_chunk = events[-1]
        assert last_chunk["usage"] is None

    def test_stream_options_default_excludes_usage(self, mock_module):
        """Test that without stream_options, usage is NOT included (per OpenAI spec)."""
        request = ChatCompletionRequest(
            model="test-model",
            messages=[ChatMessage(role="user", content="Hello")],
            stream=True,
            # No stream_options specified
        )

        mock_output = ModelOutputThunk(None)
        mock_output._computed = False
        mock_output._generate_type = mock_output._generate_type.ASYNC

        async def mock_astream():
            mock_output._computed = True
            return "response"

        mock_output.astream = mock_astream
        mock_output.is_computed = lambda: mock_output._computed
        mock_output.usage = {
            "prompt_tokens": 5,
            "completion_tokens": 3,
            "total_tokens": 8,
        }
        mock_module.serve.return_value = mock_output

        # Create test app
        app = FastAPI()
        app.add_api_route(
            "/v1/chat/completions", make_chat_endpoint(mock_module), methods=["POST"]
        )
        client = TestClient(app)

        # Make streaming request
        response = client.post(
            "/v1/chat/completions", json=request.model_dump(mode="json")
        )

        assert response.status_code == 200

        # Parse events
        events = []
        for line in response.text.strip().split("\n\n"):
            if line.startswith("data: "):
                data = line[6:]
                if data != "[DONE]":
                    events.append(json.loads(data))

        # Last chunk should NOT have usage (must explicitly request via stream_options)
        last_chunk = events[-1]
        assert last_chunk["usage"] is None

    def test_streaming_system_fingerprint_always_none(
        self, mock_module, streaming_request
    ):
        """Test that system_fingerprint is None in all streaming chunks.

        Per OpenAI spec, system_fingerprint represents a hash of backend config,
        not the model name. The model name is in chunk.model.
        We don't track backend config fingerprints yet, so it should be None.
        """
        mock_output = ModelOutputThunk(None)
        mock_output._computed = False
        mock_output._generate_type = mock_output._generate_type.ASYNC

        chunks = ["Hello", " world"]

        async def mock_astream():
            if chunks:
                chunk = chunks.pop(0)
                if not chunks:
                    mock_output._computed = True
                return chunk
            mock_output._computed = True
            return ""

        mock_output.astream = mock_astream
        mock_output.is_computed = lambda: mock_output._computed
        # Usage not needed for this test since we're not checking it
        mock_module.serve.return_value = mock_output

        # Create test app
        app = FastAPI()
        app.add_api_route(
            "/v1/chat/completions", make_chat_endpoint(mock_module), methods=["POST"]
        )
        client = TestClient(app)

        # Make streaming request
        response = client.post(
            "/v1/chat/completions", json=streaming_request.model_dump(mode="json")
        )

        assert response.status_code == 200

        # Parse all chunks
        events = []
        for line in response.text.strip().split("\n\n"):
            if line.startswith("data: "):
                data = line[6:]
                if data != "[DONE]":
                    events.append(json.loads(data))

        # All chunks should have system_fingerprint as None
        for chunk in events:
            assert chunk["system_fingerprint"] is None
            # Model name should be in the model field
            assert chunk["model"] == "test-model"

    def test_stream_options_ignored_for_non_streaming(self, mock_module):
        """Test that stream_options is ignored when stream=False (usage always included)."""
        request = ChatCompletionRequest(
            model="test-model",
            messages=[ChatMessage(role="user", content="Hello")],
            stream=False,
            stream_options={"include_usage": False},  # Should be ignored
        )

        mock_output = ModelOutputThunk("Complete response")
        mock_output.usage = {
            "prompt_tokens": 5,
            "completion_tokens": 3,
            "total_tokens": 8,
        }
        mock_module.serve.return_value = mock_output

        # Create test app
        app = FastAPI()
        app.add_api_route(
            "/v1/chat/completions", make_chat_endpoint(mock_module), methods=["POST"]
        )
        client = TestClient(app)

        # Make non-streaming request
        response = client.post(
            "/v1/chat/completions", json=request.model_dump(mode="json")
        )

        assert response.status_code == 200
        data = response.json()

        # Usage should be included regardless of stream_options (non-streaming always includes usage)
        assert data["usage"] is not None
        assert data["usage"]["total_tokens"] == 8
