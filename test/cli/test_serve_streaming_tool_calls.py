"""Unit tests for streaming with tool calls, usage fields, and error handling.

This file contains new tests added in the tool-calling PR. The main streaming
tests (from main branch) are in test_serve_streaming.py.
"""

import json
from unittest.mock import AsyncMock, Mock

import pytest

from cli.serve.models import StreamOptions
from cli.serve.streaming import stream_chat_completion_chunks
from mellea.core.base import ModelOutputThunk, ModelToolCall


class TestStreamingToolCalls:
    """Tests for streaming responses with tool calls."""

    @pytest.mark.asyncio
    async def test_streaming_tool_call_chunk_structure(self):
        """Test that tool call chunks have correct structure with index field."""
        # Create a mock tool
        mock_tool = Mock()
        mock_tool.name = "get_weather"

        # Create output with tool calls
        output = ModelOutputThunk("Checking weather...")
        output.tool_calls = {
            "get_weather": ModelToolCall(
                name="get_weather",
                func=mock_tool,
                args={"location": "San Francisco", "units": "celsius"},
            )
        }

        # Stream chunks
        chunks = []
        async for chunk_data in stream_chat_completion_chunks(
            output=output,
            completion_id="chatcmpl-test123",
            model="test-model",
            created=1234567890,
        ):
            if chunk_data.startswith("data: ") and chunk_data.strip() != "data: [DONE]":
                json_str = chunk_data[6:].strip()
                chunks.append(json.loads(json_str))

        # Should have: initial (role), content, tool_calls, final = 4 chunks
        assert len(chunks) == 4

        # Verify tool call chunk structure
        tool_call_chunk = chunks[2]
        tool_calls = tool_call_chunk["choices"][0]["delta"]["tool_calls"]
        assert len(tool_calls) == 1

        # Critical: index field must be present (OpenAI streaming spec)
        assert "index" in tool_calls[0], "tool_calls delta must include index field"
        assert tool_calls[0]["index"] == 0
        assert tool_calls[0]["id"] is not None
        assert tool_calls[0]["type"] == "function"
        assert tool_calls[0]["function"]["name"] == "get_weather"
        assert "location" in tool_calls[0]["function"]["arguments"]

    @pytest.mark.asyncio
    async def test_finish_reason_tool_calls(self):
        """Test that finish_reason is 'tool_calls' when tool calls are present."""
        mock_tool = Mock()
        mock_tool.name = "test_func"

        output = ModelOutputThunk("Response")
        output.tool_calls = {
            "test_func": ModelToolCall(
                name="test_func", func=mock_tool, args={"arg": "value"}
            )
        }

        chunks = []
        async for chunk_data in stream_chat_completion_chunks(
            output=output,
            completion_id="chatcmpl-test",
            model="test-model",
            created=1234567890,
        ):
            if chunk_data.startswith("data: ") and chunk_data.strip() != "data: [DONE]":
                json_str = chunk_data[6:].strip()
                chunks.append(json.loads(json_str))

        # Final chunk should have finish_reason="tool_calls"
        final_chunk = chunks[-1]
        assert final_chunk["choices"][0]["finish_reason"] == "tool_calls"

    @pytest.mark.asyncio
    async def test_finish_reason_stop_without_tool_calls(self):
        """Test that finish_reason is 'stop' when no tool calls are present."""
        output = ModelOutputThunk("Simple response")

        chunks = []
        async for chunk_data in stream_chat_completion_chunks(
            output=output,
            completion_id="chatcmpl-test",
            model="test-model",
            created=1234567890,
        ):
            if chunk_data.startswith("data: ") and chunk_data.strip() != "data: [DONE]":
                json_str = chunk_data[6:].strip()
                chunks.append(json.loads(json_str))

        # Final chunk should have finish_reason="stop"
        final_chunk = chunks[-1]
        assert final_chunk["choices"][0]["finish_reason"] == "stop"

    @pytest.mark.asyncio
    async def test_multiple_tool_calls_with_indices(self):
        """Test that multiple tool calls each get correct index values."""
        mock_tool1 = Mock()
        mock_tool1.name = "func1"
        mock_tool2 = Mock()
        mock_tool2.name = "func2"
        mock_tool3 = Mock()
        mock_tool3.name = "func3"

        output = ModelOutputThunk("Calling multiple functions")
        output.tool_calls = {
            "func1": ModelToolCall(name="func1", func=mock_tool1, args={"a": 1}),
            "func2": ModelToolCall(name="func2", func=mock_tool2, args={"b": 2}),
            "func3": ModelToolCall(name="func3", func=mock_tool3, args={"c": 3}),
        }

        chunks = []
        async for chunk_data in stream_chat_completion_chunks(
            output=output,
            completion_id="chatcmpl-test",
            model="test-model",
            created=1234567890,
        ):
            if chunk_data.startswith("data: ") and chunk_data.strip() != "data: [DONE]":
                json_str = chunk_data[6:].strip()
                chunks.append(json.loads(json_str))

        # Find tool call chunk
        tool_call_chunk = chunks[2]
        tool_calls = tool_call_chunk["choices"][0]["delta"]["tool_calls"]

        # Should have 3 tool calls with indices 0, 1, 2
        assert len(tool_calls) == 3
        indices = [tc["index"] for tc in tool_calls]
        assert indices == [0, 1, 2]

        # Verify each has required fields
        for tc in tool_calls:
            assert "index" in tc
            assert "id" in tc
            assert "type" in tc
            assert tc["type"] == "function"
            assert "function" in tc
            assert "name" in tc["function"]
            assert "arguments" in tc["function"]

    @pytest.mark.asyncio
    async def test_tool_call_chunk_before_final_chunk(self):
        """Test that tool call chunk is emitted before final chunk."""
        mock_tool = Mock()
        mock_tool.name = "test_func"

        output = ModelOutputThunk("Response")
        output.tool_calls = {
            "test_func": ModelToolCall(name="test_func", func=mock_tool, args={})
        }

        chunks = []
        async for chunk_data in stream_chat_completion_chunks(
            output=output,
            completion_id="chatcmpl-test",
            model="test-model",
            created=1234567890,
        ):
            if chunk_data.startswith("data: ") and chunk_data.strip() != "data: [DONE]":
                json_str = chunk_data[6:].strip()
                chunks.append(json.loads(json_str))

        # Verify chunk sequence
        assert len(chunks) == 4

        # Chunk 0: initial with role
        assert chunks[0]["choices"][0]["delta"].get("role") == "assistant"
        assert chunks[0]["choices"][0]["finish_reason"] is None

        # Chunk 1: content
        assert chunks[1]["choices"][0]["delta"].get("content") == "Response"
        assert chunks[1]["choices"][0]["finish_reason"] is None

        # Chunk 2: tool calls (before final)
        assert "tool_calls" in chunks[2]["choices"][0]["delta"]
        assert chunks[2]["choices"][0]["finish_reason"] is None

        # Chunk 3: final with finish_reason
        assert chunks[3]["choices"][0]["finish_reason"] == "tool_calls"


class TestStreamingIncrementalContent:
    """Tests for streaming with incremental content (not pre-computed)."""

    @pytest.mark.asyncio
    async def test_streaming_incremental_chunks(self):
        """Test streaming with incremental content via astream()."""
        from unittest.mock import patch

        # Create output that streams incrementally
        output = ModelOutputThunk("")

        # Mock astream to return incremental chunks
        chunks_to_stream = ["Hello", " ", "world", "!"]
        stream_index = 0

        async def mock_astream():
            nonlocal stream_index
            if stream_index < len(chunks_to_stream):
                chunk = chunks_to_stream[stream_index]
                stream_index += 1
                return chunk
            else:
                # Mark as computed by setting the value property
                output.value = "Hello world!"
                # Use object.__setattr__ to bypass property setter for _computed
                object.__setattr__(output, "_computed", True)
                return ""

        def mock_is_computed():
            return stream_index >= len(chunks_to_stream)

        # Patch the astream and is_computed methods
        with (
            patch.object(output, "astream", side_effect=mock_astream),
            patch.object(output, "is_computed", side_effect=mock_is_computed),
        ):
            # Collect streamed chunks
            collected_chunks = []
            async for chunk_data in stream_chat_completion_chunks(
                output=output,
                completion_id="chatcmpl-test",
                model="test-model",
                created=1234567890,
            ):
                if (
                    chunk_data.startswith("data: ")
                    and chunk_data.strip() != "data: [DONE]"
                ):
                    json_str = chunk_data[6:].strip()
                    parsed = json.loads(json_str)
                    delta_content = parsed["choices"][0]["delta"].get("content")
                    if delta_content:
                        collected_chunks.append(delta_content)

            # Should have initial role chunk + 4 content chunks
            # (role chunk has content=None, so not collected)
            assert collected_chunks == ["Hello", " ", "world", "!"]

    @pytest.mark.asyncio
    async def test_streaming_with_tool_calls_after_incremental_content(self):
        """Test that tool calls are emitted after incremental content streaming."""
        from unittest.mock import patch

        mock_tool = Mock()
        mock_tool.name = "test_func"

        # Create output that streams incrementally
        output = ModelOutputThunk("")
        output.tool_calls = {
            "test_func": ModelToolCall(
                name="test_func", func=mock_tool, args={"key": "value"}
            )
        }

        # Mock astream
        chunks_to_stream = ["Part1", "Part2"]
        stream_index = 0

        async def mock_astream():
            nonlocal stream_index
            if stream_index < len(chunks_to_stream):
                chunk = chunks_to_stream[stream_index]
                stream_index += 1
                return chunk
            else:
                output.value = "Part1Part2"
                object.__setattr__(output, "_computed", True)
                return ""

        def mock_is_computed():
            return stream_index >= len(chunks_to_stream)

        # Patch the astream and is_computed methods
        with (
            patch.object(output, "astream", side_effect=mock_astream),
            patch.object(output, "is_computed", side_effect=mock_is_computed),
        ):
            # Collect all chunks
            chunks = []
            async for chunk_data in stream_chat_completion_chunks(
                output=output,
                completion_id="chatcmpl-test",
                model="test-model",
                created=1234567890,
            ):
                if (
                    chunk_data.startswith("data: ")
                    and chunk_data.strip() != "data: [DONE]"
                ):
                    json_str = chunk_data[6:].strip()
                    chunks.append(json.loads(json_str))

            # Should have: initial, Part1, Part2, tool_calls, final = 5 chunks
            assert len(chunks) == 5

            # Verify sequence
            assert chunks[0]["choices"][0]["delta"].get("role") == "assistant"
            assert chunks[1]["choices"][0]["delta"].get("content") == "Part1"
            assert chunks[2]["choices"][0]["delta"].get("content") == "Part2"
            assert "tool_calls" in chunks[3]["choices"][0]["delta"]
            assert chunks[4]["choices"][0]["finish_reason"] == "tool_calls"


class TestStreamingUsageField:
    """Tests for usage field in streaming responses."""

    @pytest.mark.asyncio
    async def test_usage_included_when_stream_options_set(self):
        """Test that usage is included in final chunk when stream_options.include_usage=True."""
        output = ModelOutputThunk("Response")
        output.generation.usage = {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
        }

        chunks = []
        async for chunk_data in stream_chat_completion_chunks(
            output=output,
            completion_id="chatcmpl-test",
            model="test-model",
            created=1234567890,
            stream_options=StreamOptions(include_usage=True),
        ):
            if chunk_data.startswith("data: ") and chunk_data.strip() != "data: [DONE]":
                json_str = chunk_data[6:].strip()
                chunks.append(json.loads(json_str))

        # Final chunk should include usage
        final_chunk = chunks[-1]
        assert "usage" in final_chunk
        assert final_chunk["usage"]["prompt_tokens"] == 10
        assert final_chunk["usage"]["completion_tokens"] == 5
        assert final_chunk["usage"]["total_tokens"] == 15

    @pytest.mark.asyncio
    async def test_usage_excluded_when_stream_options_not_set(self):
        """Test that usage is excluded when stream_options is None."""
        output = ModelOutputThunk("Response")
        output.generation.usage = {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
        }

        chunks = []
        async for chunk_data in stream_chat_completion_chunks(
            output=output,
            completion_id="chatcmpl-test",
            model="test-model",
            created=1234567890,
            stream_options=None,
        ):
            if chunk_data.startswith("data: ") and chunk_data.strip() != "data: [DONE]":
                json_str = chunk_data[6:].strip()
                chunks.append(json.loads(json_str))

        # Final chunk should NOT include usage
        final_chunk = chunks[-1]
        assert "usage" not in final_chunk or final_chunk["usage"] is None

    @pytest.mark.asyncio
    async def test_usage_excluded_when_include_usage_false(self):
        """Test that usage is excluded when stream_options.include_usage=False."""
        output = ModelOutputThunk("Response")
        output.generation.usage = {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
        }

        chunks = []
        async for chunk_data in stream_chat_completion_chunks(
            output=output,
            completion_id="chatcmpl-test",
            model="test-model",
            created=1234567890,
            stream_options=StreamOptions(include_usage=False),
        ):
            if chunk_data.startswith("data: ") and chunk_data.strip() != "data: [DONE]":
                json_str = chunk_data[6:].strip()
                chunks.append(json.loads(json_str))

        # Final chunk should NOT include usage
        final_chunk = chunks[-1]
        assert "usage" not in final_chunk or final_chunk["usage"] is None


class TestStreamingErrorHandling:
    """Tests for error handling in streaming."""

    @pytest.mark.asyncio
    async def test_streaming_error_emits_error_response(self):
        """Test that streaming errors emit OpenAI-compatible error responses."""
        # Create output that will raise an error during streaming
        output = ModelOutputThunk("")
        output._computed = False

        # Use AsyncMock with side_effect to raise error
        output.astream = AsyncMock(
            side_effect=RuntimeError("Simulated streaming error")
        )

        # Collect chunks
        chunks = []
        async for chunk_data in stream_chat_completion_chunks(
            output=output,
            completion_id="chatcmpl-test",
            model="test-model",
            created=1234567890,
        ):
            chunks.append(chunk_data)

        # Should have: initial chunk, error response, [DONE]
        assert len(chunks) >= 3

        # Find error response (second-to-last before [DONE])
        error_chunk_data = chunks[-2]
        assert error_chunk_data.startswith("data: ")
        json_str = error_chunk_data[6:].strip()
        error_response = json.loads(json_str)

        # Verify error structure
        assert "error" in error_response
        assert error_response["error"]["type"] == "server_error"
        assert "Streaming error" in error_response["error"]["message"]
        assert "Simulated streaming error" in error_response["error"]["message"]

        # Should still end with [DONE]
        assert chunks[-1] == "data: [DONE]\n\n"


class TestStreamingChunkMetadata:
    """Tests for chunk metadata fields."""

    @pytest.mark.asyncio
    async def test_all_chunks_have_required_fields(self):
        """Test that all chunks have required OpenAI fields."""
        output = ModelOutputThunk("Test response")

        chunks = []
        async for chunk_data in stream_chat_completion_chunks(
            output=output,
            completion_id="chatcmpl-test123",
            model="test-model-name",
            created=1234567890,
            system_fingerprint="test-fingerprint",
        ):
            if chunk_data.startswith("data: ") and chunk_data.strip() != "data: [DONE]":
                json_str = chunk_data[6:].strip()
                chunks.append(json.loads(json_str))

        # Verify all chunks have required fields
        for chunk in chunks:
            assert chunk["id"] == "chatcmpl-test123"
            assert chunk["model"] == "test-model-name"
            assert chunk["created"] == 1234567890
            assert chunk["object"] == "chat.completion.chunk"
            assert chunk["system_fingerprint"] == "test-fingerprint"
            assert "choices" in chunk
            assert len(chunk["choices"]) == 1
            assert chunk["choices"][0]["index"] == 0

    @pytest.mark.asyncio
    async def test_done_marker_emitted(self):
        """Test that [DONE] marker is always emitted at the end."""
        output = ModelOutputThunk("Response")

        chunks = []
        async for chunk_data in stream_chat_completion_chunks(
            output=output,
            completion_id="chatcmpl-test",
            model="test-model",
            created=1234567890,
        ):
            chunks.append(chunk_data)

        # Last chunk should be [DONE]
        assert chunks[-1] == "data: [DONE]\n\n"

    @pytest.mark.asyncio
    async def test_sse_format_correct(self):
        """Test that chunks follow SSE format: 'data: {json}\\n\\n'."""
        output = ModelOutputThunk("Response")

        async for chunk_data in stream_chat_completion_chunks(
            output=output,
            completion_id="chatcmpl-test",
            model="test-model",
            created=1234567890,
        ):
            # All chunks should start with "data: "
            assert chunk_data.startswith("data: ")
            # All chunks should end with double newline
            assert chunk_data.endswith("\n\n")
