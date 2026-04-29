"""Verification that streaming tool call deltas include required index field.

This test demonstrates that our streaming implementation is compatible with
OpenAI SDK delta reassembly logic, which requires the index field.
"""

import json
from unittest.mock import Mock

import pytest

from cli.serve.app import make_chat_endpoint
from cli.serve.models import ChatCompletionRequest, ChatMessage
from mellea.core.base import ModelOutputThunk, ModelToolCall


@pytest.mark.asyncio
async def test_tool_call_delta_has_required_index_field():
    """Verify that streaming tool call deltas include the required index field.

    The OpenAI streaming spec requires each item in delta.tool_calls to carry
    an index field. Clients including the openai Python SDK, LangChain, and
    LiteLLM key their delta-reassembly state machine on this field.

    Without it, they silently drop tool calls, coalesce them incorrectly, or
    raise a TypeError depending on version.
    """
    # Create a mock module with a serve function
    mock_module = Mock()
    mock_module.__name__ = "test_module"

    # Create a mock tool
    mock_tool = Mock()
    mock_tool.name = "get_weather"

    # Create a mock output with multiple tool calls to test indexing
    mock_output = ModelOutputThunk("I'll check the weather for you.")
    mock_output.tool_calls = {
        "get_weather": ModelToolCall(
            name="get_weather",
            func=mock_tool,
            args={"location": "San Francisco", "units": "celsius"},
        ),
        "get_forecast": ModelToolCall(
            name="get_forecast",
            func=mock_tool,
            args={"location": "San Francisco", "days": 3},
        ),
    }
    mock_module.serve.return_value = mock_output

    request = ChatCompletionRequest(
        model="test-model",
        messages=[ChatMessage(role="user", content="What's the weather?")],
        stream=True,
    )

    endpoint = make_chat_endpoint(mock_module)
    response = await endpoint(request)

    # Collect all chunks
    chunks = []
    async for chunk_data in response.body_iterator:
        chunk_str = (
            bytes(chunk_data).decode("utf-8")
            if isinstance(chunk_data, (bytes, memoryview))
            else chunk_data
        )

        if chunk_str.startswith("data: "):
            json_str = chunk_str[6:].strip()
            if json_str and json_str != "[DONE]":
                chunks.append(json.loads(json_str))

    # Find the tool call chunk
    tool_call_chunk = None
    for chunk in chunks:
        if chunk["choices"][0]["delta"].get("tool_calls"):
            tool_call_chunk = chunk
            break

    assert tool_call_chunk is not None, "Should have a tool call chunk"

    tool_calls = tool_call_chunk["choices"][0]["delta"]["tool_calls"]
    assert len(tool_calls) == 2, "Should have 2 tool calls"

    # Verify REQUIRED index field is present on each tool call delta
    for i, tc in enumerate(tool_calls):
        assert "index" in tc, f"tool_calls[{i}] must include index field"
        assert isinstance(tc["index"], int), "index must be an integer"
        assert tc["index"] == i, f"tool_calls[{i}] should have index={i}"

        # Verify other fields are present (id, type, function)
        assert "id" in tc, f"tool_calls[{i}] should have id"
        assert "type" in tc, f"tool_calls[{i}] should have type"
        assert tc["type"] == "function", f"tool_calls[{i}] type should be 'function'"
        assert "function" in tc, f"tool_calls[{i}] should have function"
        assert "name" in tc["function"], f"tool_calls[{i}].function should have name"
        assert "arguments" in tc["function"], (
            f"tool_calls[{i}].function should have arguments"
        )


@pytest.mark.asyncio
async def test_single_tool_call_has_index_zero():
    """Verify that a single tool call has index=0."""
    mock_module = Mock()
    mock_module.__name__ = "test_module"

    mock_tool = Mock()
    mock_tool.name = "search"

    mock_output = ModelOutputThunk("Searching...")
    mock_output.tool_calls = {
        "search": ModelToolCall(name="search", func=mock_tool, args={"query": "test"})
    }
    mock_module.serve.return_value = mock_output

    request = ChatCompletionRequest(
        model="test-model",
        messages=[ChatMessage(role="user", content="Search for test")],
        stream=True,
    )

    endpoint = make_chat_endpoint(mock_module)
    response = await endpoint(request)

    chunks = []
    async for chunk_data in response.body_iterator:
        chunk_str = (
            bytes(chunk_data).decode("utf-8")
            if isinstance(chunk_data, (bytes, memoryview))
            else chunk_data
        )

        if chunk_str.startswith("data: "):
            json_str = chunk_str[6:].strip()
            if json_str and json_str != "[DONE]":
                chunks.append(json.loads(json_str))

    # Find the tool call chunk
    tool_call_chunk = None
    for chunk in chunks:
        if chunk["choices"][0]["delta"].get("tool_calls"):
            tool_call_chunk = chunk
            break

    assert tool_call_chunk is not None
    tool_calls = tool_call_chunk["choices"][0]["delta"]["tool_calls"]
    assert len(tool_calls) == 1
    assert tool_calls[0]["index"] == 0, "Single tool call should have index=0"
