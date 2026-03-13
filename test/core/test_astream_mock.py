"""Deterministic Mock Tests for ModelOutputThunk.astream() incremental return behavior.

Tests that astream() returns only the incremental content added during each call.
All astream() chunks concatenated should equal the full final value. Calling
astream() on a computed MOT raises RuntimeError. Uses manual queue injection to
bypass LLM calls and network operations, guaranteeing determinism.
"""

import asyncio
from typing import Any

import pytest

from mellea.core.base import CBlock, GenerateType, ModelOutputThunk


async def mock_process(mot: ModelOutputThunk, chunk: Any) -> None:
    """Mock process function that simply appends the chunk to the underlying value."""
    if mot._underlying_value is None:
        mot._underlying_value = ""
    if chunk is not None:
        mot._underlying_value += chunk


async def mock_post_process(mot: ModelOutputThunk) -> None:
    """Mock post-process function (does nothing)."""


def create_manual_mock_thunk() -> ModelOutputThunk:
    """Helper to create a mock ModelOutputThunk where we manually populate the queue."""
    mot = ModelOutputThunk(value=None)
    mot._action = CBlock("mock_action")
    mot._generate_type = GenerateType.ASYNC
    mot._process = mock_process
    mot._post_process = mock_post_process
    mot._chunk_size = 0  # Read exactly what is available
    return mot


@pytest.mark.asyncio
async def test_astream_returns_incremental_chunks():
    """Test that astream() returns only new content, not accumulated content."""
    mot = create_manual_mock_thunk()

    # Drop the first chunk and pull it
    mot._async_queue.put_nowait("chunk1 ")
    chunk1 = await mot.astream()
    assert chunk1 == "chunk1 "

    # Drop the second chunk and pull it
    mot._async_queue.put_nowait("chunk2 ")
    chunk2 = await mot.astream()
    assert chunk2 == "chunk2 "

    # Drop the third chunk and pull it
    mot._async_queue.put_nowait("chunk3 ")
    chunk3 = await mot.astream()
    assert chunk3 == "chunk3 "

    # Send completion sentinel
    mot._async_queue.put_nowait(None)

    # Wait until fully consumed
    while not mot.is_computed():
        await mot.astream()

    final_val = await mot.avalue()
    assert final_val == "chunk1 chunk2 chunk3 "


@pytest.mark.asyncio
async def test_astream_multiple_calls_accumulate_correctly():
    """Test that multiple astream() calls accumulate to the final value."""
    # Simulating a scenario where queue chunks outpace the reading loop
    mot = create_manual_mock_thunk()

    # Drop multiple items at once to simulate fast network
    mot._async_queue.put_nowait("c")
    mot._async_queue.put_nowait("h")
    mot._async_queue.put_nowait("u")

    # Calling astream should drain all currently queued items ("chu")
    chunk1 = await mot.astream()
    assert chunk1 == "chu"

    mot._async_queue.put_nowait("n")
    mot._async_queue.put_nowait("k")
    mot._async_queue.put_nowait(None)

    chunk2 = await mot.astream()
    # astream() returns only the incremental content added during this call
    assert chunk2 == "nk"

    assert mot.is_computed()
    # All astream() chunks concatenated should equal the full value
    assert chunk1 + chunk2 == "chunk"
    assert mot.value == "chunk"


@pytest.mark.asyncio
async def test_astream_beginning_length_tracking():
    """Test that beginning_length is correctly tracked across astream calls."""
    mot = create_manual_mock_thunk()

    mot._async_queue.put_nowait("AAA")
    chunk1 = await mot.astream()
    assert chunk1 == "AAA"

    mot._async_queue.put_nowait("BBB")
    chunk2 = await mot.astream()
    # verify incremental length tracking works
    assert not chunk2.startswith(chunk1)
    assert chunk2 == "BBB"


@pytest.mark.asyncio
async def test_astream_empty_beginning():
    """Test astream when _underlying_value starts as None."""
    mot = create_manual_mock_thunk()

    mot._async_queue.put_nowait("First")
    # At the start, _underlying_value is None, beginning_length is 0
    chunk = await mot.astream()

    # Because beginning length was 0, astream returns the full chunk
    assert chunk == "First"
    assert mot._underlying_value == "First"


@pytest.mark.asyncio
async def test_astream_computed_raises_error():
    """Test that astream raises RuntimeError when already computed."""
    # Precomputed thunk is already computed
    mot = ModelOutputThunk(value="Hello, world!")

    # astream() on a computed MOT now raises RuntimeError
    with pytest.raises(RuntimeError, match="Streaming has finished"):
        await mot.astream()


@pytest.mark.asyncio
async def test_astream_final_call_returns_full_value():
    """Test that the final astream call returns the full value when computed."""
    mot = create_manual_mock_thunk()

    mot._async_queue.put_nowait("part1")
    chunk1 = await mot.astream()
    assert chunk1 == "part1"

    mot._async_queue.put_nowait("part2")
    chunk2 = await mot.astream()
    assert chunk2 == "part2"

    mot._async_queue.put_nowait("part3")
    mot._async_queue.put_nowait(None)

    # Calling astream here processes "part3" and `None`, flagging it as done
    chunk3 = await mot.astream()

    # The final astream() call returns only the incremental content, not the full value
    assert chunk3 == "part3"

    # All chunks concatenated equal the full value
    assert chunk1 + chunk2 + chunk3 == "part1part2part3"
    assert mot.value == "part1part2part3"
