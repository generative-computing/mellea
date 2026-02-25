"""Regression tests for astream() exception handling.

When a backend error occurs during streaming, the Exception object lands in the
async queue.  Before the fix, astream() would either pass it to _process (crash)
or post_processing would hit a KeyError on _meta keys that were never set.

These tests verify that astream() cleanly propagates the original exception
after running _post_process for telemetry cleanup.
"""

import asyncio

import pytest

from mellea.core.base import CBlock, GenerateType, ModelOutputThunk


def _make_streaming_mot():
    """Create a ModelOutputThunk wired up for streaming with stub callbacks."""
    mot = ModelOutputThunk(value=None)
    mot._generate_type = GenerateType.ASYNC
    mot._chunk_size = 1

    process_calls: list = []

    async def _process(mot, chunk):
        process_calls.append(chunk)
        text = chunk if isinstance(chunk, str) else str(chunk)
        if mot._underlying_value is None:
            mot._underlying_value = text
        else:
            mot._underlying_value += text

    post_process_called = asyncio.Event()

    async def _post_process(mot):
        post_process_called.set()

    mot._process = _process
    mot._post_process = _post_process

    return mot, process_calls, post_process_called


async def test_astream_propagates_exception_from_queue():
    """Exception in the queue is re-raised after cleanup, not passed to _process."""
    mot, process_calls, post_process_called = _make_streaming_mot()

    original_error = RuntimeError("backend connection lost")
    await mot._async_queue.put(original_error)

    with pytest.raises(RuntimeError, match="backend connection lost"):
        await mot.astream()

    # _process must never have seen the Exception object
    assert original_error not in process_calls
    # _post_process ran for telemetry cleanup
    assert post_process_called.is_set()


async def test_astream_propagates_exception_after_valid_chunks():
    """Valid chunks before the exception are processed; exception still raised."""
    mot, process_calls, post_process_called = _make_streaming_mot()

    await mot._async_queue.put("hello ")
    await mot._async_queue.put("world")
    await mot._async_queue.put(ValueError("mid-stream failure"))

    with pytest.raises(ValueError, match="mid-stream failure"):
        await mot.astream()

    # Valid chunks were processed
    assert process_calls == ["hello ", "world"]
    # Accumulated value reflects only the valid chunks
    assert mot._underlying_value == "hello world"
    # Cleanup still ran
    assert post_process_called.is_set()


async def test_astream_skips_none_and_exception_in_chunk_loop():
    """Belt-and-suspenders: stray None/Exception objects in the middle of the
    chunk list are skipped rather than passed to _process."""
    mot, process_calls, _ = _make_streaming_mot()

    await mot._async_queue.put("good chunk")
    await mot._async_queue.put(None)

    mot._action = CBlock("test")

    result = await mot.astream()

    assert process_calls == ["good chunk"]
    assert mot.is_computed()
    assert result is not None
