"""Tests that exceptions during generation propagate correctly through ModelOutputThunk.astream().

Regression test for issue #577: post_process in a finally block was swallowing
the original generation exception by raising a secondary error from post_process
(which assumes system invariants that don't hold during failures).
"""

import asyncio

import pytest

from mellea.core.base import CBlock, GenerateType, ModelOutputThunk


async def _noop_process(mot, chunk):
    """Minimal process callback that appends chunk text to the thunk's value."""
    if mot._underlying_value is None:
        mot._underlying_value = ""
    mot._underlying_value += str(chunk)


async def _failing_post_process(mot):
    """A post_process that fails, simulating real backends which assume invariants."""
    raise RuntimeError("post_process failed due to broken invariants")


def _make_streaming_thunk(post_process=None):
    """Create a ModelOutputThunk wired up for async streaming without a real backend."""
    mot = ModelOutputThunk(value=None)
    mot._generate_type = GenerateType.ASYNC
    mot._process = _noop_process
    mot._post_process = post_process or _failing_post_process
    mot._action = CBlock("test")
    mot._chunk_size = 0  # Don't require minimum chunks
    return mot


@pytest.mark.asyncio
async def test_astream_propagates_generation_exception():
    """When the backend puts an Exception on the queue, astream must raise that exact exception.

    Before the fix for #577, a finally block called post_process on error, which
    would itself fail and swallow the original generation error.
    """
    original_error = ValueError("connection reset by peer")
    mot = _make_streaming_thunk()

    # Simulate backend putting an error on the queue
    await mot._async_queue.put(original_error)

    with pytest.raises(ValueError, match="connection reset by peer"):
        await mot.astream()


@pytest.mark.asyncio
async def test_astream_exception_is_not_from_post_process():
    """Ensure the raised exception is the generation error, not a post_process error.

    This is the core of issue #577: post_process failures must not mask generation errors.
    """
    generation_error = ConnectionError("server unavailable")
    mot = _make_streaming_thunk(post_process=_failing_post_process)

    await mot._async_queue.put(generation_error)

    # Must get ConnectionError, NOT RuntimeError from _failing_post_process
    with pytest.raises(ConnectionError, match="server unavailable"):
        await mot.astream()


@pytest.mark.asyncio
async def test_astream_post_process_only_called_on_success():
    """Verify post_process is called on successful completion, not on error."""
    post_process_called = False

    async def _tracking_post_process(mot):
        nonlocal post_process_called
        post_process_called = True

    # Error path: post_process should NOT be called
    mot = _make_streaming_thunk(post_process=_tracking_post_process)
    await mot._async_queue.put(RuntimeError("generation failed"))

    with pytest.raises(RuntimeError, match="generation failed"):
        await mot.astream()

    assert not post_process_called, (
        "post_process should not be called when generation fails"
    )

    # Success path: post_process SHOULD be called
    post_process_called = False
    mot = _make_streaming_thunk(post_process=_tracking_post_process)
    await mot._async_queue.put("hello")
    await mot._async_queue.put(None)  # Sentinel for completion

    await mot.astream()

    assert post_process_called, "post_process should be called on successful completion"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
