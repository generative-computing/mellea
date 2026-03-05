"""Tests that exceptions during generation propagate correctly through ModelOutputThunk.astream().

Regression test for issue #577: post_process in a finally block was swallowing
the original generation exception by raising a secondary error from post_process
(which assumes system invariants that don't hold during failures).
"""

import pytest

from mellea.core.base import CBlock, GenerateType, ModelOutputThunk


async def _noop_process(mot, chunk):
    if mot._underlying_value is None:
        mot._underlying_value = ""
    mot._underlying_value += str(chunk)


async def _failing_post_process(mot):
    raise RuntimeError("post_process failed due to broken invariants")


def _make_thunk(post_process=_failing_post_process):
    mot = ModelOutputThunk(value=None)
    mot._generate_type = GenerateType.ASYNC
    mot._process = _noop_process
    mot._post_process = post_process
    mot._action = CBlock("test")
    mot._chunk_size = 0
    return mot


@pytest.mark.parametrize(
    "error",
    [ValueError("connection reset by peer"), ConnectionError("server unavailable")],
)
async def test_astream_propagates_generation_exception(error):
    """The original generation error must propagate, not a secondary error from post_process."""
    mot = _make_thunk()
    await mot._async_queue.put(error)

    with pytest.raises(type(error), match=str(error)):
        await mot.astream()


async def test_astream_post_process_only_called_on_success():
    """post_process must be called on success but not on error."""
    post_process_called = False

    async def _tracking_post_process(mot):
        nonlocal post_process_called
        post_process_called = True

    # Error path: post_process should NOT be called
    mot = _make_thunk(post_process=_tracking_post_process)
    await mot._async_queue.put(RuntimeError("generation failed"))

    with pytest.raises(RuntimeError, match="generation failed"):
        await mot.astream()

    assert not post_process_called, (
        "post_process should not be called when generation fails"
    )

    # Success path: post_process SHOULD be called
    post_process_called = False
    mot = _make_thunk(post_process=_tracking_post_process)
    await mot._async_queue.put("hello")
    await mot._async_queue.put(None)  # sentinel for completion

    await mot.astream()

    assert post_process_called, "post_process should be called on successful completion"
