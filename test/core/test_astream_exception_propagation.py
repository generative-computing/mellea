"""Tests that exceptions during generation propagate correctly through ModelOutputThunk.astream().

Regression test for issue #577: post_process in a finally block was swallowing
the original generation exception by raising a secondary error from post_process
(which assumes system invariants that don't hold during failures).
"""

import datetime
import importlib.util

import pytest

from mellea.core.base import CBlock, GenerateType, ModelOutputThunk

_otel_available = importlib.util.find_spec("opentelemetry") is not None


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
    mot._start = datetime.datetime.now()
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


@pytest.mark.skipif(
    not _otel_available,
    reason="opentelemetry not installed — install mellea[telemetry]",
)
async def test_astream_closes_telemetry_span_on_error():
    """Telemetry span must be ended and error recorded when generation fails."""
    from unittest.mock import MagicMock

    mock_span = MagicMock()
    mot = _make_thunk()
    mot._meta["_telemetry_span"] = mock_span

    error = ConnectionError("server unavailable")
    await mot._async_queue.put(error)

    with pytest.raises(ConnectionError, match="server unavailable"):
        await mot.astream()

    # Span should have been ended and cleaned up
    mock_span.record_exception.assert_called_once_with(error)
    mock_span.set_status.assert_called_once()
    mock_span.end.assert_called_once()
    assert "_telemetry_span" not in mot._meta


async def test_astream_no_span_leak_when_no_telemetry():
    """When no telemetry span is present, error propagation still works."""
    mot = _make_thunk()
    assert "_telemetry_span" not in mot._meta

    error = ValueError("test error")
    await mot._async_queue.put(error)

    with pytest.raises(ValueError, match="test error"):
        await mot.astream()
