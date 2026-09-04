# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for mellea.helpers.async_helpers."""

import asyncio
import datetime

import pytest

from mellea.core.base import ModelOutputThunk
from mellea.helpers.async_helpers import (
    DEFAULT_CHUNK_TIMEOUT,
    ClientCache,
    get_current_event_loop,
    send_to_queue,
)

# --- send_to_queue ---


def _make_thunk(*, streaming: bool = False) -> ModelOutputThunk:
    """A bare thunk whose `_gen.queue` receives `send_to_queue` output.

    With `streaming=True` the thunk is primed so `_record_ttfb()` stamps `ttfb_ms`
    at first-chunk receipt.
    """
    mot: ModelOutputThunk = ModelOutputThunk(value=None)
    if streaming:
        mot.generation.streaming = True
        mot._gen.start = datetime.datetime.now()
    return mot


class TestSendToQueue:
    async def test_coroutine_single_value(self):
        """Coroutine returning a non-iterator value is put into queue followed by sentinel."""

        async def produce():
            return "result"

        mot = _make_thunk()
        q = mot._gen.queue
        await send_to_queue(produce(), mot)
        assert await q.get() == "result"
        assert await q.get() is None  # sentinel

    async def test_coroutine_returning_async_iterator(self):
        """Coroutine returning an async iterator streams items then sentinel."""

        async def produce():
            async def _gen():
                yield "a"
                yield "b"

            return _gen()

        mot = _make_thunk()
        q = mot._gen.queue
        await send_to_queue(produce(), mot)
        assert await q.get() == "a"
        assert await q.get() == "b"
        assert await q.get() is None

    async def test_async_iterator_directly(self):
        """Passing an async iterator (not wrapped in coroutine) streams items."""

        async def _gen():
            yield 1
            yield 2

        mot = _make_thunk()
        q = mot._gen.queue
        await send_to_queue(_gen(), mot)
        assert await q.get() == 1
        assert await q.get() == 2
        assert await q.get() is None

    async def test_exception_propagated_to_queue(self):
        """Exceptions during generation are put into queue instead of raising."""

        async def explode():
            raise ValueError("boom")

        mot = _make_thunk()
        q = mot._gen.queue
        await send_to_queue(explode(), mot)
        item = await q.get()
        assert isinstance(item, ValueError)
        assert str(item) == "boom"

    async def test_iterator_exception_propagated(self):
        """Exception mid-iteration is captured and put into queue."""

        async def _gen():
            yield "ok"
            raise RuntimeError("mid-stream")

        mot = _make_thunk()
        q = mot._gen.queue
        await send_to_queue(_gen(), mot)
        assert await q.get() == "ok"
        item = await q.get()
        assert isinstance(item, RuntimeError)

    async def test_chunk_timeout_fires(self):
        """A stalling iterator puts TimeoutError in the queue; no sentinel follows."""

        async def _stalling_gen():
            yield "first"
            await asyncio.sleep(1)  # longer than chunk_timeout
            yield "never"  # pragma: no cover

        mot = _make_thunk()
        q = mot._gen.queue
        await send_to_queue(_stalling_gen(), mot, chunk_timeout=0.05)

        assert await q.get() == "first"
        item = await q.get()
        assert isinstance(item, TimeoutError)
        assert "STREAM_TIMEOUT" in str(item)
        assert q.empty()  # no trailing sentinel after a timeout

    async def test_chunk_timeout_calls_timeout_callback(self):
        """A stream-guard timeout notifies the backend before returning."""
        timeout_called = False

        def on_timeout() -> None:
            nonlocal timeout_called
            timeout_called = True

        async def _stalling_gen():
            yield "first"
            await asyncio.sleep(1)
            yield "never"  # pragma: no cover

        mot = _make_thunk()
        q = mot._gen.queue
        await send_to_queue(
            _stalling_gen(), mot, chunk_timeout=0.05, on_timeout=on_timeout
        )

        assert await q.get() == "first"
        assert isinstance(await q.get(), TimeoutError)
        assert timeout_called is True

    async def test_chunk_timeout_callback_error_does_not_mask_timeout(self):
        """A backend cleanup failure must not replace the stream timeout."""

        def on_timeout() -> None:
            raise RuntimeError("cleanup failed")

        async def _stalling_gen():
            await asyncio.sleep(1)
            yield "never"  # pragma: no cover

        mot = _make_thunk()
        q = mot._gen.queue
        await send_to_queue(
            _stalling_gen(), mot, chunk_timeout=0.05, on_timeout=on_timeout
        )

        item = await q.get()
        assert isinstance(item, TimeoutError)
        assert "STREAM_TIMEOUT" in str(item)

    async def test_completed_stream_does_not_call_timeout_callback(self):
        """Normal completion leaves the backend cancellation hook untouched."""
        timeout_called = False

        def on_timeout() -> None:
            nonlocal timeout_called
            timeout_called = True

        async def _completed_gen():
            yield "done"

        mot = _make_thunk()
        q = mot._gen.queue
        await send_to_queue(
            _completed_gen(), mot, chunk_timeout=0.05, on_timeout=on_timeout
        )

        assert await q.get() == "done"
        assert await q.get() is None
        assert timeout_called is False

    async def test_backend_timeout_does_not_call_timeout_callback(self):
        """A backend-raised TimeoutError is not mistaken for the stream guard."""
        timeout_called = False

        def on_timeout() -> None:
            nonlocal timeout_called
            timeout_called = True

        async def _backend_timeout():
            raise TimeoutError("backend timeout")
            yield  # pragma: no cover

        mot = _make_thunk()
        q = mot._gen.queue
        await send_to_queue(
            _backend_timeout(), mot, chunk_timeout=1, on_timeout=on_timeout
        )

        item = await q.get()
        assert isinstance(item, TimeoutError)
        assert str(item) == "backend timeout"
        assert timeout_called is False

    async def test_chunk_timeout_none_disables_timeout(self):
        """chunk_timeout=None allows a slow-but-completing iterator to finish normally."""

        async def _slow_gen():
            yield "a"
            await asyncio.sleep(0.05)
            yield "b"

        mot = _make_thunk()
        q = mot._gen.queue
        await send_to_queue(_slow_gen(), mot, chunk_timeout=None)

        assert await q.get() == "a"
        assert await q.get() == "b"
        assert await q.get() is None  # sentinel present on clean completion

    async def test_ttfb_stamped_at_first_chunk_for_streaming(self):
        """A streaming response stamps ttfb_ms on the thunk at first-chunk receipt."""

        async def _gen():
            yield "a"
            yield "b"

        mot = _make_thunk(streaming=True)
        await send_to_queue(_gen(), mot)

        # Stamped during production, before any consumer dequeues.
        assert mot.generation.ttfb_ms is not None
        assert mot.generation.ttfb_ms >= 0

    async def test_ttfb_not_stamped_for_non_iterator(self):
        """A non-iterator (non-streaming) response leaves ttfb_ms unset."""

        async def produce():
            return "result"

        mot = _make_thunk(streaming=True)
        await send_to_queue(produce(), mot)

        assert await mot._gen.queue.get() == "result"
        assert mot.generation.ttfb_ms is None

    async def test_chunk_intervals_captured_per_chunk(self):
        """send_to_queue records one receipt interval per chunk: None first, then floats."""

        async def _gen():
            yield "a"
            yield "b"
            yield "c"

        mot = _make_thunk(streaming=True)
        await send_to_queue(_gen(), mot)

        intervals = list(mot._gen.chunk_intervals)
        assert len(intervals) == 3
        assert intervals[0] is None
        assert all(isinstance(i, float) and i >= 0 for i in intervals[1:])

    def test_default_chunk_timeout_value(self):
        """DEFAULT_CHUNK_TIMEOUT is 120 seconds."""
        assert DEFAULT_CHUNK_TIMEOUT == 120.0


# --- get_current_event_loop ---


class TestGetCurrentEventLoop:
    async def test_returns_loop_when_running(self):
        loop = get_current_event_loop()
        assert loop is not None
        assert loop is asyncio.get_running_loop()

    def test_returns_none_when_no_loop(self):
        assert get_current_event_loop() is None


# --- ClientCache ---


class TestClientCache:
    def test_put_and_get(self):
        cache = ClientCache(capacity=3)
        cache.put(1, "a")
        assert cache.get(1) == "a"

    def test_evicts_lru(self):
        cache = ClientCache(capacity=2)
        cache.put(1, "a")
        cache.put(2, "b")
        cache.put(3, "c")  # evicts key 1
        assert cache.get(1) is None
        assert cache.get(2) == "b"
        assert cache.get(3) == "c"

    def test_access_refreshes_lru_order(self):
        cache = ClientCache(capacity=2)
        cache.put(1, "a")
        cache.put(2, "b")
        cache.get(1)  # refresh key 1 — now key 2 is LRU
        cache.put(3, "c")  # evicts key 2
        assert cache.get(1) == "a"
        assert cache.get(2) is None
        assert cache.get(3) == "c"

    def test_overwrite_existing_key(self):
        cache = ClientCache(capacity=2)
        cache.put(1, "old")
        cache.put(1, "new")
        assert cache.get(1) == "new"
        assert cache.current_size() == 1


if __name__ == "__main__":
    pytest.main([__file__])
