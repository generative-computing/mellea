# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for mellea.helpers.async_helpers."""

import asyncio
import datetime
import threading
import time

import pytest

from mellea.core.base import ModelOutputThunk
from mellea.helpers import async_helpers
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


class TestClientCacheClose:
    def test_put_evicts_and_closes_via_callback(self):
        closed = []
        done = threading.Event()

        async def aclose(client):
            closed.append(client)
            done.set()

        cache = ClientCache(capacity=2, aclose=aclose)
        cache.put(1, "a")
        cache.put(2, "b")
        cache.put(3, "c")  # evicts key 1

        assert done.wait(timeout=5.0), "evicted client was never closed"
        assert closed == ["a"]

    def test_put_does_not_wait_for_the_evicted_close(self):
        """Eviction happens on `generate`'s path; it must not stall the caller."""
        release = threading.Event()
        started = threading.Event()
        finished = threading.Event()

        async def aclose(client):
            started.set()
            await asyncio.to_thread(release.wait, 5.0)
            finished.set()

        cache = ClientCache(capacity=1, aclose=aclose)
        cache.put(1, "a")
        cache.put(2, "b")  # evicts "a"

        assert started.wait(timeout=5.0), "evicted client's close was never scheduled"
        assert not finished.is_set(), "put() waited for the close to finish"
        release.set()
        assert finished.wait(timeout=5.0)

    def test_put_without_aclose_does_not_error_on_eviction(self):
        cache = ClientCache(capacity=1)
        cache.put(1, "a")
        cache.put(2, "b")  # evicts "a"; no aclose configured, nothing to close
        assert cache.get(2) == "b"

    def test_clear_closes_every_entry(self):
        closed = []

        async def aclose(client):
            closed.append(client)

        cache = ClientCache(capacity=3, aclose=aclose)
        cache.put(1, "a")
        cache.put(2, "b")
        cache.clear()

        assert sorted(closed) == ["a", "b"]
        assert cache.current_size() == 0

    def test_clear_empties_cache_even_if_closer_raises(self):
        """A raising closer must not leave a stale entry behind."""

        async def aclose(client):
            raise RuntimeError("boom")

        cache = ClientCache(capacity=1, aclose=aclose)
        cache.put(1, "a")
        cache.clear()  # the closer's error is caught and logged, not raised

        assert cache.current_size() == 0

    def test_clear_bounds_its_wait_on_an_unbound_client(self, monkeypatch):
        """`close()`'s own path: a busy background loop must time out, not hang."""
        monkeypatch.setattr(async_helpers, "CLIENT_CLOSE_TIMEOUT", 0.1)
        release = threading.Event()
        finished = threading.Event()

        async def aclose(client):
            await asyncio.to_thread(release.wait, 10.0)
            finished.set()

        cache = ClientCache(capacity=1, aclose=aclose)
        cache.put(None, "a")  # the key every sync-constructed backend uses

        try:
            start = time.monotonic()
            cache.clear()
            elapsed = time.monotonic() - start

            assert elapsed < 5.0, "clear() waited without a timeout"
            assert cache.current_size() == 0
        finally:
            release.set()

        # Timing out abandons the wait, not the close.
        assert finished.wait(timeout=5.0), "the close was cancelled, not just unwaited"

    def test_clear_without_aclose_just_empties(self):
        cache = ClientCache(capacity=2)
        cache.put(1, "a")
        cache.put(2, "b")
        cache.clear()
        assert cache.current_size() == 0

    async def test_aclear_closes_current_loop_and_unbound_entries(self):
        closed = []

        async def aclose(client):
            closed.append(client)

        loop = asyncio.get_running_loop()
        cache = ClientCache(capacity=2, aclose=aclose)
        cache.put(loop, "a")
        cache.put(None, "b")
        await cache.aclear()

        assert sorted(closed) == ["a", "b"]
        assert cache.current_size() == 0

    async def test_clear_from_inside_the_owning_loop_schedules_the_close(self):
        """The sync path can't await the running loop's own client, but must not drop it."""
        closed = []
        done = asyncio.Event()

        async def aclose(client):
            closed.append(client)
            done.set()

        cache = ClientCache(capacity=2, aclose=aclose)
        cache.put(asyncio.get_running_loop(), "a")
        cache.clear()

        assert cache.current_size() == 0
        assert closed == [], "clear() cannot await a client owned by the running loop"
        await asyncio.wait_for(done.wait(), timeout=5.0)
        assert closed == ["a"]

    async def test_aclear_drives_a_stopped_loop_to_close_its_entry(self):
        """A loop that is neither running nor closed can only be driven off-loop."""
        ran_on = []

        async def aclose(client):
            ran_on.append(asyncio.get_running_loop())

        stopped_loop = asyncio.new_event_loop()
        try:
            cache = ClientCache(capacity=2, aclose=aclose)
            cache.put(stopped_loop, "a")
            await cache.aclear()

            # `run_until_complete` from inside this coroutine's loop would raise and be
            # swallowed, leaving the client to GC.
            assert ran_on == [stopped_loop], "the stopped loop's client was not closed"
            assert cache.current_size() == 0
        finally:
            stopped_loop.close()

    async def test_aclear_closes_an_entry_owned_by_a_foreign_loop(self):
        """A client on another thread's loop closes there, without blocking this one."""
        ran_on = []

        async def aclose(client):
            ran_on.append(asyncio.get_running_loop())
            await asyncio.sleep(0.2)  # holds the foreign loop, not ours

        foreign_loop = asyncio.new_event_loop()
        thread = threading.Thread(target=foreign_loop.run_forever, daemon=True)
        thread.start()

        ticks = 0

        async def tick():
            nonlocal ticks
            while True:
                await asyncio.sleep(0.01)
                ticks += 1

        try:
            cache = ClientCache(capacity=2, aclose=aclose)
            cache.put(foreign_loop, "a")

            ticker = asyncio.create_task(tick())
            await cache.aclear()
            ticker.cancel()

            assert ran_on == [foreign_loop]
            assert cache.current_size() == 0
            # A blocking wait would have parked this thread before `tick` ever ran.
            assert ticks > 0, "aclear() blocked its own event loop while closing"
        finally:
            foreign_loop.call_soon_threadsafe(foreign_loop.stop)
            thread.join(timeout=5.0)
            foreign_loop.close()


if __name__ == "__main__":
    pytest.main([__file__])
