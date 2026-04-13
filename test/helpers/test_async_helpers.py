"""Unit tests for mellea.helpers.async_helpers."""

import asyncio

import pytest

from mellea.core import ModelOutputThunk
from mellea.helpers.async_helpers import (
    ClientCache,
    get_current_event_loop,
    send_to_queue,
    wait_for_all_mots,
)

# --- send_to_queue ---


class TestSendToQueue:
    async def test_coroutine_single_value(self):
        """Coroutine returning a non-iterator value is put into queue followed by sentinel."""

        async def produce():
            return "result"

        q: asyncio.Queue = asyncio.Queue()
        await send_to_queue(produce(), q)
        assert await q.get() == "result"
        assert await q.get() is None  # sentinel

    async def test_coroutine_returning_async_iterator(self):
        """Coroutine returning an async iterator streams items then sentinel."""

        async def produce():
            async def _gen():
                yield "a"
                yield "b"

            return _gen()

        q: asyncio.Queue = asyncio.Queue()
        await send_to_queue(produce(), q)
        assert await q.get() == "a"
        assert await q.get() == "b"
        assert await q.get() is None

    async def test_async_iterator_directly(self):
        """Passing an async iterator (not wrapped in coroutine) streams items."""

        async def _gen():
            yield 1
            yield 2

        q: asyncio.Queue = asyncio.Queue()
        await send_to_queue(_gen(), q)
        assert await q.get() == 1
        assert await q.get() == 2
        assert await q.get() is None

    async def test_exception_propagated_to_queue(self):
        """Exceptions during generation are put into queue instead of raising."""

        async def explode():
            raise ValueError("boom")

        q: asyncio.Queue = asyncio.Queue()
        await send_to_queue(explode(), q)
        item = await q.get()
        assert isinstance(item, ValueError)
        assert str(item) == "boom"

    async def test_iterator_exception_propagated(self):
        """Exception mid-iteration is captured and put into queue."""

        async def _gen():
            yield "ok"
            raise RuntimeError("mid-stream")

        q: asyncio.Queue = asyncio.Queue()
        await send_to_queue(_gen(), q)
        assert await q.get() == "ok"
        item = await q.get()
        assert isinstance(item, RuntimeError)


# --- wait_for_all_mots ---


class TestWaitForAllMots:
    async def test_resolves_all_thunks(self):
        """All pre-computed MOTs are awaited concurrently."""
        mots = [ModelOutputThunk("a"), ModelOutputThunk("b"), ModelOutputThunk("c")]
        await wait_for_all_mots(mots)
        for mot in mots:
            assert mot.value is not None

    async def test_empty_list(self):
        """Empty list is a no-op."""
        await wait_for_all_mots([])


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

    def test_get_missing_returns_none(self):
        cache = ClientCache(capacity=3)
        assert cache.get(999) is None

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

    def test_current_size(self):
        cache = ClientCache(capacity=5)
        assert cache.current_size() == 0
        cache.put(1, "a")
        cache.put(2, "b")
        assert cache.current_size() == 2


if __name__ == "__main__":
    pytest.main([__file__])
