# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Async helper functions for managing concurrent model output thunks.

Provides `send_to_queue`, which feeds a backend response coroutine or async iterator
into an `asyncio.Queue` (including sentinel and error forwarding); `wait_for_all_mots`,
which gathers multiple `ModelOutputThunk` computations in a single `asyncio.gather`
call; `get_current_event_loop`, a safe wrapper that returns `None` instead of
raising when no event loop is running; and `close_client_for_loop` /
`aclose_client_for_loop`, which close a loop-bound async client on the loop that owns
it. These utilities are used internally by backends that operate in async contexts.
"""

from __future__ import annotations

import asyncio
import time
from collections import OrderedDict
from collections.abc import AsyncIterator, Callable, Coroutine, Hashable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..core import ModelOutputThunk

CLIENT_CLOSE_TIMEOUT: float = 5.0
"""Seconds to wait for an async client to finish closing before giving up."""


DEFAULT_CHUNK_TIMEOUT: float = 120.0
"""Default per-chunk timeout for streaming responses, in seconds.

This value applies to every chunk including the first (time-to-first-token).
Slow local inference (large models on CPU, heavily queued servers) can take
well over 60 s before producing the first token — set `ModelOption.STREAM_TIMEOUT`
to a higher value or `None` for those deployments.

This timeout only activates when the backend returns an `AsyncIterator` (i.e.
streaming responses). Non-streaming coroutines that resolve to a plain response
object bypass the per-chunk loop entirely and are unaffected by this value.
"""


async def send_to_queue(
    co: Coroutine[Any, Any, AsyncIterator | Any] | AsyncIterator,
    mot: ModelOutputThunk,
    *,
    chunk_timeout: float | None = DEFAULT_CHUNK_TIMEOUT,
    on_timeout: Callable[[], None] | None = None,
) -> None:
    """Processes the output of an async chat request by sending the output to the thunk's queue.

    Args:
        co: A coroutine or async iterator producing the backend response.
        mot: The `ModelOutputThunk` this stream feeds. Its `_gen.queue` receives the
            results — a sentinel `None` is appended on normal completion; an exception
            instance (including `TimeoutError`) is appended on error (a timeout does
            **not** append a trailing sentinel — the exception item is the stream
            terminator). Time-to-first-byte is stamped on the thunk at first-chunk
            receipt via its `_record_ttfb()`, and per-chunk receipt intervals are
            recorded on `_gen.chunk_intervals`.
        chunk_timeout: Maximum seconds to wait for each chunk from the backend iterator,
            including the first (time-to-first-token). Only applies when the backend
            response is an `AsyncIterator`; non-streaming coroutines are unaffected.
            If no chunk arrives within this window a `TimeoutError` is forwarded to
            the queue and the stream is aborted. `None` disables the timeout.
            Defaults to `DEFAULT_CHUNK_TIMEOUT` (120 s). Note that `0` sets the
            deadline to "now" and aborts immediately — use `None` to disable.
        on_timeout: Optional non-blocking callback invoked when this function's
            per-chunk guard expires. Backends may use it to cooperatively stop work
            running outside the event loop. Callback errors are logged and suppressed
            so the original stream timeout remains the consumer-visible error.

    Raises:
        TimeoutError: Re-raised verbatim when the backend itself raises `TimeoutError`
            (i.e. the timeout did not originate from *this* function's per-chunk guard).
            Stream-guard timeouts are forwarded into the queue rather than raised.
    """
    aqueue = mot._gen.queue
    try:
        if isinstance(co, Coroutine):
            aresponse = await co
        else:
            # Some backends (hf) don't actually return their iterator from an
            # async function. As a result, there's no coroutine to wait for here.
            aresponse = co

        if isinstance(aresponse, AsyncIterator):
            ait = aiter(aresponse)
            prev_receipt: float | None = None
            while True:
                cm: asyncio.Timeout | None = None
                try:
                    async with asyncio.timeout(chunk_timeout) as cm:
                        item = await anext(ait)
                except StopAsyncIteration:
                    break
                except TimeoutError:
                    if cm is None or not cm.expired():
                        raise  # backend's own TimeoutError — forward verbatim
                    if on_timeout is not None:
                        try:
                            on_timeout()
                        except Exception as e:
                            from ..core import MelleaLogger

                            MelleaLogger.get_logger().debug(
                                f"Failed to notify backend of stream timeout: {e}"
                            )
                    await aqueue.put(
                        TimeoutError(
                            f"Stream timed out after {chunk_timeout}s without a chunk "
                            "(covers time-to-first-token and inter-chunk gaps). "
                            "Set ModelOption.STREAM_TIMEOUT to a larger value or None to disable."
                        )
                    )
                    close = getattr(ait, "aclose", None) or getattr(ait, "close", None)
                    if close is not None:
                        from ..core import MelleaLogger

                        try:
                            result = close()
                            if asyncio.iscoroutine(result):
                                await result
                        except Exception as e:
                            MelleaLogger.get_logger().debug(
                                f"Failed to close stalled stream iterator: {e}"
                            )
                    return
                mot._record_ttfb()  # idempotent: only the first chunk records TTFB
                now = time.perf_counter()
                mot._gen.chunk_intervals.append(
                    (now - prev_receipt) * 1000 if prev_receipt is not None else None
                )
                prev_receipt = now
                await aqueue.put(item)
        else:
            await aqueue.put(aresponse)

        # Always add a sentinel value to indicate end of stream.
        await aqueue.put(None)

    # Typically, nothing awaits this function directly (only through the queue).
    # As a result, we have to be careful about catching all errors and propagating
    # them to the queue.
    except Exception as e:
        await aqueue.put(e)


async def wait_for_all_mots(mots: list[ModelOutputThunk]) -> None:
    """Helper function to make waiting for multiple ModelOutputThunks to be computed easier.

    All ModelOutputThunks must be from the same event loop. This should always be the case in sampling
    functions, session functions, and top-level mellea functions.

    Args:
        mots: List of `ModelOutputThunk` objects to await concurrently.
    """
    coroutines: list[Coroutine[Any, Any, str]] = []
    for mot in mots:
        coroutines.append(mot.avalue())

    await asyncio.gather(*coroutines)


def get_current_event_loop() -> None | asyncio.AbstractEventLoop:
    """Get the current event loop without having to catch exceptions.

    Returns:
        The running event loop, or `None` if no loop is running.
    """
    loop = None
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        pass
    return loop


def close_client_for_loop(
    client: Any,
    loop: asyncio.AbstractEventLoop | None,
    aclose: Callable[[Any], Coroutine[Any, Any, None]],
) -> None:
    """Close an async client from synchronous code, on whichever loop owns it.

    Async HTTP clients bind their transports to the event loop that first drove
    them, so `aclose()` has to run on that same loop. Which loop it is decides
    what is possible here:

    - `loop` is `None` (client never used, so never bound): run the coroutine on
      Mellea's own background loop.
    - `loop` is running on another thread: schedule it there and wait up to
      `CLIENT_CLOSE_TIMEOUT`.
    - `loop` exists but was never started: drive it with `run_until_complete`.
    - `loop` is already closed: nothing can await on it, so the client's sockets
      cannot be reclaimed. Logged at debug and skipped.

    Never raises — cleanup runs on teardown paths where a failure to close is not
    worth masking the caller's own outcome.

    Args:
        client: The async client to close.
        loop: The event loop the client is bound to, or `None` if it was never used.
        aclose: Callable returning the coroutine that closes `client`, e.g.
            `lambda c: c.aclose()`.
    """
    from ..core import MelleaLogger
    from .event_loop_helper import _run_async_in_thread

    logger = MelleaLogger.get_logger()
    try:
        if loop is None:
            _run_async_in_thread(aclose(client))
        elif loop.is_closed():
            logger.debug(
                f"Cannot close {type(client).__name__}: the event loop it was bound "
                "to is already closed, so its connections cannot be reclaimed."
            )
        elif loop.is_running():
            if loop is get_current_event_loop():
                # Blocking on the running loop from inside it would deadlock.
                logger.debug(
                    f"Cannot close {type(client).__name__} synchronously from inside "
                    "its own event loop; use the async close path instead."
                )
            else:
                asyncio.run_coroutine_threadsafe(aclose(client), loop).result(
                    CLIENT_CLOSE_TIMEOUT
                )
        else:
            loop.run_until_complete(aclose(client))
    except Exception as e:
        logger.debug(f"Failed to close {type(client).__name__}: {e}")


async def aclose_client_for_loop(
    client: Any,
    loop: asyncio.AbstractEventLoop | None,
    aclose: Callable[[Any], Coroutine[Any, Any, None]],
) -> None:
    """Close an async client from async code, on whichever loop owns it.

    The async counterpart to `close_client_for_loop`. When `client` is bound to the
    running loop it is awaited directly — the one case the synchronous helper cannot
    handle, since blocking on the current loop from inside it would deadlock. Any
    other loop is delegated to `close_client_for_loop`.

    Never raises, for the same reason `close_client_for_loop` doesn't.

    Args:
        client: The async client to close.
        loop: The event loop the client is bound to, or `None` if it was never used.
        aclose: Callable returning the coroutine that closes `client`, e.g.
            `lambda c: c.aclose()`.
    """
    current = get_current_event_loop()
    if loop is not None and loop is current:
        from ..core import MelleaLogger

        try:
            await aclose(client)
        except Exception as e:
            MelleaLogger.get_logger().debug(
                f"Failed to close {type(client).__name__}: {e}"
            )
        return
    close_client_for_loop(client, loop, aclose)


class ClientCache:
    """A simple [LRU](https://en.wikipedia.org/wiki/Cache_replacement_policies#Least_Recently_Used_(LRU)) cache.

    Used to keep track of clients for backends where the client is tied to a specific
    event loop. Keys are the event loop each client is bound to (or `None` for a client
    created outside any loop), so an entry can be closed on the right loop when it is
    evicted or cleared. Holding the loop object rather than its `id()` also stops a
    recycled address from handing a fresh loop a client bound to a dead one.

    Args:
        capacity (int): Maximum number of entries to hold before evicting the least recently used.
        aclose (Callable[[Any], Coroutine[Any, Any, None]] | None): Optional callable
            returning the coroutine that closes a cached client, e.g.
            `lambda c: c.aclose()`. When set, it is invoked on eviction and by
            `clear()`; without it, evicted clients keep their connections open until
            garbage collection.

    Attributes:
        cache (OrderedDict): Ordered dictionary storing cached key-value pairs in LRU
            order; always initialised empty at construction.
    """

    def __init__(
        self,
        capacity: int,
        *,
        aclose: Callable[[Any], Coroutine[Any, Any, None]] | None = None,
    ):
        """Initialize the client LRU cache with the given capacity and optional close callback."""
        self.capacity = capacity
        self.cache: OrderedDict = OrderedDict()
        self.aclose = aclose

    def current_size(self) -> int:
        """Just return the size of the key set. This isn't necessarily safe.

        Returns:
            Number of entries currently in the cache.
        """
        return len(self.cache.keys())

    def get(self, key: Hashable) -> Any | None:
        """Gets a value from the cache.

        Args:
            key: Cache key; the event loop the client is bound to, or `None`.

        Returns:
            The cached value, or `None` if the key is not present.
        """
        if key not in self.cache:
            return None
        else:
            # Move the accessed item to the end (most recent)
            value = self.cache.pop(key)
            self.cache[key] = value
            return value

    def put(self, key: Hashable, value: Any) -> None:
        """Put a value into the cache, closing the evicted entry if one is displaced.

        Args:
            key: Cache key; the event loop the client is bound to, or `None`.
            value: Value to store.
        """
        if key in self.cache:
            # If the key exists, move it to the end (most recent)
            self.cache.pop(key)
        elif len(self.cache) >= self.capacity:
            # If the cache is full, remove the least recently used item
            evicted_key, evicted_value = self.cache.popitem(last=False)
            self._close_entry(evicted_key, evicted_value)
        # Add the new key-value pair to the end (most recent)
        self.cache[key] = value

    def clear(self) -> None:
        """Close every cached client and empty the cache.

        Entries are dropped from the cache before being closed, so a failure to close
        one client cannot leave a stale entry behind.
        """
        entries = list(self.cache.items())
        self.cache.clear()
        for key, value in entries:
            self._close_entry(key, value)

    async def aclear(self) -> None:
        """Close every cached client and empty the cache, from async code.

        The async counterpart to `clear()`. Prefer this when a running event loop owns
        one of the cached clients: that client is awaited directly, which `clear()`
        cannot do without deadlocking on its own loop.
        """
        entries = list(self.cache.items())
        self.cache.clear()
        if self.aclose is None:
            return
        for key, value in entries:
            await aclose_client_for_loop(value, self._loop_of(key), self.aclose)

    def _close_entry(self, key: Hashable, value: Any) -> None:
        """Close a single evicted or cleared entry, if a close callback is configured.

        Args:
            key: The entry's cache key.
            value: The client that was stored under `key`.
        """
        if self.aclose is None:
            return
        close_client_for_loop(value, self._loop_of(key), self.aclose)

    @staticmethod
    def _loop_of(key: Hashable) -> asyncio.AbstractEventLoop | None:
        """Return the event loop a cache key refers to, or `None` if it isn't one.

        Keys are normally loops (or `None`), but `ClientCache` is public and accepts
        any hashable, so a non-loop key is treated as unbound rather than an error.

        Args:
            key: The cache key to interpret.

        Returns:
            The event loop `key` refers to, or `None`.
        """
        return key if isinstance(key, asyncio.AbstractEventLoop) else None
