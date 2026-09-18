# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Async helper functions for managing concurrent model output thunks.

Provides `send_to_queue`, which feeds a backend response coroutine or async iterator
into an `asyncio.Queue` (including sentinel and error forwarding); `wait_for_all_mots`,
which gathers multiple `ModelOutputThunk` computations in a single `asyncio.gather`
call; `get_current_event_loop`, a safe wrapper that returns `None` instead of
raising when no event loop is running; and `ClientCache`, a thread-safe LRU cache of
loop-bound async clients that closes each entry on the loop that owns it when the entry
is evicted or cleared. These utilities are used internally by backends that operate in
async contexts.
"""

from __future__ import annotations

import asyncio
import threading
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


def get_current_event_loop() -> asyncio.AbstractEventLoop | None:
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


_pending_closes: set[asyncio.Task] = set()
"""Strong references to in-flight fire-and-forget close tasks.

A task scheduled on a loop nobody awaits is only weakly referenced by that loop, so
without this it can be garbage collected mid-close.
"""


def _log_close_failure(client: Any, future: Any) -> None:
    """Log, at debug, an exception left on a completed fire-and-forget close.

    Args:
        client: The client whose close was scheduled, used for the message.
        future: The finished future or task to inspect. A cancelled one is ignored.
    """
    from ..core import MelleaLogger

    try:
        if future.cancelled():
            return
        exc = future.exception()
    except Exception:  # pragma: no cover - defensive; e.g. a foreign future type
        return
    if exc is not None:
        MelleaLogger.get_logger().debug(
            f"Failed to close {type(client).__name__}: {exc}"
        )


def _close_client_for_loop(
    client: Any,
    loop: asyncio.AbstractEventLoop | None,
    aclose: Callable[[Any], Coroutine[Any, Any, None]],
    *,
    wait: bool = True,
) -> None:
    """Close an async client from synchronous code, on whichever loop owns it.

    Async HTTP clients bind their transports to the event loop that first drove
    them, so `aclose()` has to run on that same loop. Which loop it is decides
    what is possible here:

    - `loop` is `None` (client created outside any loop, so driven by Mellea's own
      background loop): schedule it there, waiting up to `CLIENT_CLOSE_TIMEOUT` only
      if `wait`. This is the path every sync-constructed backend takes on `close()`,
      so the wait is bounded: a background loop busy with a generate call costs a
      timeout, not a hang. A wait that times out leaves the close scheduled rather
      than cancelling it, so the client is still closed on the right loop.
    - `loop` is the loop this call is running inside: the close is scheduled on it as
      a task. Blocking on it from here would deadlock, so `wait` cannot be honoured.
    - `loop` is running on another thread: schedule it there, waiting up to
      `CLIENT_CLOSE_TIMEOUT` only if `wait`.
    - `loop` exists but is not running: drive it with `run_until_complete`. This
      blocks even when `wait` is false — a stopped loop has no thread to hand the
      work to — but a client close does no network I/O, so it returns promptly.
    - `loop` is already closed: nothing can await on it, so the client's sockets
      cannot be reclaimed by this or any other path. Logged at debug and skipped;
      the operating system gets them back when the client is garbage collected.

    Never raises — cleanup runs on teardown paths where a failure to close is not
    worth masking the caller's own outcome.

    Args:
        client: The async client to close.
        loop: The event loop the client is bound to, or `None` if it was created
            outside of one.
        aclose: Callable returning the coroutine that closes `client`, e.g.
            `lambda c: c.aclose()`.
        wait: Whether to block until the close finishes. Pass `False` from paths that
            run on a caller's event loop — an eviction during `generate`, say — where
            a stalled loop matters more than knowing the client is fully closed.
    """
    from ..core import MelleaLogger
    from .event_loop_helper import _schedule_async_in_thread

    logger = MelleaLogger.get_logger()
    try:
        if loop is None:
            background = _schedule_async_in_thread(aclose(client))
            if wait:
                background.result(CLIENT_CLOSE_TIMEOUT)
            else:
                background.add_done_callback(lambda f: _log_close_failure(client, f))
        elif loop.is_closed():
            logger.debug(
                f"Cannot close {type(client).__name__}: the event loop it was bound "
                "to is already closed, so its connections cannot be reclaimed until "
                "the client is garbage collected."
            )
        elif loop.is_running():
            if loop is get_current_event_loop():
                # Blocking on the running loop from inside it would deadlock, so the
                # close is scheduled on it instead of being dropped.
                task = loop.create_task(aclose(client))
                _pending_closes.add(task)
                task.add_done_callback(_pending_closes.discard)
                task.add_done_callback(lambda f: _log_close_failure(client, f))
            else:
                future = asyncio.run_coroutine_threadsafe(aclose(client), loop)
                if wait:
                    future.result(CLIENT_CLOSE_TIMEOUT)
                else:
                    future.add_done_callback(lambda f: _log_close_failure(client, f))
        else:
            loop.run_until_complete(aclose(client))
    except Exception as e:
        logger.debug(f"Failed to close {type(client).__name__}: {e}")


async def _aclose_client_for_loop(
    client: Any,
    loop: asyncio.AbstractEventLoop | None,
    aclose: Callable[[Any], Coroutine[Any, Any, None]],
) -> None:
    """Close an async client from async code, on whichever loop owns it.

    The async counterpart to `_close_client_for_loop`. It waits for the close in every
    case the synchronous helper has to either block a thread or give up on:

    - `client` is bound to the running loop: awaited directly. The synchronous helper
      can only schedule this one, since blocking on the current loop would deadlock.
    - `client` is bound to another running loop, or to Mellea's background loop
      (`loop` is `None`): scheduled there and awaited through a wrapped future, so
      this coroutine's own loop keeps running while the close proceeds.
    - `client` is bound to a loop that is stopped but not closed: only
      `run_until_complete` can drive such a loop, and that cannot run inside this
      coroutine's own loop, so it is handed to a worker thread and awaited there.

    A client bound to an **already-closed** loop is the one case nothing can fix: with
    no loop left to await on, its sockets are reclaimed only when it is garbage
    collected. That is logged at debug and skipped, exactly as in the sync path.

    Never raises, for the same reason `_close_client_for_loop` doesn't.

    Args:
        client: The async client to close.
        loop: The event loop the client is bound to, or `None` if it was created
            outside of one.
        aclose: Callable returning the coroutine that closes `client`, e.g.
            `lambda c: c.aclose()`.
    """
    from ..core import MelleaLogger
    from .event_loop_helper import _schedule_async_in_thread

    logger = MelleaLogger.get_logger()
    try:
        if loop is None:
            await asyncio.wait_for(
                asyncio.wrap_future(_schedule_async_in_thread(aclose(client))),
                CLIENT_CLOSE_TIMEOUT,
            )
        elif loop.is_closed():
            logger.debug(
                f"Cannot close {type(client).__name__}: the event loop it was bound "
                "to is already closed, so its connections cannot be reclaimed until "
                "the client is garbage collected."
            )
        elif loop is get_current_event_loop():
            await aclose(client)
        elif loop.is_running():
            await asyncio.wait_for(
                asyncio.wrap_future(
                    asyncio.run_coroutine_threadsafe(aclose(client), loop)
                ),
                CLIENT_CLOSE_TIMEOUT,
            )
        else:
            # A loop that is neither running nor closed has no thread to schedule on;
            # only `run_until_complete` can drive it, and calling that from inside this
            # coroutine's own running loop raises. Hand it to a worker thread, where
            # no loop is running, so the sync helper can drive it there.
            await asyncio.to_thread(_close_client_for_loop, client, loop, aclose)
    except Exception as e:
        logger.debug(f"Failed to close {type(client).__name__}: {e}")


class ClientCache:
    """A simple [LRU](https://en.wikipedia.org/wiki/Cache_replacement_policies#Least_Recently_Used_(LRU)) cache.

    Used to keep track of clients for backends where the client is tied to a specific
    event loop. Keys are the event loop each client is bound to (or `None` for a client
    created outside any loop), so an entry can be closed on the right loop when it is
    evicted or cleared. Holding the loop object rather than its `id()` also stops a
    recycled address from handing a fresh loop a client bound to a dead one.

    Every method is safe to call from multiple threads. Build clients through
    `get_or_create` rather than a `get`/`put` pair: only `get_or_create` holds the lock
    across the miss and the insert, which is what keeps two threads from each
    constructing a client under the same key.

    Args:
        capacity (int): Maximum number of entries to hold before evicting the least recently used.
        aclose (Callable[[Any], Coroutine[Any, Any, None]] | None): Optional callable
            returning the coroutine that closes a cached client, e.g.
            `lambda c: c.aclose()`. When set, it is scheduled on eviction and awaited
            by `clear()`/`aclear()`; without it, evicted clients keep their
            connections open until garbage collection.

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
        self._lock = threading.Lock()

    def current_size(self) -> int:
        """Just return the size of the key set. This isn't necessarily safe.

        Returns:
            Number of entries currently in the cache.
        """
        return len(self.cache.keys())

    def get(self, key: Hashable) -> Any | None:
        """Gets a value from the cache.

        A hit here is only worth acting on if nothing else can insert under `key`
        meanwhile — use `get_or_create` to build a client on a miss.

        Args:
            key: Cache key; the event loop the client is bound to, or `None`.

        Returns:
            The cached value, or `None` if the key is not present.
        """
        with self._lock:
            return self._get_locked(key)

    def put(self, key: Hashable, value: Any) -> None:
        """Put a value into the cache, closing the least-recently-used entry if one is evicted.

        LRU eviction is the only case that closes anything. Replacing the value under a
        key that is *already* cached just drops the old reference: a same-key `put`
        means another caller built a client for that key between this caller's `get`
        miss and this `put`, so the value displaced here is the one that caller is
        about to use. Closing it would break their in-flight request; its connections
        come back when they drop it and it is garbage collected. `get_or_create` avoids
        that window altogether, and is the preferred way in.

        The evicted client's close is *scheduled* on the loop that owns it rather than
        awaited: entries are added during `generate`, so blocking here would stall the
        caller's event loop for as long as the close took.

        Closing on eviction assumes no more than `capacity` event loops drive a backend
        at once, which holds for Mellea's single-background-loop design. Past that, an
        evicted client's transport could be closed while a request is still in flight on
        its loop. The alternative — dropping the reference and leaving the close to
        garbage collection — is what leaked sockets for the life of the process, so the
        close is scheduled rather than skipped.

        Args:
            key: Cache key; the event loop the client is bound to, or `None`.
            value: Value to store.
        """
        with self._lock:
            evicted = self._put_locked(key, value)
        if evicted is not None:
            self._close_entry(*evicted, wait=False)

    def get_or_create(self, key: Hashable, factory: Callable[[], Any]) -> Any:
        """Get the value cached under `key`, building and caching one if there is none.

        Prefer this to a `get`/`put` pair. `get_current_event_loop()` returns `None`
        whenever no loop is running, so every synchronous call path shares the one `None`
        key: two threads calling a backend synchronously would otherwise both miss, both
        construct a client, and the second `put` would displace the first client without
        closing it — leaking its connections for the life of the process. Holding the
        lock across the miss, the construction and the insert makes the second thread
        find the client the first one built.

        `factory` runs with the lock held, so concurrent construction for the same cache
        is serialised and `factory` must not call back into this cache. If it raises, the
        exception propagates and nothing is cached.

        Args:
            key: Cache key; the event loop the client is bound to, or `None`.
            factory: Zero-argument callable returning a value to cache under `key`.
                Called only on a miss. A `None` it returns is cached but reads back as a
                miss, the same conflation `get` has.

        Returns:
            The value already cached under `key`, or the newly built one.
        """
        with self._lock:
            value = self._get_locked(key)
            if value is not None:
                return value
            value = factory()
            evicted = self._put_locked(key, value)
        if evicted is not None:
            self._close_entry(*evicted, wait=False)
        return value

    def clear(self) -> None:
        """Close every cached client and empty the cache.

        Entries are dropped from the cache before being closed, so a failure to close
        one client cannot leave a stale entry behind.

        Called from inside a running event loop, an entry owned by *that* loop can only
        be scheduled for closing, not awaited — use `aclear()` there to be sure the
        client is closed before this returns.
        """
        with self._lock:
            entries = list(self.cache.items())
            self.cache.clear()
        for key, value in entries:
            self._close_entry(key, value)

    async def aclear(self) -> None:
        """Close every cached client and empty the cache, from async code.

        The async counterpart to `clear()`. Prefer this when the running event loop owns
        one of the cached clients: that client is awaited directly, which `clear()` can
        only schedule, since blocking on its own loop would deadlock.

        A client bound to a loop that is already closed cannot be closed here either —
        there is no loop left to await on. Those entries are dropped and logged at
        debug; their sockets come back on garbage collection.
        """
        with self._lock:
            entries = list(self.cache.items())
            self.cache.clear()
        if self.aclose is None:
            return
        for key, value in entries:
            await _aclose_client_for_loop(value, self._loop_of(key), self.aclose)

    def _get_locked(self, key: Hashable) -> Any | None:
        """Return the value under `key`, marking it most recently used.

        The caller must hold `self._lock`.

        Args:
            key: Cache key; the event loop the client is bound to, or `None`.

        Returns:
            The cached value, or `None` if the key is not present.
        """
        if key not in self.cache:
            return None
        # Move the accessed item to the end (most recent)
        value = self.cache.pop(key)
        self.cache[key] = value
        return value

    def _put_locked(self, key: Hashable, value: Any) -> tuple[Hashable, Any] | None:
        """Insert an entry as most recently used, reporting whatever it evicted.

        The caller must hold `self._lock`. The evicted entry is handed back instead of
        closed here so its close can run with the lock released — closing a client bound
        to a stopped loop blocks, and no other thread should have to wait on that.

        Args:
            key: Cache key; the event loop the client is bound to, or `None`.
            value: Value to store.

        Returns:
            The `(key, value)` pair evicted to make room, or `None` if none was.
        """
        evicted: tuple[Hashable, Any] | None = None
        if key in self.cache:
            # If the key exists, move it to the end (most recent)
            self.cache.pop(key)
        elif len(self.cache) >= self.capacity:
            # If the cache is full, remove the least recently used item
            evicted = self.cache.popitem(last=False)
        # Add the new key-value pair to the end (most recent)
        self.cache[key] = value
        return evicted

    def _close_entry(self, key: Hashable, value: Any, *, wait: bool = True) -> None:
        """Close a single evicted or cleared entry, if a close callback is configured.

        Args:
            key: The entry's cache key.
            value: The client that was stored under `key`.
            wait: Whether to block until the client is closed. `False` on eviction,
                which can happen on a caller's event loop.
        """
        if self.aclose is None:
            return
        _close_client_for_loop(value, self._loop_of(key), self.aclose, wait=wait)

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
