"""Async helper functions."""

import asyncio
from collections import OrderedDict
from collections.abc import AsyncIterator, Coroutine
from typing import Any, TypeVar

from mellea.stdlib.base import ModelOutputThunk


async def send_to_queue(
    co: Coroutine[Any, Any, AsyncIterator | Any] | AsyncIterator, aqueue: asyncio.Queue
) -> None:
    """Processes the output of an async chat request by sending the output to an async queue."""
    try:
        if isinstance(co, Coroutine):
            aresponse = await co
        else:
            # Some backends (hf) don't actually return their iterator from an
            # async function. As a result, there's no coroutine to wait for here.
            aresponse = co

        if isinstance(aresponse, AsyncIterator):
            async for item in aresponse:
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


async def wait_for_all_mots(mots: list[ModelOutputThunk]):
    """Helper function to make waiting for multiple ModelOutputThunks to be computed easier.

    All ModelOutputThunks must be from the same event loop. This should always be the case in sampling
    functions, session functions, and top-level mellea functions.
    """
    coroutines: list[Coroutine[Any, Any, str]] = []
    for mot in mots:
        coroutines.append(mot.avalue())

    await asyncio.gather(*coroutines)


def get_current_event_loop() -> None | asyncio.AbstractEventLoop:
    """Get the current event loop without having to catch exceptions."""
    loop = None
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        pass
    return loop


class ClientCache:
    """A simple [LRU](https://en.wikipedia.org/wiki/Cache_replacement_policies#Least_Recently_Used_(LRU)) cache.

    Used to keep track of clients for backends where the client is tied to a specific event loop.
    """

    def __init__(self, capacity: int):
        """Initializes the LRU cache with a certain capacity.

        The `ClientCache` either contains a value or it doesn't.
        """
        self.capacity = capacity
        self.cache: OrderedDict = OrderedDict()

    def current_size(self):
        """Just return the size of the key set. This isn't necessarily safe."""
        return len(self.cache.keys())

    def get(self, key: int) -> Any | None:
        """Gets a value from the cache."""
        if key not in self.cache:
            return None
        else:
            # Move the accessed item to the end (most recent)
            value = self.cache.pop(key)
            self.cache[key] = value
            return value

    def put(self, key: int, value: Any):
        """Put a value into the cache."""
        if key in self.cache:
            # If the key exists, move it to the end (most recent)
            self.cache.pop(key)
        elif len(self.cache) >= self.capacity:
            # If the cache is full, remove the least recently used item
            self.cache.popitem(last=False)
        # Add the new key-value pair to the end (most recent)
        self.cache[key] = value
