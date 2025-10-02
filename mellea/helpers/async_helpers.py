"""Async helper functions."""

import asyncio
from collections.abc import AsyncIterator, Coroutine
from typing import Any

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
