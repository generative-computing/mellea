import asyncio
from collections.abc import AsyncIterator, Coroutine
from typing import Any


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
