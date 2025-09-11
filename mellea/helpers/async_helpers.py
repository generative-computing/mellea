import asyncio
from collections.abc import AsyncIterator, Coroutine
from typing import Any


async def send_to_queue(
    co: Coroutine[Any, Any, AsyncIterator] | Coroutine[Any, Any, Any],
    aqueue: asyncio.Queue,
) -> None:
    """Processes the output of an async chat request by sending the output to an async queue."""
    aresponse = await co

    if isinstance(aresponse, AsyncIterator):
        async for item in aresponse:
            await aqueue.put(item)

    else:
        await aqueue.put(aresponse)

    # Always add a sentinel value to indicate end of stream.
    await aqueue.put(None)
