# pytest: ollama, e2e

"""Example of streaming a response with the `async for` iterator.

Demonstrates:
- Consuming a streamed response with `async for delta in response`
- Using `async with` so an early exit cancels the in-flight generation

Prerequisites:
    ollama serve
    ollama pull granite4.2:3b

Usage:
    uv run python docs/examples/async/async-iterator.py
"""

import asyncio

from mellea.backends.model_options import ModelOption
from mellea.core.base import ModelOutputThunk
from mellea.stdlib.session import start_session

# Create a regular session. Works with functional interface as well.
m = start_session()


async def main() -> None:
    response: ModelOutputThunk[str] = await m.ainstruct(
        "Say 'We're Streaming Now!' and then add a fun fact!",
        strategy=None,  # Cannot perform lazy compute / top level streaming if using a strategy.
        model_options={
            ModelOption.STREAM: True  # Set streaming to True for top level streaming.
        },
    )

    # Iterate the thunk to receive each delta as it arrives. `async with` cancels
    # the generation if we leave the loop early.
    async with response:
        async for delta in response:
            print(delta)


asyncio.run(main())
