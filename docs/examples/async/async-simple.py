# pytest: ollama, qualitative, llm

"""Example of how to use async with lazy compute and streaming."""

import asyncio

from mellea.backends.model_options import ModelOption
from mellea.core.base import ModelOutputThunk
from mellea.stdlib.session import start_session

# Create a regular session. Works with functional interface as well.
m = start_session()


async def main():
    response: ModelOutputThunk[str] = await m.ainstruct(
        "Say 'We're Streaming Now!' and then add a fun fact!",
        strategy=None,  # Cannot perform lazy compute / top level streaming if using a strategy.
        model_options={
            ModelOption.STREAM: True  # Set streaming to True for top level streaming.
        },
        # await_result=True  # Set await_result to True to prevent lazy compute / top-level streaming.
    )

    # We can stream the response as it happens.
    while not response.is_computed():
        print(await response.astream())


asyncio.run(main())
