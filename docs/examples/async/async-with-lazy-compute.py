# pytest: ollama, qualitative, llm

"""Example of how to use async with lazy compute and streaming."""

import asyncio
import re

from mellea.backends.model_options import ModelOption
from mellea.core.base import ModelOutputThunk
from mellea.stdlib.session import start_session

INGREDIENT_RE = re.compile(r"\d+\.\s*(.+)\n")


# Create a regular session. Works with functional interface as well.
m = start_session()


async def main():
    ingredients: ModelOutputThunk[str] = await m.ainstruct(
        "Write a list of ingredients I need for to bake an apple pie.",
        strategy=None,  # Cannot perform lazy compute / top level streaming if using a strategy.
        model_options={
            ModelOption.STREAM: True,  # Set streaming to True for top level streaming.
            ModelOption.SEED: 3,
        },
        # await_result=True  # Set await_result to True to make ingredients computed at return time.
    )

    assert not ingredients.is_computed(), "ingredients should not be computed"

    buffer = ""

    print("Ingredients:")
    # We can stream the response and extract ingredients as they arrive.
    while not ingredients.is_computed():
        buffer += await ingredients.astream()

        # Extract any complete ingredient lines from the buffer.
        for match in INGREDIENT_RE.finditer(buffer):
            print(f"  - {match.group(1)}")

        # Keep only the unmatched tail so we don't re-extract.
        last_newline = buffer.rfind("\n")
        if last_newline != -1:
            buffer = buffer[last_newline + 1 :]


asyncio.run(main())
