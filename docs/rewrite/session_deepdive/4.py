import mellea.stdlib.functional as mfuncs
from mellea.stdlib.base import SimpleContext, CBlock, Context
from mellea.backends.ollama import OllamaModelBackend
from mellea.backends import Backend
import asyncio


async def main(backend: Backend, ctx: Context):
    response, next_context = await backend.generate_from_context(
        CBlock("What is 1+1?"),
        ctx=ctx,  # TODO we should rationalize ctx and context acress mfuncs and base/backend.
    )

    print(f"Currently computed: {response.is_computed()}")
    print(await response.avalue())
    print(f"Currently computed: {response.is_computed()}")


asyncio.run(main(OllamaModelBackend("granite4:latest"), SimpleContext()))
