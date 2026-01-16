import mellea.stdlib.functional as mfuncs
from mellea.core import CBlock, Context, Backend
from mellea.stdlib.context import SimpleContext
from mellea.backends.ollama import OllamaModelBackend
import asyncio


async def main(backend: Backend, ctx: Context):
    response, next_context = await mfuncs.aact(
        CBlock("What is 1+1?"), context=ctx, backend=backend
    )

    print(response.value)


asyncio.run(main(OllamaModelBackend("granite4:latest"), SimpleContext()))
