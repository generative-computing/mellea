import asyncio

import mellea.stdlib.functional as mfuncs
from mellea.backends.ollama import OllamaModelBackend
from mellea.core import Backend, Context
from mellea.stdlib.components import Instruction
from mellea.stdlib.context import SimpleContext


async def main(backend: Backend, ctx: Context):
    response, _next_context = await mfuncs.aact(
        action=Instruction("What is 1+1?"),
        context=ctx,
        backend=backend,
        await_result=True,
    )

    print(response.value)


asyncio.run(main(OllamaModelBackend("granite4:latest"), SimpleContext()))
