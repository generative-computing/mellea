import mellea.stdlib.functional as mfuncs
from mellea.stdlib.base import SimpleContext, CBlock, Context
from mellea.backends.ollama import OllamaModelBackend
from mellea.backends import Backend
import asyncio


async def main(backend: Backend, ctx: Context):
    response, next_context = await mfuncs.aact(CBlock("What is 1+1?"), 
                                        context=ctx, 
                                        backend=backend)
    
    print(response.value)

asyncio.run(main(OllamaModelBackend("granite4:latest"), SimpleContext()))