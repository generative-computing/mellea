import mellea.stdlib.functional as mfuncs
from mellea.stdlib.base import SimpleContext, CBlock, Context, SimpleComponent
from mellea.backends.ollama import OllamaModelBackend
from mellea.backends import Backend
import asyncio


async def main(backend: Backend, ctx: Context):
    x, _ = await backend.generate_from_context(CBlock("What is 1+1?"), ctx=ctx)

    y, _ = await backend.generate_from_context(CBlock("What is 2+2?"), ctx=ctx)

    response, _ = await backend.generate_from_context(
        SimpleComponent(instruction="What is x+y?", x=x, y=y),
        ctx=ctx # TODO we should rationalize ctx and context acress mfuncs and base/backend.
    )

    print(f"x currently computed: {x.is_computed()}")
    print(f"y currently computed: {y.is_computed()}")
    print(f"response currently computed: {response.is_computed()}")
    print(await response.avalue())
    print(f"response currently computed: {response.is_computed()}")


asyncio.run(main(OllamaModelBackend("granite4:latest"), SimpleContext()))