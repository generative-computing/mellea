import asyncio

from mellea.backends.ollama import OllamaModelBackend
from mellea.core import Backend, CBlock, Context
from mellea.stdlib.context import SimpleContext


async def main(backend: Backend, ctx: Context):
    # This is not actually an async function; the computation ends immediately. It must be awaited because we create the thunk.
    # TODO clean up the above comment.
    response, _next_context = await backend.generate_from_context(
        CBlock("What is 1+1?"),
        ctx=ctx,  # TODO we should rationalize ctx and context acress mfuncs and base/backend.
    )

    print(f"Currently computed: {response.is_computed()}")
    print(await response.avalue())
    print(f"Currently computed: {response.is_computed()}")


asyncio.run(main(OllamaModelBackend("granite4:latest"), SimpleContext()))
