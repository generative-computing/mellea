import asyncio
from mellea.core import Context, CBlock, ModelOutputThunk, Backend
from mellea.backends.ollama import OllamaModelBackend
from mellea.stdlib.context import SimpleContext


async def main(backend: Backend, ctx: Context):
    """
    In this example, we show how executing multiple MOTs in parallel should work.
    """
    m_states = "Missouri", "Minnesota", "Montana", "Massachusetts"

    poem_thunks = []
    for state_name in m_states:
        mot, ctx = await backend.generate_from_context(
            CBlock(f"Write a poem about {state_name}"), ctx
        )
        poem_thunks.append(mot)

    # Notice that what we have now is a list of ModelOutputThunks, none of which are computed.
    for poem_thunk in poem_thunks:
        assert type(poem_thunk) == ModelOutputThunk
        print(f"Computed: {poem_thunk.is_computed()}")

    # Let's run all of these in parallel.
    await asyncio.gather(*[c.avalue() for c in poem_thunks])

    # Print out the final results, which are now computed.
    for poem_thunk in poem_thunks:
        print(f"Computed: {poem_thunk.is_computed()}")

    # And let's print out the final results.
    for poem_thunk in poem_thunks:
        print(poem_thunk.value)


backend = OllamaModelBackend(model_id="granite4:latest")
asyncio.run(main(backend, SimpleContext()))
