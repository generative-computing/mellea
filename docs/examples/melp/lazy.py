import asyncio
from mellea.stdlib.span import Span, SimpleComponent
from mellea.stdlib.base import SimpleContext, Context, CBlock, ModelOutputThunk
from mellea.backends import Backend
from mellea.backends.ollama import OllamaModelBackend

backend = OllamaModelBackend("granite4:latest")


async def main(backend: Backend, ctx: Context):
    s1 = CBlock("What is 1+1? Respond with the number only.")
    s1_out, _ = await backend.generate_from_context(action=s1, ctx=SimpleContext())

    s2 = CBlock("What is 2+2? Respond with the number only.")
    s2_out, _ = await backend.generate_from_context(action=s2, ctx=SimpleContext())

    sc1 = SimpleComponent(
        instruction="What is x+y? Respond with the number only", x=s1_out, y=s2_out
    )

    print(await s1_out.avalue())
    print(await s2_out.avalue())

    sc1_out, _ = await backend.generate_from_context(action=sc1, ctx=SimpleContext())

    print(await sc1_out.avalue())


asyncio.run(main(backend, SimpleContext()))
