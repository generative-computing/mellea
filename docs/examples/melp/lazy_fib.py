# pytest: ollama, llm

import asyncio

from mellea.backends.ollama import OllamaModelBackend
from mellea.core import Backend, CBlock, Context, ModelOutputThunk
from mellea.stdlib.components import SimpleComponent
from mellea.stdlib.context import SimpleContext

backend = OllamaModelBackend("granite4:latest")


async def fib(backend: Backend, ctx: Context, x: CBlock, y: CBlock) -> ModelOutputThunk:
    sc = SimpleComponent(
        instruction="What is x+y? Respond with the number only.", x=x, y=y
    )
    mot, _ = await backend.generate_from_context(action=sc, ctx=SimpleContext())
    return mot


async def fib_main(backend: Backend, ctx: Context):
    fibs = []
    for i in range(20):
        if i == 0 or i == 1:
            fibs.append(CBlock(f"{i}"))
        else:
            mot = await fib(backend, ctx, fibs[i - 1], fibs[i - 2])
            fibs.append(mot)

    print(await fibs[-1].avalue())  # type: ignore[attr-defined]
    # for x in fibs:
    #     match x:
    #         case ModelOutputThunk():
    #             n = await x.avalue()
    #             print(n)
    #         case CBlock():
    #             print(x.value)


asyncio.run(fib_main(backend, SimpleContext()))
