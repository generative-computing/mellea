import asyncio
from mellea.core import Context, CBlock, ModelOutputThunk

from mellea.stdlib.components import SimpleComponent
from mellea.stdlib.context import SimpleContext

from mellea.core import Backend
from mellea.backends.ollama import OllamaModelBackend

backend = OllamaModelBackend("granite4:latest")


async def fib(backend: Backend, ctx: Context, x: CBlock, y: CBlock) -> ModelOutputThunk:
    sc = SimpleComponent(
        instruction="What is x+y? Respond with the number only.", x=x, y=y
    )
    mot, _ = await backend.generate_from_context(action=sc, ctx=SimpleContext())
    return mot


async def main(backend: Backend, ctx: Context):
    fibs = []
    for i in range(100):
        if i == 0 or i == 1:
            fibs.append(CBlock(f"{i}"))
        else:
            fibs.append(await fib(backend, ctx, fibs[i - 1], fibs[i - 2]))

    for x in fibs:
        match x:
            case ModelOutputThunk():
                print(await x.avalue())
            case CBlock():
                print(x.value)


asyncio.run(main(backend, SimpleContext()))
