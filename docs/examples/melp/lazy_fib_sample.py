# pytest: ollama, llm

import asyncio
from mellea.core import Context, CBlock, ModelOutputThunk

from mellea.stdlib.components import SimpleComponent
from mellea.stdlib.context import SimpleContext

from mellea.core import Backend
from mellea.backends.ollama import OllamaModelBackend
from typing import Tuple

backend = OllamaModelBackend("granite4:latest")


async def _fib_sample(
    backend: Backend, ctx: Context, x: CBlock, y: CBlock
) -> ModelOutputThunk | None:
    sc = SimpleComponent(
        instruction="What is x+y? Respond with the number only.", x=x, y=y
    )
    answer_mot, _ = await backend.generate_from_context(action=sc, ctx=SimpleContext())

    # This is a fundamental thing: it means computation must occur.
    # We need to be able to read this off at c.g. construction time.
    value = await answer_mot.avalue()

    try:
        int(value)
        return answer_mot
    except:
        return None


async def fib_sampling_version(
    backend: Backend, ctx: Context, x: CBlock, y: CBlock
) -> ModelOutputThunk | None:
    for i in range(5):
        sample = await _fib_sample(backend, ctx, x, y)
        if sample is not None:
            return sample
        else:
            continue
    return None


async def fib_sampling_version_main(backend: Backend, ctx: Context):
    fibs = []
    for i in range(20):
        if i == 0 or i == 1:
            fibs.append(CBlock(f"{i}"))
        else:
            mot = await fib_sampling_version(backend, ctx, fibs[i - 1], fibs[i - 2])
            fibs.append(mot)

    for x_i, x in enumerate(fibs):
        match x:
            case ModelOutputThunk():
                n = await x.avalue()
                print(n)
            case CBlock():
                print(x.value)


asyncio.run(fib_sampling_version_main(backend, SimpleContext()))
