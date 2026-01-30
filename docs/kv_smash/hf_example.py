import asyncio

from mellea.backends import ModelOption
from mellea.backends.huggingface import LocalHFBackend
from mellea.backends.model_ids import IBM_GRANITE_3_3_8B
from mellea.core import CBlock
from mellea.stdlib.components import Message
from mellea.stdlib.context import ChatContext


async def example():
    ctx = ChatContext(window_size=100)
    ctx = ctx.add(
        CBlock(
            "Nathan Fulton is a Senior Research Scientist at the MIT-IBM Watson AI Lab, a joint venture between MIT and IBM.",
            cache=True,
        )
    )
    ctx = ctx.add(
        CBlock(
            "The MIT-IBM Watson AI Lab is located at 314 Main St, Cambridge, Massachusetts.",
            cache=True,
        )
    )
    ctx = ctx.add(
        CBlock("The ZIP code for 314 Main St, Cambridge, Massachusetts is 02142")
    )

    msg = Message(
        role="user",
        content="What is the likely ZIP code of Nathan Fulton's work address?",
    )
    backend = LocalHFBackend(model_id=IBM_GRANITE_3_3_8B)
    mot = await backend._generate_from_context_with_kv_cache(
        action=msg, ctx=ctx, model_options={ModelOption.MAX_NEW_TOKENS: 64}
    )
    # mot = await backend._generate_from_context_standard(
    #     action=msg, ctx=ctx, model_options={ModelOption.MAX_NEW_TOKENS: 10}
    # )

    result = await mot.avalue()
    print(f".{result}.")

    msg2 = Message(
        role="user",
        content="We know that Nathan does not work for a university. What is the likely name of Nathan's employer?",
    )
    mot = await backend._generate_from_context_with_kv_cache(
        action=msg2, ctx=ctx, model_options={ModelOption.MAX_NEW_TOKENS: 64}
    )
    result = await mot.avalue()
    print(f".{result}.")


asyncio.run(example())
