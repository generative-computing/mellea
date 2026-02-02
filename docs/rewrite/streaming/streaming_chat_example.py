import asyncio

from mellea import start_session
from mellea.backends.model_options import ModelOption
from mellea.core.base import CBlock


async def stream_chat(prompt: str) -> str:
    m = start_session()

    mot, _ = await m.backend.generate_from_context(
        action=CBlock(value=prompt), ctx=m.ctx, model_options={ModelOption.STREAM: True}
    )

    while not mot.is_computed():
        print(await mot.astream(), flush=True)

    print("\n\nFINAL ANSWER")
    print(mot.value)

    return str(mot.value)


if __name__ == "__main__":
    asyncio.run(stream_chat("Write a tight 8-line poem about granite and winter."))
