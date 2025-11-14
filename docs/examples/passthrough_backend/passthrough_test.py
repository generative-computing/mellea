import asyncio
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice

from mellea.backends.passthrough import PassthroughBackend
from mellea.stdlib.base import CBlock, ChatContext

def test_generate(conversation: list[dict], opt1: int, opt2: bool) -> ChatCompletion:
    return ChatCompletion(
        id="random",
        created=12345,
        model="model",
        object="chat.completion",
        choices=[
            Choice(
                finish_reason="stop",
                index=0,
                message=ChatCompletionMessage(
                    role="assistant",
                    content=f"response with opts: [opt1: {opt1}, opt2: {opt2}]"
                )
            )
        ]
    )

async def test_generate_from_raw(prompts: list[str]) -> list[str]:
    responses = []
    for prompt in prompts:
        responses.append(f"result of {prompt}")
    return responses


# TODO: JAL. Fix typing here.
b = PassthroughBackend(generate=test_generate, generate_raw=test_generate_from_raw)

async def gen_test():
    out, ctx = b.generate_from_context(CBlock("hello"), ctx=ChatContext(), model_options={"opt1": 1, "opt2": False})
    assert out.value is None
    await out.avalue()
    assert out.value is not None
    print(out)

asyncio.run(gen_test())

print(b.generate_from_raw([CBlock("1"), CBlock("2"), CBlock("3")], ctx=ChatContext()))