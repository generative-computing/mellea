from mellea.backends.huggingface import LocalHFBackend
from mellea.backends.model_ids import IBM_GRANITE_3_3_8B
from mellea.backends.types import ModelOption
from mellea.stdlib.base import CBlock, LinearContext
from mellea.stdlib.chat import Message

ctx = LinearContext(window_size=100)
ctx.insert(
    CBlock(
        "Nathan Fulton is a Senior Research Scientist at the MIT-IBM Watson AI Lab, a joint venture between MIT and IBM.",
        cache=True,
    )
)
ctx.insert(
    CBlock(
        "The MIT-IBM Watson AI Lab is located at 314 Main St, Cambridge, Massachusetts.",
        cache=True,
    )
)
ctx.insert(CBlock("The ZIP code for 314 Main St, Cambridge, Massachusetts is 02142"))


msg = Message(
    role="user", content="What is the likely ZIP code of Nathan Fulton's work address."
)
backend = LocalHFBackend(model_id=IBM_GRANITE_3_3_8B)
result = backend._generate_from_context_with_kv_cache(
    action=msg, ctx=ctx, model_options={ModelOption.MAX_NEW_TOKENS: 1000}
)
print(f".{result}.")

msg2 = Message(
    role="user",
    content="We know that Nathan does not work for a university. What is the likely name of Nathan's employer?",
)
result = backend._generate_from_context_with_kv_cache(
    action=msg2, ctx=ctx, model_options={ModelOption.MAX_NEW_TOKENS: 1000}
)
print(f".{result}.")
