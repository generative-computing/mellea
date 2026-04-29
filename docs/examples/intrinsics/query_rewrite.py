# pytest: huggingface, e2e

"""Example usage of the query rewrite intrinsic for RAG applications.

To run this script from the root of the Mellea source tree, use the command:
```
uv run python docs/examples/intrinsics/query_rewrite.py
```
"""

from mellea import model_ids, start_backend
from mellea.stdlib.components import Message
from mellea.stdlib.components.intrinsic import rag

ctx, backend = start_backend(
    "hf", model_id=model_ids.IBM_GRANITE_4_MICRO_3B, context_type="chat"
)
# NOTE: This example can also be run with the OpenAIBackend using a GraniteSwitch model. See docs/examples/granite-switch/.

ctx = (
    ctx.add(Message("assistant", "Welcome to pet questions!"))
    .add(Message("user", "I have two pets, a dog named Rex and a cat named Lucy."))
    .add(
        Message(
            "assistant",
            "Rex spends a lot of time in the backyard and outdoors, "
            "and Luna is always inside.",
        )
    )
    .add(
        Message(
            "user",
            "Sounds good! Rex must love exploring outside, while Lucy "
            "probably enjoys her cozy indoor life.",
        )
    )
)

next_user_turn = "But is he more likely to get fleas because of that?"
ctx_with_question = ctx.add(Message("user", next_user_turn))

print(f"Original user question: {next_user_turn}")

result = rag.rewrite_question(None, ctx_with_question, backend)
print(f"Rewritten user question: {result}")
