# pytest: huggingface, requires_heavy_ram, llm

"""
Example usage of the query rewrite intrinsic for RAG applications.

To run this script from the root of the Mellea source tree, use the command:
```
uv run python docs/examples/intrinsics/query_rewrite.py
```
"""

from mellea.backends.huggingface import LocalHFBackend
from mellea.stdlib.context import ChatContext
from mellea.stdlib.components import Message
from mellea.stdlib.components.intrinsic import rag

backend = LocalHFBackend(model_id="ibm-granite/granite-4.0-micro")
context = (
    ChatContext()
    .add(Message("assistant", "Welcome to pet questions!"))
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

print(f"Original user question: {next_user_turn}")

result = rag.rewrite_question(next_user_turn, context, backend)
print(f"Rewritten user question: {result}")
