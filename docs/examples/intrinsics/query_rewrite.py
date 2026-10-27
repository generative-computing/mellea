# pytest: huggingface, e2e
"""Rewriting a follow-up question into a standalone retrieval query with the query-rewrite adapter.

Demonstrates `rag.rewrite_question()`, which uses the conversation history to
rewrite an under-specified last user question (here, "is he more likely to get
fleas?") into a self-contained query suitable for retrieval.

The script uses `LocalHFBackend`. The same adapter function also runs on
`OpenAIBackend` with a Granite Switch model; see `docs/examples/granite-switch/`.
Runs the model locally, so it needs a GPU or Apple Silicon Mac; the model and
adapter are downloaded from Hugging Face on first run.

Prerequisites:
    uv sync --extra hf

Usage:
    uv run python docs/examples/intrinsics/query_rewrite.py
"""

from mellea import model_ids, start_backend
from mellea.stdlib.components import Message
from mellea.stdlib.components.intrinsic import rag

ctx, backend = start_backend(
    "hf", model_id=model_ids.IBM_GRANITE_4_1_3B, context_type="chat"
)
# NOTE: This example can also be run with the OpenAIBackend using a GraniteSwitch model. See docs/examples/granite-switch/.

ctx_with_question = (
    ctx.add(Message("assistant", "Welcome to pet questions!"))
    .add(
        Message(
            "user",
            "I have two pets, a dog named Rex and a cat named Lucy. "
            "Rex spends a lot of time in the backyard and outdoors, "
            "and Lucy is always inside.",
        )
    )
    .add(
        Message(
            "assistant",
            "Sounds good! Rex must love exploring outside, while Lucy "
            "probably enjoys her cozy indoor life.",
        )
    )
    .add(Message("user", "But is he more likely to get fleas because of that?"))
)

print("Original user question: 'But is he more likely to get fleas because of that?'")

result = rag.rewrite_question(None, ctx_with_question, backend)
print(f"Rewritten user question: {result}")
