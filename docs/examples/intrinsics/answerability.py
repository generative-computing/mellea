# pytest: huggingface, e2e
"""Checking whether documents can answer a question with the answerability adapter.

Demonstrates `rag.check_answerability()`, which scores whether a set of retrieved
documents contains enough information to answer a question. The script runs the
check twice: once with a document that contains the answer and once with a
document that does not.

The script uses `LocalHFBackend`. The same adapter function also runs on
`OpenAIBackend` with a Granite Switch model; see `docs/examples/granite-switch/`.
Runs the model locally, so it needs a GPU or Apple Silicon Mac; the model and
adapter are downloaded from Hugging Face on first run.

Prerequisites:
    uv sync --extra hf

Usage:
    uv run python docs/examples/intrinsics/answerability.py
"""

from mellea import model_ids, start_backend
from mellea.stdlib.components.intrinsic import rag

ctx, backend = start_backend(
    "hf", model_id=model_ids.IBM_GRANITE_4_1_3B, context_type="chat"
)
# NOTE: This example can also be run with the OpenAIBackend using a GraniteSwitch model. See docs/examples/granite-switch/.

result = rag.check_answerability(
    "What is the square root of 4?",
    documents=["The square root of 4 is 2."],
    context=ctx,
    backend=backend,
)
print(f"Result of answerability check when answer is in documents: {result}")

result = rag.check_answerability(
    "What is the square root of 4?",
    documents=["The square root of 8 is not 2."],
    context=ctx,
    backend=backend,
)
print(f"Result of answerability check when answer is not in documents: {result}")
