# pytest: e2e, huggingface, skip
"""Run the answerability adapter function through a local Granite Switch checkpoint.

Demonstrates `rag.check_answerability()` against a Granite Switch checkpoint loaded
with `LocalHFBackend(load_embedded_adapters=True)`. The adapters are already baked
into the checkpoint, so no separate adapter weights are downloaded.

Runs the model locally, so it needs a GPU or Apple Silicon Mac; the checkpoint is
downloaded from Hugging Face on first run.

Prerequisites:
    uv sync --extra hf

Usage:
    uv run python docs/examples/granite-switch/answerability_local_hf.py
"""

from mellea.backends.huggingface import LocalHFBackend
from mellea.backends.model_ids import IBM_GRANITE_SWITCH_4_1_3B_PREVIEW
from mellea.stdlib.components import Document, Message
from mellea.stdlib.components.intrinsic import rag
from mellea.stdlib.context import ChatContext

backend = LocalHFBackend(
    model_id=IBM_GRANITE_SWITCH_4_1_3B_PREVIEW, load_embedded_adapters=True
)

context = ChatContext().add(Message("assistant", "Hello! How can I help you?"))
question = "What is the square root of 4?"
documents = [Document("The square root of 4 is 2.")]

result = rag.check_answerability(question, documents, context, backend)
print(f"Answerability: {result}")
