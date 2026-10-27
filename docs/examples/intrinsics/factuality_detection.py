# pytest: huggingface, e2e
"""Detecting a factually incorrect response with the factuality-detection adapter.

Demonstrates `guardian.factuality_detection()`, which returns "yes" if an assistant
response is factually incorrect given the supporting document attached to the
context, and "no" otherwise. The document can be attached to the context or passed
explicitly with `documents=`.

The script uses `LocalHFBackend`. The same adapter function also runs on
`OpenAIBackend` with a Granite Switch model; see `docs/examples/granite-switch/`.
Runs the model locally, so it needs a GPU or Apple Silicon Mac; the model and
adapter are downloaded from Hugging Face on first run.

Prerequisites:
    uv sync --extra hf

Usage:
    uv run python docs/examples/intrinsics/factuality_detection.py
"""

from mellea import model_ids, start_backend
from mellea.stdlib.components import Document, Message
from mellea.stdlib.components.intrinsic import guardian

user_text = "Is Ozzy Osbourne still alive?"
response_text = "Yes, Ozzy Osbourne is alive in 2025 and preparing for another world tour, continuing to amaze fans with his energy and resilience."

document = Document(
    # Context says Ozzy Osbourne is dead, but the response says he is alive.
    "Ozzy Osbourne passed away on July 22, 2025, at the age of 76 from a heart attack. "
    "He died at his home in Buckinghamshire, England, with contributing conditions "
    "including coronary artery disease and Parkinson's disease. His final "
    "performance took place earlier that month in Birmingham."
)

# Create the backend.
ctx, backend = start_backend(
    "hf", model_id=model_ids.IBM_GRANITE_4_1_3B, context_type="chat"
)
# NOTE: This example can also be run with the OpenAIBackend using a GraniteSwitch model. See docs/examples/granite-switch/.

ctx = (
    ctx.add(document)
    .add(Message("user", user_text))
    .add(Message("assistant", response_text))
)

# string "yes" or "no"
result = guardian.factuality_detection(ctx, backend)
# Alternatively, instead of attaching the document to the context above, pass it
# explicitly via documents= (replaces any docs on the last assistant turn):
#   result = guardian.factuality_detection(ctx, backend, documents=[document])
print(f"Result of factuality detection: {result}")
