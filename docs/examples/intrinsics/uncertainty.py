# pytest: huggingface, e2e
"""Example usage of the uncertainty/certainty intrinsic.

Evaluates how certain the model is about its response to a user question.
The context should contain a user question followed by an assistant answer.

For Ollama, see `uncertainty_ollama.py`.

Runs the model locally, so it needs a GPU or Apple Silicon Mac; the model and
adapter are downloaded from Hugging Face on first run.

Prerequisites:
    uv sync --extra hf

Usage:
    uv run python docs/examples/intrinsics/uncertainty.py
"""

from mellea import model_ids, start_backend
from mellea.stdlib import functional as mfuncs
from mellea.stdlib.components.intrinsic import core

ctx, backend = start_backend(
    "hf", model_id=model_ids.IBM_GRANITE_4_1_3B, context_type="chat"
)
# NOTE: This example can also be run with the OpenAIBackend using a GraniteSwitch model. See docs/examples/granite-switch/.

response, ctx = mfuncs.chat("What is 2 + 2?", ctx, backend)  # type: ignore
print(f"Response: {response.content}")

result = core.check_certainty(ctx, backend)  # type: ignore
print(f"Certainty score: {result}")
