# pytest: e2e, ollama

"""Example usage of the uncertainty/certainty intrinsic with Ollama.

Evaluates how certain the model is about its response to a user question.
The context should contain a user question followed by an assistant answer.

Ollama bundles one adapter per model. For this single-adapter example, the
uncertainty aLoRA tag is used for normal chat and the certainty helper, so
Ollama can retain one model identity.

Requires `mellea[switch]` to download the adapter's `io.yaml`.

To run this script from the root of the Mellea source tree, use the command:
```
uv run python docs/examples/intrinsics/uncertainty_ollama.py
```
"""

import os

from mellea import start_backend
from mellea.stdlib import functional as mfuncs
from mellea.stdlib.components.intrinsic import core

ctx, backend = start_backend(
    "ollama",
    # Before its invocation tokens, this bundled aLoRA behaves as the base model.
    model_id=os.environ["MELLEA_OLLAMA_UNCERTAINTY_MODEL"],
    context_type="chat",
    # The certainty helper uses the same model identity.
    adapter_models={"uncertainty": os.environ["MELLEA_OLLAMA_UNCERTAINTY_MODEL"]},
)

# Add the exchange whose answer the adapter will score.
response, ctx = mfuncs.chat("What is 2 + 2?", ctx, backend)  # type: ignore
print(f"Response: {response.content}")

# This call uses the mapped bundled adapter model, not the normal chat model.
result = core.check_certainty(ctx, backend)  # type: ignore
print(f"Certainty score: {result}")
