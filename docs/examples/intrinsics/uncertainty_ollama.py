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
import subprocess
from pathlib import Path

from mellea import start_backend
from mellea.stdlib import functional as mfuncs
from mellea.stdlib.components.intrinsic import core

adapter_model = os.environ.get("MELLEA_OLLAMA_UNCERTAINTY_MODEL")
if adapter_model is None:
    builder = (
        Path(__file__).parents[3] / "test/scripts/build_ollama_uncertainty_adapter.sh"
    )
    adapter_model = subprocess.run(
        [builder], check=True, stdout=subprocess.PIPE, text=True
    ).stdout.strip()

ctx, backend = start_backend(
    "ollama",
    # Before its invocation tokens, this bundled aLoRA behaves as the base model.
    model_id=adapter_model,
    # The bundled tag cannot identify the Hugging Face adapter directory itself.
    adapter_base_model_name="granite-4.1-3b",
    context_type="chat",
    # The certainty helper uses the same model identity.
    adapter_models={"uncertainty": adapter_model},
)

# Add the exchange whose answer the adapter will score.
response, ctx = mfuncs.chat("What is 2 + 2?", ctx, backend)  # type: ignore
print(f"Response: {response.content}")

# This call uses the mapped bundled adapter model, not the normal chat model.
result = core.check_certainty(ctx, backend)  # type: ignore
print(f"Certainty score: {result}")
