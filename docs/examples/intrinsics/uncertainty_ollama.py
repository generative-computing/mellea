# pytest: e2e, ollama

"""Example usage of the uncertainty/certainty intrinsic with Ollama.

Evaluates how certain the model is about its response to a user question.
The context should contain a user question followed by an assistant answer.

Ollama bundles one adapter per model, so the uncertainty adapter is served by
its own model tag (`granite4.1:3b` plus the uncertainty aLoRA). Pass that tag
via `adapter_models`; normal chat still uses the base model.

Requires `mellea[switch]` to download the adapter's `io.yaml`.

To run this script from the root of the Mellea source tree, use the command:
```
uv run python docs/examples/intrinsics/uncertainty_ollama.py
```
"""

import os

from mellea import model_ids, start_backend
from mellea.stdlib import functional as mfuncs
from mellea.stdlib.components.intrinsic import core

ctx, backend = start_backend(
    "ollama",
    # Regular chat uses this base model.
    model_id=model_ids.IBM_GRANITE_4_1_3B,
    context_type="chat",
    # The certainty helper routes only its adapter call to this bundled aLoRA model.
    adapter_models={"uncertainty": os.environ["MELLEA_OLLAMA_UNCERTAINTY_MODEL"]},
)

# Add the exchange whose answer the adapter will score.
response, ctx = mfuncs.chat("What is 2 + 2?", ctx, backend)  # type: ignore
print(f"Response: {response.content}")

# This call uses the mapped bundled adapter model, not the normal chat model.
result = core.check_certainty(ctx, backend)  # type: ignore
print(f"Certainty score: {result}")
