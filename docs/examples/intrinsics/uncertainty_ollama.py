# pytest: e2e, ollama, qualitative

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

from mellea import model_ids, start_backend
from mellea.stdlib import functional as mfuncs
from mellea.stdlib.components.intrinsic import core

ctx, backend = start_backend(
    "ollama",
    model_id=model_ids.IBM_GRANITE_4_1_3B,
    context_type="chat",
    adapter_models={"uncertainty": "gabegoodhart/granite4.1-uncertainty:3b"},
)

response, ctx = mfuncs.chat("What is 2 + 2?", ctx, backend)  # type: ignore
print(f"Response: {response.content}")

result = core.check_certainty(ctx, backend)  # type: ignore
print(f"Certainty score: {result}")
