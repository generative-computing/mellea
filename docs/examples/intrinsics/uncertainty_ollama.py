# pytest: e2e, ollama
"""Example usage of the uncertainty/certainty intrinsic with Ollama.

Evaluates how certain the model is about its response to a user question.
The context should contain a user question followed by an assistant answer.

Ollama bundles one adapter per model. For this single-adapter example, the
uncertainty aLoRA tag is used for normal chat and the certainty helper, so
Ollama can retain one model identity (mapped through `adapter_models`). Unless
`MELLEA_OLLAMA_UNCERTAINTY_MODEL` names an existing tag, the script builds one by
running `test/scripts/build_ollama_uncertainty_adapter.sh` (needs `git`, `uv`, and
network access to download the base model and adapter). If the build fails, the
script prints "SKIPPED" and exits with status 0.

Prerequisites:
    uv sync --extra switch   # downloads the adapter's io.yaml
    ollama serve

Usage:
    uv run python docs/examples/intrinsics/uncertainty_ollama.py
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
    try:
        adapter_model = subprocess.run(
            [builder], check=True, stdout=subprocess.PIPE, text=True
        ).stdout.strip()
    except subprocess.CalledProcessError as exc:
        print(
            "SKIPPED: Ollama does not support the uncertainty aLoRA adapter "
            f"on this installation ({exc})."
        )
        raise SystemExit(0) from exc

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
