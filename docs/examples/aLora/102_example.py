# pytest: skip, huggingface, e2e
# SKIP REASON: Requires user input (blocks on stdin in an infinite loop).
"""Interactive diagnosis loop using a fully custom, non-catalog aLoRA adapter.

Demonstrates loading and calling the custom `stembolts` adapter (Hugging Face:
`nfulton/stembolts`) through the helpers in `stembolts_intrinsic.py`. Each turn
runs the adapter-backed intrinsic with a fresh `ChatContext` and prints the raw
JSON result (`defective_part` and `diag_likelihood`). The loop never exits on its
own; stop it with Ctrl+C.

The example uses Granite 3.3 2B because that is the available `stembolts`
adapter variant; Granite 4.0 and 4.1 variants are not currently available. To
train and upload your own variant, see `README.md` in this directory.

Runs the model locally, so it needs a GPU or Apple Silicon Mac; the model and
adapter are downloaded from Hugging Face on first run.

Prerequisites:
    uv sync --extra hf

Usage:
    uv run python docs/examples/aLora/102_example.py
"""

from stembolts_intrinsic import (
    async_stembolt_failure_analysis,
    stembolt_failure_analysis,
)

from mellea.backends.cache import SimpleLRUCache
from mellea.backends.huggingface import LocalHFBackend
from mellea.stdlib.context import ChatContext

if __name__ == "__main__":
    # nfulton/stembolts currently provides this adapter for Granite 3.3 2B;
    # no Granite 4.0 or 4.1 variant is available in the repository.
    backend = LocalHFBackend(
        model_id="ibm-granite/granite-3.3-2b-instruct", cache=SimpleLRUCache(5)
    )

    welcome_msg = (
        "==   Welcome to the Self-Sealing Stembolt Part Diagnostic System.   =="
    )
    print("=" * len(welcome_msg))
    print("=" * len(welcome_msg))
    print(welcome_msg)
    print("=" * len(welcome_msg))
    print("=" * len(welcome_msg))

    mechanics_notes = None
    while True:
        if mechanics_notes is None:
            mechanics_notes = "Oil seepage when stembolt is oriented diagonally, even when oil pin is tightened"
            print(f"Mechanic: {mechanics_notes}")
        else:
            mechanics_notes = input("Mechanic: ")

        result, ctx = stembolt_failure_analysis(
            notes=mechanics_notes, ctx=ChatContext(), backend=backend
        )
        print(f"Assistant: {result}")
