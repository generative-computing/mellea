# pytest: e2e, openai, skip

"""Example: `m.instruct` against an LLMON-trained model over chat-completions.

The LLMON behavior lives in the model's fine-tuned weights, not a server chat
template (verified against `llmon95:latest`, whose Ollama template is a bare
passthrough — see LLMON-SUMMARY.md). Mellea sends its message through the ordinary
OpenAI-compatible chat backend — no new backend and no formatter swap.

The LLMON round-trip is automatic and model-driven: because the served model id is
in Mellea's allowlist (`llmon_models.is_llmon_model`), a plain `m.instruct(prompt)`
is encoded into a Machine-LLMON `instr`/`data`/`req` object on the way out (in
`Backend.generate_from_context`) and the reply is stripped back to plain text on
the way in (in `ModelOutputThunk._set_parsed_repr`). No `instruction_cls` and no
per-call flag are needed; `thunk.parsed_repr` is always plain text, never markup.

(To opt into LLMON hard-mode output grammar, pass
`instruction_cls=LLMONInstruction, instruction_kwargs={"constrain_output": True}`;
the automatic encode/strip still apply.)

Requires the `llmon` extra (`pip install "mellea[llmon]"`) and a served LLMON
model reachable at `LLMON_TEST_BASE_URL` (default `http://localhost:8000`; for the
tested Ollama model use `http://localhost:11434` and `llmon95:latest`).

To run this script from the root of the Mellea source tree:
    uv run python docs/examples/llmon/served_instruct_round_trip.py
"""

import os
import sys

import requests

BASE_URL = os.environ.get("LLMON_TEST_BASE_URL", "http://localhost:8000")
try:
    requests.get(f"{BASE_URL}/v1/models", timeout=2)
except requests.ConnectionError:
    # Detected by docs/examples/conftest.py subprocess runner as a skip.
    print(f"Skipped: LLMON server not reachable at {BASE_URL}", file=sys.stderr)
    raise SystemExit(1)

import mellea
from mellea.backends.openai import OpenAIBackend
from mellea.core import SamplingResult

MODEL_ID = os.environ.get("LLMON_TEST_MODEL_ID", "llmon95:latest")

m = mellea.MelleaSession(
    backend=OpenAIBackend(model_id=MODEL_ID, base_url=f"{BASE_URL}/v1", api_key="EMPTY")
)


def summarize_feedback(text: str) -> str:
    """Summarize customer feedback via the automatic LLMON round-trip."""
    result = m.instruct(
        "Summarise this customer feedback in one sentence: {{text}}",
        requirements=[
            "The summary must be a single sentence.",
            "Include both positive and negative aspects if both are present.",
        ],
        user_variables={"text": text},
    )

    # The default sampling strategy validates the requirements, so `instruct` may
    # return a SamplingResult; its `.result` is the chosen output thunk.
    thunk = result.result if isinstance(result, SamplingResult) else result

    # For an LLMON model, parsed_repr is the detokenized plain text (never markup).
    return str(thunk.parsed_repr)


if __name__ == "__main__":
    summary = summarize_feedback("loved the product, but shipping was very slow")
    print(f"Summary: {summary}")
