# pytest: e2e, openai, skip

"""Example: `m.instruct` against an LLMON-trained model over chat-completions.

The LLMON behavior lives in the model's fine-tuned weights, not a server chat
template (verified against `llmon95:latest`, whose Ollama template is a bare
passthrough). Mellea sends its message through the ordinary OpenAI-compatible chat
backend — no new backend.

Because `llmon95:latest` is a registered LLMON model, the backend renders with
`LLMONFormatter` by default: a plain `m.instruct(prompt)` is sent as its instruction
text, and the reply is extracted back to plain text, which becomes the thunk's
`value`. The text the model generated is kept on `thunk.raw.text`. An instruction's
requirements and grounding context reach an LLMON model only as its chat template's
message fields, which OpenAI-compatible servers drop, so this example sends the
instruction alone. To use another LLMON model, pass
`formatter=LLMONFormatter(model_id)` to the backend.

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

MODEL_ID = os.environ.get("LLMON_TEST_MODEL_ID", "llmon95:latest")

m = mellea.MelleaSession(
    backend=OpenAIBackend(model_id=MODEL_ID, base_url=f"{BASE_URL}/v1", api_key="EMPTY")
)


def summarize_feedback(text: str) -> str:
    """Summarize customer feedback via the automatic LLMON round-trip."""
    thunk = m.instruct(
        "Summarise this customer feedback in one sentence: {{text}}",
        user_variables={"text": text},
    )

    # For an LLMON model, value is the extracted reply (never markup).
    return thunk.value


if __name__ == "__main__":
    summary = summarize_feedback("loved the product, but shipping was very slow")
    print(f"Summary: {summary}")
