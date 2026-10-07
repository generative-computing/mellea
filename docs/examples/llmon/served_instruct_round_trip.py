# pytest: e2e, openai, skip

"""Example: `m.instruct` against an LLMON-trained model over chat-completions.

The LLMON behavior lives in the model's fine-tuned weights, not a server chat
template (verified against `llmon95:latest`, whose Ollama template is a bare
passthrough — see LLMON-SUMMARY.md). Mellea sends its message through the ordinary
OpenAI-compatible chat backend — no new backend and no formatter swap — and
`LLMONInstruction` owns the LLMON work on the Mellea side.

Reply-in: `instruction_cls=LLMONInstruction` makes the round-trip robust to a
model that returns *raw* Machine LLMON. Its `_parse` runs `parse_llmon_reply`,
which extracts the assistant block's plain text when raw tokens are present
(robust to partial/garbled fragments) and passes already-plain prose through
unchanged. The returned value is always plain text, never markup.

Out-direction (optional): `llmonize_input=True` on the instruction encodes it as a
Machine-LLMON `instr`/`data` object so the model gets explicit instruction/data
boundaries, instead of the plain prose sent by default. It is off here; flip it to
compare. This example drives it via `instruction_cls`, which uses the default
(plain) input path.

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
from mellea.stdlib.components.llmon_instruction import LLMONInstruction, LLMONResult

MODEL_ID = os.environ.get("LLMON_TEST_MODEL_ID", "my-llmon-model")

m = mellea.MelleaSession(
    backend=OpenAIBackend(model_id=MODEL_ID, base_url=f"{BASE_URL}/v1", api_key="EMPTY")
)


def summarize_feedback(text: str) -> str:
    """Summarize customer feedback via the served LLMON round-trip."""
    result = m.instruct(
        "Summarise this customer feedback in one sentence: {{text}}",
        requirements=[
            "The summary must be a single sentence.",
            "Include both positive and negative aspects if both are present.",
        ],
        user_variables={"text": text},
        # Parse any raw Machine-LLMON reply back into an LLMONResult Component;
        # already-detokenized prose passes through unchanged.
        instruction_cls=LLMONInstruction,
    )

    # The default sampling strategy validates the requirements, so `instruct` may
    # return a SamplingResult; its `.result` is the chosen output thunk.
    thunk = result.result if isinstance(result, SamplingResult) else result

    parsed = thunk.parsed_repr
    assert isinstance(parsed, LLMONResult)
    return str(parsed)


if __name__ == "__main__":
    summary = summarize_feedback("loved the product, but shipping was very slow")
    print(f"Summary: {summary}")
