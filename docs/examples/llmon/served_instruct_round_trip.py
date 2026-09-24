# pytest: e2e, openai, skip

"""Example: `m.instruct` against an LLMON model served behind chat-completions.

The LLMON model is served with its `chat_template.jinja` installed server-side,
so the *server* llmonizes the prompt (drawing block ids, emitting the
`instr`/`req` blocks, wiring the `exec` graph) and detokenizes the reply. Mellea
sends a plain `{role, content}` message through the ordinary OpenAI-compatible
chat backend — no new backend and no formatter swap.

NOTE: the outgoing message shape is unverified (see LLMON-STATUS.md). Some LLMON
checkpoints expect structured instruction/data/exec fields on the message dict
rather than a flat content string; if the server returns nothing sensible, inspect
its chat_template.jinja to see whether it reads named fields.

`instruction_cls=LLMONInstruction` makes the round-trip robust to a server that
returns *raw* Machine LLMON in the reply: its `_parse` runs `parse_llmon_reply`,
which extracts the assistant block's text when raw tokens are present and passes
already-detokenized prose through unchanged. A plain `m.instruct` (no
`instruction_cls`) also works whenever the server returns detokenized prose.

Requires the `llmon` extra (`pip install "mellea[llmon]"`) and a served LLMON
model reachable at `LLMON_TEST_BASE_URL` (default `http://localhost:8000`).

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
