# pytest: e2e, huggingface, skip

"""Example: `m.instruct` against an LLMON-trained model with `LocalHFBackend`.

The granite LLMON models write LLMON in their chat template, from message
fields that only `LocalHFBackend` passes on, so LLMON runs on that backend alone:
other backends raise `ValueError` for an LLMON model. Because the model is a
registered LLMON model, `LocalHFBackend` renders with `LLMONFormatter` by default:
the instruction's grounding context reaches the model as a `data` block of its own,
labeled with its key. The backend decodes the reply without special tokens, so the
thunk's `value` is the model's text, with no LLMON markup.

Requires the `hf` and `llmon` extras (`pip install "mellea[hf,llmon]"`) and access to
the model on Hugging Face. Its float32 weights take about 14 GB.

To run this script from the root of the Mellea source tree:
    uv run python docs/examples/llmon/instruct_round_trip.py
"""

import mellea
from mellea.backends.huggingface import LocalHFBackend
from mellea.stdlib.context import ChatContext

MODEL_ID = "ibm-granite/granite-4.1-3b-llmon-preview"

m = mellea.MelleaSession(backend=LocalHFBackend(model_id=MODEL_ID), ctx=ChatContext())


def summarize_feedback(text: str) -> str:
    """Summarize customer feedback, which the model reads as data."""
    thunk = m.instruct(
        "Summarise this customer feedback in one sentence.",
        grounding_context={"feedback": text},
    )

    # The reply is decoded without special tokens, so value is plain text.
    return thunk.value


if __name__ == "__main__":
    summary = summarize_feedback("loved the product, but shipping was very slow")
    print(f"Summary: {summary}")
