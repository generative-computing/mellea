# pytest: ollama, e2e

"""Soft LLMON round-trip for `m.instruct`.

Calls `m.instruct` normally; the session's `LLMONFormatter` llmonizes the
instruction on the way out (id-tagged `instr`/`req` blocks plus an `exec` object),
and passing `instruction_cls=LLMONInstruction` parses the llmonized reply back
into a Component. This is the "soft" path — the model is prompted to emit LLMON
but decoding is not constrained to the grammar.

Two pieces cooperate:

- `LLMONFormatter` on the backend renders every `Instruction` as LLMON on the way
  out, so a plain `m.instruct` is llmonized with no special component.
- `instruction_cls=LLMONInstruction` makes `m.instruct` build an `LLMONInstruction`
  whose `_parse` runs the LLMON parser on the reply and returns an `LLMONResult`
  whose `str(...)` is the reply's output-block text — closing the round-trip.

See `LLMON-FORMATTER-AND-COMPONENT-FOR-SOFT-INSTRUCT-ROUND-TRIP.md` (issue #1623).
"""

import mellea
from mellea.core import SamplingResult
from mellea.formatters import LLMONFormatter
from mellea.stdlib.components.llmon_instruction import LLMONInstruction, LLMONResult

# Construct the session's backend with the LLMON formatter so every instruction is
# rendered as LLMON on the way out.
m = mellea.start_session(
    "ollama", "granite3.3:8b", formatter=LLMONFormatter(model_id="granite3.3:8b")
)


def summarize_feedback(text: str) -> str:
    """Summarize customer feedback via a llmonized instruction, parsed back."""
    result = m.instruct(
        "Summarise this customer feedback in one sentence: {{text}}",
        requirements=[
            "The summary must be a single sentence.",
            "Include both positive and negative aspects if both are present.",
        ],
        user_variables={"text": text},
        # Parse the llmonized reply back into an LLMONResult Component.
        instruction_cls=LLMONInstruction,
    )

    # The default sampling strategy validates the requirements, so `instruct` may
    # return a SamplingResult; its `.result` is the chosen output thunk.
    thunk = result.result if isinstance(result, SamplingResult) else result

    # The reply parses back into an LLMONResult Component (the round-trip target).
    # `thunk.value` is the raw llmonized reply; `thunk.parsed_repr` is the parsed
    # Component, and `str(...)` on it is the extracted output-block text.
    parsed = thunk.parsed_repr
    assert isinstance(parsed, LLMONResult)
    return str(parsed)


if __name__ == "__main__":
    summary = summarize_feedback(
        "The product quality is excellent, but shipping took two weeks."
    )
    print(summary)
