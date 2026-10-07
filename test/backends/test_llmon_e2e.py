# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""E2E round-trip against a served LLMON-trained model over chat-completions.

An LLMON model's behavior lives in its fine-tuned weights, not a server chat
template, so Mellea drives it through the ordinary OpenAI-compatible chat backend
and lets `LLMONInstruction` own the LLMON work (robust reply parsing on the way
back, optional Machine-LLMON encoding on the way out). This exercises that full
served path end-to-end — the thing the unit tests in `test_llmon_instruction.py`
deliberately do not cover.

Gating (all three must pass, else the module skips cleanly):

1. The `llmon` extra (`pip install "mellea[llmon]"`, import root `lang`) must be
   installed.
2. A real served model must be named. There is **no committed LLMON model yet**,
   so `_LLMON_MODEL_ID` defaults to the `TODOREPLACEME` sentinel below and the
   module skips until one is wired in. Point it at a real tag by exporting
   `LLMON_TEST_MODEL_ID` (and, for the tested local Ollama path,
   `LLMON_TEST_BASE_URL=http://localhost:11434`, `LLMON_TEST_MODEL_ID=llmon95:latest`).
3. The server at `LLMON_TEST_BASE_URL` must be reachable (checked per-module by
   the `llmon_session` fixture).
"""

import os

import pytest

# LLMON support is an optional extra; skip the whole module when it is absent.
pytest.importorskip("lang.json.llmon_converters", reason="install mellea[llmon]")

import requests

import mellea
from mellea.backends.openai import OpenAIBackend
from mellea.core import SamplingResult
from mellea.stdlib.components.llmon_instruction import LLMONInstruction, LLMONResult

# Served endpoint + model. The model id is a placeholder sentinel until a real
# LLMON model is wired into CI: `_is_placeholder` below gates the module off while
# it is still TODOREPLACEME, so the test never silently runs against a non-LLMON
# model. Override via env to run locally (see the module docstring).
#
# TODO(llmon): replace TODOREPLACEME with the canonical served LLMON model id (and
# a reachable default base url) once one is published, then drop the placeholder
# skip below.
_LLMON_BASE_URL = os.environ.get("LLMON_TEST_BASE_URL", "http://localhost:8000")
_LLMON_MODEL_ID = os.environ.get("LLMON_TEST_MODEL_ID", "TODOREPLACEME")


def _is_placeholder(model_id: str) -> bool:
    """True while no real LLMON model has been named (still the sentinel)."""
    return model_id.strip().upper() == "TODOREPLACEME"


pytestmark = [
    pytest.mark.openai,
    pytest.mark.e2e,
    pytest.mark.qualitative,  # asserts on live LLM output, not deterministic
    pytest.mark.skipif(
        _is_placeholder(_LLMON_MODEL_ID),
        reason=(
            "No LLMON model defined — set LLMON_TEST_MODEL_ID (and "
            "LLMON_TEST_BASE_URL) to a served LLMON model. Default is the "
            "TODOREPLACEME placeholder."
        ),
    ),
]


@pytest.fixture(scope="module")
def llmon_session() -> mellea.MelleaSession:
    """A MelleaSession bound to the served LLMON model.

    Skips the module when the endpoint is unreachable so a missing local server is
    a skip, not a hard failure.
    """
    try:
        requests.get(f"{_LLMON_BASE_URL}/v1/models", timeout=2)
    except requests.ConnectionError:
        pytest.skip(
            f"LLMON server not reachable at {_LLMON_BASE_URL}", allow_module_level=True
        )

    return mellea.MelleaSession(
        backend=OpenAIBackend(
            model_id=_LLMON_MODEL_ID, base_url=f"{_LLMON_BASE_URL}/v1", api_key="EMPTY"
        )
    )


def test_served_instruct_round_trip_returns_plain_text(llmon_session):
    """`m.instruct` with `instruction_cls=LLMONInstruction` yields plain text.

    Runs with `llmonize_input=True`, so `format_for_llm` encodes the instruction
    into a Machine-LLMON `instr`/`data` object before it is sent — the wire input
    is LLMON markup, not plain prose. Whether the model replies with raw
    Machine-LLMON or already-detokenized prose, `LLMONInstruction._parse` must
    still land on an `LLMONResult` whose string value is plain text with no LLMON
    markup.
    """
    prompt = "Summarise this customer feedback in one sentence: {{text}}"
    user_variables = {"text": "loved the product, but shipping was very slow"}

    result = llmon_session.instruct(
        prompt,
        requirements=[
            "The summary must be a single sentence.",
            "Include both positive and negative aspects if both are present.",
        ],
        user_variables=user_variables,
        instruction_cls=LLMONInstruction,
        # Encode the input as Machine LLMON on the way out (opt-in; default False).
        instruction_kwargs={"llmonize_input": True},
    )

    thunk = result.result if isinstance(result, SamplingResult) else result
    parsed = thunk.parsed_repr

    # Print the round-trip so `pytest -s` shows what Mellea and LLMON exchanged:
    # the prompt template + user vars the test supplied, the exact chat payload
    # Mellea actually sent on the wire (`generate_log.prompt` — here the
    # LLMON-encoded messages, since llmonize_input=True), the raw bytes the model
    # emitted (possibly Machine LLMON), and the plain text LLMON parsing extracted.
    wire_messages = thunk._generate_log.prompt if thunk._generate_log else None

    print("\n" + "=" * 72)
    print("LLMON ROUND-TRIP")
    print("=" * 72)
    print(f"MODEL:            {_LLMON_MODEL_ID}  @  {_LLMON_BASE_URL}")
    print(f"PROMPT TEMPLATE:  {prompt}")
    print(f"USER VARIABLES:   {user_variables}")
    print("-" * 72)
    print("WIRE INPUT (generate_log.prompt — LLMON-encoded messages sent to model):")
    if isinstance(wire_messages, list):
        for i, msg in enumerate(wire_messages):
            role = msg.get("role", "?") if isinstance(msg, dict) else "?"
            content = msg.get("content", msg) if isinstance(msg, dict) else msg
            print(f"  [{i}] role={role}")
            print(content)
    else:
        print(wire_messages)
    print("-" * 72)
    print("RAW MODEL OUTPUT (thunk.value, pre-parse):")
    print(thunk.value)
    print("-" * 72)
    print("PARSED RESULT (LLMONResult, str()):")
    print(str(parsed))
    print("=" * 72)

    assert isinstance(parsed, LLMONResult)
    text = str(parsed)
    assert text  # non-empty answer
    # The parsed value is always plain text — never raw LLMON markup.
    assert "<|open|>" not in text
