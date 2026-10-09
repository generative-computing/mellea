# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""E2E round-trip against a served LLMON-trained model over chat-completions.

An LLMON model's behavior lives in its fine-tuned weights, not a server chat
template, so Mellea drives it through the ordinary OpenAI-compatible chat backend
with an `LLMONFormatter`: a plain `m.instruct(prompt)` is sent as LLMON built from
every part of the instruction, ending in an `exec` block, and the response the
formatter extracts from the reply becomes the thunk's `value`. This exercises that
full served path end-to-end, which the unit tests in
`test/formatters/test_llmon_formatter.py` and
`test/backends/test_formatter_output_unit.py` do not.

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
from mellea.formatters import LLMONFormatter

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

    The formatter is passed explicitly, so a model named through the environment
    gets LLMON even when it is not a registered LLMON model. Skips the module when
    the endpoint is unreachable so a missing local server is a skip, not a hard
    failure.
    """
    try:
        requests.get(f"{_LLMON_BASE_URL}/v1/models", timeout=2)
    except requests.ConnectionError:
        pytest.skip(
            f"LLMON server not reachable at {_LLMON_BASE_URL}", allow_module_level=True
        )

    return mellea.MelleaSession(
        backend=OpenAIBackend(
            model_id=_LLMON_MODEL_ID,
            formatter=LLMONFormatter(_LLMON_MODEL_ID),
            base_url=f"{_LLMON_BASE_URL}/v1",
            api_key="EMPTY",
        )
    )


def _print_round_trip(
    title: str, model_id: str, prompt: str, user_variables: dict, thunk
) -> None:
    """Print one LLMON round-trip so `pytest -s` shows what crossed the wire.

    Shows the prompt template and user variables the test supplied, the chat
    payload Mellea sent (`generate_log.prompt`, the messages as rendered by
    `LLMONFormatter`), the text the model generated (`raw.text`, possibly with LLMON
    markup), and the response extracted from it (`value`). Values are printed
    directly (not `repr`'d) so embedded newlines render as newlines.
    """
    wire_messages = thunk._generate_log.prompt if thunk._generate_log else None

    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)
    print(f"MODEL:            {model_id}  @  {_LLMON_BASE_URL}")
    print(f"PROMPT TEMPLATE:  {prompt}")
    print(f"USER VARIABLES:   {user_variables}")
    print("-" * 72)
    print("WIRE INPUT (generate_log.prompt — messages sent to model):")
    if isinstance(wire_messages, list):
        for i, msg in enumerate(wire_messages):
            role = msg.get("role", "?") if isinstance(msg, dict) else "?"
            content = msg.get("content", msg) if isinstance(msg, dict) else msg
            print(f"  [{i}] role={role}")
            print(content)
    else:
        print(wire_messages)
    print("-" * 72)
    print("GENERATED TEXT (thunk.raw.text, or thunk.value when nothing was extracted):")
    print(thunk.raw.text if thunk.raw.text is not None else thunk.value)
    print("-" * 72)
    print("EXTRACTED RESPONSE (thunk.value):")
    print(thunk.value)
    print("=" * 72)


def test_served_instruct_round_trip_returns_plain_text(llmon_session):
    """A plain `m.instruct` against an LLMON model sends LLMON and returns plain text.

    The prompt must carry every part of the instruction (description and
    requirements) and an `exec` block. Whether the model replies with raw LLMON or
    already-detokenized prose, the thunk's `value` must be plain text with no LLMON
    markup.
    """
    prompt = "Summarise this customer feedback in one sentence: {{text}}"
    user_variables = {"text": "loved the product, but shipping was very slow"}
    requirements = [
        "The summary must be a single sentence.",
        "Include both positive and negative aspects if both are present.",
    ]

    result = llmon_session.instruct(
        prompt, requirements=requirements, user_variables=user_variables
    )

    thunk = result.result if isinstance(result, SamplingResult) else result

    _print_round_trip(
        "LLMON MODEL — LLMONFormatter round trip",
        _LLMON_MODEL_ID,
        prompt,
        user_variables,
        thunk,
    )

    assert thunk._generate_log is not None
    wire = thunk._generate_log.prompt
    sent = wire[-1]["content"] if isinstance(wire, list) else str(wire)
    for part in [
        "Summarise this customer feedback in one sentence: "
        "loved the product, but shipping was very slow",
        *requirements,
    ]:
        assert part in sent
    assert "exec" in sent

    assert thunk.value  # non-empty answer
    # The response is always plain text, never raw LLMON markup.
    assert "<|open|>" not in thunk.value
