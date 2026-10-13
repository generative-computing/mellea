# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""E2E tests on a granite LLMON preview model with `LocalHFBackend`.

The model's chat template writes each turn into its own LLMON blocks. An
instruction's grounding context and requirements are sent as the template's message
fields, so they reach the model as `data` and `req` blocks of their own rather than
nested in the instruction. For tools, Mellea sends the assistant's tool call back as
the turn's `tool_calls` and the tool's result as plain text for the template to
wrap, and the tool-calling test runs that whole loop on the real weights: the model
calls a tool, Mellea parses the call and runs the tool, the result goes back, and
the model answers from it. Each test's answer has to contain a made-up code the
model cannot know, so an answer that contains it came from what Mellea sent.

The served path is covered by `test_llmon_e2e.py`, and the messages Mellea builds by
the unit tests in `test/formatters/test_llmon_formatter.py`.
"""

import re

import pytest

from test.predicates import require_gpu

pytest.importorskip("torch", reason="torch not installed — install mellea[hf]")
# LLMON support is an optional extra (`pip install "mellea[llmon]"`).
pytest.importorskip("lang.json.llmon_converters", reason='install "mellea[llmon]"')

# The weights are float32: about 14 GB.
pytestmark = [pytest.mark.huggingface, pytest.mark.e2e, require_gpu(min_vram_gb=18)]

from mellea import MelleaSession
from mellea.backends import ModelOption, tool
from mellea.backends.huggingface import LocalHFBackend
from mellea.backends.tools import convert_tools_to_json
from mellea.formatters import LLMONFormatter
from mellea.stdlib.components import Message
from mellea.stdlib.context import ChatContext
from mellea.stdlib.functional import call_tools
from test.conftest import hf_skip

_MODEL_ID = "ibm-granite/granite-4.1-3b-llmon-preview"
_CODE = "WH-7Q4Z-LY"


@pytest.fixture(scope="module")
def backend():
    """Shared LLMON preview model for all tests in this module."""
    with hf_skip():
        backend = LocalHFBackend(
            model_id=_MODEL_ID, formatter=LLMONFormatter(_MODEL_ID)
        )
    yield backend

    from test.conftest import cleanup_gpu_backend

    cleanup_gpu_backend(backend, "huggingface-llmon")


@pytest.fixture(scope="function")
def session(backend):
    """Fresh chat session for each test."""
    session = MelleaSession(backend, ctx=ChatContext())
    yield session
    session.reset()


@pytest.mark.qualitative
def test_instruction_answers_from_its_grounding_context(backend, session):
    """An instruction's parts reach the model as the chat template's own blocks, and
    the model answers from its grounding context."""
    grounding = f"The Lyon warehouse code is {_CODE}."
    answer = session.instruct(
        "What is the warehouse code for Lyon?",
        grounding_context={"codes": grounding},
        requirements=["Answer with the code only."],
        # The default strategy validates requirements with an LLM-as-a-judge turn,
        # which LLMONFormatter does not send.
        strategy=None,
        model_options={ModelOption.MAX_NEW_TOKENS: 50},
    )

    assert answer._generate_log is not None
    sent = answer._generate_log.prompt
    assert isinstance(sent, list)
    prompt = backend._tokenizer.apply_chat_template(
        sent, tokenize=False, add_generation_prompt=True
    )
    assert isinstance(prompt, str)
    print(f"\n{prompt}\n\n{answer.value}")
    # One instruction block, the template's own, with no LLMON nested inside it; the
    # grounding context and the requirement are blocks of their own.
    assert prompt.count("<|open|>instr") == 1
    block = r"<\|open\|>{tag}<\|:\|>[^<]+<\|close\|>{text}<\|open_end\|>"
    assert re.search(block.format(tag="data", text=re.escape(grounding)), prompt)
    assert re.search(
        block.format(tag="req", text="Answer with the code only\\."), prompt
    )

    # The model answers from the grounding context.
    assert answer.value is not None
    assert _CODE in answer.value


@pytest.mark.qualitative
def test_tool_call_round_trip_answers_from_the_tool_result(backend, session):
    """The model calls a tool, Mellea runs it and sends the result back, and the
    model answers from it."""

    @tool
    def lookup_warehouse_code(city: str) -> str:
        """Look up the internal warehouse code for a city.

        Args:
            city: The city whose warehouse code to look up.
        """
        return _CODE if "lyon" in city.lower() else "No warehouse in that city."

    options = {
        ModelOption.TOOLS: [lookup_warehouse_code],
        ModelOption.MAX_NEW_TOKENS: 200,
    }

    # The model writes a tool call, which Mellea parses.
    reply = session.act(
        Message("user", "What is the warehouse code for Lyon?"),
        model_options=options,
        tool_calls=True,
    )
    assert reply.tool_calls is not None
    assert [call.name for call in reply.tool_calls] == ["lookup_warehouse_code"]

    # Mellea runs the tool.
    (result,) = call_tools(reply, backend)
    assert result.content == _CODE

    answer = session.act(result, model_options=options, tool_calls=True)

    # The call goes back as the assistant turn's `tool_calls`, not its text, and the
    # chat template writes it out once; the result is wrapped as plain text.
    assert answer._generate_log is not None
    sent = answer._generate_log.prompt
    assert isinstance(sent, list)
    assert sent[-2]["role"] == "assistant"
    assert sent[-2]["content"] == ""
    prompt = backend._tokenizer.apply_chat_template(
        sent,
        tools=convert_tools_to_json({"lookup_warehouse_code": lookup_warehouse_code}),
        tokenize=False,
        add_generation_prompt=True,
    )
    assert isinstance(prompt, str)
    print(f"\n{prompt}\n\n{answer.value}")
    assert prompt.count("<tool_call>") == 1
    assert f"<tool_response>{_CODE}</tool_response>" in prompt

    # The model answers from the result.
    assert answer.value is not None
    assert _CODE in answer.value
