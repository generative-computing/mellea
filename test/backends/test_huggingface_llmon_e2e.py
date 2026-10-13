# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""E2E tests on a granite LLMON preview model with `LocalHFBackend`.

The model's chat template writes each turn into its own LLMON blocks. An
instruction's grounding context and requirements are sent as the template's message
fields, so they reach the model as `data` and `req` blocks of their own rather than
nested in the instruction. A requirement's LLM-as-a-judge check sends the output and
the requirement as `data`, and the validation test runs the judge-and-repair loop on
the real weights. For tools, Mellea sends the assistant's tool call back as
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
from mellea.core import ModelOutputThunk, Requirement
from mellea.formatters import LLMONFormatter
from mellea.stdlib.components import Instruction, Message
from mellea.stdlib.context import ChatContext
from mellea.stdlib.functional import call_tools
from mellea.stdlib.sampling import MultiTurnStrategy
from test.conftest import hf_skip

_MODEL_ID = "ibm-granite/granite-4.1-3b-llmon-preview"
_CODE = "WH-7Q4Z-LY"
_REVIEW = (
    "I bought the X100 blender last month for $89. It is powerful and crushes ice "
    "easily, but it is very loud and the shipping took three weeks."
)
# A chat template block of `tag` holding exactly `text` (a regex), with any id.
_BLOCK = r"<\|open\|>{tag}<\|:\|>[^<]+<\|close\|>{text}<\|open_end\|>"


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
        # Only the generation is under test; validation has a test of its own.
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
    assert re.search(_BLOCK.format(tag="data", text=re.escape(grounding)), prompt)
    assert re.search(
        _BLOCK.format(tag="req", text="Answer with the code only\\."), prompt
    )

    # The model answers from the grounding context.
    assert answer.value is not None
    assert _CODE in answer.value


@pytest.mark.qualitative
@pytest.mark.parametrize(
    ("summary", "meets"),
    [
        ("The X100 is powerful but loud, and shipping took three weeks.", True),
        ("The X100 blender is powerful and crushes ice easily, but it is loud.", False),
    ],
    ids=["meets", "misses"],
)
async def test_judge_verdict_matches_the_output(backend, summary, meets):
    """The LLM-as-a-judge check sees the output and the requirement as data, and
    says whether the output meets the requirement."""
    requirement = Requirement("Mention the shipping time.")
    instruction = Instruction(
        "Summarize this customer review.",
        grounding_context={"review": _REVIEW},
        requirements=[requirement],
    )
    ctx = ChatContext().add(instruction).add(ModelOutputThunk(summary))

    verdict = await requirement.validate(backend, ctx)

    assert verdict.thunk is not None and verdict.thunk._generate_log is not None
    sent = verdict.thunk._generate_log.prompt
    assert isinstance(sent, list)
    prompt = backend._tokenizer.apply_chat_template(
        sent, tokenize=False, add_generation_prompt=True
    )
    assert isinstance(prompt, str)
    print(f"\n{prompt}\n\n{verdict.thunk.value}")
    # The judge's turn: the output and the requirement are data, not a `req` block,
    # which would constrain the judge's own reply.
    judge_turn = prompt[prompt.rfind("<|open|>instr") :]
    for text in (f"Output: {summary}", "Requirement: Mention the shipping time."):
        assert re.search(_BLOCK.format(tag="data", text=re.escape(text)), judge_turn)
    assert "<|open|>req" not in judge_turn

    assert verdict.as_bool() is meets


@pytest.mark.qualitative
def test_missed_requirement_is_repaired(session):
    """The judge catches a requirement the model missed, and a repair turn gets the
    model to meet it."""
    result = session.instruct(
        "Summarize this customer review.",
        grounding_context={"review": _REVIEW},
        # The model's first summary names the product, as the review does.
        requirements=['Refer to the product only as "the device".'],
        # The default strategy re-sends the same prompt, which greedy decoding answers
        # the same way; this one shows the model its miss and the failed requirement.
        strategy=MultiTurnStrategy(loop_budget=3),
        return_sampling_results=True,
        model_options={ModelOption.MAX_NEW_TOKENS: 120},
    )
    attempts = [g.value or "" for g in result.sample_generations]
    print(f"\n{attempts}")

    def meets(text: str) -> bool:
        lowered = text.lower()
        return "device" in lowered and not any(
            name in lowered for name in ("blender", "x100")
        )

    assert not meets(attempts[0]), "the model met the requirement on its first try"
    (_, first_verdict) = result.sample_validations[0][0]
    assert not first_verdict.as_bool()
    assert result.success
    assert meets(attempts[-1])


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
