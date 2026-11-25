# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""`@learnable` on `AdapterFunction`: the student's input is the call, rendered.

A learnable function's record holds the call as the student backend renders
it, taken before the body runs; `sft_rows` / `preference_rows` train on it. So
the prompt is the function of the arguments whatever the teacher does: it may
generate from another conversation, or make no model call at all, and its
privileged arguments never reach the request. Serving
(`fn.policy(adapter_student(backend))`) sends that same call.
"""

import json
import os
from typing import Literal
from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch

import pytest
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice
from openai.types.completion_usage import CompletionUsage
from pydantic import BaseModel

from mellea.backends import ModelOption
from mellea.backends.adapters._core import (
    Adapter as _AdapterCore,
    EmbeddedBinding,
    Identity,
)
from mellea.backends.adapters.catalog import AdapterType
from mellea.backends.adapters.io_contracts import get_io_contract
from mellea.backends.openai import OpenAIBackend
from mellea.backends.tools import MelleaTool
from mellea.core import Backend, ComputedModelOutputThunk, GenerateLog, ModelOutputThunk
from mellea.formatters import TemplateFormatter
from mellea.formatters.granite.intrinsics.input import IntrinsicsRewriter
from mellea.stdlib import functional as mfuncs
from mellea.stdlib.components import Intrinsic, Message
from mellea.stdlib.context import ChatContext
from mellea.stdlib.learnable import (
    AdapterFunction,
    Recorder,
    Score,
    adapter_student,
    capture,
    learnable,
    preference_rows,
    read_rows,
    sft_rows,
)

STATE = {
    "type": "function",
    "function": {
        "name": "get_state",
        "description": "The order's status now.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
}
get_state = MelleaTool("get_state", dict, STATE)

TEACHER_INSTRUCTIONS = "You are the writer. SECRET-INSTRUCTIONS. Write his line."


class _TeacherBackend(Backend):
    """Answers from a list, recording what it was sent (as real backends do)."""

    def __init__(self, replies: list[str]) -> None:
        self._model_id = "teacher"
        self._provider = "teacher"
        self.replies = list(replies)
        self.formatter = TemplateFormatter(model_id="teacher")
        self.model_options: dict = {}

    async def _generate_from_context(
        self, action, ctx, *, format=None, model_options=None, tool_calls=False
    ):
        mot = ModelOutputThunk(value=self.replies.pop(0))
        mot._call.context = ctx.view_for_generation() or []
        mot._call.action = action
        mot._call.model_options = model_options
        mot._generate_log = GenerateLog()
        return mot, ctx.add(action).add(mot)

    async def _generate_from_raw(self, actions, ctx, **kwargs):
        raise NotImplementedError


def _student() -> OpenAIBackend:
    """A backend for rendering only: no request is sent."""
    return OpenAIBackend(
        model_id="granite-switch",
        api_key="fake-key",
        base_url="http://localhost:9999/v1",
    )


def _chat() -> ChatContext:
    return (
        ChatContext()
        .add(Message("system", "You are a support agent."))
        .add(Message("user", "where is my order"))
    )


def _completion(content: str) -> ChatCompletion:
    return ChatCompletion(
        id="t",
        created=0,
        model="granite-switch",
        object="chat.completion",
        choices=[
            Choice(
                index=0,
                finish_reason="stop",
                message=ChatCompletionMessage(role="assistant", content=content),
            )
        ],
        usage=CompletionUsage(prompt_tokens=10, completion_tokens=2, total_tokens=12),
    )


def _serving_backend(fn: AdapterFunction) -> OpenAIBackend:
    """A backend with `fn`'s adapter registered, its io.yaml `fn`'s own."""
    backend = _student()
    adapter = _AdapterCore(
        identity=Identity(name=fn.name, adapter_type="lora", capability=fn.name),
        io_contract=get_io_contract(fn.name),
        weights=EmbeddedBinding(),
    )
    backend.add_adapter(adapter, config=fn.io_config())
    return backend


def _mocked(content: str):
    """A patch of the OpenAI client, and the create mock it answers with."""
    create = AsyncMock(return_value=_completion(content))
    client = MagicMock()
    client.chat.completions.create = create
    return (
        patch.object(
            OpenAIBackend,
            "_async_client",
            new_callable=PropertyMock,
            return_value=client,
        ),
        create,
    )


# --- the developer's view ---------------------------------------------------


@learnable(tools=[get_state], privileged=("question",))
def answer(chat: ChatContext, *, question: str) -> str:
    """His answer to what his partner just said."""
    teacher = _TeacherBackend([f"Shipped. ({question})"])
    # The writer reads its own instructions and the privileged question, never
    # the student's conversation.
    ctx = ChatContext().add(Message("system", TEACHER_INSTRUCTIONS))
    out, _ = mfuncs.act(Message("user", question), ctx, teacher, strategy=None)
    return [out]


@learnable
def move(chat: ChatContext) -> Literal["fire", "forward", "turn"]:
    """His move this tic."""
    # No model call: an RL policy's choice, as hand-built thunks with scores.
    words = ["fire", "forward", "turn"]
    probs = [0.7, 0.2, 0.1]
    return (
        [ComputedModelOutputThunk(ModelOutputThunk(w)) for w in words],
        [Score(value=p, keep=i == 0) for i, p in enumerate(probs)],
    )


@learnable(privileged=("secret",))
def plan(chat: ChatContext, *, goal: str, secret: str) -> str:
    """The plan toward `goal`."""
    return [ComputedModelOutputThunk(ModelOutputThunk(f"go to {goal}"))]


# --- 1. a teacher generating from a different conversation ------------------


def test_prompt_is_the_rendered_argument_not_the_teachers_conversation():
    student = _student()
    with Recorder(student=student) as rec:
        got = answer(_chat(), question="where is my order, really")
    assert got == "Shipped. (where is my order, really)"
    (r,) = rec.records
    want = answer.render(student, {"chat": _chat()})
    assert r.request == want
    assert want.messages == [
        {"role": "system", "content": "You are a support agent."},
        {"role": "user", "content": "where is my order"},
    ]
    # The teacher's trace is its own conversation.
    assert r.thunks[0].context[0]["content"] == TEACHER_INSTRUCTIONS
    (row,) = sft_rows([r.to_json()], answer)
    assert row["prompt"] == want.messages
    assert row["completion"][0]["content"] == got
    assert "SECRET-INSTRUCTIONS" not in json.dumps(row["prompt"])


# --- 2. a teacher with no model call ----------------------------------------


def test_prompt_of_a_teacher_with_no_model_call():
    student = _student()
    with Recorder(student=student) as rec:
        assert move(_chat()) == "fire"
    (r,) = rec.records
    assert r.thunks[0].context == []  # no backend produced it: no trace
    assert r.request is not None
    assert r.request.messages == answer.render(student, {"chat": _chat()}).messages
    assert r.request.constraint == {"labels": ["fire", "forward", "turn"]}
    assert r.request.tools is None and r.request.tool_choice is None
    (row,) = sft_rows([r.to_json()], move)
    assert row["prompt"] == r.request.messages
    assert row["completion"] == [{"role": "assistant", "content": "fire"}]
    prefs = preference_rows([r.to_json()], move)
    assert [p["rejected"][0]["content"] for p in prefs] == ["forward", "turn"]
    assert all(p["prompt"] == r.request.messages for p in prefs)


# --- 3. privileged values never appear in the request -----------------------


def test_privileged_values_never_reach_the_request():
    student = _student()
    with Recorder(student=student) as rec:
        assert plan(_chat(), goal="the BFG", secret="SECRET-VALUE") == "go to the BFG"
    (r,) = rec.records
    assert r.arguments["secret"] == "SECRET-VALUE"  # recorded, by name
    sent = json.dumps(r.request.to_json())
    assert "SECRET-VALUE" not in sent and "secret" not in sent
    assert r.request.messages[-1] == {
        "role": "user",
        "content": json.dumps({"goal": "the BFG"}, sort_keys=True),
    }
    (row,) = sft_rows([r.to_json()], plan)
    assert "SECRET-VALUE" not in json.dumps(row["prompt"])


def test_rows_recorded_without_a_student_backend_cannot_be_trained_on():
    with Recorder() as rec:
        move(_chat())
    (r,) = rec.records
    assert r.request is None
    with pytest.raises(ValueError, match="student backend"):
        sft_rows([r.to_json()], move)


def test_capture_records_the_request(tmp_path):
    out = tmp_path / "rows.jsonl.gz"
    student = _student()
    summary = capture([("e0", lambda: move(_chat()))], out, student=student, log=None)
    assert summary["calls"] == 1
    (row,) = read_rows(out)
    assert row["request"] == move.render(student, {"chat": _chat()}).to_json()


# --- 4. tools reach the row and the served request --------------------------


def test_tools_reach_the_row_and_the_served_request():
    student = _student()
    with Recorder(student=student) as rec:
        answer(_chat(), question="where?")
    (r,) = rec.records
    assert r.request.tools == [STATE]
    assert r.request.tool_choice == "none"
    (row,) = sft_rows([r.to_json()], answer)
    assert row["tools"] == [STATE]

    config = answer.io_config()
    assert config["tools"] == [STATE] and config["tool_choice"] == "none"
    assert config["output"] == {"type": "text"}
    IntrinsicsRewriter(config_dict=config)  # a valid io.yaml

    backend = _serving_backend(answer)
    mock, create = _mocked("Shipped.")
    with mock, answer.policy(adapter_student(backend)):
        assert answer(_chat(), question=None) == "Shipped."
    sent = create.call_args.kwargs
    assert sent["tools"] == [STATE] and sent["tool_choice"] == "none"
    assert sent["messages"] == r.request.messages
    assert sent["extra_body"]["chat_template_kwargs"]["adapter_name"] == answer.name


def test_io_yaml_tools_serve_a_call_that_passes_none():
    backend = _serving_backend(answer)
    mock, create = _mocked("Shipped.")
    with mock:
        mot, _ = mfuncs.act(
            Intrinsic(answer.name, adapter_types=(AdapterType.LORA,)),
            _chat(),
            backend,
            tool_calls=True,
            strategy=None,
        )
    assert mot.value == "Shipped."
    assert create.call_args.kwargs["tools"] == [STATE]
    assert create.call_args.kwargs["tool_choice"] == "none"


def test_a_json_output_is_parsed():
    class Order(BaseModel):
        where: str

    @learnable
    def order(chat: ChatContext) -> Order:
        """Where to go."""
        return [ComputedModelOutputThunk(ModelOutputThunk('{"where": "the BFG"}'))]

    assert order(_chat()) == Order(where="the BFG")
    config = order.io_config()
    assert json.loads(config["response_format"]) == Order.model_json_schema()
    assert order.render(_student(), {"chat": _chat()}).constraint == {
        "json_schema": Order.model_json_schema()
    }


def test_the_teacher_fires_the_invocation_hook():
    seen = []
    with patch(
        "mellea.backends.adapters.adapter._fire_invocation_complete",
        side_effect=lambda **kw: seen.append(kw),
    ):
        move(_chat())
    assert seen == [
        {
            "name": move.name,
            "revision": None,
            "binding_type": "teacher",
            "adapter_type": "teacher",
            "outcome": "success",
            "error": None,
        }
    ]


# --- 5. render_request is what _generate_from_context sends -----------------


async def test_render_request_equals_what_the_intrinsic_path_sends():
    backend = _serving_backend(answer)
    action, ctx, opts = answer.call({"chat": _chat()})
    want = backend.render_request(action, ctx, model_options=opts, tool_calls=True)
    mock, create = _mocked("Shipped.")
    with mock:
        mot, _ = await mfuncs.aact(
            action, ctx, backend, model_options=opts, tool_calls=True, strategy=None
        )
        await mot.avalue()
    sent = create.call_args.kwargs
    assert sent["messages"] == want.messages
    assert sent["tools"] == want.tools
    assert sent["tool_choice"] == want.tool_choice
    assert want.adapter == answer.name


async def test_render_request_equals_what_the_chat_path_sends():
    backend = _student()
    ctx = _chat()
    action = Message("user", "and deaths?")
    opts = {ModelOption.TOOLS: [get_state], ModelOption.SYSTEM_PROMPT: "Be brief."}
    want = backend.render_request(action, ctx, model_options=opts, tool_calls=True)
    mock, create = _mocked("Three.")
    with mock:
        mot, _ = await mfuncs.aact(
            action, ctx, backend, model_options=opts, tool_calls=True, strategy=None
        )
        await mot.avalue()
    sent = create.call_args.kwargs
    assert sent["messages"] == want.messages
    assert sent["tools"] == want.tools == [STATE]
    assert want.messages[0] == {"role": "system", "content": "Be brief."}
    assert want.adapter is None


# --- 6. round trip: the training prompt is what the served adapter receives -


@pytest.mark.e2e
@pytest.mark.openai
@pytest.mark.vllm
@pytest.mark.skipif(
    not os.environ.get("MELLEA_TEST_VLLM_URL"),
    reason="needs MELLEA_TEST_VLLM_URL: a vLLM server's /v1 URL, serving "
    "MELLEA_TEST_VLLM_MODEL (a Granite Switch checkpoint, whose embedded "
    "adapter MELLEA_TEST_VLLM_ADAPTER is served)",
)
def test_sft_prompt_tokenizes_as_the_served_request():
    """`sft_rows` rendered with the base tokenizer equals what vLLM tokenizes
    for the request the adapter is served with."""
    import requests
    from transformers import AutoTokenizer

    url = os.environ["MELLEA_TEST_VLLM_URL"].rstrip("/")
    model = os.environ["MELLEA_TEST_VLLM_MODEL"]
    adapter = os.environ.get("MELLEA_TEST_VLLM_ADAPTER", answer.name)
    tok = AutoTokenizer.from_pretrained(model)

    student = OpenAIBackend(model_id=model, api_key="unused", base_url=url)
    with Recorder(student=student) as rec:
        answer(_chat(), question="where?")
    (row,) = sft_rows([rec.records[0].to_json()], answer)
    trained = tok.apply_chat_template(
        row["prompt"],
        tools=row["tools"],
        add_generation_prompt=True,
        tokenize=True,
        adapter_name=adapter,
    )
    if isinstance(trained, dict):
        trained = trained["input_ids"]

    served = _serving_backend(answer)
    mock, create = _mocked("Shipped.")
    with mock, answer.policy(adapter_student(served)):
        answer(_chat(), question=None)
    sent = create.call_args.kwargs
    got = requests.post(
        url.removesuffix("/v1") + "/tokenize",
        json={
            "model": model,
            "messages": sent["messages"],
            "tools": sent["tools"],
            "add_generation_prompt": True,
            "chat_template_kwargs": {"adapter_name": adapter},
        },
        timeout=30,
    ).json()["tokens"]
    assert got == list(trained)


def test_a_decision_may_carry_the_tools_it_declares():
    @learnable(tools=[get_state])
    def pick(chat: ChatContext) -> Literal["a", "b"]:
        """A pick."""
        return [ComputedModelOutputThunk(ModelOutputThunk("a"))]

    request = {
        "messages": [{"role": "user", "content": "pick"}],
        "tools": [STATE["function"]],
    }
    IntrinsicsRewriter(config_dict=pick.io_config()).transform(request)
    undeclared = {**pick.io_config(), "tools": None, "tool_choice": None}
    with pytest.raises(ValueError, match="do not support tools"):
        IntrinsicsRewriter(config_dict=undeclared).transform(request)
    req = pick.render(_student(), {"chat": _chat()})
    assert req.tools == [STATE] and req.constraint == {"labels": ["a", "b"]}


def test_adapter_student_serves_the_checkpoints_name_for_the_function():
    backend = _student()
    adapter = _AdapterCore(
        identity=Identity(name="answers", adapter_type="lora", capability="answers"),
        io_contract=get_io_contract("answers"),
        weights=EmbeddedBinding(),
    )
    backend.add_adapter(adapter, config={**answer.io_config(), "name": "answers"})
    mock, create = _mocked("Shipped.")
    with mock, answer.policy(adapter_student(backend, name="answers")):
        with Recorder(student=_student()) as rec:
            assert answer(_chat(), question=None) == "Shipped."
    sent = create.call_args.kwargs
    assert sent["extra_body"]["chat_template_kwargs"]["adapter_name"] == "answers"
    (r,) = rec.records
    assert r.policy is not None and r.policy.by == "adapter:answers"
    assert sent["messages"] == r.request.messages and sent["tools"] == r.request.tools
