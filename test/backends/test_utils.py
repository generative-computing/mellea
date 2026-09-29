# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for backends/utils.py — get_value accessor and to_tool_calls parser."""

import logging
from dataclasses import dataclass

import pytest

from mellea.backends.tools import MelleaTool
from mellea.backends.utils import (
    get_value,
    populate_response_metadata_openai_shape,
    to_tool_calls,
)
from mellea.core import ModelToolCall
from mellea.core.base import GenerationMetadata, ModelOutputThunk

# --- get_value ---


def test_get_value_dict_present():
    assert get_value({"a": 1, "b": 2}, "a") == 1


def test_get_value_dict_missing():
    assert get_value({"a": 1}, "missing") is None


def test_get_value_object_attribute():
    obj = type("Obj", (), {"x": "hello"})()
    assert get_value(obj, "x") == "hello"


def test_get_value_object_missing_attribute():
    obj = type("Obj", (), {})()
    assert get_value(obj, "nonexistent") is None


def test_get_value_dict_none_value():
    # Explicitly stored None should come back as None (same as get())
    assert get_value({"k": None}, "k") is None


@dataclass
class _DC:
    score: float
    label: str


def test_get_value_dataclass():
    dc = _DC(score=0.9, label="positive")
    assert get_value(dc, "score") == 0.9
    assert get_value(dc, "label") == "positive"


# --- to_tool_calls ---


def _make_tool_registry() -> dict:
    def add(x: int, y: int) -> int:
        """Add two integers."""
        return x + y

    def greet(name: str) -> str:
        """Greet a person."""
        return f"Hello, {name}!"

    return {
        "add": MelleaTool.from_callable(add),
        "greet": MelleaTool.from_callable(greet),
    }


def _tool_call_json(name: str, args: dict) -> str:
    import json

    return json.dumps([{"name": name, "arguments": args}])


def test_to_tool_calls_single_call():
    registry = _make_tool_registry()
    raw = _tool_call_json("add", {"x": 3, "y": 4})
    result = to_tool_calls(registry, raw)
    assert result is not None
    assert len(result) == 1
    mtc = result[0]
    assert isinstance(mtc, ModelToolCall)
    assert mtc.name == "add"
    assert mtc.args == {"x": 3, "y": 4}


def test_to_tool_calls_returns_none_when_no_calls():
    registry = _make_tool_registry()
    result = to_tool_calls(registry, "no tool call here")
    assert result is None


def test_to_tool_calls_unknown_tool_skipped():
    registry = _make_tool_registry()
    raw = _tool_call_json("nonexistent_fn", {"arg": "val"})
    # Unknown tool is skipped — result should be None (empty dict → None)
    result = to_tool_calls(registry, raw)
    assert result is None


def test_to_tool_calls_empty_params_cleared():
    """When the tool has no parameters, hallucinated args should be stripped."""

    def noop() -> str:
        """Does nothing."""
        return "done"

    registry = {"noop": MelleaTool.from_callable(noop)}
    raw = _tool_call_json("noop", {"hallucinated": "arg"})
    result = to_tool_calls(registry, raw)
    assert result is not None
    assert result[0].args == {}


def test_to_tool_calls_string_arg_coerced_to_int():
    """validate_tool_arguments coerces strings to int when strict=False."""
    registry = _make_tool_registry()
    raw = _tool_call_json("add", {"x": "5", "y": "10"})
    result = to_tool_calls(registry, raw)
    assert result is not None
    assert result[0].args["x"] == 5
    assert result[0].args["y"] == 10


# --- to_tool_calls: Granite XML function format (#1689) ---


def _tool_call_xml(name: str, args: dict[str, str]) -> str:
    params = "".join(f"<parameter={k}>\n{v}\n</parameter>\n" for k, v in args.items())
    return f"<tool_call>\n<function={name}>\n{params}</function>\n</tool_call>"


def test_to_tool_calls_granite_xml_call_coerced_to_schema():
    registry = _make_tool_registry()
    result = to_tool_calls(registry, _tool_call_xml("add", {"x": "3", "y": "4"}))
    assert result is not None
    assert len(result) == 1
    assert result[0].name == "add"
    assert result[0].args == {"x": 3, "y": 4}


def test_to_tool_calls_granite_xml_array_param_decoded_from_json():
    """The template renders list/dict arguments with `tojson`, so they arrive as JSON text."""

    def tag(labels: list[str]) -> str:
        """Tag an item.

        Args:
            labels: the labels to apply
        """
        return ",".join(labels)

    registry = {"tag": MelleaTool.from_callable(tag)}
    result = to_tool_calls(registry, _tool_call_xml("tag", {"labels": '["a", "b"]'}))
    assert result is not None
    assert result[0].args == {"labels": ["a", "b"]}


def test_to_tool_calls_json_text_for_string_param_stays_a_string():
    """Only object/array parameters are JSON-decoded."""

    def note(text: str) -> str:
        """Store a note.

        Args:
            text: the note text
        """
        return text

    registry = {"note": MelleaTool.from_callable(note)}
    result = to_tool_calls(registry, _tool_call_xml("note", {"text": '{"a": 1}'}))
    assert result is not None
    assert result[0].args == {"text": '{"a": 1}'}


def test_to_tool_calls_granite_xml_optional_model_param_decoded():
    """`Optional[Model]` renders as `anyOf: [{type: object}, {type: null}]`."""
    from pydantic import BaseModel

    class Point(BaseModel):
        x: int
        y: int

    def plot(point: Point | None = None) -> str:
        """Plot a point.

        Args:
            point: the point to plot
        """
        return str(point)

    registry = {"plot": MelleaTool.from_callable(plot)}
    raw = _tool_call_xml("plot", {"point": '{"x": 1, "y": 2}'})
    result = to_tool_calls(registry, raw)
    assert result is not None
    assert result[0].args == {"point": {"x": 1, "y": 2}}


def test_decode_json_container_args_list_type_inside_any_of():
    """External schemas (LangChain, smolagents) can put a list-valued `type` in an `anyOf` branch."""
    from mellea.backends.utils import _decode_json_container_args

    properties = {"x": {"anyOf": [{"type": ["array", "null"]}]}}
    assert _decode_json_container_args({"x": "[1, 2]"}, properties) == {"x": [1, 2]}


def test_to_tool_calls_undecodable_array_value_left_for_validation():
    """Text that is not JSON is passed through; validation reports the mismatch."""

    def tag(labels: list[str]) -> str:
        """Tag an item.

        Args:
            labels: the labels to apply
        """
        return ",".join(labels)

    registry = {"tag": MelleaTool.from_callable(tag)}
    result = to_tool_calls(registry, _tool_call_xml("tag", {"labels": "a, b"}))
    assert result is not None
    assert result[0].args == {"labels": "a, b"}


def test_to_tool_calls_warns_on_unparsable_tool_call_markup(
    caplog: pytest.LogCaptureFixture,
):
    registry = _make_tool_registry()
    with caplog.at_level(logging.WARNING, logger="mellea"):
        result = to_tool_calls(
            registry, "<tool_call>\nadd three and four\n</tool_call>"
        )
    assert result is None
    assert any("<tool_call>" in r.getMessage() for r in caplog.records)


def test_to_tool_calls_warns_when_some_tool_calls_are_dropped(
    caplog: pytest.LogCaptureFixture,
):
    """A malformed call next to a good one is dropped, and the drop is logged."""
    registry = _make_tool_registry()
    raw = (
        "<tool_call><function=add><parameter=x>1</parameter></tool_call>"
        + _tool_call_xml("greet", {"name": "Ada"})
    )
    with caplog.at_level(logging.WARNING, logger="mellea"):
        result = to_tool_calls(registry, raw)
    assert result is not None
    assert [(r.name, r.args) for r in result] == [("greet", {"name": "Ada"})]
    assert any("<tool_call>" in r.getMessage() for r in caplog.records)


@pytest.mark.integration
def test_to_tool_calls_round_trips_real_granite_template() -> None:
    """Parse the tool-call markup the real granite-4.2-3b chat template renders.

    Loads the template only (no GPU, no model weights); skips if not locally cached.
    """
    from test.backends.test_huggingface_filter_options import (
        _GRANITE_THINKING_MODEL_ID,
        _try_load_granite_tokenizer,
    )

    tok = _try_load_granite_tokenizer(_GRANITE_THINKING_MODEL_ID)
    if tok is None:
        pytest.skip(f"{_GRANITE_THINKING_MODEL_ID} not in local HF cache")

    def tag(labels: list[str], note: str) -> str:
        """Tag an item.

        Args:
            labels: the labels to apply
            note: a free-text note
        """
        return note

    registry = {**_make_tool_registry(), "tag": MelleaTool.from_callable(tag)}
    calls = [
        {"name": "add", "arguments": {"x": 3, "y": 4}},
        {"name": "tag", "arguments": {"labels": ["a", "b"], "note": "first\nsecond"}},
    ]
    rendered = tok.apply_chat_template(
        [
            {"role": "user", "content": "Add 3 and 4, then tag it."},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": str(i), "type": "function", "function": c}
                    for i, c in enumerate(calls)
                ],
            },
        ],
        tokenize=False,
    )
    assert "<function=add>" in rendered  # the template really uses the XML format

    result = to_tool_calls(registry, rendered)
    assert result is not None
    assert [(r.name, r.args) for r in result] == [
        (c["name"], c["arguments"]) for c in calls
    ]


# --- to_chat ---


def test_to_chat_basic_message():
    from mellea.backends.utils import to_chat
    from mellea.formatters.template_formatter import TemplateFormatter as ChatFormatter
    from mellea.stdlib.components import Message
    from mellea.stdlib.context import ChatContext

    ctx = ChatContext()
    ctx = ctx.add(Message("user", "hello"))
    action = Message("user", "next question")
    formatter = ChatFormatter(model_id="test")

    result = to_chat(action, ctx, formatter, system_prompt=None)
    assert isinstance(result, list)
    assert len(result) == 2
    assert result[0]["role"] == "user"
    assert result[0]["content"] == "hello"
    assert result[1]["role"] == "user"
    assert result[1]["content"] == "next question"


def test_to_chat_attaches_reasoning_content_for_assistant_thinking():
    """An assistant Message carrying `.thinking` must have it forwarded as
    `reasoning_content` on the wire dict.
    """
    from mellea.backends.utils import to_chat
    from mellea.formatters.template_formatter import TemplateFormatter as ChatFormatter
    from mellea.stdlib.components import Message
    from mellea.stdlib.context import ChatContext

    ctx = ChatContext()
    ctx = ctx.add(Message("user", "hello"))
    ctx = ctx.add(Message("assistant", "the answer", thinking="reasoning trace"))
    action = Message("user", "next question")
    formatter = ChatFormatter(model_id="test")

    result = to_chat(action, ctx, formatter, system_prompt=None)
    assistant_msg = next(m for m in result if m["role"] == "assistant")
    assert assistant_msg["reasoning_content"] == "reasoning trace"


def test_to_chat_known_reasoning_content_wins_over_provider_fields():
    """Regression/contract test: `reasoning_content` is set from `Message.thinking`
    before `merge_provider_fields` runs (same known-fields-first pattern as
    `tool_calls`/`tool_call_id`), so an author-declared `provider_fields`
    collision on the same key is silently dropped (debug-logged, not raised) —
    Mellea's own value always wins.
    """
    from mellea.backends.utils import to_chat
    from mellea.formatters.template_formatter import TemplateFormatter as ChatFormatter
    from mellea.stdlib.components import Message
    from mellea.stdlib.context import ChatContext

    ctx = ChatContext()
    ctx = ctx.add(Message("user", "hello"))
    ctx = ctx.add(
        Message(
            "assistant",
            "the answer",
            thinking="reasoning trace",
            provider_fields={"huggingface": {"reasoning_content": "author override"}},
        )
    )
    action = Message("user", "next question")
    formatter = ChatFormatter(model_id="test")

    result = to_chat(action, ctx, formatter, system_prompt=None)
    assistant_msg = next(m for m in result if m["role"] == "assistant")
    assert assistant_msg["reasoning_content"] == "reasoning trace"


def test_to_chat_omits_reasoning_content_when_no_thinking():
    """An assistant Message with no captured reasoning must not get a
    `reasoning_content` key at all (not even an empty string)."""
    from mellea.backends.utils import to_chat
    from mellea.formatters.template_formatter import TemplateFormatter as ChatFormatter
    from mellea.stdlib.components import Message
    from mellea.stdlib.context import ChatContext

    ctx = ChatContext()
    ctx = ctx.add(Message("user", "hello"))
    ctx = ctx.add(Message("assistant", "the answer"))
    action = Message("user", "next question")
    formatter = ChatFormatter(model_id="test")

    result = to_chat(action, ctx, formatter, system_prompt=None)
    assistant_msg = next(m for m in result if m["role"] == "assistant")
    assert "reasoning_content" not in assistant_msg


def test_to_chat_ignores_thinking_on_non_assistant_message():
    """`.thinking` on a non-assistant Message (never set by real code paths, but
    not type-guarded against) must not be forwarded as `reasoning_content` — the
    Granite template only consumes that key on the assistant branch, so this is
    otherwise inert, but the wire dict should not carry stray, unconsumed keys."""
    from mellea.backends.utils import to_chat
    from mellea.formatters.template_formatter import TemplateFormatter as ChatFormatter
    from mellea.stdlib.components import Message
    from mellea.stdlib.context import ChatContext

    ctx = ChatContext()
    ctx = ctx.add(Message("user", "hello", thinking="stray thinking"))
    action = Message("user", "next question")
    formatter = ChatFormatter(model_id="test")

    result = to_chat(action, ctx, formatter, system_prompt=None)
    user_msg = next(m for m in result if m["content"] == "hello")
    assert "reasoning_content" not in user_msg


def test_to_chat_with_system_prompt():
    from mellea.backends.utils import to_chat
    from mellea.formatters.template_formatter import TemplateFormatter as ChatFormatter
    from mellea.stdlib.components import Message
    from mellea.stdlib.context import ChatContext

    ctx = ChatContext()
    ctx = ctx.add(Message("user", "hi"))
    action = Message("user", "q")
    formatter = ChatFormatter(model_id="test")

    result = to_chat(action, ctx, formatter, system_prompt="You are helpful.")
    assert result[0]["role"] == "system"
    assert result[0]["content"] == "You are helpful."
    assert len(result) == 3  # system + user context + user action


# --- populate_response_metadata_openai_shape ---


def _mot_with_generation() -> ModelOutputThunk:
    mot = ModelOutputThunk("x")
    mot.generation = GenerationMetadata(model="gpt-4o", provider="openai")
    return mot


@pytest.mark.parametrize(
    "choices, expected_finish_reasons",
    [
        ([{"finish_reason": "stop"}], ["stop"]),
        ([{"finish_reason": "stop"}, {"finish_reason": "length"}], ["stop", "length"]),
    ],
    ids=["single", "multi"],
)
def test_populate_response_metadata_full(choices, expected_finish_reasons):
    """Dict response with valid choices populates all three fields."""
    mot = _mot_with_generation()
    populate_response_metadata_openai_shape(
        mot, {"model": "gpt-4o", "id": "chatcmpl-abc", "choices": choices}
    )
    assert mot.generation.response_model == "gpt-4o"
    assert mot.generation.response_id == "chatcmpl-abc"
    assert mot.generation.finish_reasons == expected_finish_reasons


def test_populate_response_metadata_object_response():
    """Object responses (not dicts) work — verifies use of get_value, not [] access."""
    mot = _mot_with_generation()
    Choice = type("Choice", (), {"finish_reason": "length"})
    Resp = type("Resp", (), {"model": "gpt-4o", "id": "resp-1", "choices": [Choice()]})
    populate_response_metadata_openai_shape(mot, Resp())
    assert mot.generation.response_model == "gpt-4o"
    assert mot.generation.response_id == "resp-1"
    assert mot.generation.finish_reasons == ["length"]


@pytest.mark.parametrize(
    "response",
    [
        None,
        {"model": "m", "id": "i", "choices": []},
        {"model": "m", "id": "i", "choices": [{"finish_reason": None}]},
    ],
    ids=["none", "empty-choices", "all-none-reasons"],
)
def test_populate_response_metadata_finish_reasons_stays_none(response):
    """No-op or empty extraction leaves finish_reasons as None (NOT [])."""
    mot = _mot_with_generation()
    populate_response_metadata_openai_shape(mot, response)
    assert mot.generation.finish_reasons is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
