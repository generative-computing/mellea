"""Unit tests for backends/utils.py — get_value accessor and to_tool_calls parser."""

from dataclasses import dataclass

import pytest

from mellea.backends.tools import MelleaTool
from mellea.backends.utils import get_value, to_tool_calls
from mellea.core import ModelToolCall

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
    assert "add" in result
    mtc = result["add"]
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
    assert result["noop"].args == {}


def test_to_tool_calls_string_arg_coerced_to_int():
    """validate_tool_arguments coerces strings to int when strict=False."""
    registry = _make_tool_registry()
    raw = _tool_call_json("add", {"x": "5", "y": "10"})
    result = to_tool_calls(registry, raw)
    assert result is not None
    assert result["add"].args["x"] == 5
    assert result["add"].args["y"] == 10


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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
