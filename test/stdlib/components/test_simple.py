"""Unit tests for SimpleComponent — kwargs rendering, type validation, JSON output."""

import json

import pytest

from mellea.core import CBlock, ModelOutputThunk
from mellea.stdlib.components.simple import SimpleComponent

# --- constructor & type checking ---


def test_init_converts_strings_to_cblocks():
    sc = SimpleComponent(task="write a poem")
    assert isinstance(sc._kwargs["task"], CBlock)
    assert sc._kwargs["task"].value == "write a poem"


def test_init_accepts_cblock_directly():
    cb = CBlock("already a block")
    sc = SimpleComponent(thing=cb)
    assert sc._kwargs["thing"] is cb


def test_init_rejects_non_string_non_component():
    with pytest.raises(AssertionError):
        SimpleComponent(bad=42)


def test_init_rejects_non_string_key():
    # We can't pass non-string keys via kwargs syntax; test _kwargs_type_check directly
    sc = SimpleComponent(ok="fine")
    with pytest.raises(AssertionError):
        sc._kwargs_type_check({123: CBlock("v")})


def test_init_multiple_kwargs():
    sc = SimpleComponent(task="summarise", context="some text")
    assert len(sc._kwargs) == 2
    assert set(sc._kwargs.keys()) == {"task", "context"}


# --- parts() ---


def test_parts_returns_all_values():
    sc = SimpleComponent(a="one", b="two")
    parts = sc.parts()
    assert len(parts) == 2
    assert all(isinstance(p, CBlock) for p in parts)


def test_parts_empty():
    sc = SimpleComponent()
    assert sc.parts() == []


# --- make_simple_string ---


def test_make_simple_string_single():
    kwargs = {"task": CBlock("do something")}
    result = SimpleComponent.make_simple_string(kwargs)
    assert result == "<|task|>do something</|task|>"


def test_make_simple_string_multiple():
    # Use ordered dict (Python 3.7+ guarantees insertion order)
    kwargs = {"a": CBlock("first"), "b": CBlock("second")}
    result = SimpleComponent.make_simple_string(kwargs)
    assert "<|a|>first</|a|>" in result
    assert "<|b|>second</|b|>" in result
    assert "\n" in result


def test_make_simple_string_empty():
    assert SimpleComponent.make_simple_string({}) == ""


# --- make_json_string ---


def test_make_json_string_cblock():
    kwargs = {"key": CBlock("value")}
    result = json.loads(SimpleComponent.make_json_string(kwargs))
    assert result == {"key": "value"}


def test_make_json_string_model_output_thunk():
    mot = ModelOutputThunk(value="output text")
    kwargs = {"out": mot}
    result = json.loads(SimpleComponent.make_json_string(kwargs))
    assert result == {"out": "output text"}


def test_make_json_string_nested_component():
    inner = SimpleComponent(x="nested")
    kwargs = {"inner": inner}
    result = json.loads(SimpleComponent.make_json_string(kwargs))
    assert "inner" in result


def test_make_json_string_empty():
    result = json.loads(SimpleComponent.make_json_string({}))
    assert result == {}


# --- format_for_llm ---


def test_format_for_llm_returns_json_string():
    sc = SimpleComponent(topic="ocean", style="poetic")
    formatted = sc.format_for_llm()
    parsed = json.loads(formatted)
    assert parsed["topic"] == "ocean"
    assert parsed["style"] == "poetic"


# --- _parse ---


def test_parse_returns_value():
    sc = SimpleComponent(x="whatever")
    mot = ModelOutputThunk(value="result")
    assert sc._parse(mot) == "result"


def test_parse_none_returns_empty_string():
    sc = SimpleComponent(x="whatever")
    mot = ModelOutputThunk(value=None)
    assert sc._parse(mot) == ""


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
