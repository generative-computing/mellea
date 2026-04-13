from unittest.mock import Mock

import pytest

from mellea.core import ModelOutputThunk, ModelToolCall
from mellea.stdlib.context import ChatContext
from mellea.stdlib.requirements.tool_reqs import (
    _name2str,
    tool_arg_validator,
    uses_tool,
)


def _ctx_with_tool_calls(tool_calls: dict[str, ModelToolCall] | None) -> ChatContext:
    """Helper: build a ChatContext whose last output has the given tool_calls."""
    ctx = ChatContext()
    return ctx.add(ModelOutputThunk(value="", tool_calls=tool_calls))


def _make_tool_call(name: str, args: dict) -> ModelToolCall:
    """Helper: build a ModelToolCall with a mock func."""
    return ModelToolCall(name=name, func=Mock(), args=args)


# --- _name2str ---


def test_name2str():
    """Test handling when no Python code is present."""

    def test123():
        pass

    assert _name2str(test123) == "test123"
    assert _name2str("test1234") == "test1234"


def test_name2str_type_error():
    with pytest.raises(TypeError, match="Expected Callable or str"):
        _name2str(123)  # type: ignore[arg-type]


# --- uses_tool ---


def test_uses_tool_present():
    ctx = _ctx_with_tool_calls({"get_weather": _make_tool_call("get_weather", {})})
    req = uses_tool("get_weather")
    result = req.validation_fn(ctx)
    assert result.as_bool() is True


def test_uses_tool_absent():
    ctx = _ctx_with_tool_calls({"get_weather": _make_tool_call("get_weather", {})})
    req = uses_tool("send_email")
    result = req.validation_fn(ctx)
    assert result.as_bool() is False


def test_uses_tool_no_tool_calls():
    ctx = _ctx_with_tool_calls(None)
    req = uses_tool("get_weather")
    result = req.validation_fn(ctx)
    assert result.as_bool() is False
    assert "no tool calls" in result.reason.lower()


def test_uses_tool_callable_input():
    def my_tool():
        pass

    ctx = _ctx_with_tool_calls({"my_tool": _make_tool_call("my_tool", {})})
    req = uses_tool(my_tool)
    result = req.validation_fn(ctx)
    assert result.as_bool() is True


def test_uses_tool_check_only():
    req = uses_tool("get_weather", check_only=True)
    assert req.check_only is True


# --- tool_arg_validator ---


def test_tool_arg_validator_valid():
    ctx = _ctx_with_tool_calls(
        {"search": _make_tool_call("search", {"query": "hello", "limit": 10})}
    )
    req = tool_arg_validator(
        description="limit must be positive",
        tool_name="search",
        arg_name="limit",
        validation_fn=lambda v: v > 0,
    )
    result = req.validation_fn(ctx)
    assert result.as_bool() is True


def test_tool_arg_validator_failed_validation():
    ctx = _ctx_with_tool_calls(
        {"search": _make_tool_call("search", {"query": "hello", "limit": -1})}
    )
    req = tool_arg_validator(
        description="limit must be positive",
        tool_name="search",
        arg_name="limit",
        validation_fn=lambda v: v > 0,
    )
    result = req.validation_fn(ctx)
    assert result.as_bool() is False


def test_tool_arg_validator_missing_tool():
    ctx = _ctx_with_tool_calls(
        {"search": _make_tool_call("search", {"query": "hello"})}
    )
    req = tool_arg_validator(
        description="check email tool",
        tool_name="send_email",
        arg_name="to",
        validation_fn=lambda v: True,
    )
    result = req.validation_fn(ctx)
    assert result.as_bool() is False
    assert "send_email" in result.reason


def test_tool_arg_validator_missing_arg():
    ctx = _ctx_with_tool_calls(
        {"search": _make_tool_call("search", {"query": "hello"})}
    )
    req = tool_arg_validator(
        description="limit must exist",
        tool_name="search",
        arg_name="limit",
        validation_fn=lambda v: True,
    )
    result = req.validation_fn(ctx)
    assert result.as_bool() is False
    assert "limit" in result.reason


def test_tool_arg_validator_no_tool_calls():
    ctx = _ctx_with_tool_calls(None)
    req = tool_arg_validator(
        description="check tool",
        tool_name="search",
        arg_name="query",
        validation_fn=lambda v: True,
    )
    result = req.validation_fn(ctx)
    assert result.as_bool() is False


def test_tool_arg_validator_no_tool_name_all_pass():
    ctx = _ctx_with_tool_calls(
        {
            "tool_a": _make_tool_call("tool_a", {"x": 5}),
            "tool_b": _make_tool_call("tool_b", {"x": 10}),
        }
    )
    req = tool_arg_validator(
        description="x must be positive",
        tool_name=None,
        arg_name="x",
        validation_fn=lambda v: v > 0,
    )
    result = req.validation_fn(ctx)
    assert result.as_bool() is True


def test_tool_arg_validator_no_tool_name_one_fails():
    ctx = _ctx_with_tool_calls(
        {
            "tool_a": _make_tool_call("tool_a", {"x": 5}),
            "tool_b": _make_tool_call("tool_b", {"x": -1}),
        }
    )
    req = tool_arg_validator(
        description="x must be positive",
        tool_name=None,
        arg_name="x",
        validation_fn=lambda v: v > 0,
    )
    result = req.validation_fn(ctx)
    assert result.as_bool() is False


def test_tool_arg_validator_no_tool_name_arg_missing_everywhere():
    """Documents current behavior: when tool_name=None and no tool call contains
    the target arg_name, validation silently passes (the for-loop completes
    without failing). This is arguably a latent bug — the validator never runs."""
    ctx = _ctx_with_tool_calls({"tool_a": _make_tool_call("tool_a", {"y": 5})})
    req = tool_arg_validator(
        description="x must be positive",
        tool_name=None,
        arg_name="x",
        validation_fn=lambda v: v > 0,
    )
    result = req.validation_fn(ctx)
    assert result.as_bool() is True
