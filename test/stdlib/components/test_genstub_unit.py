"""Unit tests for genstub pure-logic helpers — no backend, no LLM required.

Covers describe_function, get_argument, bind_function_arguments,
create_response_format, GenerativeStub.format_for_llm, and @generative routing.
"""

from typing import Literal

import pytest

from mellea import generative
from mellea.core import TemplateRepresentation, ValidationResult
from mellea.stdlib.components.genstub import (
    ArgPreconditionRequirement,
    Arguments,
    AsyncGenerativeStub,
    Function,
    PreconditionException,
    SyncGenerativeStub,
    bind_function_arguments,
    create_response_format,
    describe_function,
    get_argument,
)
from mellea.stdlib.requirements.requirement import reqify

# --- describe_function ---


def test_describe_function_name():
    def greet(name: str) -> str:
        """Say hello."""
        return f"Hello {name}"

    result = describe_function(greet)
    assert result["name"] == "greet"


def test_describe_function_signature_includes_params():
    def add(x: int, y: int) -> int:
        return x + y

    result = describe_function(add)
    assert "x" in result["signature"]
    assert "y" in result["signature"]


def test_describe_function_docstring():
    def noop() -> None:
        """Does nothing."""

    result = describe_function(noop)
    assert result["docstring"] == "Does nothing."


def test_describe_function_no_docstring():
    def bare():
        pass

    result = describe_function(bare)
    assert result["docstring"] is None


# --- get_argument ---


def test_get_argument_string_value_quoted():
    def fn(name: str) -> None:
        pass

    arg = get_argument(fn, "name", "Alice")
    assert arg._argument_dict["value"] == '"Alice"'
    assert arg._argument_dict["name"] == "name"


def test_get_argument_int_value_not_quoted():
    def fn(count: int) -> None:
        pass

    arg = get_argument(fn, "count", 42)
    assert arg._argument_dict["value"] == 42
    assert "int" in str(arg._argument_dict["annotation"])


def test_get_argument_no_annotation_falls_back_to_runtime_type():
    # No annotation on kwargs — should fall back to type(val)
    def fn(**kwargs) -> None:
        pass

    arg = get_argument(fn, "x", 3.14)
    assert "float" in str(arg._argument_dict["annotation"])


# --- bind_function_arguments ---


def test_bind_function_arguments_basic():
    def fn(x: int, y: int) -> int:
        return x + y

    result = bind_function_arguments(fn, x=1, y=2)
    assert result == {"x": 1, "y": 2}


def test_bind_function_arguments_with_defaults():
    def fn(x: int, y: int = 10) -> int:
        return x + y

    result = bind_function_arguments(fn, x=5)
    assert result == {"x": 5, "y": 10}


def test_bind_function_arguments_missing_required_raises():
    def fn(x: int, y: int) -> int:
        return x + y

    with pytest.raises(TypeError, match="missing required parameter"):
        bind_function_arguments(fn, x=1)


def test_bind_function_arguments_no_params():
    def fn() -> str:
        return "hi"

    result = bind_function_arguments(fn)
    assert result == {}


# --- create_response_format ---


def test_create_response_format_class_name_derived_from_func():
    def get_sentiment() -> str: ...

    model = create_response_format(get_sentiment)
    assert "GetSentiment" in model.__name__


def test_create_response_format_result_field_accessible():
    def score_text() -> float: ...

    model = create_response_format(score_text)
    instance = model(result=0.9)
    assert instance.result == 0.9


def test_create_response_format_literal_type():
    def classify() -> Literal["pos", "neg"]: ...

    model = create_response_format(classify)
    instance = model(result="pos")
    assert instance.result == "pos"


# --- GenerativeStub.format_for_llm ---


def test_generative_stub_format_for_llm_returns_template_repr():
    @generative
    def summarise(text: str) -> str:
        """Summarise the given text."""

    result = summarise.format_for_llm()
    assert isinstance(result, TemplateRepresentation)


def test_generative_stub_format_for_llm_includes_function_name():
    @generative
    def my_function(x: int) -> int: ...

    result = my_function.format_for_llm()
    assert result.args["function"]["name"] == "my_function"


def test_generative_stub_format_for_llm_includes_docstring():
    @generative
    def documented() -> str:
        """This is the docstring."""

    result = documented.format_for_llm()
    assert result.args["function"]["docstring"] == "This is the docstring."


def test_generative_stub_format_for_llm_no_args_until_called():
    @generative
    def fn() -> str: ...

    result = fn.format_for_llm()
    assert result.args["arguments"] is None


# --- @generative decorator routing ---


def test_generative_sync_function_returns_sync_stub():
    @generative
    def sync_fn() -> str: ...

    assert isinstance(sync_fn, SyncGenerativeStub)


def test_generative_async_function_returns_async_stub():
    @generative
    async def async_fn() -> str: ...

    assert isinstance(async_fn, AsyncGenerativeStub)


def test_generative_disallowed_param_name_raises():
    with pytest.raises(ValueError, match="disallowed parameter names"):

        @generative
        def fn(backend: str) -> str: ...


# --- Arguments (CBlock subclass rendering bound args) ---


def test_arguments_renders_text():
    def fn(name: str, count: int) -> None:
        pass

    args = [get_argument(fn, "name", "Alice"), get_argument(fn, "count", 3)]
    block = Arguments(args)
    assert "name" in block.value
    assert "count" in block.value


def test_arguments_stores_meta_by_name():
    def fn(x: int) -> None:
        pass

    args = [get_argument(fn, "x", 5)]
    block = Arguments(args)
    assert "x" in block._meta


def test_arguments_empty_list():
    block = Arguments([])
    assert block.value == ""


# --- Function (wraps callable with metadata) ---


def test_function_stores_callable():
    def greet(name: str) -> str:
        """Say hi."""
        return f"hi {name}"

    f = Function(greet)
    assert f._func is greet
    assert f._function_dict["name"] == "greet"
    assert f._function_dict["docstring"] == "Say hi."


# --- ArgPreconditionRequirement (requirement wrapper) ---


def test_arg_precondition_delegates_description():
    req = reqify("must be non-empty")
    wrapper = ArgPreconditionRequirement(req)
    assert wrapper.description == req.description


def test_arg_precondition_copy():
    from copy import copy

    req = reqify("be valid")
    wrapper = ArgPreconditionRequirement(req)
    copied = copy(wrapper)
    assert isinstance(copied, ArgPreconditionRequirement)
    assert copied.req is req


def test_arg_precondition_deepcopy():
    from copy import deepcopy

    req = reqify("be clean")
    wrapper = ArgPreconditionRequirement(req)
    cloned = deepcopy(wrapper)
    assert isinstance(cloned, ArgPreconditionRequirement)
    assert cloned.description == req.description


# --- PreconditionException ---


def test_precondition_exception_message():
    vr = ValidationResult(result=False, reason="failed check")
    exc = PreconditionException("precondition failed", [vr])
    assert "precondition failed" in str(exc)
    assert exc.validation == [vr]


# --- GenerativeStub._parse ---


def test_genstub_parse_json_to_result():
    import json

    from mellea.core import ModelOutputThunk

    @generative
    def classify(text: str) -> str: ...

    mot = ModelOutputThunk(value=json.dumps({"result": "positive"}))
    parsed = classify._parse(mot)
    assert parsed == "positive"


def test_genstub_parse_int_result():
    import json

    from mellea.core import ModelOutputThunk

    @generative
    def compute(x: int) -> int: ...

    mot = ModelOutputThunk(value=json.dumps({"result": 42}))
    parsed = compute._parse(mot)
    assert parsed == 42


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
