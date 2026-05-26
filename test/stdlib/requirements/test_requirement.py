import json
from unittest.mock import patch

import pytest

from mellea.core import ModelOutputThunk, Requirement
from mellea.stdlib.context import ChatContext
from mellea.stdlib.requirements import LLMaJRequirement, simple_validate
from mellea.stdlib.requirements.requirement import (
    ALoraRequirement,
    check,
    req,
    reqify,
    requirement_check_to_bool,
)
from mellea.stdlib.session import start_session

ctx = ChatContext()
ctx = ctx.add(ModelOutputThunk("test"))


@pytest.mark.ollama
@pytest.mark.e2e
async def test_llmaj_validation_req_output_field():
    m = start_session(ctx=ctx)
    req = Requirement("Must output test.")
    assert req._output is None

    _ = await req.validate(m.backend, ctx=ctx)
    assert req._output is None, (
        "requirement's output shouldn't be updated during/after validation"
    )


@pytest.mark.ollama
@pytest.mark.e2e
async def test_llmaj_requirement_uses_requirement_template():
    m = start_session(ctx=ctx)
    req = LLMaJRequirement("Must output test.")
    assert req._output is None

    _ = await req.validate(m.backend, ctx=ctx)
    assert req._output is None, (
        "requirement's output shouldn't be updated during/after validation"
    )


def test_simple_validate_bool():
    validation_func = simple_validate(lambda x: False, reason="static reason")
    val_result = validation_func(ctx)

    assert not val_result.as_bool(), (
        "validation result should be False given the lambda func passed to simple_validate"
    )
    assert val_result.reason == "static reason"


def test_simple_validate_bool_string():
    validation_func = simple_validate(lambda x: (False, "dynamic reason"))
    val_result = validation_func(ctx)

    assert not bool(val_result), (
        "validation result should be False given the lambda func passed to simple_validate"
    )
    assert val_result.reason == "dynamic reason"


def test_simple_validate_invalid():
    validation_func = simple_validate(lambda x: None)  # type: ignore

    with pytest.raises(ValueError):
        validation_func(ctx)


# --- requirement_check_to_bool ---


def test_requirement_check_to_bool_above_threshold():
    assert requirement_check_to_bool('{"requirement_check": {"score": 0.8}}') is True


def test_requirement_check_to_bool_below_threshold():
    assert requirement_check_to_bool('{"requirement_check": {"score":0.3}}') is False


def test_requirement_check_to_bool_at_threshold():
    """0.5 is NOT > 0.5, so should return False."""
    assert requirement_check_to_bool('{"requirement_check": {"score": 0.5}}') is False


def test_requirement_check_to_bool_missing_key():
    assert requirement_check_to_bool('{"other_field": 1.0}') is False


def test_requirement_check_to_bool_invalid_json():
    with pytest.raises(json.JSONDecodeError):
        requirement_check_to_bool("not json")


# --- reqify ---


def test_reqify_string():
    result = reqify("must be valid")
    assert isinstance(result, Requirement)
    assert result.description == "must be valid"


def test_reqify_requirement():
    original = Requirement("must be valid")
    result = reqify(original)
    assert result is original


def test_reqify_invalid_type():
    with pytest.raises(Exception, match="reqify takes a str or requirement"):
        reqify(123)  # type: ignore[arg-type]


# --- req / check shorthands ---


def test_req_shorthand():
    result = req("must be valid")
    assert isinstance(result, Requirement)
    assert result.description == "must be valid"


def test_check_shorthand():
    result = check("must be valid")
    assert isinstance(result, Requirement)
    assert result.check_only is True


# --- simple_validate edge case ---


def test_simple_validate_none_output():
    """Context with no output should return False without calling the fn."""
    empty_ctx = ChatContext()
    validation_func = simple_validate(lambda x: True)
    result = validation_func(empty_ctx)
    assert result.as_bool() is False


# --- LLMaJRequirement ---


def test_llmaj_requirement_use_aloras_false():
    r = LLMaJRequirement("must be valid")
    assert r.use_aloras is False


# --- ALoraRequirement ---


@patch("mellea.stdlib.requirements.requirement.Intrinsic.__init__")
def test_alora_requirement_default_intrinsic(mock_intrinsic_init):
    mock_intrinsic_init.return_value = None
    r = ALoraRequirement("must be valid")
    assert r.use_aloras is True
    assert r.description == "must be valid"
    # Intrinsic.__init__ is unbound; mock receives self as first positional arg.
    mock_intrinsic_init.assert_called_once_with(
        r,
        intrinsic_name="requirement-check",
        intrinsic_kwargs={"requirement": "must be valid"},
    )


@patch("mellea.stdlib.requirements.requirement.Intrinsic.__init__")
def test_alora_requirement_custom_intrinsic(mock_intrinsic_init):
    mock_intrinsic_init.return_value = None
    r = ALoraRequirement("must be valid", intrinsic_name="custom_check")
    assert r.use_aloras is True
    mock_intrinsic_init.assert_called_once_with(
        r,
        intrinsic_name="custom_check",
        intrinsic_kwargs={"requirement": "must be valid"},
    )


if __name__ == "__main__":
    pytest.main([__file__])
