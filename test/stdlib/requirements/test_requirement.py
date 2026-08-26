# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mellea.backends.adapters import AdapterSchemaMismatchError
from mellea.core import ModelOutputThunk, Requirement, TemplateRepresentation
from mellea.formatters.template_formatter import TemplateFormatter
from mellea.stdlib.components import Message
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
    assert req._validation_target is None

    _ = await req.validate(m.backend, ctx=ctx)
    assert req._validation_target is None, (
        "requirement's validation target shouldn't be bound during/after validation"
    )


@pytest.mark.ollama
@pytest.mark.e2e
async def test_llmaj_requirement_uses_requirement_template():
    m = start_session(ctx=ctx)
    req = LLMaJRequirement("Must output test.")
    assert req._validation_target is None

    _ = await req.validate(m.backend, ctx=ctx)
    assert req._validation_target is None, (
        "requirement's validation target shouldn't be bound during/after validation"
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


def test_requirement_check_to_bool_raises_on_schema_mismatch():
    """Wrong top-level key must raise, not silently return False."""
    with pytest.raises(AdapterSchemaMismatchError):
        requirement_check_to_bool('{"other_field": 1.0}')


def test_pre_schema_change_output_raises():
    """Old output shape (requirement_likelihood) must raise AdapterSchemaMismatchError."""
    with pytest.raises(AdapterSchemaMismatchError):
        requirement_check_to_bool('{"requirement_likelihood": 0.9}')


def test_requirement_check_to_bool_missing_score_raises():
    """Missing nested score key must raise AdapterSchemaMismatchError."""
    with pytest.raises(AdapterSchemaMismatchError):
        requirement_check_to_bool('{"requirement_check": {"other_key": 0.9}}')


def test_requirement_check_to_bool_null_score_raises():
    """Null score must raise AdapterSchemaMismatchError, not TypeError."""
    with pytest.raises(AdapterSchemaMismatchError):
        requirement_check_to_bool('{"requirement_check": {"score": null}}')


def test_requirement_check_to_bool_string_score_raises():
    """String score must raise AdapterSchemaMismatchError, not TypeError."""
    with pytest.raises(AdapterSchemaMismatchError):
        requirement_check_to_bool('{"requirement_check": {"score": "0.9"}}')


def test_requirement_check_to_bool_invalid_json():
    with pytest.raises(json.JSONDecodeError):
        requirement_check_to_bool("not json")


def test_requirement_check_to_bool_non_object_raises():
    """A top-level JSON array or scalar is a ValueError, not a schema mismatch.

    The replaced code raised an undocumented AttributeError here instead
    (`list.get` on the parsed result).
    """
    with pytest.raises(ValueError, match="must be a JSON object"):
        requirement_check_to_bool("[1, 2]")


def test_requirement_check_to_bool_nan_score_raises():
    """NaN would silently evaluate as False without the finiteness guard."""
    with pytest.raises(AdapterSchemaMismatchError):
        requirement_check_to_bool('{"requirement_check": {"score": NaN}}')


def test_requirement_check_to_bool_inf_score_raises():
    """Infinite score must raise, not silently pass as True."""
    with pytest.raises(AdapterSchemaMismatchError):
        requirement_check_to_bool('{"requirement_check": {"score": Infinity}}')


def test_requirement_check_to_bool_score_above_range_raises():
    with pytest.raises(AdapterSchemaMismatchError):
        requirement_check_to_bool('{"requirement_check": {"score": 1.5}}')


def test_requirement_check_to_bool_score_below_range_raises():
    with pytest.raises(AdapterSchemaMismatchError):
        requirement_check_to_bool('{"requirement_check": {"score": -0.1}}')


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


# --- validation target binding (issue #426) ---


def test_parts_empty_until_target_bound():
    r = Requirement("must mention Paris")
    assert r.parts() == [], "an unbound requirement has no parts"


def test_bind_validation_target_leaves_original_unbound():
    """Binding happens on a copy so a requirement can be reused across validations."""
    r = Requirement("must mention Paris")
    target = ModelOutputThunk("The capital of France is Paris.")

    bound = r._bind_validation_target(target)

    assert bound is not r
    assert r._validation_target is None
    assert bound._validation_target is target
    assert bound.parts() == [target], (
        "the bound target must be exposed as a part so generate_walk can await it"
    )


def test_format_for_llm_carries_the_target_span():
    """The judge prompt gets the thunk itself, not a detached string copy of it."""
    r = Requirement("must mention Paris")
    target = ModelOutputThunk("The capital of France is Paris.")

    representation = r._bind_validation_target(target).format_for_llm()

    assert isinstance(representation, TemplateRepresentation)
    assert representation.args["output"] is target
    assert representation.args["description"] == "must mention Paris"


def test_format_for_llm_prefers_component_parsed_repr():
    """A parsed `Component` repr beats the raw generated string."""
    r = Requirement("must be polite")
    parsed = Message("assistant", "parsed message content")
    target = ModelOutputThunk("raw string value")
    target.parsed_repr = parsed

    representation = r._bind_validation_target(target).format_for_llm()

    assert isinstance(representation, TemplateRepresentation)
    assert representation.args["output"] is parsed


def test_format_for_llm_without_bound_target_raises():
    with pytest.raises(AssertionError, match="Object protocol error"):
        Requirement("must mention Paris").format_for_llm()


# --- judge prompt rendering ---


@pytest.fixture
def formatter():
    return TemplateFormatter(model_id="ibm-granite/granite-3.3-8b-instruct")


def test_requirement_prompt_inlines_the_output(formatter):
    r = Requirement("must mention Paris")
    target = ModelOutputThunk("The capital of France is Paris.")

    rendered = formatter.print(r._bind_validation_target(target))

    assert "The capital of France is Paris." in rendered
    assert "must mention Paris" in rendered


# --- the judgement request carries the conversation (issue #426, defect 4) ---


async def test_validate_hands_the_backend_a_context_that_renders_the_output():
    """The context reaching the backend must render the output under judgement.

    This is what makes adapter-backed requirement checking work at all:
    `_generate_from_intrinsic` builds the `requirement-check` conversation from
    `ctx.view_for_generation()`. Under the old throwaway `SimpleContext` that view was
    empty, so the adapter was handed only the injected requirement-check message and
    never saw the assistant turn it was supposed to judge.
    """
    seen: list = []
    target = ModelOutputThunk("The capital of France is Paris.")
    validation_ctx = (
        ChatContext().add(Message("user", "capital of France?")).add(target)
    )

    async def capture(action, ctx, **kwargs):
        seen.append((action, ctx))
        # A thunk constructed with a value is already computed.
        return ModelOutputThunk("yes"), ctx

    backend = MagicMock()
    backend.generate_from_context = capture

    result = await Requirement("must mention Paris").validate(backend, validation_ctx)

    assert result.as_bool() is True
    bound_req, passed_ctx = seen[0]
    view = passed_ctx.view_for_generation()
    assert view is not None and target in view, (
        "the judged output must be visible to the model, not merely present in as_list()"
    )
    assert any("capital of France?" in str(node) for node in view), (
        "the conversation that produced the output must reach the judge too"
    )
    assert bound_req._validation_target is target


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


@patch("mellea.stdlib.requirements.requirement.Intrinsic.__init__")
async def test_alora_validate_propagates_schema_mismatch(mock_intrinsic_init):
    """AdapterSchemaMismatchError from output_to_bool propagates uncaught through validate().

    This documents that the LLMaJ fallback in ALoraRequirement covers only
    generation errors, not output-parsing schema mismatches.
    """
    mock_intrinsic_init.return_value = None
    req = ALoraRequirement("must satisfy requirement")

    mock_thunk = MagicMock()
    mock_thunk.__str__ = MagicMock(return_value='{"requirement_likelihood": 0.9}')
    mock_thunk.avalue = AsyncMock(return_value=None)
    mock_thunk.value = '{"requirement_likelihood": 0.9}'

    mock_backend = MagicMock()
    mock_backend.generate_from_context = AsyncMock(return_value=(mock_thunk, ctx))

    with pytest.raises(AdapterSchemaMismatchError):
        await req.validate(mock_backend, ctx=ctx)


if __name__ == "__main__":
    pytest.main([__file__])
