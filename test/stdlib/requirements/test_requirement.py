import pytest

from mellea.core import ModelOutputThunk, Requirement
from mellea.stdlib.context import ChatContext
from mellea.stdlib.requirements import LLMaJRequirement, simple_validate
from mellea.stdlib.session import start_session

ctx = ChatContext()
ctx = ctx.add(ModelOutputThunk("test"))


async def test_llmaj_validation_req_output_field():
    m = start_session(ctx=ctx)
    req = Requirement("Must output test.")
    assert req._output is None

    _ = await req.validate(m.backend, ctx=ctx)
    assert req._output is None, (
        "requirement's output shouldn't be updated during/after validation"
    )


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


if __name__ == "__main__":
    pytest.main([__file__])
