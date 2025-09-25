import os
import pydantic
import pytest
from typing_extensions import Annotated

from mellea import MelleaSession
from mellea.backends.sglang import LocalSGLangBackend
from mellea.backends.types import ModelOption
import mellea.backends.model_ids as model_ids
from mellea.stdlib.base import CBlock, LinearContext
from mellea.stdlib.requirement import (
    LLMaJRequirement,
    Requirement,
    ValidationResult,
    default_output_to_bool,
)


@pytest.fixture(scope="module")
def backend():
    """Shared vllm backend for all tests in this module."""
    backend = LocalSGLangBackend(
        model_id=model_ids.MISTRALAI_MISTRAL_0_3_7B,
       )
    return backend

@pytest.fixture(scope="function")
def session(backend):
    """Fresh HuggingFace session for each test."""
    session = MelleaSession(backend, ctx=LinearContext())
    yield session
    session.reset()


@pytest.mark.qualitative
def test_tool(session):

    tool_call_history = []
    def get_temperature(location: str) -> int:
        """Returns today's temperature of the given city in Celsius.

        Args:
            location: a city name.
        """
        tool_call_history.append(location)
        return 21

    output = session.instruct(
        "What is today's temperature in Boston? Answer in Celsius. Reply the number only.",
        model_options={
            ModelOption.TOOLS: [get_temperature,],
            ModelOption.MAX_NEW_TOKENS: 1000,
        },
        tool_calls = True,
    )

    assert output.tool_calls is not None

    result = output.tool_calls["get_temperature"].call_func()
    print(result)

    assert len(tool_call_history) > 0
    assert tool_call_history[0].lower() == "boston"
    assert 21 == result


if __name__ == "__main__":
    import pytest

    pytest.main([__file__])
