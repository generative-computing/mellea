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
        model_id=model_ids.QWEN3_0_6B,
        # formatter=TemplateFormatter(model_id="ibm-granite/granite-4.0-tiny-preview"),
       )
    return backend

@pytest.fixture(scope="function")
def session(backend):
    """Fresh HuggingFace session for each test."""
    session = MelleaSession(backend, ctx=LinearContext())
    yield session
    session.reset()


@pytest.mark.qualitative
def test_system_prompt(session):
    result = session.chat(
        "Where are we going?",
        model_options={ModelOption.SYSTEM_PROMPT: "Talk like a pirate."},
    )
    print(result)


@pytest.mark.qualitative
def test_instruct(session):
    result = session.instruct("Compute 1+1.")
    print(result)



if __name__ == "__main__":
    import pytest

    pytest.main([__file__])
