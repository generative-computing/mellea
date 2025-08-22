# test/rits_backend_tests/test_watsonx_integration.py
import os
from mellea import MelleaSession
from mellea.stdlib.base import CBlock, LinearContext, ModelOutputThunk
from mellea.backends.watsonx import WatsonxAIBackend
from mellea.backends.formatter import TemplateFormatter
from mellea.backends.types import ModelOption

import pydantic
from typing_extensions import Annotated
import pytest


@pytest.fixture(scope="module")
def backend():
    """Shared Watson backend for all tests in this module."""
    return WatsonxAIBackend(
        model_id="ibm/granite-3-3-8b-instruct",
        formatter=TemplateFormatter(model_id="ibm-granite/granite-3.3-8b-instruct"),
    )


@pytest.fixture(scope="function")
def session(backend):
    """Fresh Watson session for each test."""
    session = MelleaSession(backend, ctx=LinearContext(is_chat_context=True))
    yield session
    session.reset()




def test_instruct(session):
    result = session.instruct("Compute 1+1.")
    assert isinstance(result, ModelOutputThunk)
    assert "2" in result.value  # type: ignore

def test_multiturn(session):
    session.instruct("What is the capital of France?")
    answer = session.instruct("Tell me the answer to the previous question.")
    assert "Paris" in answer.value  # type: ignore

def test_format(session):
    class Person(pydantic.BaseModel):
        name: str
        # it does not support regex patterns in json schema
        email_address: str
        # email_address: Annotated[
        #     str,
        #     pydantic.StringConstraints(pattern=r"[a-zA-Z]{5,10}@example\.com"),
        # ]

    class Email(pydantic.BaseModel):
        to: Person
        subject: str
        body: str

    output = session.instruct(
        "Write a short email to Olivia, thanking her for organizing a sailing activity. Her email server is example.com. No more than two sentences. ",
        format=Email,
        model_options={ModelOption.MAX_NEW_TOKENS: 2**8},
    )
    print("Formatted output:")
    email = Email.model_validate_json(
        output.value
    )  # this should succeed because the output should be JSON because we passed in a format= argument...
    print(email)

    print("address:", email.to.email_address)
    # this is not guaranteed, due to the lack of regexp pattern
    # assert "@" in email.to.email_address
    # assert email.to.email_address.endswith("example.com")
    pass

def test_generate_from_raw(session):
    prompts = ["what is 1+1?", "what is 2+2?", "what is 3+3?", "what is 4+4?"]

    results = session.backend._generate_from_raw(
        actions=[CBlock(value=prompt) for prompt in prompts], generate_logs=None
    )

    assert len(results) == len(prompts)


if __name__ == "__main__":
    import pytest

    pytest.main([__file__])
