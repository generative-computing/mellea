import json

import pydantic
import pytest
from typing_extensions import Annotated

from mellea import SimpleContext, start_session
from mellea.backends.types import ModelOption
from mellea.stdlib.base import CBlock
from mellea.stdlib.requirement import Requirement


@pytest.fixture(scope="function")
def session():
    """Fresh Ollama session for each test."""
    session = start_session()
    yield session
    session.reset()


@pytest.mark.qualitative
def test_simple_instruct(session):
    result = session.instruct(
        "Write an email to Hendrik trying to sell him self-sealing stembolts."
    )
    assert result.value.startswith("Subject")
    assert "chat_response" in result._meta
    assert result._meta["chat_response"].message.role == "assistant"


@pytest.mark.qualitative
def test_instruct_with_requirement(session):
    response = session.instruct(
        "Write an email to Hendrik convincing him to buy some self-sealing stembolts."
    )

    email_word_count_req = Requirement(
        f"The email should be at most 100",
        validation_fn=lambda x: len(" ".split(x.last_output().value)) <= 100,
    )

    happy_tone_req = Requirement(
        "The email should sound happy in tone.",
        output_to_bool=lambda x: "happy" in x.value,
    )

    sad_tone_req = Requirement("The email should sound sad in tone.")

    results = session.validate(
        reqs=[email_word_count_req, happy_tone_req, sad_tone_req]
    )
    print(results)

@pytest.mark.qualitative
def test_chat(session):
    output_message = session.chat("What is 1+1?")
    assert "2" in output_message.content, (
        f"Expected a message with content containing 2 but found {output_message}"
    )

@pytest.mark.qualitative
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

@pytest.mark.qualitative
def test_generate_from_raw(session):
    prompts = ["what is 1+1?", "what is 2+2?", "what is 3+3?", "what is 4+4?"]

    results = session.backend._generate_from_raw(
        actions=[CBlock(value=prompt) for prompt in prompts], generate_logs=None
    )

    assert len(results) == len(prompts)


@pytest.mark.xfail(reason="ollama sometimes fails generated structured outputs")
def test_generate_from_raw_with_format(session):
    prompts = ["what is 1+1?", "what is 2+2?", "what is 3+3?", "what is 4+4?"]

    class Answer(pydantic.BaseModel):
        name: str
        value: int

    results = session.backend._generate_from_raw(
        actions=[CBlock(value=prompt) for prompt in prompts],
        format=Answer,
        generate_logs=None,
    )

    assert len(results) == len(prompts)

    random_result = results[0]
    try:
        answer = Answer.model_validate_json(random_result.value)
    except pydantic.ValidationError as e:
        assert False, (
            f"formatting directive failed for {random_result.value}: {e.json()}"
        )


if __name__ == "__main__":
    pytest.main([__file__])
