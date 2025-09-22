import asyncio
import pydantic
import pytest
from typing_extensions import Annotated

from mellea import MelleaSession
from mellea.backends.aloras.huggingface.granite_aloras import add_granite_aloras
from mellea.backends.cache import SimpleLRUCache
from mellea.backends.formatter import TemplateFormatter
from mellea.backends.huggingface import LocalHFBackend
from mellea.backends.types import ModelOption
from mellea.stdlib.base import CBlock, LinearContext, SimpleContext
from mellea.stdlib.requirement import (
    ALoraRequirement,
    LLMaJRequirement,
    Requirement,
    ValidationResult,
    default_output_to_bool,
)


@pytest.fixture(scope="module")
def backend():
    """Shared HuggingFace backend for all tests in this module."""
    backend = LocalHFBackend(
        model_id="ibm-granite/granite-3.2-8b-instruct",
        formatter=TemplateFormatter(model_id="ibm-granite/granite-4.0-tiny-preview"),
        cache=SimpleLRUCache(5),
    )
    add_granite_aloras(backend)
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
def test_constraint_alora(session, backend):
    answer = session.instruct(
        "Corporate wants you to find the difference between these two strings: aaaaaaaaaa aaaaabaaaa. Be concise and don't write code to answer the question.",
        model_options={
            ModelOption.MAX_NEW_TOKENS: 300
        },  # Until aloras get a bit better, try not to abruptly end generation.
    )

    async def alora_generate():
        alora_output = backend.get_aloras()[
            0
        ].generate_using_strings(
            input="Find the difference between these two strings: aaaaaaaaaa aaaaabaaaa",
            response=str(answer),
            constraint="The answer mention that there is a b in the middle of one of the strings but not the other.",
            force_yn=False,  # make sure that the alora naturally output Y and N without constrained generation
        )
        await alora_output.avalue()
        assert alora_output.value in ["Y", "N"], alora_output
    asyncio.run(alora_generate())

@pytest.mark.qualitative
def test_constraint_lora_with_requirement(session, backend):
    answer = session.instruct(
        "Corporate wants you to find the difference between these two strings: aaaaaaaaaa aaaaabaaaa"
    )
    assert session.backend._cache is not None  # type: ignore
    assert session.backend._use_caches
    assert backend._cache.current_size() != 0
    validation_outputs = session.validate(
        "The answer should mention that there is a b in the middle of one of the strings but not the other."
    )
    assert len(validation_outputs) == 1
    val_result = validation_outputs[0]
    assert isinstance(val_result, ValidationResult)
    assert str(val_result.reason) in ["Y", "N"]


@pytest.mark.qualitative
def test_constraint_lora_override(session, backend):
    backend.default_to_constraint_checking_alora = False  # type: ignore
    answer = session.instruct(
        "Corporate wants you to find the difference between these two strings: aaaaaaaaaa aaaaabaaaa"
    )
    validation_outputs = session.validate(
        "The answer should mention that there is a b in the middle of one of the strings but not the other."
    )
    assert len(validation_outputs) == 1
    val_result = validation_outputs[0]
    assert isinstance(val_result, ValidationResult)
    assert isinstance(default_output_to_bool(str(val_result.reason)), bool)
    backend.default_to_constraint_checking_alora = True


@pytest.mark.qualitative
def test_constraint_lora_override_does_not_override_alora(session, backend):
    backend.default_to_constraint_checking_alora = False  # type: ignore
    answer = session.instruct(
        "Corporate wants you to find the difference between these two strings: aaaaaaaaaa aaaaabaaaa"
    )
    validation_outputs = session.validate(
        ALoraRequirement(
            "The answer should mention that there is a b in the middle of one of the strings but not the other."
        )
    )
    assert len(validation_outputs) == 1
    val_result = validation_outputs[0]
    assert isinstance(val_result, ValidationResult)
    assert str(val_result.reason) in ["Y", "N"]
    backend.default_to_constraint_checking_alora = True


@pytest.mark.qualitative
def test_llmaj_req_does_not_use_alora(session, backend):
    backend.default_to_constraint_checking_alora = True  # type: ignore
    answer = session.instruct(
        "Corporate wants you to find the difference between these two strings: aaaaaaaaaa aaaaabaaaa"
    )
    validation_outputs = session.validate(
        LLMaJRequirement(
            "The answer should mention that there is a b in the middle of one of the strings but not the other."
        )
    )
    assert len(validation_outputs) == 1
    val_result = validation_outputs[0]
    assert isinstance(val_result, ValidationResult)
    assert str(val_result.reason) not in ["Y", "N"]

@pytest.mark.qualitative
def test_instruct(session):
    result = session.instruct("Compute 1+1.")
    print(result)


@pytest.mark.qualitative
def test_multiturn(session):
    session.instruct("Compute 1+1")
    beta = session.instruct(
        "Take the result of the previous sum and find the corresponding letter in the greek alphabet."
    )
    assert "β" in str(beta).lower()
    words = session.instruct("Now list five English words that start with that letter.")
    print(words)

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
        email_address: Annotated[
            str, pydantic.StringConstraints(pattern=r"[a-zA-Z]{5,10}@example\.com")
        ]

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
    assert "@" in email.to.email_address, "The @ sign should be in the meail address."
    assert email.to.email_address.endswith("example.com"), (
        "The email address should be at example.com"
    )

@pytest.mark.qualitative
def test_generate_from_raw(session):
    prompts = ["what is 1+1?", "what is 2+2?", "what is 3+3?", "what is 4+4?"]

    results = session.backend._generate_from_raw(
        actions=[CBlock(value=prompt) for prompt in prompts], generate_logs=None
    )

    assert len(results) == len(prompts)


@pytest.mark.qualitative
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

def test_async_parallel_requests(session):
    async def parallel_requests():
        model_opts = {ModelOption.STREAM: True}
        mot1 = session.backend.generate_from_context(CBlock("Say Hello."), SimpleContext(), model_options=model_opts)
        mot2 = session.backend.generate_from_context(CBlock("Say Goodbye!"), SimpleContext(), model_options=model_opts)

        m1_val = None
        m2_val = None
        if not mot1.is_computed():
            m1_val = await mot1.astream()
        if not mot2.is_computed():
            m2_val = await mot2.astream()
        
        assert m1_val is not None, "should be a string val after generation"
        assert m2_val is not None, "should be a string val after generation"

        m1_final_val = await mot1.avalue()
        m2_final_val = await mot2.avalue()

        # Ideally, we would be able to assert that m1_final_val != m1_val, but sometimes the first streaming response
        # contains the full response.
        assert m1_final_val.startswith(m1_val), "final val should contain the first streamed chunk"
        assert m2_final_val.startswith(m2_val), "final val should contain the first streamed chunk"

        assert m1_final_val == mot1.value
        assert m2_final_val == mot2.value
    asyncio.run(parallel_requests())

def test_async_avalue(session):
    async def avalue():
        mot1 = session.backend.generate_from_context(CBlock("Say Hello."), SimpleContext())
        m1_final_val = await mot1.avalue()
        assert m1_final_val is not None
        assert m1_final_val == mot1.value
    asyncio.run(avalue())

if __name__ == "__main__":
    import pytest

    pytest.main([__file__])
