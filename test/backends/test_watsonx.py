# test/rits_backend_tests/test_watsonx_integration.py
import asyncio
import os

import pydantic
import pytest

from mellea import MelleaSession
from mellea.backends.formatter import TemplateFormatter
from mellea.backends.types import ModelOption
from mellea.backends.watsonx import WatsonxAIBackend
from mellea.stdlib.base import CBlock, ModelOutputThunk, ChatContext, SimpleContext


@pytest.fixture(scope="module")
def backend():
    """Shared Watson backend for all tests in this module."""
    if int(os.environ.get("CICD", 0)) == 1:
        pytest.skip("Skipping watsonx tests.")
    else:
        return WatsonxAIBackend(
            model_id="ibm/granite-3-3-8b-instruct",
            formatter=TemplateFormatter(model_id="ibm-granite/granite-3.3-8b-instruct"),
        )


@pytest.fixture(scope="function")
def session(backend: WatsonxAIBackend):
    if int(os.environ.get("CICD", 0)) == 1:
        pytest.skip("Skipping watsonx tests.")
    else:
        """Fresh Watson session for each test."""
        session = MelleaSession(backend, ctx=ChatContext())
        yield session
        session.reset()


@pytest.mark.qualitative
def test_filter_chat_completions_kwargs(backend: WatsonxAIBackend):
    """Detect changes to the WatsonxAI TextChatParameters."""

    known_keys = [
        "frequency_penalty",
        "logprobs",
        "top_logprobs",
        "presence_penalty",
        "response_format",
        "temperature",
        "max_tokens",
        "max_completion_tokens",
        "time_limit",
        "top_p",
        "n",
        "logit_bias",
        "seed",
        "stop",
        "guided_choice",
        "guided_regex",
        "guided_grammar",
        "guided_json",
    ]
    test_dict = {key: 1 for key in known_keys}

    # Make sure keys that we think should be in the TextChatParameters are there.
    filtered_dict = backend.filter_chat_completions_kwargs(test_dict)

    for key in known_keys:
        assert key in filtered_dict

    # Make sure unsupported keys still get filtered out.
    incorrect_dict = {"random": 1}
    filtered_incorrect_dict = backend.filter_chat_completions_kwargs(incorrect_dict)
    assert "random" not in filtered_incorrect_dict


@pytest.mark.qualitative
def test_instruct(session: MelleaSession):
    result = session.instruct("Compute 1+1.")
    assert isinstance(result, ModelOutputThunk)
    assert "2" in result.value  # type: ignore


@pytest.mark.qualitative
def test_multiturn(session: MelleaSession):
    session.instruct("What is the capital of France?")
    answer = session.instruct("Tell me the answer to the previous question.")
    assert "Paris" in answer.value  # type: ignore


@pytest.mark.qualitative
def test_chat(session):
    output_message = session.chat("What is 1+1?")
    assert "2" in output_message.content, (
        f"Expected a message with content containing 2 but found {output_message}"
    )


@pytest.mark.qualitative
def test_format(session: MelleaSession):
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
def test_generate_from_raw(session: MelleaSession):
    prompts = ["what is 1+1?", "what is 2+2?", "what is 3+3?", "what is 4+4?"]

    results = session.backend.generate_from_raw(
        actions=[CBlock(value=prompt) for prompt in prompts], ctx=session.ctx
    )

    assert len(results) == len(prompts)


@pytest.mark.qualitative
async def test_async_parallel_requests(session):
    model_opts = {ModelOption.STREAM: True}
    mot1, _ = session.backend.generate_from_context(
        CBlock("Say Hello."), SimpleContext(), model_options=model_opts
    )
    mot2, _ = session.backend.generate_from_context(
        CBlock("Say Goodbye!"), SimpleContext(), model_options=model_opts
    )

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
    assert m1_final_val.startswith(m1_val), (
        "final val should contain the first streamed chunk"
    )
    assert m2_final_val.startswith(m2_val), (
        "final val should contain the first streamed chunk"
    )

    assert m1_final_val == mot1.value
    assert m2_final_val == mot2.value


@pytest.mark.qualitative
async def test_async_avalue(session):
    mot1, _ = session.backend.generate_from_context(
        CBlock("Say Hello."), SimpleContext()
    )
    m1_final_val = await mot1.avalue()
    assert m1_final_val is not None
    assert m1_final_val == mot1.value


def test_client_cache(backend):
    first_client = backend._model

    async def get_client_async():
        return backend._model

    second_client = asyncio.run(get_client_async())

    items_in_cache = backend._client_cache.cache.values()
    assert len(items_in_cache) == 2, (
        "should be two clients in the cache since _async_client was called from two event loops"
    )
    assert first_client in items_in_cache
    assert second_client in items_in_cache

    third_client = backend._model
    assert third_client == first_client, (
        "clients in sync code should be the same if haven't been pushed out of the cache"
    )

    fourth_client = asyncio.run(get_client_async())
    assert fourth_client in backend._client_cache.cache.values()
    assert len(backend._client_cache.cache.values()) == 2


if __name__ == "__main__":
    import pytest

    pytest.main([__file__])
