# test/rits_backend_tests/test_openai_integration.py
import asyncio
import os

import openai
import pydantic
import pytest
from typing_extensions import Annotated

from mellea import MelleaSession
from mellea.backends.formatter import TemplateFormatter
from mellea.backends.model_ids import META_LLAMA_3_2_1B
from mellea.backends.openai import OpenAIBackend
from mellea.backends.types import ModelOption
from mellea.stdlib.base import CBlock, ChatContext, ModelOutputThunk, SimpleContext


@pytest.fixture(scope="module")
def backend(gh_run: int):
    """Shared OpenAI backend configured for Ollama."""
    if gh_run == 1:
        return OpenAIBackend(
            model_id=META_LLAMA_3_2_1B.ollama_name,  # type: ignore
            formatter=TemplateFormatter(model_id=META_LLAMA_3_2_1B.hf_model_name),  # type: ignore
            base_url=f"http://{os.environ.get('OLLAMA_HOST', 'localhost:11434')}/v1",
            api_key="ollama",
        )
    else:
        return OpenAIBackend(
            model_id="granite3.3:8b",
            formatter=TemplateFormatter(model_id="ibm-granite/granite-3.2-8b-instruct"),
            base_url=f"http://{os.environ.get('OLLAMA_HOST', 'localhost:11434')}/v1",
            api_key="ollama",
        )


@pytest.fixture(scope="function")
def m_session(backend):
    """Fresh OpenAI session for each test."""
    session = MelleaSession(backend, ctx=ChatContext())
    yield session
    session.reset()


@pytest.mark.qualitative
def test_instruct(m_session):
    result = m_session.instruct("Compute 1+1.")
    assert isinstance(result, ModelOutputThunk)
    assert "2" in result.value  # type: ignore


@pytest.mark.qualitative
def test_multiturn(m_session):
    m_session.instruct("What is the capital of France?")
    answer = m_session.instruct("Tell me the answer to the previous question.")
    assert "Paris" in answer.value  # type: ignore

    # def test_api_timeout_error(self):
    #     self.m.reset()
    #     # Mocking the client to raise timeout error is needed for full coverage
    #     # This test assumes the exception is properly propagated
    #     with self.assertRaises(Exception) as context:
    #         self.m.instruct("This should trigger a timeout.")
    #     assert "APITimeoutError" in str(context.exception)
    #     self.m.reset()

    # def test_model_id_usage(self):
    #     self.m.reset()
    #     result = self.m.instruct("What model are you using?")
    #     assert "granite3.3:8b" in result.value
    #     self.m.reset()


@pytest.mark.qualitative
def test_chat(m_session):
    output_message = m_session.chat("What is 1+1?")
    assert "2" in output_message.content, (
        f"Expected a message with content containing 2 but found {output_message}"
    )


@pytest.mark.qualitative
def test_format(m_session):
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

    output = m_session.instruct(
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
async def test_generate_from_raw(m_session):
    prompts = ["what is 1+1?", "what is 2+2?", "what is 3+3?", "what is 4+4?"]

    with pytest.raises(openai.BadRequestError):
        results = await m_session.backend.generate_from_raw(
            actions=[CBlock(value=prompt) for prompt in prompts], ctx=m_session.ctx
        )


# Default OpenAI implementation doesn't support structured outputs for the completions API.
# def test_generate_from_raw_with_format(self):
#     prompts = ["what is 1+1?", "what is 2+2?", "what is 3+3?", "what is 4+4?"]

#     class Answer(pydantic.BaseModel):
#         name: str
#         value: int

#     results = self.m.backend._generate_from_raw(
#         actions=[CBlock(value=prompt) for prompt in prompts],
#         format=Answer,
#         generate_logs=None,
#     )

#     assert len(results) == len(prompts)

#     random_result = results[0]
#     try:
#         answer = Answer.model_validate_json(random_result.value)
#     except pydantic.ValidationError as e:
#         assert False, f"formatting directive failed for {random_result.value}: {e.json()}"


async def test_async_parallel_requests(m_session):
    model_opts = {ModelOption.STREAM: True}
    mot1, _ = await m_session.backend.generate_from_context(
        CBlock("Say Hello."), SimpleContext(), model_options=model_opts
    )
    mot2, _ = await m_session.backend.generate_from_context(
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


async def test_async_avalue(m_session):
    mot1, _ = await m_session.backend.generate_from_context(
        CBlock("Say Hello."), SimpleContext()
    )
    m1_final_val = await mot1.avalue()
    assert m1_final_val is not None
    assert m1_final_val == mot1.value


def test_client_cache(backend):
    first_client = backend._async_client

    async def get_client_async():
        return backend._async_client

    second_client = asyncio.run(get_client_async())

    items_in_cache = backend._client_cache.cache.values()
    assert len(items_in_cache) == 2, (
        "should be two clients in the cache since _async_client was called from two event loops"
    )
    assert first_client in items_in_cache
    assert second_client in items_in_cache
    assert first_client is not second_client

    third_client = backend._async_client
    assert third_client is first_client, (
        "clients in sync code should be the same if haven't been pushed out of the cache"
    )

    fourth_client = asyncio.run(get_client_async())
    assert fourth_client in backend._client_cache.cache.values()
    assert len(backend._client_cache.cache.values()) == 2


async def test_reasoning_effort_conditional_passing(backend):
    """Test that reasoning_effort is only passed to API when not None."""
    from unittest.mock import AsyncMock, MagicMock, patch

    ctx = ChatContext()
    ctx = ctx.add(CBlock(value="Test"))

    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message = MagicMock()
    mock_response.choices[0].message.content = "Response"
    mock_response.choices[0].message.role = "assistant"

    # Test 1: reasoning_effort should NOT be passed when not specified
    with patch.object(
        backend._async_client.chat.completions, "create", new_callable=AsyncMock
    ) as mock_create:
        mock_create.return_value = mock_response
        await backend.generate_from_chat_context(
            CBlock(value="Hi"), ctx, model_options={}
        )
        call_kwargs = mock_create.call_args.kwargs
        assert "reasoning_effort" not in call_kwargs, (
            "reasoning_effort should not be passed when not specified"
        )

    # Test 2: reasoning_effort SHOULD be passed when specified
    with patch.object(
        backend._async_client.chat.completions, "create", new_callable=AsyncMock
    ) as mock_create:
        mock_create.return_value = mock_response
        await backend.generate_from_chat_context(
            CBlock(value="Hi"), ctx, model_options={ModelOption.THINKING: "medium"}
        )
        call_kwargs = mock_create.call_args.kwargs
        assert call_kwargs.get("reasoning_effort") == "medium", (
            "reasoning_effort should be passed with correct value when specified"
        )


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-k", "generate_from_raw"])
