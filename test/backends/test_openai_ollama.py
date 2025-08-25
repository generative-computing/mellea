# test/rits_backend_tests/test_openai_integration.py
import pydantic
import pytest
from typing_extensions import Annotated

from mellea import MelleaSession
from mellea.backends.formatter import TemplateFormatter
from mellea.backends.model_ids import META_LLAMA_3_2_1B
from mellea.backends.openai import OpenAIBackend
from mellea.backends.types import ModelOption
from mellea.stdlib.base import CBlock, LinearContext, ModelOutputThunk


@pytest.fixture(scope="module")
def backend(gh_run: int):
    """Shared OpenAI backend configured for Ollama."""
    if gh_run == 1:
        return OpenAIBackend(
        model_id=META_LLAMA_3_2_1B,
        formatter=TemplateFormatter(model_id=META_LLAMA_3_2_1B),
        base_url="http://localhost:11434/v1",
        api_key="ollama",
    )
    else:
        return OpenAIBackend(
            model_id="granite3.3:8b",
            formatter=TemplateFormatter(model_id="ibm-granite/granite-3.2-8b-instruct"),
            base_url="http://localhost:11434/v1",
            api_key="ollama",
        )


@pytest.fixture(scope="function")
def m_session(backend):
    """Fresh OpenAI session for each test."""
    session = MelleaSession(backend, ctx=LinearContext(is_chat_context=True))
    yield session
    session.reset()

@pytest.mark.llm
def test_instruct(m_session):
    result = m_session.instruct("Compute 1+1.")
    assert isinstance(result, ModelOutputThunk)
    assert "2" in result.value  # type: ignore

@pytest.mark.llm
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

@pytest.mark.llm
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

    # Ollama doesn't support batch requests. Cannot run this test unless we switch backend providers.
    # def test_generate_from_raw(self):
    #     prompts = ["what is 1+1?", "what is 2+2?", "what is 3+3?", "what is 4+4?"]

    #     results = self.m.backend._generate_from_raw(
    #         actions=[CBlock(value=prompt) for prompt in prompts], generate_logs=None
    #     )

    #     assert len(results) == len(prompts)

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


if __name__ == "__main__":
    import pytest

    pytest.main([__file__])
