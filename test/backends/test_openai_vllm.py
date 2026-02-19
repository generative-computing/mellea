# test/rits_backend_tests/test_openai_integration.py
import os
import signal
import subprocess
import time

import openai
import pydantic
import pytest
import requests

# Mark all tests in this module with backend and resource requirements
pytestmark = [
    pytest.mark.openai,
    pytest.mark.llm,
    pytest.mark.vllm,
    pytest.mark.requires_gpu,
    pytest.mark.requires_heavy_ram,
    # Skip entire module in CI since all 8 tests are qualitative
    pytest.mark.skipif(
        int(os.environ.get("CICD", 0)) == 1,
        reason="Skipping vLLM tests in CI - all qualitative tests",
    ),
]

# Try to import vLLM backend - skip all tests if not available

import mellea.backends.model_ids as model_ids
from mellea import MelleaSession
from mellea.backends import ModelOption
from mellea.backends.model_ids import IBM_GRANITE_4_MICRO_3B
from mellea.backends.openai import OpenAIBackend
from mellea.core import CBlock, ModelOutputThunk
from mellea.formatters import TemplateFormatter
from mellea.stdlib.context import ChatContext


@pytest.fixture(scope="module")
def vllm_process():
    """Shared vllm process for all tests in this module."""
    process = None
    try:
        process = subprocess.Popen(
            [
                "vllm",
                "serve",
                IBM_GRANITE_4_MICRO_3B.hf_model_name,
                "--enable-lora",
                "--dtype",
                "bfloat16",
                "--max-lora-rank",
                "64",
                "--enable-prefix-caching",
            ],
            # the process will have a new session id, so
            # entire process tree is killable at once
            start_new_session=True,
        )
        url = "http://127.0.0.1:8000/ping"
        timeout = 600  # vllm initialization takes quite a while
        start_time = time.time()

        # Wait for readiness message
        while True:
            if process.poll() is not None:
                raise RuntimeError("vLLM server exited before startup.")

            try:
                response = requests.get(url, timeout=2)
                if response.status_code == 200:
                    break
            except requests.RequestException:
                pass

            if time.time() - start_time > timeout:
                raise TimeoutError(
                    f"Timed out waiting for server health check at {url}"
                )

        yield process

    # --- Teardown (always runs) ---
    finally:
        try:
            os.killpg(process.pid, signal.SIGTERM)  # kill the session group
            process.wait(timeout=30)
        except Exception:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except Exception:
                pass
            process.wait()


@pytest.fixture(scope="module")
def backend(gh_run: int, vllm_process: subprocess.Popen):
    """Shared OpenAI backend configured for Ollama."""
    return OpenAIBackend(
        model_id=IBM_GRANITE_4_MICRO_3B.hf_model_name,  # type: ignore
        formatter=TemplateFormatter(model_id=IBM_GRANITE_4_MICRO_3B.hf_model_name),  # type: ignore
        base_url="http://0.0.0.0:8000/v1",
        api_key="EMPTY",
    )


@pytest.fixture(scope="function")
def m_session(backend: OpenAIBackend):
    """Fresh OpenAI session for each test."""
    session = MelleaSession(backend, ctx=ChatContext())
    yield session
    session.reset()


@pytest.mark.qualitative
def test_instruct(m_session: MelleaSession) -> None:
    result = m_session.instruct("Compute 1+1.")
    assert isinstance(result, ModelOutputThunk)
    assert "2" in result.value  # type: ignore


@pytest.mark.qualitative
def test_multiturn(m_session: MelleaSession) -> None:
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
def test_chat(m_session: MelleaSession) -> None:
    output_message = m_session.chat("What is 1+1?")
    assert "2" in output_message.content, (
        f"Expected a message with content containing 2 but found {output_message}"
    )


@pytest.mark.qualitative
def test_chat_stream(m_session: MelleaSession) -> None:
    output_message = m_session.chat(
        "What is 1+1?", model_options={ModelOption.STREAM: True}
    )
    assert "2" in output_message.content, (
        f"Expected a message with content containing 2 but found {output_message}"
    )


@pytest.mark.qualitative
def test_format(m_session: MelleaSession) -> None:
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
        output.value  # type: ignore
    )  # this should succeed because the output should be JSON because we passed in a format= argument...
    print(email)

    print("address:", email.to.email_address)
    # this is not guaranteed, due to the lack of regexp pattern
    # assert "@" in email.to.email_address
    # assert email.to.email_address.endswith("example.com")


@pytest.mark.qualitative
async def test_generate_from_raw(m_session: MelleaSession) -> None:
    prompts = ["what is 1+1?", "what is 2+2?", "what is 3+3?", "what is 4+4?"]

    results = await m_session.backend.generate_from_raw(
        actions=[CBlock(value=prompt) for prompt in prompts], ctx=m_session.ctx
    )

    assert len(results) == len(prompts)
    assert results[0].value is not None


@pytest.mark.qualitative
async def test_generate_from_raw_with_format(m_session: MelleaSession) -> None:
    prompts = ["what is 1+1?", "what is 2+2?", "what is 3+3?", "what is 4+4?"]

    class Answer(pydantic.BaseModel):
        name: str
        value: int

    results = await m_session.backend.generate_from_raw(
        actions=[CBlock(value=prompt) for prompt in prompts],
        format=Answer,
        ctx=m_session.ctx,
    )

    assert len(results) == len(prompts)

    random_result = results[0]
    try:
        Answer.model_validate_json(random_result.value)  # type: ignore
    except pydantic.ValidationError as e:
        assert False, (
            f"formatting directive failed for {random_result.value}: {e.json()}"
        )


if __name__ == "__main__":
    pytest.main([__file__])
