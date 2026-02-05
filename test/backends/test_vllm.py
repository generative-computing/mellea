import asyncio
import os
from typing import Annotated

import pydantic
import pytest

# Mark all tests in this module with backend and resource requirements
pytestmark = [
    pytest.mark.vllm,
    pytest.mark.llm,
    pytest.mark.requires_gpu,
    pytest.mark.requires_heavy_ram,
    # Skip entire module in CI since all 8 tests are qualitative
    pytest.mark.skipif(
        int(os.environ.get("CICD", 0)) == 1,
        reason="Skipping vLLM tests in CI - all qualitative tests",
    ),
]

import mellea.backends.model_ids as model_ids
from mellea import MelleaSession
from mellea.backends import ModelOption
from mellea.backends.vllm import LocalVLLMBackend
from mellea.core import CBlock
from mellea.stdlib.context import ChatContext, SimpleContext


@pytest.fixture(scope="module")
def backend():
    """Shared vllm backend for all tests in this module."""
    # Import cleanup dependencies at top to avoid scoping issues
    import gc
    import time

    import torch

    if os.environ.get("VLLM_USE_V1", -1) != "0":
        pytest.skip("skipping vllm tests; tests require `export VLLM_USE_V1=0`")

    backend = LocalVLLMBackend(
        model_id=model_ids.QWEN3_0_6B,
        # formatter=TemplateFormatter(model_id="ibm-granite/granite-4.0-tiny-preview"),
        model_options={
            # made smaller for a testing environment with smaller gpus.
            # such an environment could possibly be running other gpu applications, including slack
            "gpu_memory_utilization": 0.8,
            "max_model_len": 8192,
            "max_num_seqs": 8,
        },
    )
    yield backend
    # Cleanup: shutdown vLLM engine and release GPU memory

    # Shutdown the background loop
    backend._underlying_model.shutdown_background_loop()

    # Delete the model and clear references
    del backend._underlying_model
    del backend

    # Force garbage collection and clear CUDA cache
    gc.collect()

    if torch.cuda.is_available():
        # Synchronize to ensure all CUDA operations complete
        torch.cuda.synchronize()
        # Empty the cache multiple times to ensure memory is released
        torch.cuda.empty_cache()

        # Set environment variable to help with memory fragmentation
        # This tells PyTorch to use expandable memory segments
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

        # Reset peak memory stats
        torch.cuda.reset_peak_memory_stats()
        # Reset accumulated memory stats
        torch.cuda.reset_accumulated_memory_stats()

        # Multiple cleanup passes with delays to allow CUDA runtime to release memory
        for _ in range(3):
            gc.collect()
            torch.cuda.empty_cache()
            time.sleep(1)


@pytest.fixture(scope="function")
def session(backend):
    """Fresh HuggingFace session for each test."""
    session = MelleaSession(backend, ctx=ChatContext())
    yield session
    session.reset()


@pytest.mark.qualitative
def test_system_prompt(session) -> None:
    result = session.chat(
        "Where are we going?",
        model_options={ModelOption.SYSTEM_PROMPT: "Talk like a pirate."},
    )
    print(result)


@pytest.mark.qualitative
def test_instruct(session) -> None:
    result = session.instruct("Compute 1+1.")
    print(result)


@pytest.mark.qualitative
def test_multiturn(session) -> None:
    session.instruct("Compute 1+1")
    session.instruct(
        "Take the result of the previous sum and find the corresponding letter in the greek alphabet."
    )
    words = session.instruct("Now list five English words that start with that letter.")
    print(words)


@pytest.mark.qualitative
def test_format(session) -> None:
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
async def test_generate_from_raw(session) -> None:
    prompts = ["what is 1+1?", "what is 2+2?", "what is 3+3?", "what is 4+4?"]

    results = await session.backend.generate_from_raw(
        actions=[CBlock(value=prompt) for prompt in prompts], ctx=session.ctx
    )

    assert len(results) == len(prompts)
    assert results[0].value is not None


@pytest.mark.qualitative
async def test_generate_from_raw_with_format(session) -> None:
    prompts = ["what is 1+1?", "what is 2+2?", "what is 3+3?", "what is 4+4?"]

    class Answer(pydantic.BaseModel):
        name: str
        value: int

    results = await session.backend.generate_from_raw(
        actions=[CBlock(value=prompt) for prompt in prompts],
        ctx=session.ctx,
        format=Answer,
        model_options={ModelOption.MAX_NEW_TOKENS: 100},
    )

    assert len(results) == len(prompts)

    random_result = results[0]
    try:
        Answer.model_validate_json(random_result.value)
    except pydantic.ValidationError as e:
        assert False, (
            f"formatting directive failed for {random_result.value}: {e.json()}"
        )


@pytest.mark.qualitative
def test_async_parallel_requests(session) -> None:
    async def parallel_requests():
        model_opts = {ModelOption.STREAM: True}
        mot1, _ = await session.backend.generate_from_context(
            CBlock("Say Hello."), SimpleContext(), model_options=model_opts
        )
        mot2, _ = await session.backend.generate_from_context(
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

    asyncio.run(parallel_requests())


@pytest.mark.qualitative
def test_async_avalue(session) -> None:
    async def avalue():
        mot1, _ = await session.backend.generate_from_context(
            CBlock("Say Hello."), SimpleContext()
        )
        m1_final_val = await mot1.avalue()
        assert m1_final_val is not None
        assert m1_final_val == mot1.value

    asyncio.run(avalue())


if __name__ == "__main__":
    import pytest

    pytest.main([__file__])
