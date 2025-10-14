import asyncio
import pytest

from mellea import MelleaSession, generative
from mellea.backends import ModelOption
from mellea.backends.litellm import LiteLLMBackend
from mellea.stdlib.base import CBlock, SimpleContext
from mellea.stdlib.chat import Message
from mellea.stdlib.sampling import RejectionSamplingStrategy


@pytest.fixture(scope="function")
def session():
    """Fresh Ollama session for each test."""
    session = MelleaSession(LiteLLMBackend())
    yield session
    session.reset()


@pytest.mark.qualitative
def test_litellm_ollama_chat(session):
    res = session.chat("What is 1+1?")
    assert res is not None
    assert isinstance(res, Message)
    assert "2" in res.content, (
        f"Expected a message with content containing 2 but found {res}"
    )

@pytest.mark.qualitative
def test_litellm_ollama_instruct(session):
    res = session.instruct(
        "Write an email to the interns.",
        requirements=["be funny"],
        strategy=RejectionSamplingStrategy(loop_budget=3),
    )
    assert res is not None
    assert isinstance(res.value, str)


@pytest.mark.qualitative
def test_litellm_ollama_instruct_options(session):
    model_options={
        ModelOption.SEED: 123,
        ModelOption.TEMPERATURE: 0.5,
        ModelOption.THINKING: True,
        ModelOption.MAX_NEW_TOKENS: 100,
        "reasoning_effort": True,
        "homer_simpson": "option should be kicked out",
    }

    res = session.instruct(
        "Write an email to the interns.",
        requirements=["be funny"],
        model_options=model_options,
    )
    assert res is not None
    assert isinstance(res.value, str)

    # make sure that homer_simpson is in the logged model_options
    assert "homer_simpson" in res._generate_log.model_options

    # make sure the backend function filters out the model option when passing to the generate call
    backend = session.backend
    assert isinstance(backend, LiteLLMBackend)
    assert "homer_simpson" not in backend._make_backend_specific_and_remove(model_options)


@pytest.mark.qualitative
def test_gen_slot(session):
    @generative
    def is_happy(text: str) -> bool:
        """Determine if text is of happy mood."""

    h = is_happy(session, text="I'm enjoying life.")

    assert isinstance(h, bool)
    # should yield to true - but, of course, is model dependent
    assert h is True

@pytest.mark.qualitative
async def test_async_parallel_requests(session):
    model_opts = {ModelOption.STREAM: True}
    mot1, _ = session.backend.generate_from_context(CBlock("Say Hello."), SimpleContext(), model_options=model_opts)
    mot2, _ = session.backend.generate_from_context(CBlock("Say Goodbye!"), SimpleContext(), model_options=model_opts)

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

@pytest.mark.qualitative
async def test_async_avalue(session):
    mot1, _ = session.backend.generate_from_context(CBlock("Say Hello."), SimpleContext())
    m1_final_val = await mot1.avalue()
    assert m1_final_val is not None
    assert m1_final_val == mot1.value

if __name__ == "__main__":
    import pytest

    pytest.main([__file__])
