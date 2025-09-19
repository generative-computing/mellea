import pytest

from mellea import MelleaSession, generative
from mellea.backends import ModelOption
from mellea.backends.litellm import LiteLLMBackend
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
    res = session.chat("hello world")
    assert res is not None
    assert isinstance(res, Message)


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
    res = session.instruct(
        "Write an email to the interns.",
        requirements=["be funny"],
        model_options={
            ModelOption.SEED: 123,
            ModelOption.TEMPERATURE: 0.5,
            ModelOption.THINKING: True,
            ModelOption.MAX_NEW_TOKENS: 100,
            "reasoning_effort": True,
            "stream": False,
            "homer_simpson": "option should be kicked out",
        },
    )
    assert res is not None
    assert isinstance(res.value, str)
    # make sure that homer_simpson is ignored for generation
    assert "homer_simpson" not in session.ctx.last_output_and_logs()[1].model_options


@pytest.mark.qualitative
def test_gen_slot(session):
    @generative
    def is_happy(text: str) -> bool:
        """Determine if text is of happy mood."""

    h = is_happy(session, text="I'm enjoying life.")

    assert isinstance(h, bool)
    # should yield to true - but, of course, is model dependent
    assert h is True


if __name__ == "__main__":
    import pytest

    pytest.main([__file__])
