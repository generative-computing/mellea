import asyncio
import os
import pytest

from mellea import MelleaSession, generative
from mellea.backends import ModelOption
from mellea.backends.litellm import LiteLLMBackend
from mellea.stdlib.base import CBlock, SimpleContext
from mellea.stdlib.chat import Message
from mellea.stdlib.sampling import RejectionSamplingStrategy
from mellea.backends import model_ids


_MODEL_ID = f"ollama_chat/{model_ids.IBM_GRANITE_4_MICRO_3B.ollama_name}"


@pytest.fixture(scope="function")
def backend(gh_run: int):
    """Shared OpenAI backend configured for Ollama."""
    if gh_run == 1:
        # LiteLLM prepends 127.0.0.1 with a `/` which causes issues.
        url = os.environ.get("OLLAMA_HOST", None)
        if url is None:
            url = "http://localhost:11434"
        else:
            url = url.replace("127.0.0.1", "http://localhost")

        return LiteLLMBackend(
            model_id=_MODEL_ID,
            base_url=url,
            model_options={"api_base": url},
        )
    else:
        return LiteLLMBackend(model_id=_MODEL_ID)


@pytest.fixture(scope="function")
def session(backend):
    """Fresh Ollama session for each test."""
    session = MelleaSession(backend=backend)
    yield session
    session.reset()


# Use capsys to check that the logging is working.
def test_make_backend_specific_and_remove():
    # Doesn't need to be a real model here; just a provider that LiteLLM knows about.
    backend = LiteLLMBackend(model_id="ollama_chat/")

    params = {
        "max_tokens": 1,
        "stream": 1,
        ModelOption.TEMPERATURE: 1,
        "unknown_parameter": 1,  # Unknown / non-OpenAI parameter
        "web_search_options": 1,  # Standard OpenAI parameter not supported by Ollama.
    }

    mellea = backend._simplify_and_merge(params)
    backend_specific = backend._make_backend_specific_and_remove(mellea)

    # All of these options should be in the model options that get passed to LiteLLM since it handles the dropping.
    assert "max_completion_tokens" in backend_specific, (
        "max_tokens should get remapped to max_completion_tokens for ollama_chat/"
    )
    assert "stream" in backend_specific
    assert "temperature" in backend_specific
    assert "unknown_parameter" in backend_specific
    assert "web_search_options" in backend_specific

    # TODO: Investigate why this isn't working on github action runners.
    #       Add the capsys or capfd fixture back.
    # out = capsys.readouterr()
    # # Check for the specific warning logs.
    # assert "supported by the current model/provider: web_search_options" in out.out
    # assert "mellea won't validate the following params that may cause issues: unknown_parameter" in out.out

    # Do a quick test for the Watsonx specific scenario.
    backend = LiteLLMBackend(model_id="watsonx/")
    watsonx_params = {"max_tokens": 1}

    # Make sure we make it Mellea specific correctly.
    watsonx_mellea = backend._simplify_and_merge(watsonx_params)
    assert ModelOption.MAX_NEW_TOKENS in watsonx_mellea

    watsonx_backend_specific = backend._make_backend_specific_and_remove(watsonx_mellea)
    assert "max_tokens" in watsonx_backend_specific


@pytest.mark.qualitative
def test_litellm_ollama_chat(session):
    res = session.chat("What is 1+1?")
    assert res is not None
    assert isinstance(res, Message)
    assert "2" in res.content, (
        f"Expected a message with content containing 2 but found {res}"
    )


def test_litellm_ollama_instruct(session):
    res = session.instruct(
        "Write an email to the interns.",
        requirements=["be funny"],
        strategy=RejectionSamplingStrategy(loop_budget=3),
    )
    assert res is not None
    assert isinstance(res.value, str)


def test_litellm_ollama_instruct_options(session):
    model_options = {
        ModelOption.SEED: 123,
        ModelOption.TEMPERATURE: 0.5,
        ModelOption.MAX_NEW_TOKENS: 100,
        
        # Ollama thinking controls currently broken on Granite; see 
        # https://github.com/ollama/ollama/issues/10983
        # TODO: Re-enable when this upstream bug gets fixed.
        #ModelOption.THINKING: True,
        #"reasoning_effort": True,
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


@pytest.mark.qualitative
def test_gen_slot(session):
    @generative
    def is_happy(text: str) -> bool:
        """Determine if text is of happy mood."""

    h = is_happy(session, text="I'm enjoying life.")

    assert isinstance(h, bool)
    # should yield to true - but, of course, is model dependent
    assert h is True


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


async def test_async_avalue(session):
    mot1, _ = session.backend.generate_from_context(
        CBlock("Say Hello."), SimpleContext()
    )
    m1_final_val = await mot1.avalue()
    assert m1_final_val is not None
    assert m1_final_val == mot1.value


if __name__ == "__main__":
    import pytest

    pytest.main([__file__])
