import asyncio

import pytest

# Mark all tests in this module as requiring Watsonx via LiteLLM
pytestmark = [
    pytest.mark.litellm,
    pytest.mark.watsonx,
    pytest.mark.llm,
    pytest.mark.requires_api_key,
]

from mellea import MelleaSession
from mellea.backends.litellm import LiteLLMBackend
from mellea.core import CBlock


@pytest.fixture(scope="function")
def session():
    """Fresh Ollama session for each test."""
    session = MelleaSession(LiteLLMBackend(model_id="watsonx/ibm/granite-4-h-small"))
    yield session
    session.reset()


def test_has_potential_event_loop_errors(session) -> None:
    """This test is specific to litellm backends that use watsonx/. It can be removed once that bug is fixed."""
    backend: LiteLLMBackend = session.backend
    potential_err = backend._has_potential_event_loop_errors()
    assert not potential_err, "first invocation in an event loop shouldn't flag errors"

    potential_err = backend._has_potential_event_loop_errors()
    assert not potential_err, (
        "second invocation in the same event loop shouldn't flag errors"
    )

    async def new_event_loop() -> bool:
        return backend._has_potential_event_loop_errors()

    err_expected = asyncio.run(new_event_loop())
    assert err_expected, "invocation in a new event loop should flag an error"


@pytest.mark.qualitative
def test_multiple_sync_funcs(session) -> None:
    session.chat("first")
    session.chat("second")


@pytest.mark.qualitative
async def test_generate_from_raw(session) -> None:
    prompts = ["what is 1+1?", "what is 2+2?", "what is 3+3?", "what is 4+2+2?"]

    results = await session.backend.generate_from_raw(
        actions=[CBlock(value=prompt) for prompt in prompts], ctx=session.ctx
    )

    assert len(results) == 1, (
        "litellm converts a batch request for watsonx into a single message"
    )
    assert results[0].value is not None


@pytest.mark.qualitative
@pytest.mark.xfail(
    reason="litellm has a bug with watsonx; once that is fixed, this should pass."
)
async def test_multiple_async_funcs(session) -> None:
    """If this test passes, remove the _has_potential_event_loop_errors func from litellm."""
    session.chat(
        "first sync"
    )  # Do one sync first in this function so that it should always fail.
    await session.achat("first async")
    await session.achat("second async")


if __name__ == "__main__":
    import pytest

    pytest.main([__file__])
