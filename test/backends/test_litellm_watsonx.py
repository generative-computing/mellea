import asyncio
import pytest

from mellea import MelleaSession
from mellea.backends.litellm import LiteLLMBackend

@pytest.fixture(scope="function")
def session():
    """Fresh Ollama session for each test."""
    session = MelleaSession(
        LiteLLMBackend(
            model_id="watsonx/ibm/granite-4-h-small",
        ), 
    )
    yield session
    session.reset()

def test_has_potential_event_loop_errors(session):
    """This test is specific to litellm backends that use watsonx/. It can be removed once that bug is fixed."""

    backend: LiteLLMBackend = session.backend
    potential_err = backend._has_potential_event_loop_errors()
    assert not potential_err, "first invocation in an event loop shouldn't flag errors"

    potential_err = backend._has_potential_event_loop_errors()
    assert not potential_err, "second invocation in the same event loop shouldn't flag errors"


    async def new_event_loop() -> bool:
        return backend._has_potential_event_loop_errors()

    err_expected = asyncio.run(new_event_loop())
    assert err_expected, "invocation in a new event loop should flag an error"

@pytest.mark.qualitative
def test_multiple_sync_funcs(session):
    session.chat("first")
    session.chat("second")

@pytest.mark.qualitative
@pytest.mark.xfail(reason="litellm has a bug with watsonx; once that is fixed, this should pass.")
async def test_multiple_async_funcs(session):
    """If this test passes, remove the _has_potential_event_loop_errors func from litellm."""
    session.chat("first sync")  # Do one sync first in this function so that it should always fail.
    await session.achat("first async")
    await session.achat("second async")

if __name__ == "__main__":
    import pytest

    pytest.main([__file__])
