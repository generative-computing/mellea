import os

import pytest

from mellea.backends.types import ModelOption
from mellea.stdlib.base import ModelOutputThunk
from mellea.stdlib.chat import Message
from mellea.stdlib.session import start_session

@pytest.fixture(scope="module")
def m_session(gh_run):
    if gh_run == 1:
        m = start_session(
            "ollama",
            model_id="llama3.2:1b",
            model_options={ModelOption.MAX_NEW_TOKENS: 5},
        )
    else:
        m = start_session(
            "ollama",
            model_id="granite3.3:8b",
            model_options={ModelOption.MAX_NEW_TOKENS: 5},
        )
    yield m
    del m

def test_start_session_watsonx(gh_run):
    if gh_run == 1:
        pytest.skip("Skipping watsonx tests.")
    else:
        m = start_session(backend_name="watsonx")
        response = m.instruct("testing")
        assert isinstance(response, ModelOutputThunk)
        assert response.value is not None

def test_start_session_openai_with_kwargs(gh_run):
    if gh_run == 1:
        m = start_session(
        "openai",
        model_id="llama3.2:1b",
        base_url=f"http://{os.environ.get('OLLAMA_HOST', 'localhost:11434')}/v1",
        api_key="ollama",
    )
    else:
        m = start_session(
            "openai",
            model_id="granite3.3:8b",
            base_url=f"http://{os.environ.get('OLLAMA_HOST', 'localhost:11434')}/v1",
            api_key="ollama",
        )
    initial_ctx = m.ctx
    response = m.instruct("testing")
    assert isinstance(response, ModelOutputThunk)
    assert response.value is not None
    assert initial_ctx is not m.ctx

async def test_aact(m_session):
    initial_ctx = m_session.ctx
    out = await m_session.aact(Message(role="user", content="Hello!"))
    assert m_session.ctx is not initial_ctx
    assert out.value is not None

async def test_ainstruct(m_session):
    initial_ctx = m_session.ctx
    out = await m_session.ainstruct("Write a sentence.")
    assert m_session.ctx is not initial_ctx
    assert out.value is not None

if __name__ == "__main__":
    pytest.main([__file__])
