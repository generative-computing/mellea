import asyncio
import os

import pytest

from mellea.backends.types import ModelOption
from mellea.stdlib.base import ChatContext, ModelOutputThunk
from mellea.stdlib.chat import Message
from mellea.stdlib.session import start_session

# We edit the context type in the async tests below. Don't change the scope here.
@pytest.fixture(scope="function")
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

async def test_async_await_with_chat_context(m_session):
    m_session.ctx = ChatContext()

    m1 = Message(role="user", content="1")
    m2 = Message(role="user", content="2")
    r1 = await m_session.aact(m1)
    r2 = await m_session.aact(m2)

    # This should be the order of these items in the session's context.
    history = [r2, m2, r1, m1]

    ctx = m_session.ctx
    for i in range(len(history)):
        assert ctx.node_data is history[i]
        ctx = ctx.previous_node

    # Ensure we made it back to the root.
    assert ctx.is_root_node == True

async def test_async_without_waiting_with_chat_context(m_session):
    m_session.ctx = ChatContext()

    m1 = Message(role="user", content="1")
    m2 = Message(role="user", content="2")
    co1 = m_session.aact(m1)
    co2 = m_session.aact(m2)
    _, _ = await asyncio.gather(co2, co1)

    ctx = m_session.ctx
    assert len(ctx.view_for_generation()) == 2


if __name__ == "__main__":
    pytest.main([__file__])
