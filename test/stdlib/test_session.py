# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import os

import pytest

from mellea.backends import ModelOption
from mellea.core import ModelOutputThunk
from mellea.stdlib.components import Message
from mellea.stdlib.context import ChatContext, SimpleContext
from mellea.stdlib.session import MelleaSession, start_session
from test.predicates import require_api_key

# Mark all tests as requiring Ollama (start_session defaults to Ollama)
pytestmark = [pytest.mark.ollama, pytest.mark.e2e]


# We edit the context type in the async tests below. Don't change the scope here.
@pytest.fixture(scope="module")
def m_session(gh_run):
    m = start_session(model_options={ModelOption.MAX_NEW_TOKENS: 5})
    yield m
    del m


@pytest.mark.watsonx
@require_api_key("WATSONX_API_KEY", "WATSONX_URL", "WATSONX_PROJECT_ID")
def test_start_session_watsonx(gh_run):
    if gh_run == 1:
        pytest.skip("Skipping watsonx tests.")
    else:
        m = start_session(backend_name="watsonx")
        response = m.instruct("testing")
        assert isinstance(response, ModelOutputThunk)
        assert response.value is not None


def test_start_session_openai_with_kwargs(m_session):
    initial_ctx = m_session.ctx
    response = m_session.instruct("testing")
    assert isinstance(response, ModelOutputThunk)
    assert response.value is not None
    assert initial_ctx is not m_session.ctx


async def test_aact(m_session):
    initial_ctx = m_session.ctx
    # aact defaults to strategy=None; pass await_result=True to compute the
    # result (matching act's synchronous contract) and to keep the
    # module-scoped session's context free of an uncomputed ModelOutputThunk,
    # which later deepcopy/clone-based tests cannot handle.
    out = await m_session.aact(
        Message(role="user", content="Hello!"), await_result=True
    )
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
    r1 = await m_session.aact(m1, strategy=None, await_result=True)
    r2 = await m_session.aact(m2, strategy=None, await_result=True)

    # This should be the order of these items in the session's context.
    history = [r2, m2, r1, m1]

    ctx = m_session.ctx
    for i in range(len(history)):
        assert ctx.node_data is history[i]  # type: ignore
        ctx = ctx.previous_node  # type: ignore

    # Ensure we made it back to the root.
    assert ctx.is_root_node  # type: ignore


async def test_async_without_waiting_with_simple_context(m_session):
    # Concurrent fire-and-forget aact calls must use SimpleContext. With aact's
    # default strategy=None (and await_result=False), each call adds an
    # uncomputed ModelOutputThunk to its context; on a shared ChatContext two
    # gathered calls race and one reads the other's uncomputed thunk during
    # rendering (assert c.is_computed() failure). SimpleContext keeps each turn
    # independent, which is the documented context for parallel generation.
    m_session.ctx = SimpleContext()

    m1 = Message(role="user", content="1")
    m2 = Message(role="user", content="2")
    co1 = m_session.aact(m1)
    co2 = m_session.aact(m2)
    r2, r1 = await asyncio.gather(co2, co1)

    # Both fire-and-forget generations resolve to real values once awaited.
    assert await r1.avalue() is not None
    assert await r2.avalue() is not None


def test_session_copy_with_context_ops(m_session):
    # Start from a clean ChatContext. The module-scoped session is mutated by
    # earlier async tests, so reset here to make this test order-independent
    # and to get the chained history the previous_node assertions below rely on.
    m_session.ctx = ChatContext()

    out = m_session.instruct("What is 2x2?")
    main_ctx = m_session.ctx

    m1 = m_session.clone()
    out1 = m1.instruct("Multiply by 3.")

    m2 = m_session.clone()
    out2 = m2.instruct("Multiply by 4.")

    # Assert that each context is the correct one.
    assert m_session.ctx is main_ctx
    assert m_session.ctx is not m1.ctx
    assert m_session.ctx is not m2.ctx
    assert m1.ctx is not m2.ctx

    # Assert that node data is correct.
    assert m_session.ctx.node_data is out
    assert m1.ctx.node_data is out1
    assert m2.ctx.node_data is out2

    # Assert that the new sessions still branch off the original one.
    assert m1.ctx.previous_node.previous_node is m_session.ctx
    assert m2.ctx.previous_node.previous_node is m_session.ctx


class TestPowerup:
    def hello(m: MelleaSession):  # type: ignore
        return "hello"


def test_powerup(m_session):
    MelleaSession.powerup(TestPowerup)

    assert "hello" == m_session.hello()


if __name__ == "__main__":
    pytest.main([__file__])
