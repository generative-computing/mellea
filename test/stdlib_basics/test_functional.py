import pytest

from mellea.backends.types import ModelOption
from mellea.stdlib.base import ModelOutputThunk
from mellea.stdlib.chat import Message
from mellea.stdlib.functional import aact, ainstruct, avalidate, instruct
from mellea.stdlib.requirement import req
from mellea.stdlib.session import start_session


@pytest.fixture(scope="module")
def m_session(gh_run):
    m = start_session(model_options={ModelOption.MAX_NEW_TOKENS: 5})
    yield m
    del m


def test_func_context(m_session):
    initial_ctx = m_session.ctx
    backend = m_session.backend

    out, ctx = instruct("Write a sentence.", initial_ctx, backend)
    assert initial_ctx is not ctx
    assert ctx._data is out


async def test_aact(m_session):
    initial_ctx = m_session.ctx
    backend = m_session.backend

    out, ctx = await aact(Message(role="user", content="hello"), initial_ctx, backend)

    assert initial_ctx is not ctx
    assert ctx._data is out


async def test_ainstruct(m_session):
    initial_ctx = m_session.ctx
    backend = m_session.backend

    out, ctx = await ainstruct("Write a sentence", initial_ctx, backend)

    assert initial_ctx is not ctx
    assert ctx._data is out


async def test_avalidate(m_session):
    initial_ctx = m_session.ctx
    backend = m_session.backend

    val_result = await avalidate(
        reqs=[req("Be formal."), req("Avoid telling jokes.")],
        context=initial_ctx,
        backend=backend,
        output=ModelOutputThunk("Here is an output."),
    )

    assert len(val_result) == 2
    assert val_result[0] is not None


if __name__ == "__main__":
    pytest.main([__file__])
