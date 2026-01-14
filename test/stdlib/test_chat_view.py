import pytest

from mellea.stdlib.context import ChatContext
from mellea.stdlib.components import Message, as_chat_history
from mellea.stdlib.session import start_session


@pytest.fixture(scope="function")
def linear_session():
    """Session with linear context for chat tests."""
    session = start_session(ctx=ChatContext())
    yield session
    session.reset()


@pytest.fixture(scope="function")
def simple_session():
    """Session with simple context for chat tests."""
    session = start_session()
    yield session
    session.reset()


def test_chat_view_linear_ctx(linear_session):
    linear_session.chat("What is 1+1?")
    linear_session.chat("What is 2+2?")
    assert len(as_chat_history(linear_session.ctx)) == 4
    assert all([type(x) == Message for x in as_chat_history(linear_session.ctx)])
    assert len(linear_session.ctx.view_for_generation()) == 4


# @pytest.mark.skip("linearize() returns [] for a SimpleContext... that's going to be annoying.")
def test_chat_view_simple_ctx(simple_session):
    simple_session.chat("What is 1+1?")
    simple_session.chat("What is 2+2?")
    assert len(as_chat_history(simple_session.ctx)) == 4
    assert all([type(x) == Message for x in as_chat_history(simple_session.ctx)])
    assert len(simple_session.ctx.view_for_generation()) == 0


if __name__ == "__main__":
    pytest.main([__file__])
