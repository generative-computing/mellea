import pytest
from mellea.stdlib.base import ModelOutputThunk, LinearContext
from mellea.stdlib.chat import as_chat_history, Message
from mellea.stdlib.session import start_session


@pytest.fixture(scope="function")
def linear_session():
    """Session with linear context for chat tests."""
    session = start_session(ctx=LinearContext())
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

@pytest.mark.skip("linearize() returns [] for a SimpleContext... that's going to be annoying.")
def test_chat_view_simple_ctx(simple_session):
    simple_session.chat("What is 1+1?")
    simple_session.chat("What is 2+2?")
    assert len(as_chat_history(simple_session.ctx)) == 2
    assert all([type(x) == Message for x in as_chat_history(simple_session.ctx)])

if __name__ == "__main__":
    pytest.main([__file__])
