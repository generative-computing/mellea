import pytest
from mellea.stdlib.base import ModelOutputThunk, LinearContext
from mellea.stdlib.chat import as_chat_history, Message
from mellea.stdlib.session import start_session


def test_chat_view_linear_ctx():
    m = start_session(ctx=LinearContext())
    m.chat("What is 1+1?")
    m.chat("What is 2+2?")
    assert len(as_chat_history(m.ctx)) == 4
    assert all([type(x) == Message for x in as_chat_history(m.ctx)])


def test_chat_view_simple_ctx():
    m = start_session()
    m.chat("What is 1+1?")
    m.chat("What is 2+2?")
    assert len(as_chat_history(m.ctx)) == 2
    assert all([type(x) == Message for x in as_chat_history(m.ctx)])


if __name__ == "__main__":
    pytest.main([__file__])
