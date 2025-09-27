

import pytest

from mellea.backends.types import ModelOption
from mellea.stdlib.base import CBlock
from mellea.stdlib.funcs import instruct
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

def test_func_context(m_session):
    initial_ctx = m_session.ctx
    backend = m_session.backend

    out, ctx = instruct("Write a sentence.", initial_ctx, backend)
    assert initial_ctx is not ctx
    assert ctx._data is out

if __name__ == "__main__":
    pytest.main([__file__])