import copy
import pytest

from mellea.backends.types import ModelOption
from mellea.stdlib.base import ModelOutputThunk
from mellea.stdlib.session import MelleaSession, start_session

# Use generated ModelOutputThunks to fully test copying. This can technically be done without a
# backend, but it simplifies test setup.
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

def test_model_output_thunk_copy(m_session: MelleaSession):
    """Basic tests for copying ModelOutputThunk. Add checks if needed."""
    out = m_session.instruct("Hello!")
    copied = copy.copy(out)

    assert out is not copied
    assert copied._generate is None
    assert copied._meta is out._meta

    empty = ModelOutputThunk("")
    copy.copy(empty) # Make sure no errors happen.



def test_model_output_thunk_deepcopy(m_session: MelleaSession):
    """Basic tests for deepcopying ModelOutputThunk. Add checks if needed."""
    out = m_session.instruct("Goodbye!")
    deepcopied = copy.deepcopy(out)

    assert out is not deepcopied
    assert deepcopied._generate is None
    assert deepcopied._meta is not out._meta

    empty = ModelOutputThunk("")
    copy.deepcopy(empty) # Make sure no errors happen.


if __name__ == "__main__":
    pytest.main([__file__])
