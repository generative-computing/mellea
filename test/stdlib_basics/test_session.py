import os

import pytest

from mellea.stdlib.base import ModelOutputThunk
from mellea.stdlib.session import start_session


def test_start_session_watsonx():
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
    response = m.instruct("testing")
    assert isinstance(response, ModelOutputThunk)
    assert response.value is not None


if __name__ == "__main__":
    pytest.main([__file__])
