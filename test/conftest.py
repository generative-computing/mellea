import os

import pytest

from mellea.backends.huggingface import LocalHFBackend
from mellea.backends.ollama import OllamaModelBackend
from mellea.backends.openai import OpenAIBackend
from mellea.stdlib.session import MelleaSession


@pytest.fixture(scope="session")
def gh_run() -> int:
    return int(os.environ.get("GITHUB_ACTION", 0))  # type: ignore


def pytest_runtest_setup(item):
    # Runs tests *not* marked with `@pytest.mark.llm` to run normally.
    if not item.get_closest_marker("llm"):
        return

    gh_run = int(os.environ.get("GITHUB_ACTION", 0))

    if gh_run == 1:
        pytest.xfail(
            reason="Skipping LLM test: got env variable GITHUB_ACTION == 1. Used only in gh workflows."
        )

    # # Check if there is a session fixture.
    # try:
    #     session: MelleaSession = item._request.getfixturevalue("m_session")
    # except Exception:
    #     # Skip test cause all llm marked tests need a session fixture.
    #     pytest.skip("`llm` marked tests requires a `m_session` fixture.")
    # # Get the Ollama name.
    # if isinstance(session.backend, OllamaModelBackend) or isinstance(session.backend, OpenAIBackend):
    #     model_id = session.backend.model_id.ollama_name
    #     # Skip tests of the model name is llama 1b
    #     if model_id == "llama3.2:1b":
    #         pytest.skip(
    #             "Skipping LLM test: got model_id == llama3.2:1b in ollama. Used only in gh workflows."
    #         )
    # elif isinstance(session.backend, LocalHFBackend):
    #     model_id = session.backend.model_id.hf_model_name
    #     # Skip tests of the model name is llama 1b
    #     if model_id == "unsloth/Llama-3.2-1B":
    #         pytest.skip(
    #             "Skipping LLM test: got model_id == unsloth/Llama-3.2-1B in hf. Used only in gh workflows."
    #         )
