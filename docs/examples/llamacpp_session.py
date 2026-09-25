# pytest: llamacpp, e2e

"""Example: run a Mellea session against a local llama.cpp server.

llama-server speaks the OpenAI chat-completions API, so Mellea reaches it through
the `openai` backend with a `base_url`. There is no separate llama.cpp backend.

Prerequisites:
    Start the server first (downloads ~2.2 GB on the first run):

        ./scripts/start_llamacpp.sh

    That serves Granite 4.2 3b on http://127.0.0.1:8080 under the alias
    `ibm-granite/granite-4.2-3b`, which is the name `IBM_GRANITE_4_2_3B` resolves
    to for the openai backend. Set LLAMACPP_BASE_URL to point somewhere else.
"""

import os

from mellea import start_session
from mellea.backends.model_ids import IBM_GRANITE_4_2_3B

BASE_URL = os.environ.get("LLAMACPP_BASE_URL", "http://127.0.0.1:8080/v1")

# OpenAIBackend requires an api_key even when the server ignores it, so pass a
# placeholder rather than leaving OPENAI_API_KEY to leak in from the environment.
with start_session(
    "openai", IBM_GRANITE_4_2_3B, base_url=BASE_URL, api_key="llamacpp"
) as m:
    email = m.instruct(
        "Write an email inviting the interns to a lunch party.",
        requirements=["Be under 100 words.", "Do not use the word 'synergy'."],
    )
    print(email)
