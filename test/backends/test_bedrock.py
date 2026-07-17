# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os

import openai
import pytest

import mellea.backends.model_ids
import mellea.backends.model_ids as model_ids
from mellea import MelleaSession
from mellea.backends.bedrock import (
    create_bedrock_litellm_backend,
    create_bedrock_openai_backend,
)
from mellea.backends.openai import OpenAIBackend
from mellea.stdlib.context import ChatContext
from test.predicates import require_api_key

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.bedrock,
    require_api_key("AWS_BEARER_TOKEN_BEDROCK"),
]


def _is_bedrock_model(model_id: model_ids.ModelIdentifier):
    return model_id.bedrock_name is not None


def test_model_ids_exist():
    bedrock_models = [
        getattr(mellea.backends.model_ids, name)
        for name in dir(mellea.backends.model_ids)
        if "bedrock_name" in dir(getattr(mellea.backends.model_ids, name))
        and _is_bedrock_model(getattr(mellea.backends.model_ids, name))
    ]

    print(f"Found {len(bedrock_models)} bedrock-supported models.")
    for model in bedrock_models:
        print(f"Checking {model.bedrock_name}")
        m = MelleaSession(
            backend=create_bedrock_openai_backend(model_id=model), ctx=ChatContext()
        )
        print(m.chat("What is 1+1?").content)


@pytest.mark.qualitative
@pytest.mark.skipif(
    not os.environ.get("AWS_REGION_NAME")
    and not os.environ.get("AWS_DEFAULT_REGION")
    and not os.environ.get("AWS_REGION"),
    reason="No AWS region set; cannot exercise the litellm bedrock path.",
)
def test_litellm_bedrock_chat():
    """Smoke test for the litellm bedrock path.

    Skipped unless an AWS_BEARER_TOKEN_BEDROCK is present (module-level skip)
    and an AWS region is resolvable. Uses a string model_id so the test does not
    depend on `model_ids.OPENAI_GPT_OSS_*.bedrock_litellm_name` staying stable.
    """
    backend = create_bedrock_litellm_backend(
        model_id="bedrock/converse/openai.gpt-oss-20b-1:0"
    )
    m = MelleaSession(backend=backend, ctx=ChatContext())
    response = m.chat("Reply with the single word: ok")
    assert response.content is not None
    assert len(str(response.content).strip()) > 0


if __name__ == "__main__":
    test_model_ids_exist()
    # pytest.main([__file__])
