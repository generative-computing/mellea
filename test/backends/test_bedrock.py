import os

import openai
import pytest

import mellea.backends.model_ids
import mellea.backends.model_ids as model_ids
from mellea import MelleaSession
from mellea.backends.bedrock import create_bedrock_backend
from mellea.backends.openai import OpenAIBackend
from mellea.stdlib.context import ChatContext

# Skip entire module in CI since the single test is qualitative
pytestmark = [
    pytest.mark.llm,
    pytest.mark.skipif(
        "AWS_BEARER_TOKEN_BEDROCK" not in os.environ.keys(),
        reason="Skipping Bedrock backend tests if $AWS_BEARER_TOKEN_BEDROCK is not set.",
    ),
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

    non_bedrock_models = [
        getattr(mellea.backends.model_ids, name)
        for name in dir(mellea.backends.model_ids)
        if "bedrock_name" not in dir(getattr(mellea.backends.model_ids, name))
        and "ollama_name" in dir(getattr(mellea.backends.model_ids, name))
    ]

    for model in bedrock_models:
        m = MelleaSession(
            backend=create_bedrock_backend(model_id=model), ctx=ChatContext()
        )
        print(m.chat("What is 1+1?").content)


if __name__ == "__main__":
    test_model_ids_exist()
    # pytest.main([__file__])
