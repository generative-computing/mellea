"""Helpers for creating bedrock backends from openai/litellm."""

import os

from openai import OpenAI

from mellea.backends.model_ids import ModelIdentifier
from mellea.backends.openai import OpenAIBackend


def _make_region_for_uri(region: str | None):
    if region is None:
        region = "us-east-1"
    return region


def _make_mantle_uri(region: str | None = None):
    region_for_uri = _make_region_for_uri(region)
    uri = f"https://bedrock-mantle.{region_for_uri}.api.aws/v1"
    return uri


def create_bedrock_mantle_backend(
    model_id: ModelIdentifier | str, region: str | None = None
) -> OpenAIBackend:
    """Returns an OpenAI backend that points to Bedrock mantle for model `model_id`."""
    model_name = ""
    match model_id:
        case ModelIdentifier() if model_id.bedrock_name is None:
            raise Exception(
                f"We do not have a known bedrock model identifier for {model_id}. If Bedrock supports this model, please pass the model_id string directly and  open an issue to add the model id: https://github.com/generative-computing/mellea/issues/new"
            )
        case ModelIdentifier() if model_id.bedrock_name is not None:
            assert model_id.bedrock_name is not None  # for type checker help.
            model_name = model_id.bedrock_name
        case str():
            model_name = model_id
    assert model_name != ""

    assert "AWS_BEARER_TOKEN_BEDROCK" in os.environ.keys(), (
        "Using AWS Bedrock requires setting a AWS_BEARER_TOKEN_BEDROCK environment variable.\n\nTo proceed:\n"
        "\n\t1. Generate a key from the AWS console at: https://us-east-1.console.aws.amazon.com/bedrock/home?region=us-east-1#/api-keys?tab=long-term "
        "\n\t2. Run `export AWS_BEARER_TOKEN_BEDROCK=<insert your key here>\n"
        "If you need to use normal AWS credentials instead of a bedrock-specific bearer token, please open an issue: https://github.com/generative-computing/mellea/issues/new"
    )

    uri = _make_mantle_uri(region=region)

    _client = OpenAI(base_url=uri, api_key=os.environ["AWS_BEARER_TOKEN_BEDROCK"])
    _models = _client.models.list()
    if model_name not in [m.id for m in _models]:
        model_names = "\n * ".join([str(m.id) for m in _models])
        raise Exception(
            f"Model {model_name} is not supported in region {_make_region_for_uri(region=region)}.\nSupported models are:\n * {model_names}\n\nPerhaps change regions or check that model access for {model_name} is not gated on Bedrock?"
        )

    backend = OpenAIBackend(
        model_id=model_name,  # sic: do not pass the model_id itself!
        base_url=uri,
        api_key=os.environ["AWS_BEARER_TOKEN_BEDROCK"],
    )
    return backend
