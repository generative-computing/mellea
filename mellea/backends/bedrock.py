"""Helpers for creating bedrock backends from openai/litellm."""

import os

import logging

import boto3
import botocore.exceptions

# botocore logs a credential-resolution message on every boto3.Session() call. Suppress it.
logging.getLogger("botocore.credentials").setLevel(logging.WARNING)
from openai import OpenAI
from openai.pagination import SyncPage

from mellea.backends.litellm import LiteLLMBackend
from mellea.backends.model_ids import ModelIdentifier
from mellea.backends.openai import OpenAIBackend


def _assert_region(region: str | None) -> str:
    resolved_region = (
        region
        or os.environ.get("AWS_REGION_NAME")
        or os.environ.get("AWS_DEFAULT_REGION")
        or os.environ.get("AWS_REGION")
    )
    assert (
        resolved_region is not None
    ), "you must specify a region: pass `region` explicitly or set AWS_REGION_NAME, AWS_DEFAULT_REGION, or AWS_REGION."


def _assert_bedrock_auth() -> None:
    """Raises if no valid AWS credentials can be resolved.

    Accepts any credential source that boto3 supports:
    - Static env vars (AWS_ACCESS_KEY_ID + AWS_SECRET_ACCESS_KEY)
    - Named profile (AWS_PROFILE or ~/.aws/credentials)
    - ECS task role (AWS_CONTAINER_CREDENTIALS_RELATIVE_URI)
    - EC2 / ECS instance profile (IMDSv2)
    - LiteLLM-specific Bedrock API key (AWS_BEARER_TOKEN_BEDROCK)
    """
    if "AWS_BEARER_TOKEN_BEDROCK" in os.environ:
        return

    try:
        creds = boto3.Session().get_credentials()
        if creds is None:
            raise botocore.exceptions.NoCredentialsError()
        # Resolve to catch expired/invalid assume-role chains early.
        creds.get_frozen_credentials()
    except botocore.exceptions.NoCredentialsError:
        raise AssertionError(
            "No AWS credentials found. Provide one of:\n"
            "  - AWS_BEARER_TOKEN_BEDROCK (Bedrock API key)\n"
            "  - AWS_ACCESS_KEY_ID + AWS_SECRET_ACCESS_KEY\n"
            "  - AWS_PROFILE pointing to a configured profile\n"
            "  - An IAM role attached to the instance/task (EC2, ECS, Lambda)"
        )
    except botocore.exceptions.NoRegionError:
        pass  # Credentials exist; region is validated separately.


def _make_region_for_uri(region: str | None):
    if region is None:
        region = "us-east-1"
    return region


def _make_mantle_uri(region: str | None = None):
    region_for_uri = _make_region_for_uri(region)
    uri = f"https://bedrock-mantle.{region_for_uri}.api.aws/v1"
    return uri


def list_mantle_models(region: str | None = None) -> list:
    """Return all models available at a bedrock-mantle endpoint.

    Args:
        region: AWS region name (e.g. ``"us-east-1"``), or ``None`` to use the
            default region.

    Returns:
        List of model objects returned by the Bedrock Mantle models API.
    """
    uri = _make_mantle_uri(region)
    client = OpenAI(base_url=uri, api_key=os.environ["AWS_BEARER_TOKEN_BEDROCK"])
    ms = client.models.list()
    all_models = list(ms)
    assert ms.next_page_info() is None
    return all_models


def stringify_mantle_model_ids(region: str | None = None) -> str:
    """Return a human-readable list of all models available at the mantle endpoint for an AWS region.

    Args:
        region: AWS region name, or ``None`` to use the default region.

    Returns:
        Newline-separated string of model IDs prefixed with ``" * "``.
    """
    models = list_mantle_models()
    model_names = "\n * ".join([str(m.id) for m in models])
    return f" * {model_names}"


def create_bedrock_litellm_backend(
    model_id: ModelIdentifier | str, region: str | None = None, num_retries: int = 15
) -> LiteLLMBackend:
    """Returns a LiteLLM backend that points to Bedrock for model `model_id`.

    Use this instead of `create_bedrock_openai_backend` when you need auth with an AWS_ACCESS_KEY_ID.
    """
    _assert_bedrock_auth()
    _assert_region(region)

    model_name = ""
    match model_id:
        case ModelIdentifier():
            if model_id.bedrock_litellm_name is None:
                raise Exception(
                    f"We do not have a known bedrock model identifier for {model_id}. If Bedrock supports this model, please pass the model_id string directly and  open an issue to add the model id: https://github.com/generative-computing/mellea/issues/new"
                )
            else:
                model_name = model_id.bedrock_litellm_name
        case str():
            model_name = model_id
    assert (
        model_name != ""
    ), f"Model identifier {model_id} does not specify a bedrock_name."

    backend = LiteLLMBackend(model_id=model_name, num_retries=num_retries)

    # TODO litellm doesn't even appear to use this...?
    backend._base_url = None  # type: ignore
    return backend


def create_bedrock_openai_backend(
    model_id: ModelIdentifier | str, region: str | None = None
) -> OpenAIBackend:
    """Return an OpenAI backend that points to Bedrock mantle for the given model.

    Args:
        model_id (ModelIdentifier | str): The model to use, either as a
            ``ModelIdentifier`` (which must have a ``bedrock_name``) or a raw
            Bedrock model ID string.
        region (str | None): AWS region name, or ``None`` to use the default
            region (``"us-east-1"``).

    Returns:
        OpenAIBackend: An ``OpenAIBackend`` configured to call the specified model
            via AWS Bedrock Mantle.

    Raises:
        Exception: If ``model_id`` is a ``ModelIdentifier`` with no ``bedrock_name``
            set.
        AssertionError: If the ``AWS_BEARER_TOKEN_BEDROCK`` environment variable is
            not set.
        Exception: If the specified model is not available in the target region.
    """
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

    models = list_mantle_models(region)
    if model_name not in [m.id for m in models]:
        raise Exception(
            f"Model {model_name} is not supported in region {_make_region_for_uri(region=region)}.\nSupported models are:\n{stringify_mantle_model_ids(region)}\n\nPerhaps change regions or check that model access for {model_name} is not gated on Bedrock?"
        )

    backend = OpenAIBackend(
        model_id=model_name,  # sic: do not pass the model_id itself!
        base_url=uri,
        api_key=os.environ["AWS_BEARER_TOKEN_BEDROCK"],
    )
    return backend
