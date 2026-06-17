"""Helpers for creating bedrock backends from openai/litellm."""

import logging
import os
import warnings

from openai import OpenAI

from mellea.backends.litellm import LiteLLMBackend
from mellea.backends.model_ids import ModelIdentifier
from mellea.backends.openai import OpenAIBackend


def _resolve_region(region: str | None) -> str | None:
    return (
        region
        or os.environ.get("AWS_REGION_NAME")
        or os.environ.get("AWS_DEFAULT_REGION")
        or os.environ.get("AWS_REGION")
    )


def _assert_region(region: str | None) -> None:
    if _resolve_region(region) is None:
        raise ValueError(
            "you must specify a region: pass `region` explicitly or set AWS_REGION_NAME, AWS_DEFAULT_REGION, or AWS_REGION."
        )


def _assert_bedrock_auth() -> None:
    """Raises if no valid AWS credentials can be resolved.

    Accepts any credential source that boto3 supports:
    - Static env vars (AWS_ACCESS_KEY_ID + AWS_SECRET_ACCESS_KEY)
    - Named profile (AWS_PROFILE or ~/.aws/credentials)
    - ECS task role (AWS_CONTAINER_CREDENTIALS_RELATIVE_URI)
    - EC2 / ECS instance profile (IMDSv2)
    - LiteLLM-specific Bedrock API key (AWS_BEARER_TOKEN_BEDROCK)

    Raises:
        ImportError: If boto3 is not installed (install via the `litellm` extra).
        RuntimeError: If no AWS credentials can be resolved.
    """
    if "AWS_BEARER_TOKEN_BEDROCK" in os.environ:
        return

    try:
        import boto3
        import botocore.exceptions
    except ImportError as e:
        raise ImportError(
            "boto3 is required to validate AWS credentials. "
            "Please `pip install mellea[litellm]` (which includes boto3) "
            "or set AWS_BEARER_TOKEN_BEDROCK to skip credential validation."
        ) from e

    # botocore logs a credential-resolution message on every boto3.Session() call. Suppress it.
    logging.getLogger("botocore.credentials").setLevel(logging.WARNING)

    try:
        creds = boto3.Session().get_credentials()
        if creds is None:
            raise botocore.exceptions.NoCredentialsError()
        # Resolve to catch expired/invalid assume-role chains early.
        creds.get_frozen_credentials()
    except botocore.exceptions.NoCredentialsError:
        raise RuntimeError(
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
        region: AWS region name (e.g. `"us-east-1"`), or `None` to use the
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
        region: AWS region name, or `None` to use the default region.

    Returns:
        Newline-separated string of model IDs prefixed with `" * "`.
    """
    models = list_mantle_models()
    model_names = "\n * ".join([str(m.id) for m in models])
    return f" * {model_names}"


def create_bedrock_litellm_backend(
    model_id: ModelIdentifier | str, region: str | None = None, num_retries: int = 3
) -> LiteLLMBackend:
    """Returns a LiteLLM backend that points to Bedrock for model `model_id`.

    Use this instead of `create_bedrock_openai_backend` when you need auth with an AWS_ACCESS_KEY_ID.

    Args:
        model_id: A `ModelIdentifier` (must have `bedrock_litellm_name`) or a raw
            litellm-format Bedrock model ID string (e.g. `"bedrock/..."`).
        region: AWS region. If `None`, falls back to AWS_REGION_NAME /
            AWS_DEFAULT_REGION / AWS_REGION env vars.
        num_retries: Retry budget for LiteLLM. LiteLLM uses exponential backoff,
            so keep this low to avoid long hangs on persistent failures.

    Raises:
        ValueError: If no region can be resolved or `model_id` does not specify a
            bedrock litellm name.
        RuntimeError: If no AWS credentials can be resolved.
    """
    _assert_bedrock_auth()
    _assert_region(region)

    model_name = ""
    match model_id:
        case ModelIdentifier():
            if model_id.bedrock_litellm_name is None:
                raise ValueError(
                    f"We do not have a known bedrock model identifier for {model_id}. If Bedrock supports this model, please pass the model_id string directly and open an issue to add the model id: https://github.com/generative-computing/mellea/issues/new"
                )
            else:
                model_name = model_id.bedrock_litellm_name
        case str():
            model_name = model_id
    if model_name == "":
        raise ValueError(
            f"Model identifier {model_id} does not specify a bedrock_name."
        )

    # Pass the resolved region through model_options so litellm picks it up even
    # when `region` was supplied explicitly rather than via env vars.
    model_options: dict = {"num_retries": num_retries}
    resolved_region = _resolve_region(region)
    if resolved_region is not None:
        model_options["aws_region_name"] = resolved_region

    return LiteLLMBackend(
        model_id=model_name, model_options=model_options, num_retries=num_retries
    )


def create_bedrock_openai_backend(
    model_id: ModelIdentifier | str, region: str | None = None
) -> OpenAIBackend:
    """Return an OpenAI backend that points to Bedrock mantle for the given model.

    Args:
        model_id (ModelIdentifier | str): The model to use, either as a
            `ModelIdentifier` (which must have a `bedrock_name`) or a raw
            Bedrock model ID string.
        region (str | None): AWS region name, or `None` to use the default
            region (`"us-east-1"`).

    Returns:
        OpenAIBackend: An `OpenAIBackend` configured to call the specified model
            via AWS Bedrock Mantle.

    Raises:
        ValueError: If `model_id` is a `ModelIdentifier` with no `bedrock_name`
            set, or if the specified model is not available in the target region.
        RuntimeError: If the `AWS_BEARER_TOKEN_BEDROCK` environment variable is
            not set.
    """
    model_name = ""
    match model_id:
        case ModelIdentifier() if model_id.bedrock_name is None:
            raise ValueError(
                f"We do not have a known bedrock model identifier for {model_id}. If Bedrock supports this model, please pass the model_id string directly and open an issue to add the model id: https://github.com/generative-computing/mellea/issues/new"
            )
        case ModelIdentifier() if model_id.bedrock_name is not None:
            assert model_id.bedrock_name is not None  # for type checker help.
            model_name = model_id.bedrock_name
        case str():
            model_name = model_id
    if model_name == "":
        raise ValueError(
            f"Model identifier {model_id} does not specify a bedrock_name."
        )

    if "AWS_BEARER_TOKEN_BEDROCK" not in os.environ:
        raise RuntimeError(
            "Using AWS Bedrock requires setting a AWS_BEARER_TOKEN_BEDROCK environment variable.\n\nTo proceed:\n"
            "\n\t1. Generate a key from the AWS console at: https://us-east-1.console.aws.amazon.com/bedrock/home?region=us-east-1#/api-keys?tab=long-term "
            "\n\t2. Run `export AWS_BEARER_TOKEN_BEDROCK=<insert your key here>`\n"
            "If you need to use normal AWS credentials instead of a bedrock-specific bearer token, use create_bedrock_litellm_backend instead."
        )

    uri = _make_mantle_uri(region=region)

    models = list_mantle_models(region)
    if model_name not in [m.id for m in models]:
        raise ValueError(
            f"Model {model_name} is not supported in region {_make_region_for_uri(region=region)}.\nSupported models are:\n{stringify_mantle_model_ids(region)}\n\nPerhaps change regions or check that model access for {model_name} is not gated on Bedrock?"
        )

    backend = OpenAIBackend(
        model_id=model_name,  # sic: do not pass the model_id itself!
        base_url=uri,
        api_key=os.environ["AWS_BEARER_TOKEN_BEDROCK"],
    )
    return backend


def create_bedrock_mantle_backend(
    model_id: ModelIdentifier | str, region: str | None = None
) -> OpenAIBackend:
    """Deprecated alias for `create_bedrock_openai_backend`.

    .. deprecated::
        Use `create_bedrock_openai_backend` instead. This shim will be removed
        in a future release.
    """
    warnings.warn(
        "create_bedrock_mantle_backend is deprecated; use create_bedrock_openai_backend instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return create_bedrock_openai_backend(model_id=model_id, region=region)
