"""Unit tests for bedrock helpers — URI construction, ModelIdentifier matching, env var validation.

These tests cover the pure-logic helpers and error paths in bedrock.py without
requiring AWS credentials or network access. Happy-path tests mock only the
network boundary (list_mantle_models) and verify the rest of the pipeline.
"""

from dataclasses import dataclass
from unittest.mock import patch

import pytest

from mellea.backends.bedrock import (
    _make_mantle_uri,
    _make_region_for_uri,
    create_bedrock_litellm_backend,
    create_bedrock_openai_backend,
    stringify_mantle_model_ids,
)
from mellea.backends.litellm import LiteLLMBackend
from mellea.backends.model_ids import ModelIdentifier
from mellea.backends.openai import OpenAIBackend


@dataclass
class _FakeModel:
    """Minimal stand-in for the model objects returned by list_mantle_models."""

    id: str


_FAKE_MODELS = [_FakeModel("granite-3.3-8b"), _FakeModel("llama-4-scout")]

# --- _make_region_for_uri ---


def test_region_default_when_none():
    assert _make_region_for_uri(None) == "us-east-1"


# --- _make_mantle_uri ---


def test_mantle_uri_default_region():
    uri = _make_mantle_uri()
    assert uri == "https://bedrock-mantle.us-east-1.api.aws/v1"


def test_mantle_uri_custom_region():
    uri = _make_mantle_uri("ap-northeast-1")
    assert uri == "https://bedrock-mantle.ap-northeast-1.api.aws/v1"


# --- create_bedrock_openai_backend error paths ---


def test_model_identifier_without_bedrock_name_raises():
    mid = ModelIdentifier(hf_model_name="some/model")
    with pytest.raises(Exception, match="do not have a known bedrock model identifier"):
        create_bedrock_openai_backend(mid)


def test_missing_env_var_raises(monkeypatch):
    monkeypatch.delenv("AWS_BEARER_TOKEN_BEDROCK", raising=False)
    monkeypatch.delenv("AWS_ACCESS_KEY_ID", raising=False)
    mid = ModelIdentifier(bedrock_name="some.model-id")
    with pytest.raises(RuntimeError, match="AWS_BEARER_TOKEN_BEDROCK"):
        create_bedrock_openai_backend(mid)


# --- create_bedrock_openai_backend happy paths (mock network boundary) ---


@patch("mellea.backends.bedrock.list_mantle_models", return_value=_FAKE_MODELS)
def test_create_backend_with_model_identifier(mock_list, monkeypatch):
    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "fake-token")
    mid = ModelIdentifier(bedrock_name="granite-3.3-8b")
    backend = create_bedrock_openai_backend(mid, region="eu-west-1")

    assert isinstance(backend, OpenAIBackend)
    assert backend.model_id == "granite-3.3-8b"
    mock_list.assert_called_once_with("eu-west-1")


@patch("mellea.backends.bedrock.list_mantle_models", return_value=_FAKE_MODELS)
def test_create_backend_with_string_model_id(mock_list, monkeypatch):
    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "fake-token")
    backend = create_bedrock_openai_backend("llama-4-scout")
    assert backend.model_id == "llama-4-scout", "model id should be llama-4-scout"


@patch("mellea.backends.bedrock.list_mantle_models", return_value=_FAKE_MODELS)
def test_model_not_in_region_raises(mock_list, monkeypatch):
    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "fake-token")
    with pytest.raises(Exception, match="not supported in region"):
        create_bedrock_openai_backend("nonexistent-model")


# --- stringify_mantle_model_ids ---


@patch("mellea.backends.bedrock.list_mantle_models", return_value=_FAKE_MODELS)
def test_stringify_mantle_model_ids_passes_region(mock_list):
    stringify_mantle_model_ids("eu-west-1")
    mock_list.assert_called_once_with("eu-west-1")


# --- create_bedrock_litellm_backend ---


@patch("mellea.backends.bedrock._assert_bedrock_auth")
def test_litellm_backend_num_retries_single_source(mock_auth, monkeypatch):
    """num_retries is wired through model_options only, not as a direct arg.

    Passing it both ways would forward num_retries to litellm.acompletion twice
    (once via the constructor param, once via model_specific_options), raising a
    TypeError at generation time. Keeping it solely in model_options is the
    single source of truth.
    """
    monkeypatch.setenv("AWS_REGION", "us-east-1")
    backend = create_bedrock_litellm_backend(
        "bedrock/converse/openai.gpt-oss-20b-1:0", num_retries=7
    )

    assert isinstance(backend, LiteLLMBackend)
    assert backend.model_options["num_retries"] == 7


@patch("mellea.backends.bedrock._assert_bedrock_auth")
def test_litellm_backend_resolved_region_in_model_options(mock_auth, monkeypatch):
    monkeypatch.setenv("AWS_REGION", "ap-northeast-1")
    backend = create_bedrock_litellm_backend("bedrock/converse/openai.gpt-oss-20b-1:0")
    assert backend.model_options["aws_region_name"] == "ap-northeast-1"
