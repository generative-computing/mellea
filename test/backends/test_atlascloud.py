# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for Atlas Cloud backend helpers."""

from unittest.mock import patch

import pytest

from mellea.backends.atlascloud import (
    ATLASCLOUD_DEFAULT_BASE_URL,
    ATLASCLOUD_DEFAULT_MODEL,
    AtlasCloudBackend,
    create_atlascloud_openai_backend,
)
from mellea.stdlib.context import ChatContext
from mellea.stdlib.session import backend_name_to_class
from mellea.stdlib.start_backend import start_backend


def test_atlascloud_backend_uses_env_defaults(monkeypatch):
    """AtlasCloudBackend resolves key, base URL, and default model from Atlas settings."""
    monkeypatch.setenv("ATLASCLOUD_API_KEY", "atlas-key")
    monkeypatch.delenv("ATLASCLOUD_API_BASE", raising=False)
    monkeypatch.delenv("ATLASCLOUD_BASE_URL", raising=False)

    with patch(
        "mellea.backends.openai.is_vllm_server_with_structured_output",
        return_value=False,
    ):
        backend = AtlasCloudBackend()

    assert backend.model_id == ATLASCLOUD_DEFAULT_MODEL
    assert backend._api_key == "atlas-key"
    assert str(backend._client.base_url) == ATLASCLOUD_DEFAULT_BASE_URL + "/"
    assert backend._provider == "atlascloud"


def test_atlascloud_backend_prefers_explicit_over_env(monkeypatch):
    """Explicit API configuration takes precedence over environment defaults."""
    monkeypatch.setenv("ATLASCLOUD_API_KEY", "env-key")
    monkeypatch.setenv("ATLASCLOUD_API_BASE", "https://env.example/v1")

    with patch(
        "mellea.backends.openai.is_vllm_server_with_structured_output",
        return_value=False,
    ):
        backend = create_atlascloud_openai_backend(
            "deepseek-ai/deepseek-v4-pro",
            api_key="explicit-key",
            base_url="https://explicit.example/v1",
        )

    assert backend.model_id == "deepseek-ai/deepseek-v4-pro"
    assert backend._api_key == "explicit-key"
    assert str(backend._client.base_url) == "https://explicit.example/v1/"


def test_atlascloud_backend_env_base_url_alias(monkeypatch):
    """ATLASCLOUD_BASE_URL is accepted as a base URL alias."""
    monkeypatch.setenv("ATLASCLOUD_API_KEY", "atlas-key")
    monkeypatch.delenv("ATLASCLOUD_API_BASE", raising=False)
    monkeypatch.setenv("ATLASCLOUD_BASE_URL", "https://alias.example/v1")

    with patch(
        "mellea.backends.openai.is_vllm_server_with_structured_output",
        return_value=False,
    ):
        backend = AtlasCloudBackend()

    assert str(backend._client.base_url) == "https://alias.example/v1/"


def test_atlascloud_backend_requires_atlas_api_key(monkeypatch):
    """Missing Atlas credentials raise an Atlas-specific error."""
    monkeypatch.delenv("ATLASCLOUD_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(ValueError, match="ATLASCLOUD_API_KEY"):
        AtlasCloudBackend()


def test_backend_name_to_class_resolves_atlascloud_aliases():
    """Atlas Cloud backend aliases resolve to the same backend class."""
    assert backend_name_to_class("atlascloud") is AtlasCloudBackend
    assert backend_name_to_class("atlas-cloud") is AtlasCloudBackend
    assert backend_name_to_class("atlas") is AtlasCloudBackend


def test_start_backend_can_create_atlascloud_backend(monkeypatch):
    """start_backend can create a chat context with the Atlas Cloud backend."""
    monkeypatch.setenv("ATLASCLOUD_API_KEY", "atlas-key")

    with patch(
        "mellea.backends.openai.is_vllm_server_with_structured_output",
        return_value=False,
    ):
        ctx, backend = start_backend(
            "atlascloud", "qwen/qwen3.5-flash", context_type="chat"
        )

    assert isinstance(ctx, ChatContext)
    assert isinstance(backend, AtlasCloudBackend)
    assert backend.model_id == "qwen/qwen3.5-flash"


def test_start_backend_atlascloud_uses_atlas_default_model(monkeypatch):
    """start_backend uses the Atlas Cloud default model when none is specified."""
    monkeypatch.setenv("ATLASCLOUD_API_KEY", "atlas-key")

    with patch(
        "mellea.backends.openai.is_vllm_server_with_structured_output",
        return_value=False,
    ):
        _ctx, backend = start_backend("atlascloud")

    assert isinstance(backend, AtlasCloudBackend)
    assert backend.model_id == ATLASCLOUD_DEFAULT_MODEL
