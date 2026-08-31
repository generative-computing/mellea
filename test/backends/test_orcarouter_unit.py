# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for OrcaRouterBackend — no API calls required.

Covers the provider-branded defaults (base URL, env var, provider tag) and the
model_id resolution shared with the OpenAI backend.
"""

import pytest

from mellea.backends import model_ids
from mellea.backends.orcarouter import ORCAROUTER_BASE_URL, OrcaRouterBackend


def _make_backend(**kwargs) -> OrcaRouterBackend:
    """Return an OrcaRouterBackend with a fake API key."""
    model_id = kwargs.pop("model_id", "orcarouter/auto")
    return OrcaRouterBackend(
        model_id=model_id, api_key=kwargs.pop("api_key", "fake-key"), **kwargs
    )


def test_defaults_to_orcarouter_base_url():
    backend = _make_backend()
    assert backend._base_url == ORCAROUTER_BASE_URL
    assert backend._base_url == "https://api.orcarouter.ai/v1"


def test_provider_tag_is_orcarouter():
    backend = _make_backend()
    assert backend._provider == "orcarouter"


def test_repr_masks_api_key():
    backend = _make_backend()
    r = repr(backend)
    assert "fake-key" not in r
    assert "***" in r


def test_default_model_id_is_orcarouter_auto():
    backend = OrcaRouterBackend(api_key="fake-key")
    assert backend._model_id == model_ids.ORCAROUTER_AUTO.openai_name
    assert backend._model_id == "orcarouter/auto"


def test_model_identifier_resolves_to_openai_name():
    backend = _make_backend(model_id=model_ids.ORCAROUTER_AUTO)
    assert backend._model_id == "orcarouter/auto"


def test_missing_api_key_raises(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("ORCAROUTER_API_KEY", raising=False)
    with pytest.raises(ValueError, match="ORCAROUTER_API_KEY"):
        OrcaRouterBackend()


def test_api_key_reads_from_environment(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("ORCAROUTER_API_KEY", "env-key")
    backend = OrcaRouterBackend()
    assert backend._api_key == "env-key"


def test_explicit_api_key_takes_precedence(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("ORCAROUTER_API_KEY", "env-key")
    backend = OrcaRouterBackend(api_key="explicit-key")
    assert backend._api_key == "explicit-key"
