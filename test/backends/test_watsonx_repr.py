"""Unit tests for WatsonxAIBackend.__repr__ and __str__ masking behaviour."""

import pytest

pytest.importorskip(
    "ibm_watsonx_ai", reason="ibm_watsonx_ai not installed — install mellea[watsonx]"
)

from unittest.mock import patch

from mellea.backends.watsonx import WatsonxAIBackend


def clear_watsonx_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove all Watsonx env vars so constructor sees a clean environment."""
    monkeypatch.delenv("WATSONX_API_KEY", raising=False)
    monkeypatch.delenv("WATSONX_URL", raising=False)
    monkeypatch.delenv("WATSONX_PROJECT_ID", raising=False)


def _make_backend(monkeypatch: pytest.MonkeyPatch, **kwargs) -> "WatsonxAIBackend":
    """Build a WatsonxAIBackend with SDK internals mocked out.
    Clears all Watsonx env vars first so the constructor only sees kwargs.
    """
    clear_watsonx_env(monkeypatch)
    with (
        patch("mellea.backends.watsonx.Credentials"),
        patch("mellea.backends.watsonx.APIClient"),
        patch("mellea.backends.watsonx.ModelInference"),
    ):
        return WatsonxAIBackend(
            model_id="ibm/granite-4-h-small",
            base_url="https://example.com",
            project_id="test-project",
            **kwargs,
        )


def test_repr_masks_api_key(monkeypatch: pytest.MonkeyPatch):
    """repr redacts an explicit api_key as '***'."""
    backend = _make_backend(monkeypatch, api_key="fake-key")
    r = repr(backend)
    assert "fake-key" not in r
    assert "***" in r
    assert "test-project" not in r


def test_repr_no_key_shows_none(monkeypatch: pytest.MonkeyPatch):
    """repr shows _api_key=None when no key is provided and env is clear."""
    backend = _make_backend(monkeypatch, api_key=None)
    r = repr(backend)
    assert "***" not in r
    assert "_api_key=None" in r
    assert "test-project" not in r


def test_repr_env_key_is_masked(monkeypatch: pytest.MonkeyPatch):
    """repr redacts a key resolved from WATSONX_API_KEY env var as '***'."""
    clear_watsonx_env(monkeypatch)
    monkeypatch.setenv("WATSONX_API_KEY", "env-key")
    with (
        patch("mellea.backends.watsonx.Credentials"),
        patch("mellea.backends.watsonx.APIClient"),
        patch("mellea.backends.watsonx.ModelInference"),
    ):
        backend = WatsonxAIBackend(
            model_id="ibm/granite-4-h-small",
            api_key=None,
            base_url="https://example.com",
            project_id="test-project",
        )
    r = repr(backend)
    assert "env-key" not in r
    assert "***" in r
    assert "_api_key=None" not in r


def test_str_masks_api_key(monkeypatch: pytest.MonkeyPatch):
    """str() delegates to repr and also redacts the api_key."""
    backend = _make_backend(monkeypatch, api_key="fake-key")
    assert "fake-key" not in str(backend)
    assert "***" in str(backend)
    assert "test-project" not in str(backend)
