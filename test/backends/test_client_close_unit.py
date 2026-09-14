# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for `Backend.close()` / `Backend.aclose()` on client-holding backends.

Regression coverage for issue #349: without these, a backend's sync client and
cached async client (or, for Watsonx, its cached `ModelInference`) live for the
process lifetime, leaking sockets.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mellea.backends.openai import OpenAIBackend
from mellea.helpers.async_helpers import get_current_event_loop

# --- Ollama ---


def test_ollama_close_closes_sync_and_async_clients_and_clears_cache(
    mock_ollama_backend,
):
    backend = mock_ollama_backend()

    async_client = backend._client_cache.get(None)
    async_client._client = MagicMock()
    async_client._client.aclose = AsyncMock()

    backend.close()

    backend._client._client.close.assert_called_once()
    async_client._client.aclose.assert_awaited_once()
    assert backend._client_cache.current_size() == 0


async def test_ollama_aclose_closes_sync_and_async_clients_and_clears_cache(
    mock_ollama_backend,
):
    backend = mock_ollama_backend()

    # __init__ ran inside this test's running loop, so the async client it
    # populated the cache with is keyed on that loop, not None.
    async_client = backend._client_cache.get(get_current_event_loop())
    async_client._client = MagicMock()
    async_client._client.aclose = AsyncMock()

    await backend.aclose()

    backend._client._client.close.assert_called_once()
    async_client._client.aclose.assert_awaited_once()
    assert backend._client_cache.current_size() == 0


def test_ollama_close_is_idempotent(mock_ollama_backend):
    backend = mock_ollama_backend()
    backend.close()
    backend.close()  # must not raise on an already-empty cache
    assert backend._client_cache.current_size() == 0


# --- OpenAI ---


def _make_openai_backend() -> OpenAIBackend:
    return OpenAIBackend(
        model_id="gpt-4o", api_key="fake-key", base_url="http://localhost:9999/v1"
    )


def test_openai_close_closes_sync_and_async_clients_and_clears_cache():
    backend = _make_openai_backend()

    backend._client.close = MagicMock()
    async_client = backend._client_cache.get(None)
    async_client.close = AsyncMock()

    backend.close()

    backend._client.close.assert_called_once()
    async_client.close.assert_awaited_once()
    assert backend._client_cache.current_size() == 0


async def test_openai_aclose_closes_sync_and_async_clients_and_clears_cache():
    backend = _make_openai_backend()

    backend._client.close = MagicMock()
    # __init__ ran inside this test's running loop, so the async client it
    # populated the cache with is keyed on that loop, not None.
    async_client = backend._client_cache.get(get_current_event_loop())
    async_client.close = AsyncMock()

    await backend.aclose()

    backend._client.close.assert_called_once()
    async_client.close.assert_awaited_once()
    assert backend._client_cache.current_size() == 0


def test_openai_close_is_idempotent():
    backend = _make_openai_backend()
    backend.close()
    backend.close()
    assert backend._client_cache.current_size() == 0


# --- Watsonx ---

pytest.importorskip(
    "ibm_watsonx_ai", reason="ibm_watsonx_ai not installed — install mellea[watsonx]"
)

from mellea.backends.watsonx import WatsonxAIBackend


def _make_watsonx_backend(monkeypatch: pytest.MonkeyPatch) -> WatsonxAIBackend:
    """Build a WatsonxAIBackend with SDK internals mocked out (mirrors test_watsonx_repr.py)."""
    monkeypatch.delenv("WATSONX_API_KEY", raising=False)
    monkeypatch.delenv("WATSONX_URL", raising=False)
    monkeypatch.delenv("WATSONX_PROJECT_ID", raising=False)
    with (
        patch("mellea.backends.watsonx.Credentials"),
        patch("mellea.backends.watsonx.APIClient"),
        patch("mellea.backends.watsonx.ModelInference"),
    ):
        return WatsonxAIBackend(
            model_id="ibm/granite-4-h-small",
            base_url="https://example.com",
            project_id="test-project",
        )


def test_watsonx_close_closes_model_inference_and_clears_cache(
    monkeypatch: pytest.MonkeyPatch,
):
    backend = _make_watsonx_backend(monkeypatch)

    model_inference = backend._client_cache.get(None)
    model_inference.aclose_persistent_connection = AsyncMock()

    backend.close()

    model_inference.aclose_persistent_connection.assert_awaited_once()
    assert backend._client_cache.current_size() == 0


async def test_watsonx_aclose_closes_model_inference_and_clears_cache(
    monkeypatch: pytest.MonkeyPatch,
):
    backend = _make_watsonx_backend(monkeypatch)

    # __init__ ran inside this test's running loop, so the ModelInference it
    # populated the cache with is keyed on that loop, not None.
    model_inference = backend._client_cache.get(get_current_event_loop())
    model_inference.aclose_persistent_connection = AsyncMock()

    await backend.aclose()

    model_inference.aclose_persistent_connection.assert_awaited_once()
    assert backend._client_cache.current_size() == 0


def test_watsonx_close_is_idempotent(monkeypatch: pytest.MonkeyPatch):
    backend = _make_watsonx_backend(monkeypatch)
    backend.close()
    backend.close()
    assert backend._client_cache.current_size() == 0


if __name__ == "__main__":
    pytest.main([__file__])
