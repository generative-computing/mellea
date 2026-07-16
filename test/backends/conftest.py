# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared fixtures for backend tests."""

from unittest.mock import MagicMock, patch

import pytest

from mellea.backends.ollama import OllamaModelBackend


@pytest.fixture
def mock_ollama_backend():
    """Factory fixture: returns an OllamaModelBackend with all network calls patched out.

    No live Ollama server or pulled model is required. The returned backend has
    real client objects replaced with MagicMocks, so subsequent tests can set
    attributes such as ``backend._async_client.chat`` to control behaviour.

    Usage::

        def test_something(mock_ollama_backend):
            backend = mock_ollama_backend(model_options={ModelOption.MAX_NEW_TOKENS: 5})
            # _async_client is an event-loop-keyed property; instance assignment won't
            # override it for tests that call through _run_async_in_thread.  Patch at
            # the class level instead:
            mock_async = MagicMock()
            mock_async.chat = AsyncMock(return_value=canned_response)
            with patch.object(
                type(backend), "_async_client", new_callable=PropertyMock, return_value=mock_async
            ):
                yield MelleaSession(backend)
    """

    def _make(
        model_id: str = "granite4.1:3b",
        model_options: dict | None = None,
        timeout: float | None = None,
    ) -> OllamaModelBackend:
        with (
            patch.object(OllamaModelBackend, "_check_ollama_server", return_value=True),
            patch.object(OllamaModelBackend, "_pull_ollama_model", return_value=True),
            patch("mellea.backends.ollama.ollama.Client", return_value=MagicMock()),
            patch(
                "mellea.backends.ollama.ollama.AsyncClient", return_value=MagicMock()
            ),
        ):
            return OllamaModelBackend(
                model_id=model_id, model_options=model_options, timeout=timeout
            )

    return _make
