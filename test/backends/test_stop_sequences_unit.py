"""Unit tests for ModelOption.STOP_SEQUENCES routing across backends.

Verifies that the sentinel is mapped to the correct native parameter for each
backend that exposes a stop-sequence concept. No live model or server is used.
"""

from unittest.mock import MagicMock, patch

import pytest

from mellea.backends import ModelOption
from mellea.backends.ollama import OllamaModelBackend
from mellea.backends.openai import OpenAIBackend


def test_sentinel_value_is_unique():
    """STOP_SEQUENCES sentinel must not collide with other ModelOption values."""
    sentinels = {
        v
        for k, v in vars(ModelOption).items()
        if isinstance(v, str) and v.startswith("@@@")
    }
    assert ModelOption.STOP_SEQUENCES in sentinels
    # No duplicates among sentinel values.
    sentinel_values = [
        v
        for k, v in vars(ModelOption).items()
        if isinstance(v, str) and v.startswith("@@@")
    ]
    assert len(sentinel_values) == len(set(sentinel_values))


# --- OpenAI ---


def _make_openai_backend() -> OpenAIBackend:
    return OpenAIBackend(
        model_id="gpt-4o", api_key="fake-key", base_url="http://localhost:9999/v1"
    )


@pytest.mark.parametrize("context", ["chats", "completions"])
def test_openai_stop_sequences_round_trip(context):
    """Native ``stop`` -> sentinel -> ``stop`` for both chat and completions."""
    backend = _make_openai_backend()
    is_chat = context == "chats"
    stops = ["END", "</s>"]

    # to_mellea: native -> sentinel
    simplified = backend._simplify_and_merge({"stop": stops}, is_chat_context=is_chat)
    assert simplified[ModelOption.STOP_SEQUENCES] == stops

    # from_mellea: sentinel -> native
    backend_specific = backend._make_backend_specific_and_remove(
        {ModelOption.STOP_SEQUENCES: stops}, is_chat_context=is_chat
    )
    assert backend_specific["stop"] == stops
    assert ModelOption.STOP_SEQUENCES not in backend_specific


# --- Ollama ---


def _make_ollama_backend() -> OllamaModelBackend:
    with (
        patch.object(OllamaModelBackend, "_check_ollama_server", return_value=True),
        patch.object(OllamaModelBackend, "_pull_ollama_model", return_value=True),
        patch("mellea.backends.ollama.ollama.Client", return_value=MagicMock()),
        patch("mellea.backends.ollama.ollama.AsyncClient", return_value=MagicMock()),
    ):
        return OllamaModelBackend(model_id="granite3.3:8b")


def test_ollama_stop_sequences_round_trip():
    backend = _make_ollama_backend()
    stops = ["END", "\n\nUser:"]

    simplified = backend._simplify_and_merge({"stop": stops})
    assert simplified[ModelOption.STOP_SEQUENCES] == stops

    backend_specific = backend._make_backend_specific_and_remove(
        {ModelOption.STOP_SEQUENCES: stops}
    )
    assert backend_specific["stop"] == stops
    assert ModelOption.STOP_SEQUENCES not in backend_specific


# --- LiteLLM ---


def test_litellm_stop_sequences_round_trip():
    pytest.importorskip("litellm")
    from mellea.backends.litellm import LiteLLMBackend

    backend = LiteLLMBackend(model_id="ollama_chat/granite3.3:8b")
    stops = ["END"]

    simplified = backend._simplify_and_merge({"stop": stops})
    assert simplified[ModelOption.STOP_SEQUENCES] == stops

    backend_specific = backend._make_backend_specific_and_remove(
        {ModelOption.STOP_SEQUENCES: stops}
    )
    assert backend_specific["stop"] == stops


# --- HuggingFace and Watsonx map shape (source-level, no optional deps) ---


def _read(path: str) -> str:
    from pathlib import Path

    return Path(path).read_text()


def test_huggingface_map_entries_present():
    """Verify HF backend source registers STOP_SEQUENCES <-> stop_strings."""
    src = _read("mellea/backends/huggingface.py")
    assert '"stop_strings": ModelOption.STOP_SEQUENCES' in src
    assert 'ModelOption.STOP_SEQUENCES: "stop_strings"' in src


def test_watsonx_map_entries_present():
    """Verify Watsonx backend source registers STOP_SEQUENCES for both endpoints."""
    src = _read("mellea/backends/watsonx.py")
    assert '"stop": ModelOption.STOP_SEQUENCES' in src
    assert '"stop_sequences": ModelOption.STOP_SEQUENCES' in src
    assert 'ModelOption.STOP_SEQUENCES: "stop"' in src
    assert 'ModelOption.STOP_SEQUENCES: "stop_sequences"' in src
