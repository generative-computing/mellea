"""Unit tests for ModelOption.STOP_SEQUENCES routing across backends.

Verifies that the sentinel is mapped to the correct native parameter for each
backend that exposes a stop-sequence concept. No live model or server is used.
"""

from unittest.mock import MagicMock, patch

import pytest

from mellea.backends import ModelOption
from mellea.backends.ollama import OllamaModelBackend
from mellea.backends.openai import OpenAIBackend

# --- Input validation ---


def test_stop_sequences_rejects_bare_string():
    """A bare ``str`` is rejected — only ``list[str]`` is allowed."""
    backend = _make_openai_backend()
    with pytest.raises(TypeError, match="STOP_SEQUENCES must be a list"):
        backend._simplify_and_merge(
            {ModelOption.STOP_SEQUENCES: "END"}, is_chat_context=True
        )


def test_stop_sequences_rejects_non_string_elements():
    """Lists containing non-str elements are rejected."""
    backend = _make_openai_backend()
    with pytest.raises(TypeError, match="STOP_SEQUENCES must be a list"):
        backend._simplify_and_merge(
            {ModelOption.STOP_SEQUENCES: ["END", 42]}, is_chat_context=True
        )


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


# --- Watsonx ---


def _make_watsonx_backend():
    """Construct a WatsonxAIBackend without touching the network."""
    pytest.importorskip("ibm_watsonx_ai")
    from mellea.backends.watsonx import WatsonxAIBackend

    with (
        patch("mellea.backends.watsonx.Credentials", return_value=MagicMock()),
        patch("mellea.backends.watsonx.APIClient", return_value=MagicMock()),
        patch("mellea.backends.watsonx.ModelInference", return_value=MagicMock()),
    ):
        return WatsonxAIBackend(
            model_id="ibm/granite-3-8b-instruct",
            api_key="fake-key",
            base_url="https://example.invalid",
            project_id="fake-project",
        )


@pytest.mark.parametrize(
    "is_chat,native_key", [(True, "stop"), (False, "stop_sequences")]
)
def test_watsonx_stop_sequences_round_trip(is_chat, native_key):
    """Native -> sentinel -> native for both chat (``stop``) and completions (``stop_sequences``)."""
    backend = _make_watsonx_backend()
    stops = ["END", "</s>"]

    simplified = backend._simplify_and_merge(
        {native_key: stops}, is_chat_context=is_chat
    )
    assert simplified[ModelOption.STOP_SEQUENCES] == stops

    backend_specific = backend._make_backend_specific_and_remove(
        {ModelOption.STOP_SEQUENCES: stops}, is_chat_context=is_chat
    )
    assert backend_specific[native_key] == stops
    assert ModelOption.STOP_SEQUENCES not in backend_specific


# --- HuggingFace _generate_kwargs branch ---


def _make_hf_backend_stub():
    """Construct a LocalHFBackend without loading a model.

    Bypasses ``__init__`` because the real constructor downloads weights. We only
    need the attributes ``_generate_kwargs`` reads: ``_tokenizer`` and
    ``from_mellea_model_opts_map``.
    """
    pytest.importorskip("transformers")
    from mellea.backends.huggingface import LocalHFBackend

    backend = LocalHFBackend.__new__(LocalHFBackend)
    backend._tokenizer = MagicMock(name="tokenizer")  # type: ignore[attr-defined]
    backend.from_mellea_model_opts_map = {  # type: ignore[attr-defined]
        ModelOption.STOP_SEQUENCES: "stop_strings"
    }
    return backend


def test_hf_generate_kwargs_injects_tokenizer_when_stop_strings_present():
    backend = _make_hf_backend_stub()
    kwargs = backend._generate_kwargs({ModelOption.STOP_SEQUENCES: ["END"]})
    assert kwargs["stop_strings"] == ["END"]
    assert kwargs["tokenizer"] is backend._tokenizer


def test_hf_generate_kwargs_omits_tokenizer_when_no_stop_strings():
    backend = _make_hf_backend_stub()
    kwargs = backend._generate_kwargs({"max_new_tokens": 16})
    assert "tokenizer" not in kwargs
    assert "stop_strings" not in kwargs


def test_hf_generate_kwargs_preserves_user_supplied_tokenizer():
    backend = _make_hf_backend_stub()
    user_tokenizer = MagicMock(name="user-tokenizer")
    kwargs = backend._generate_kwargs(
        {ModelOption.STOP_SEQUENCES: ["END"], "tokenizer": user_tokenizer}
    )
    assert kwargs["tokenizer"] is user_tokenizer
