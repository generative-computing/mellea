"""Unit tests for Ollama backend pure-logic helpers — no Ollama server required.

Covers _simplify_and_merge, _make_backend_specific_and_remove,
chat_response_delta_merge, and generate_from_raw exception propagation.
"""

from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch

import ollama
import pytest

from mellea.backends import ModelOption
from mellea.backends.ollama import OllamaModelBackend, chat_response_delta_merge
from mellea.core import CBlock, ModelOutputThunk
from mellea.stdlib.context import SimpleContext


def _make_backend(
    model_options: dict | None = None, timeout: float | None = None
) -> OllamaModelBackend:
    """Return an OllamaModelBackend with all network calls patched."""
    with (
        patch.object(OllamaModelBackend, "_check_ollama_server", return_value=True),
        patch.object(OllamaModelBackend, "_pull_ollama_model", return_value=True),
        patch("mellea.backends.ollama.ollama.Client", return_value=MagicMock()),
        patch("mellea.backends.ollama.ollama.AsyncClient", return_value=MagicMock()),
    ):
        return OllamaModelBackend(
            model_id="granite3.3:8b", model_options=model_options, timeout=timeout
        )


@pytest.fixture
def backend():
    """Return an OllamaModelBackend with no pre-set model options."""
    return _make_backend()


# --- Map consistency ---


def test_from_mellea_keys_are_subset_of_to_mellea_values(backend):
    """Every key in from_mellea must appear as a value in to_mellea (maps agree)."""
    to_values = set(backend.to_mellea_model_opts_map.values())
    from_keys = set(backend.from_mellea_model_opts_map.keys())
    assert from_keys <= to_values, (
        f"from_mellea has keys absent from to_mellea values: {from_keys - to_values}"
    )


# --- _simplify_and_merge ---


def test_simplify_and_merge_none_returns_empty_dict(backend):
    result = backend._simplify_and_merge(None)
    assert result == {}


def test_simplify_and_merge_all_to_mellea_entries(backend):
    """Every to_mellea entry remaps to its ModelOption via _simplify_and_merge."""
    for backend_key, mellea_key in backend.to_mellea_model_opts_map.items():
        # STOP_SEQUENCES is validated as list[str]; other sentinels accept anything.
        value = ["STOP"] if mellea_key == ModelOption.STOP_SEQUENCES else 42
        result = backend._simplify_and_merge({backend_key: value})
        assert mellea_key in result, f"{backend_key!r} did not produce {mellea_key!r}"
        assert result[mellea_key] == value


def test_simplify_and_merge_remaps_num_predict(backend):
    """Hardcoded anchor: the most critical mapping for generation length."""
    result = backend._simplify_and_merge({"num_predict": 128})
    assert ModelOption.MAX_NEW_TOKENS in result
    assert result[ModelOption.MAX_NEW_TOKENS] == 128


def test_simplify_and_merge_per_call_overrides_backend():
    # Backend sets num_predict=128; per-call value of 256 must win.
    b = _make_backend(model_options={"num_predict": 128})
    result = b._simplify_and_merge({"num_predict": 256})
    assert result[ModelOption.MAX_NEW_TOKENS] == 256


# --- _make_backend_specific_and_remove ---


def test_make_backend_specific_all_from_mellea_entries(backend):
    """Every from_mellea entry remaps to its backend key via _make_backend_specific_and_remove."""
    for mellea_key, backend_key in backend.from_mellea_model_opts_map.items():
        result = backend._make_backend_specific_and_remove({mellea_key: 42})
        assert backend_key in result, f"{mellea_key!r} did not produce {backend_key!r}"
        assert result[backend_key] == 42


def test_make_backend_specific_remaps_max_new_tokens(backend):
    """Hardcoded anchor: the most critical mapping for generation length."""
    opts = {ModelOption.MAX_NEW_TOKENS: 64}
    result = backend._make_backend_specific_and_remove(opts)
    assert "num_predict" in result
    assert result["num_predict"] == 64


def test_make_backend_specific_removes_sentinel_keys(backend):
    opts = {ModelOption.MAX_NEW_TOKENS: 32, ModelOption.SYSTEM_PROMPT: "sys"}
    result = backend._make_backend_specific_and_remove(opts)
    # Sentinel keys not in from_mellea_model_opts_map should be removed
    assert ModelOption.SYSTEM_PROMPT not in result


# --- chat_response_delta_merge ---


def _make_delta(
    content: str,
    role: str = "assistant",
    done: bool = False,
    thinking: str | None = None,
) -> ollama.ChatResponse:
    msg = ollama.Message(role=role, content=content, thinking=thinking)
    return ollama.ChatResponse(model="test", created_at=None, message=msg, done=done)


def test_delta_merge_first_sets_chat_response():
    mot = ModelOutputThunk(value=None)
    delta = _make_delta("Hello")
    chat_response_delta_merge(mot, delta)
    assert mot._meta["chat_response"] is delta


def test_delta_merge_second_appends_content():
    mot = ModelOutputThunk(value=None)
    chat_response_delta_merge(mot, _make_delta("Hello"))
    chat_response_delta_merge(mot, _make_delta(" world"))
    assert mot._meta["chat_response"].message.content == "Hello world"


def test_delta_merge_done_propagated():
    mot = ModelOutputThunk(value=None)
    chat_response_delta_merge(mot, _make_delta("partial", done=False))
    chat_response_delta_merge(mot, _make_delta("", done=True))
    assert mot._meta["chat_response"].done is True


def test_delta_merge_role_set_from_first_delta():
    mot = ModelOutputThunk(value=None)
    chat_response_delta_merge(mot, _make_delta("hi", role="assistant"))
    chat_response_delta_merge(mot, _make_delta(" there", role=""))
    assert mot._meta["chat_response"].message.role == "assistant"


def test_delta_merge_thinking_concatenated():
    mot = ModelOutputThunk(value=None)
    chat_response_delta_merge(mot, _make_delta("reply", thinking="step 1"))
    chat_response_delta_merge(mot, _make_delta("", thinking=" step 2"))
    assert mot._meta["chat_response"].message.thinking == "step 1 step 2"


# --- timeout wiring ---


def test_timeout_default_not_passed_to_clients():
    """When timeout is omitted, it must not be forwarded to the Ollama clients."""
    with (
        patch.object(OllamaModelBackend, "_check_ollama_server", return_value=True),
        patch.object(OllamaModelBackend, "_pull_ollama_model", return_value=True),
        patch("mellea.backends.ollama.ollama.Client") as mock_client,
        patch("mellea.backends.ollama.ollama.AsyncClient") as mock_async_client,
    ):
        OllamaModelBackend(model_id="granite3.3:8b")

    _, sync_kwargs = mock_client.call_args
    assert "timeout" not in sync_kwargs
    _, async_kwargs = mock_async_client.call_args
    assert "timeout" not in async_kwargs


def test_timeout_forwarded_to_sync_and_async_clients():
    """When timeout is set, it must reach both ollama.Client and ollama.AsyncClient."""
    with (
        patch.object(OllamaModelBackend, "_check_ollama_server", return_value=True),
        patch.object(OllamaModelBackend, "_pull_ollama_model", return_value=True),
        patch("mellea.backends.ollama.ollama.Client") as mock_client,
        patch("mellea.backends.ollama.ollama.AsyncClient") as mock_async_client,
    ):
        OllamaModelBackend(model_id="granite3.3:8b", timeout=12.5)

    _, sync_kwargs = mock_client.call_args
    assert sync_kwargs.get("timeout") == 12.5
    _, async_kwargs = mock_async_client.call_args
    assert async_kwargs.get("timeout") == 12.5


def test_timeout_forwarded_to_new_async_clients_per_event_loop():
    """Newly created AsyncClients (one per event loop) must inherit the timeout."""
    backend = _make_backend(timeout=7.0)
    with patch(
        "mellea.backends.ollama.ollama.AsyncClient", return_value=MagicMock()
    ) as mock_async_client:
        backend._client_cache = type(backend._client_cache)(2)  # reset cache
        _ = backend._async_client

    _, async_kwargs = mock_async_client.call_args
    assert async_kwargs.get("timeout") == 7.0


# --- generate_from_raw exception propagation (#597) ---


async def test_generate_from_raw_propagates_exception(backend):
    """Exceptions from individual Ollama requests must propagate to the caller.

    Regression test for #597: previously, exceptions were silently converted to
    ModelOutputThunk(value="") via asyncio.gather(return_exceptions=True).
    """
    error = ConnectionError("ollama server unavailable")
    mock_async_client = MagicMock()
    mock_async_client.generate = AsyncMock(side_effect=error)

    with patch.object(
        type(backend),
        "_async_client",
        new_callable=PropertyMock,
        return_value=mock_async_client,
    ):
        with pytest.raises(ConnectionError, match="ollama server unavailable"):
            await backend.generate_from_raw(
                actions=[CBlock(value="what is 1+1?")], ctx=SimpleContext()
            )


async def test_generate_from_raw_multi_action_fail_fast(backend):
    """Any failure in a multi-action batch raises immediately; no partial results.

    Documents the all-or-nothing batch semantics introduced by #597. Previously
    the failed slot was silently replaced with ModelOutputThunk(value="") and
    the caller received a partial list. Now the first exception propagates and
    the caller receives nothing.
    """
    mock_async_client = MagicMock()
    mock_async_client.generate = AsyncMock(
        side_effect=ConnectionError("request failed")
    )

    with patch.object(
        type(backend),
        "_async_client",
        new_callable=PropertyMock,
        return_value=mock_async_client,
    ):
        with pytest.raises(ConnectionError):
            await backend.generate_from_raw(
                actions=[CBlock("a"), CBlock("b"), CBlock("c")], ctx=SimpleContext()
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
