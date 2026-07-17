# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for Ollama backend pure-logic helpers — no Ollama server required.

Covers _simplify_and_merge, _make_backend_specific_and_remove,
chat_response_delta_merge, timeout wiring, and generate_from_raw
empty-response handling (#599).
"""

from unittest.mock import AsyncMock, MagicMock, patch

import ollama
import pytest

from mellea.backends import ModelOption
from mellea.backends.ollama import OllamaModelBackend, chat_response_delta_merge
from mellea.core import CBlock, ModelOutputThunk
from mellea.stdlib.context import SimpleContext


@pytest.fixture
def backend(mock_ollama_backend):
    """Return an OllamaModelBackend with no pre-set model options."""
    return mock_ollama_backend(model_id="granite3.3:8b")


# --- Map consistency ---


def test_from_mellea_keys_are_subset_of_to_mellea_values(backend):
    """Every key in from_mellea must appear as a value in to_mellea (maps agree)."""
    to_values = set(backend.to_mellea_model_opts_map.values())
    from_keys = set(backend.from_mellea_model_opts_map.keys())
    assert from_keys <= to_values, (
        f"from_mellea has keys absent from to_mellea values: {from_keys - to_values}"
    )


# --- _strip_data_uri_prefix ---


def test_strip_data_uri_prefix_removes_prefix():
    """_strip_data_uri_prefix removes data URI prefix from base64 strings."""
    from mellea.backends.ollama import _strip_data_uri_prefix

    images = [
        "data:image/png;base64,iVBORw0KGgo...",
        "data:image/jpeg;base64,/9j/4AAQSkZJRg...",
    ]
    result = _strip_data_uri_prefix(images)
    assert result[0] == "iVBORw0KGgo..."
    assert result[1] == "/9j/4AAQSkZJRg..."


def test_strip_data_uri_prefix_handles_already_stripped():
    """_strip_data_uri_prefix leaves already-stripped base64 strings unchanged."""
    from mellea.backends.ollama import _strip_data_uri_prefix

    images = ["iVBORw0KGgo...", "/9j/4AAQSkZJRg..."]
    result = _strip_data_uri_prefix(images)
    assert result[0] == "iVBORw0KGgo..."
    assert result[1] == "/9j/4AAQSkZJRg..."


def test_strip_data_uri_prefix_mixed_input():
    """_strip_data_uri_prefix handles mixed prefixed and unprefixed strings."""
    from mellea.backends.ollama import _strip_data_uri_prefix

    images = [
        "data:image/png;base64,iVBORw0KGgo...",
        "already-stripped-base64",
        "data:image/webp;base64,UklGRiQAAABXRUJQ...",
    ]
    result = _strip_data_uri_prefix(images)
    assert result[0] == "iVBORw0KGgo..."
    assert result[1] == "already-stripped-base64"
    assert result[2] == "UklGRiQAAABXRUJQ..."


def test_strip_data_uri_prefix_empty_list():
    """_strip_data_uri_prefix handles empty list."""
    from mellea.backends.ollama import _strip_data_uri_prefix

    result = _strip_data_uri_prefix([])
    assert result == []


def test_strip_data_uri_prefix_preserves_order():
    """_strip_data_uri_prefix preserves the order of images."""
    from mellea.backends.ollama import _strip_data_uri_prefix

    images = [
        "data:image/png;base64,first",
        "data:image/png;base64,second",
        "data:image/png;base64,third",
    ]
    result = _strip_data_uri_prefix(images)
    assert result == ["first", "second", "third"]


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


def test_simplify_and_merge_per_call_overrides_backend(mock_ollama_backend):
    # Backend sets num_predict=128; per-call value of 256 must win.
    b = mock_ollama_backend(
        model_id="granite3.3:8b", model_options={"num_predict": 128}
    )
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
    assert mot.raw.response is delta


def test_delta_merge_second_appends_content():
    mot = ModelOutputThunk(value=None)
    chat_response_delta_merge(mot, _make_delta("Hello"))
    chat_response_delta_merge(mot, _make_delta(" world"))
    assert mot.raw.response.message.content == "Hello world"


def test_delta_merge_done_propagated():
    mot = ModelOutputThunk(value=None)
    chat_response_delta_merge(mot, _make_delta("partial", done=False))
    chat_response_delta_merge(mot, _make_delta("", done=True))
    assert mot.raw.response.done is True


def test_delta_merge_role_set_from_first_delta():
    mot = ModelOutputThunk(value=None)
    chat_response_delta_merge(mot, _make_delta("hi", role="assistant"))
    chat_response_delta_merge(mot, _make_delta(" there", role=""))
    assert mot.raw.response.message.role == "assistant"


def test_delta_merge_thinking_concatenated():
    mot = ModelOutputThunk(value=None)
    chat_response_delta_merge(mot, _make_delta("reply", thinking="step 1"))
    chat_response_delta_merge(mot, _make_delta("", thinking=" step 2"))
    assert mot.raw.response.message.thinking == "step 1 step 2"


@pytest.mark.asyncio
async def test_processing_initializes_and_accumulates_thinking(
    backend: OllamaModelBackend,
):
    """processing() initializes thinking and accumulates chunk thinking text."""
    mot = ModelOutputThunk(value=None)
    await backend.processing(mot, _make_delta("answer", thinking="step 1"), {})

    assert mot.thinking == "step 1"
    assert mot._underlying_value == "answer"


# --- timeout wiring ---


def test_timeout_default_forwarded_to_clients():
    """When timeout is omitted, the 300 s default must be forwarded to both Ollama clients."""
    with (
        patch.object(OllamaModelBackend, "_check_ollama_server", return_value=True),
        patch.object(OllamaModelBackend, "_pull_ollama_model", return_value=True),
        patch("mellea.backends.ollama.ollama.Client") as mock_client,
        patch("mellea.backends.ollama.ollama.AsyncClient") as mock_async_client,
    ):
        OllamaModelBackend(model_id="granite3.3:8b")

    _, sync_kwargs = mock_client.call_args
    assert sync_kwargs.get("timeout") == 300.0
    _, async_kwargs = mock_async_client.call_args
    assert async_kwargs.get("timeout") == 300.0


def test_timeout_none_not_forwarded_to_clients():
    """Explicit timeout=None must omit the key — preserves the upstream SDK default (no timeout)."""
    with (
        patch.object(OllamaModelBackend, "_check_ollama_server", return_value=True),
        patch.object(OllamaModelBackend, "_pull_ollama_model", return_value=True),
        patch("mellea.backends.ollama.ollama.Client") as mock_client,
        patch("mellea.backends.ollama.ollama.AsyncClient") as mock_async_client,
    ):
        OllamaModelBackend(model_id="granite3.3:8b", timeout=None)

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


def test_timeout_forwarded_to_new_async_clients_per_event_loop(mock_ollama_backend):
    """Newly created AsyncClients (one per event loop) must inherit the timeout."""
    backend = mock_ollama_backend(model_id="granite3.3:8b", timeout=7.0)
    with patch(
        "mellea.backends.ollama.ollama.AsyncClient", return_value=MagicMock()
    ) as mock_async_client:
        backend._client_cache = type(backend._client_cache)(2)  # reset cache
        _ = backend._async_client

    _, async_kwargs = mock_async_client.call_args
    assert async_kwargs.get("timeout") == 7.0


# --- generate_from_raw empty-response handling (#599) ---


async def test_generate_from_raw_empty_response_soft_fails(mock_ollama_backend) -> None:
    """Empty done Ollama response soft-fails instead of raising.

    The result list must have the same length as the input actions so that
    sibling results from the same gather call are not discarded. The failing
    slot must have value="" and carry the RuntimeError on `mot.error`.
    """
    backend = mock_ollama_backend()
    empty = ollama.GenerateResponse(response="", done=True)

    with (
        patch("mellea.backends.ollama.ollama.AsyncClient", return_value=MagicMock()),
        patch(
            "mellea.backends.ollama.asyncio.gather", new=AsyncMock(return_value=[empty])
        ),
    ):
        results = await backend.generate_from_raw(
            actions=[CBlock("What is 1+1?")], ctx=SimpleContext()
        )

    assert len(results) == 1, (
        "result list must match action count even on empty response"
    )
    assert results[0].value == "", "empty-response slot should have value=''"
    err = results[0].error
    assert isinstance(err, RuntimeError), (
        f"expected RuntimeError on mot.error, got {err!r}"
    )
    assert "599" in str(err), "error message should reference issue #599"
    assert "16326" in str(err), "error message should reference upstream Ollama issue"
    assert results[0].cancelled is False, (
        "soft-fail must not flip the cancelled flag — it is a sibling channel"
    )


async def test_generate_from_raw_thinking_response_not_flagged(
    mock_ollama_backend,
) -> None:
    """A thinking-model response with response="" and non-empty thinking is not an error.

    Thinking models legitimately return an empty response string alongside a non-empty
    thinking field — this must not be treated as an empty-response soft-fail.
    """
    backend = mock_ollama_backend()
    thinking_response = ollama.GenerateResponse(
        response="", thinking="Let me work this out...", done=True
    )

    with (
        patch("mellea.backends.ollama.ollama.AsyncClient", return_value=MagicMock()),
        patch(
            "mellea.backends.ollama.asyncio.gather",
            new=AsyncMock(return_value=[thinking_response]),
        ),
    ):
        results = await backend.generate_from_raw(
            actions=[CBlock("What is 1+1?")], ctx=SimpleContext()
        )

    assert len(results) == 1
    assert results[0].error is None, (
        "thinking-only response should not be flagged as a model-load race"
    )


async def test_generate_from_raw_preserves_sibling_results_on_empty(
    mock_ollama_backend,
) -> None:
    """One empty response in a batch of three does not discard the other two results."""
    backend = mock_ollama_backend()
    good = ollama.GenerateResponse(
        response="2", done=True, eval_count=1, prompt_eval_count=5
    )
    empty = ollama.GenerateResponse(response="", done=True)

    with (
        patch("mellea.backends.ollama.ollama.AsyncClient", return_value=MagicMock()),
        patch(
            "mellea.backends.ollama.asyncio.gather",
            new=AsyncMock(return_value=[good, empty, good]),
        ),
    ):
        results = await backend.generate_from_raw(
            actions=[CBlock("q1"), CBlock("q2"), CBlock("q3")], ctx=SimpleContext()
        )

    assert len(results) == 3, "all three slots should be returned"
    assert results[0].value == "2"
    assert results[0].error is None
    assert results[1].value == ""
    assert results[1].error is not None
    assert results[2].value == "2"
    assert results[2].error is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

"""Unit tests for Ollama backend pure-logic helpers — no Ollama server required.

Covers _simplify_and_merge, _make_backend_specific_and_remove,
chat_response_delta_merge, _strip_data_uri_prefix, and generate_from_raw exception propagation.
"""
