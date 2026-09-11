# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the in-memory response store and previous_response_id session chaining."""

import time
from typing import Literal
from unittest.mock import MagicMock

import pytest

import cli.serve.app as app_module
from cli.serve.app import _build_history_from_messages, _response_store, _StoredResponse
from cli.serve.models import Response, ResponseRequest, ResponseUsage

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_response(response_id: str = "resp_abc") -> Response:
    return Response(
        id=response_id,
        created_at=int(time.time()),
        model="test-model",
        status="completed",
        output=[],
        output_text="hello",
        usage=ResponseUsage(input_tokens=5, output_tokens=3, total_tokens=8),
    )


def _make_chat_message(
    role: Literal["system", "user", "assistant", "tool", "function"], content: str
):
    from mellea.serve.models import ChatMessage

    return ChatMessage(role=role, content=content)


# ---------------------------------------------------------------------------
# _build_history_from_messages
# ---------------------------------------------------------------------------


class TestBuildHistoryFromMessages:
    """_build_history_from_messages serialises messages + assistant reply."""

    def test_basic_round_trip(self):
        """Messages and assistant text are serialised into plain dicts."""
        msgs = [
            _make_chat_message("system", "You are helpful."),
            _make_chat_message("user", "Hi"),
        ]
        history = _build_history_from_messages(msgs, "Hello!")
        assert history == [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
        ]

    def test_empty_messages(self):
        """Works with no prior messages — only the assistant reply is added."""
        history = _build_history_from_messages([], "standalone reply")
        assert history == [{"role": "assistant", "content": "standalone reply"}]

    def test_assistant_text_appended_last(self):
        """The assistant entry is always the last item in history."""
        msgs = [_make_chat_message("user", "question")]
        history = _build_history_from_messages(msgs, "answer")
        assert history[-1] == {"role": "assistant", "content": "answer"}


# ---------------------------------------------------------------------------
# _StoredResponse
# ---------------------------------------------------------------------------


class TestStoredResponse:
    """_StoredResponse defaults stored_at to now."""

    def test_stored_at_defaults_to_now(self):
        before = time.time()
        entry = _StoredResponse(response=_make_response(), history=[])
        after = time.time()
        assert before <= entry.stored_at <= after

    def test_stored_at_can_be_overridden(self):
        entry = _StoredResponse(response=_make_response(), history=[], stored_at=0.0)
        assert entry.stored_at == 0.0


# ---------------------------------------------------------------------------
# _evict_expired_responses
# ---------------------------------------------------------------------------


class TestEvictExpiredResponses:
    """TTL eviction removes old entries and keeps fresh ones."""

    def test_evicts_expired_entry(self):
        _response_store.clear()
        _response_store["old"] = _StoredResponse(
            response=_make_response("old"), history=[], stored_at=0.0
        )
        app_module._response_ttl_seconds = 1800

        # Run one sweep iteration directly (skip the asyncio.sleep)
        cutoff = time.time() - app_module._response_ttl_seconds
        expired = [
            rid for rid, entry in _response_store.items() if entry.stored_at < cutoff
        ]
        for rid in expired:
            del _response_store[rid]

        assert "old" not in _response_store

    def test_keeps_fresh_entry(self):
        _response_store.clear()
        _response_store["fresh"] = _StoredResponse(
            response=_make_response("fresh"), history=[]
        )
        app_module._response_ttl_seconds = 1800

        cutoff = time.time() - app_module._response_ttl_seconds
        expired = [
            rid for rid, entry in _response_store.items() if entry.stored_at < cutoff
        ]
        for rid in expired:
            del _response_store[rid]

        assert "fresh" in _response_store
        _response_store.clear()


# ---------------------------------------------------------------------------
# make_responses_endpoint — store and previous_response_id
# ---------------------------------------------------------------------------


def _make_mock_module(reply: str = "the answer"):
    """Return a minimal module mock whose serve() returns a computed thunk."""
    thunk = MagicMock()
    thunk.is_computed.return_value = True
    thunk.value = reply
    thunk.generation = MagicMock()
    thunk.generation.usage = None

    module = MagicMock()
    module.__name__ = "test_module"
    module.serve = MagicMock(return_value=thunk)
    return module


class TestResponsesEndpointStore:
    """Endpoint stores responses and chains history via previous_response_id."""

    def setup_method(self):
        _response_store.clear()

    def teardown_method(self):
        _response_store.clear()

    @pytest.mark.asyncio
    async def test_response_is_stored_when_store_true(self):
        """A completed response is placed in _response_store when store=True."""
        module = _make_mock_module("first reply")
        endpoint = app_module.make_responses_endpoint(module)

        request = ResponseRequest(model="m", input="hello", store=True)
        response = await endpoint(request)

        assert response.id in _response_store
        stored = _response_store[response.id]
        assert stored.response.output_text == "first reply"

    @pytest.mark.asyncio
    async def test_response_not_stored_when_store_false(self):
        """No entry is written when store=False."""
        module = _make_mock_module("ephemeral")
        endpoint = app_module.make_responses_endpoint(module)

        request = ResponseRequest(model="m", input="hi", store=False)
        response = await endpoint(request)

        assert response.id not in _response_store

    @pytest.mark.asyncio
    async def test_previous_response_id_prepends_history(self):
        """A second request with previous_response_id receives the prior history."""
        module = _make_mock_module("second reply")
        endpoint = app_module.make_responses_endpoint(module)

        # Seed the store with a prior response
        prior_id = "resp_prior000"
        _response_store[prior_id] = _StoredResponse(
            response=_make_response(prior_id),
            history=[
                {"role": "user", "content": "first question"},
                {"role": "assistant", "content": "first answer"},
            ],
        )

        request = ResponseRequest(
            model="m", input="follow-up question", previous_response_id=prior_id
        )
        await endpoint(request)

        # The serve() call should have received 3 messages: 2 history + 1 new
        call_messages = module.serve.call_args.kwargs["input"]
        assert len(call_messages) == 3
        assert call_messages[0].role == "user"
        assert call_messages[0].content == "first question"
        assert call_messages[1].role == "assistant"
        assert call_messages[1].content == "first answer"
        assert call_messages[2].role == "user"
        assert call_messages[2].content == "follow-up question"

    @pytest.mark.asyncio
    async def test_unknown_previous_response_id_returns_404(self):
        """A 404 error response is returned for an unknown previous_response_id."""
        from fastapi.responses import JSONResponse

        module = _make_mock_module()
        endpoint = app_module.make_responses_endpoint(module)

        request = ResponseRequest(
            model="m", input="anything", previous_response_id="resp_doesnotexist"
        )
        result = await endpoint(request)

        assert isinstance(result, JSONResponse)
        assert result.status_code == 404

    @pytest.mark.asyncio
    async def test_stored_history_includes_assistant_reply(self):
        """History written to the store includes the assistant's reply as last entry."""
        module = _make_mock_module("my answer")
        endpoint = app_module.make_responses_endpoint(module)

        request = ResponseRequest(model="m", input="a question", store=True)
        response = await endpoint(request)

        history = _response_store[response.id].history
        assert history[-1] == {"role": "assistant", "content": "my answer"}


# ---------------------------------------------------------------------------
# GET /v1/responses/{response_id}
# ---------------------------------------------------------------------------


class TestGetResponseEndpoint:
    """GET /v1/responses/{id} retrieves stored responses or returns 404."""

    def setup_method(self):
        _response_store.clear()

    def teardown_method(self):
        _response_store.clear()

    @pytest.mark.asyncio
    async def test_returns_stored_response(self):
        """Returns the Response object for a known response_id."""
        resp = _make_response("resp_known")
        _response_store["resp_known"] = _StoredResponse(response=resp, history=[])

        result = await app_module.get_response("resp_known")
        assert result.id == "resp_known"

    @pytest.mark.asyncio
    async def test_returns_404_for_unknown_id(self):
        """Returns a 404 JSONResponse for an unknown response_id."""
        from fastapi.responses import JSONResponse

        result = await app_module.get_response("resp_unknown")
        assert isinstance(result, JSONResponse)
        assert result.status_code == 404
