# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the in-memory response store and previous_response_id session chaining."""

import asyncio
import time
from typing import Literal
from unittest.mock import AsyncMock, MagicMock

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

    def test_tool_call_id_preserved(self):
        """A tool message's tool_call_id survives the round trip."""
        tool_msg = _make_chat_message("tool", "Sunny")
        tool_msg.tool_call_id = "call_1"
        history = _build_history_from_messages(
            [_make_chat_message("user", "q"), tool_msg], "answer"
        )
        assert history[1] == {
            "role": "tool",
            "content": "Sunny",
            "tool_call_id": "call_1",
        }

    def test_assistant_tool_calls_stored(self):
        """The assistant reply's tool_calls are stored on its history entry."""
        tool_calls = [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "get_weather", "arguments": "{}"},
            }
        ]
        history = _build_history_from_messages(
            [_make_chat_message("user", "q")], "", tool_calls=tool_calls
        )
        assert history[-1] == {
            "role": "assistant",
            "content": "",
            "tool_calls": tool_calls,
        }

    def test_mid_conversation_tool_calls_preserved(self):
        """An assistant message with tool_calls inside the message list keeps
        them, so a replayed history stays pairable."""
        assistant_msg = _make_chat_message("assistant", "")
        assistant_msg.tool_calls = [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "get_weather", "arguments": "{}"},
            }
        ]
        history = _build_history_from_messages(
            [_make_chat_message("user", "q"), assistant_msg], "answer"
        )
        assert history[1]["tool_calls"] == assistant_msg.tool_calls

    def test_plain_turns_stay_two_key_dicts(self):
        """Keys are only added when set, so plain turns keep their shape."""
        msgs = [_make_chat_message("user", "Hi")]
        history = _build_history_from_messages(msgs, "Hello!")
        assert history[0] == {"role": "user", "content": "Hi"}
        assert history[1] == {"role": "assistant", "content": "Hello!"}


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

    async def _run_one_sweep(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Run the real coroutine through one sweep, then cancel it.

        The first sleep returns so one real sweep executes; the second raises
        CancelledError, which is what task.cancel() does to the sleeping loop.
        """
        sleep_mock = AsyncMock(side_effect=[None, asyncio.CancelledError()])
        monkeypatch.setattr(app_module.asyncio, "sleep", sleep_mock)
        with pytest.raises(asyncio.CancelledError):
            await app_module._evict_expired_responses()

    async def test_evicts_expired_entry(self, monkeypatch):
        _response_store.clear()
        _response_store["old"] = _StoredResponse(
            response=_make_response("old"), history=[], stored_at=0.0
        )
        app_module._response_ttl_seconds = 1800

        await self._run_one_sweep(monkeypatch)

        assert "old" not in _response_store

    async def test_keeps_fresh_entry(self, monkeypatch):
        _response_store.clear()
        _response_store["fresh"] = _StoredResponse(
            response=_make_response("fresh"), history=[]
        )
        app_module._response_ttl_seconds = 1800

        await self._run_one_sweep(monkeypatch)

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
    async def test_expires_at_set_when_store_true(self):
        """expires_at is created_at + TTL when the response is stored."""
        module = _make_mock_module("answer")
        endpoint = app_module.make_responses_endpoint(module)
        app_module._response_ttl_seconds = 1800

        before = int(time.time())
        request = ResponseRequest(model="m", input="hello", store=True)
        response = await endpoint(request)
        after = int(time.time())

        assert response.expires_at is not None
        assert before + 1800 <= response.expires_at <= after + 1800

    @pytest.mark.asyncio
    async def test_expires_at_none_when_store_false(self):
        """expires_at is None when store=False because the response is never persisted."""
        module = _make_mock_module("ephemeral")
        endpoint = app_module.make_responses_endpoint(module)

        request = ResponseRequest(model="m", input="hi", store=False)
        response = await endpoint(request)

        assert response.expires_at is None

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

    @pytest.mark.asyncio
    async def test_instructions_excluded_from_stored_history(self):
        """Instructions are not persisted, so they do not carry over to the next turn."""
        module = _make_mock_module("answer")
        endpoint = app_module.make_responses_endpoint(module)

        request = ResponseRequest(
            model="m", input="a question", instructions="Be concise.", store=True
        )
        response = await endpoint(request)

        history = _response_store[response.id].history
        assert history == [
            {"role": "user", "content": "a question"},
            {"role": "assistant", "content": "answer"},
        ]

    @pytest.mark.asyncio
    async def test_instructions_not_carried_over_across_turns(self):
        """A previous turn's instructions are not re-applied on a chained turn."""
        module = _make_mock_module()
        endpoint = app_module.make_responses_endpoint(module)

        # Turn 1: with instructions.
        request1 = ResponseRequest(
            model="m", input="Q1", instructions="Be concise.", store=True
        )
        response1 = await endpoint(request1)
        assert {"role": "system", "content": "Be concise."} not in _response_store[
            response1.id
        ].history

        # Turn 2: chain via previous_response_id, no new instructions.
        request2 = ResponseRequest(
            model="m", input="Q2", previous_response_id=response1.id
        )
        await endpoint(request2)

        call_messages = module.serve.call_args.kwargs["input"]
        # No system message: the prior instructions were not stored, and none
        # were supplied this turn.
        assert all(m.role != "system" for m in call_messages)
        assert [m.content for m in call_messages] == ["Q1", "the answer", "Q2"]

    @pytest.mark.asyncio
    async def test_new_instructions_lead_context_on_continuation(self):
        """New instructions on a chained turn lead the context; the old ones are dropped."""
        module = _make_mock_module()
        endpoint = app_module.make_responses_endpoint(module)

        # Turn 1: with instructions.
        request1 = ResponseRequest(
            model="m", input="Q1", instructions="Be concise.", store=True
        )
        response1 = await endpoint(request1)

        # Turn 2: chain with NEW, different instructions.
        request2 = ResponseRequest(
            model="m",
            input="Q2",
            instructions="Be verbose.",
            previous_response_id=response1.id,
        )
        await endpoint(request2)

        call_messages = module.serve.call_args.kwargs["input"]
        # The new instructions lead the context.
        assert call_messages[0].role == "system"
        assert call_messages[0].content == "Be verbose."
        # The prior turn's instructions must not appear anywhere.
        assert all(m.content != "Be concise." for m in call_messages)
        # Prior conversation + new input follow the new instructions.
        assert [m.content for m in call_messages[1:]] == ["Q1", "the answer", "Q2"]


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

    @pytest.mark.asyncio
    async def test_returns_404_after_expiry(self):
        """An entry still in the store but past its TTL is not served."""
        from fastapi.responses import JSONResponse

        _response_store["resp_old"] = _StoredResponse(
            response=_make_response("resp_old"), history=[], stored_at=0.0
        )
        app_module._response_ttl_seconds = 1800

        result = await app_module.get_response("resp_old")
        assert isinstance(result, JSONResponse)
        assert result.status_code == 404

    @pytest.mark.asyncio
    async def test_returns_stored_response_with_future_expires_at(self):
        """An explicit future expires_at keeps the entry servable."""
        resp = _make_response("resp_future")
        resp.expires_at = int(time.time()) + 3600
        _response_store["resp_future"] = _StoredResponse(
            response=resp, history=[], stored_at=0.0
        )

        result = await app_module.get_response("resp_future")
        assert result.id == "resp_future"


class TestStoreCap:
    """The store drops the oldest entries once it exceeds its cap."""

    def setup_method(self):
        _response_store.clear()
        self._original_cap = app_module._max_stored_responses

    def teardown_method(self):
        _response_store.clear()
        app_module._max_stored_responses = self._original_cap

    def test_enforce_store_cap_evicts_oldest(self):
        app_module._max_stored_responses = 2
        for i, rid in enumerate(["resp_a", "resp_b", "resp_c"]):
            _response_store[rid] = _StoredResponse(
                response=_make_response(rid), history=[], stored_at=float(i)
            )

        app_module._enforce_store_cap()

        assert "resp_a" not in _response_store  # oldest evicted
        assert set(_response_store) == {"resp_b", "resp_c"}

    def test_enforce_store_cap_noop_at_or_below_cap(self):
        app_module._max_stored_responses = 5
        _response_store["resp_a"] = _StoredResponse(
            response=_make_response("resp_a"), history=[]
        )

        app_module._enforce_store_cap()

        assert "resp_a" in _response_store

    @pytest.mark.asyncio
    async def test_cap_enforced_on_store_write(self):
        """Storing past the cap evicts the oldest entry on the write path."""
        app_module._max_stored_responses = 1
        module = _make_mock_module("first")
        endpoint = app_module.make_responses_endpoint(module)

        first = await endpoint(ResponseRequest(model="m", input="one"))
        second = await endpoint(ResponseRequest(model="m", input="two"))

        assert first.id not in _response_store
        assert second.id in _response_store
