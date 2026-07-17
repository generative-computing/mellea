"""Per-backend multi-turn reasoning-replay policy tests (no live model).

These tests assert the *messages array actually sent to the provider* on a
follow-up turn, exercising issue #1201's consensus replay rule end-to-end
through each backend's conversation-construction path:

- reasoning is **stripped** on a plain follow-up turn, and
- **round-tripped** when the prior assistant turn issued a tool call
  (detected by that assistant message's own ``tool_calls`` field).

The provider ``create``/``acompletion`` call is mocked, so no endpoint is
required. A separate CI-skipped e2e test (``test_reasoning_replay_e2e.py``)
verifies the same behaviour against a real reasoning model.
"""

from unittest.mock import AsyncMock, patch

import pytest
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice

from mellea.backends import ModelOption
from mellea.backends.openai import OpenAIBackend
from mellea.core import CBlock
from mellea.stdlib.components import Message
from mellea.stdlib.context import ChatContext


def _fake_openai_response() -> ChatCompletion:
    """Minimal real ChatCompletion that survives processing()/post_processing()."""
    return ChatCompletion(
        id="test",
        choices=[
            Choice(
                finish_reason="stop",
                index=0,
                message=ChatCompletionMessage(role="assistant", content="next answer"),
            )
        ],
        created=0,
        model="qwen3",
        object="chat.completion",
    )


@pytest.fixture
def openai_backend() -> OpenAIBackend:
    """OpenAIBackend with a fake key — the client is never actually contacted."""
    return OpenAIBackend(
        model_id="qwen3", api_key="fake-key", base_url="http://localhost:9999/v1"
    )


async def _capture_messages(backend: OpenAIBackend, ctx: ChatContext) -> list[dict]:
    """Run one generation with the network mocked; return the sent messages array."""
    with patch.object(
        backend._async_client.chat.completions, "create", new_callable=AsyncMock
    ) as mock_create:
        mock_create.return_value = _fake_openai_response()
        mot, _ = await backend.generate_from_chat_context(
            CBlock(value="follow up"), ctx, model_options={ModelOption.STREAM: False}
        )
        await mot.avalue()
        assert mock_create.called, "create() was never called"
        return list(mock_create.call_args.kwargs["messages"])


def _assistant_messages(messages: list[dict]) -> list[dict]:
    return [m for m in messages if m.get("role") == "assistant"]


# A tool-issuing assistant turn is detected by its own `tool_calls` field (not a
# trailing `tool`-role message), matching `should_replay_reasoning`'s policy.
_TOOL_CALLS = [
    {"id": "call_1", "type": "function", "function": {"name": "fn", "arguments": "{}"}}
]


# ---------------------------------------------------------------------------
# OpenAI-compatible path (reasoning_content wire key)
# ---------------------------------------------------------------------------


async def test_openai_strips_reasoning_on_plain_turn(openai_backend: OpenAIBackend):
    """A plain prior assistant turn: reasoning must NOT be replayed."""
    ctx = (
        ChatContext()
        .add(Message("user", "first question"))
        .add(Message("assistant", "first answer", thinking="prior reasoning"))
    )
    messages = await _capture_messages(openai_backend, ctx)
    assistants = _assistant_messages(messages)
    assert assistants, "expected the prior assistant turn in the payload"
    assert all("reasoning_content" not in m for m in assistants), (
        "reasoning must be stripped on a plain follow-up turn"
    )


async def test_openai_round_trips_reasoning_after_tool_call(
    openai_backend: OpenAIBackend,
):
    """Prior assistant turn issued a tool call: reasoning must be replayed."""
    ctx = (
        ChatContext()
        .add(Message("user", "use a tool"))
        .add(
            Message(
                "assistant",
                "calling tool",
                thinking="tool-turn reasoning",
                tool_calls=_TOOL_CALLS,
            )
        )
        .add(Message("tool", "tool output"))
    )
    messages = await _capture_messages(openai_backend, ctx)
    tool_turn = next(
        m
        for m in _assistant_messages(messages)
        if m.get("reasoning_content") is not None
    )
    assert tool_turn["reasoning_content"] == "tool-turn reasoning"


async def test_openai_only_tool_turn_reasoning_replayed(openai_backend: OpenAIBackend):
    """In a mixed history only the tool-call assistant turn carries reasoning."""
    ctx = (
        ChatContext()
        .add(Message("user", "q1"))
        .add(Message("assistant", "plain answer", thinking="plain reasoning"))
        .add(Message("user", "q2"))
        .add(
            Message(
                "assistant",
                "calling tool",
                thinking="tool reasoning",
                tool_calls=_TOOL_CALLS,
            )
        )
        .add(Message("tool", "tool output"))
    )
    messages = await _capture_messages(openai_backend, ctx)
    replayed = [
        m["reasoning_content"]
        for m in _assistant_messages(messages)
        if "reasoning_content" in m
    ]
    assert replayed == ["tool reasoning"], (
        "exactly the tool-call assistant turn's reasoning should be replayed"
    )


async def test_openai_no_thinking_never_replayed(openai_backend: OpenAIBackend):
    """An assistant turn without reasoning never grows a reasoning_content key."""
    ctx = (
        ChatContext()
        .add(Message("user", "q"))
        .add(Message("assistant", "answer"))  # no thinking
        .add(Message("tool", "out"))
    )
    messages = await _capture_messages(openai_backend, ctx)
    assert all("reasoning_content" not in m for m in _assistant_messages(messages))


# ---------------------------------------------------------------------------
# End-to-end seam: a tool-issuing turn parsed by `_parse` must carry reasoning
# all the way to the wire. The per-turn tests above hand-build the assistant
# Message; this exercises the real capture -> _parse -> policy -> serialize
# path, guarding the gap where `_parse`'s tool-call branch dropped `thinking`
# and left the round-trip with nothing to replay.
# ---------------------------------------------------------------------------


async def test_openai_tool_turn_reasoning_survives_parse_to_wire(
    openai_backend: OpenAIBackend,
):
    """`_parse` of a tool-issuing MOT yields a Message whose reasoning is replayed."""
    from typing import cast

    from mellea.core import ModelOutputThunk, ModelToolCall, RawProviderResponse

    # Simulate what a backend produces on a tool-calling turn: tool_calls set and
    # reasoning captured on the MOT. `_parse` only checks `tool_calls is not None`,
    # never the value — the cast keeps the type checker happy without a real tool.
    mot = ModelOutputThunk(value="v", tool_calls={"fn": cast(ModelToolCall, None)})
    mot.thinking = "reasoning that led to the tool call"
    mot.raw = RawProviderResponse(
        provider="openai",
        response={
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "tool_calls": [{"function": {"name": "fn"}}],
                    }
                }
            ]
        },
    )
    assistant_msg = Message("user", "use a tool")._parse(mot)
    assert assistant_msg.thinking == "reasoning that led to the tool call", (
        "_parse must carry reasoning onto the tool-issuing assistant Message"
    )

    ctx = (
        ChatContext()
        .add(Message("user", "use a tool"))
        .add(assistant_msg)
        .add(Message("tool", "tool output"))
    )
    messages = await _capture_messages(openai_backend, ctx)
    tool_turn = next(
        m
        for m in _assistant_messages(messages)
        if m.get("reasoning_content") is not None
    )
    assert tool_turn["reasoning_content"] == "reasoning that led to the tool call"


# ---------------------------------------------------------------------------
# Ollama and WatsonX build the conversation dict INLINE (not via
# message_to_openai_message) and use DIVERGENT wire keys — Ollama's native SDK
# carries reasoning under `thinking`, WatsonX under `reasoning_content`. That
# per-backend divergence is the riskiest part of the change, so it gets direct
# coverage here. Both backends assemble `conversation` fully before calling the
# client, so a mock that records the `messages=` kwarg and short-circuits the
# rest of the async pipeline captures exactly what would go on the wire.
# ---------------------------------------------------------------------------


class _StopBeforeSend(Exception):
    """Raised by the capturing mock to skip the (unmocked) response machinery."""


def _plain_ctx() -> ChatContext:
    return (
        ChatContext()
        .add(Message("user", "first question"))
        .add(Message("assistant", "first answer", thinking="prior reasoning"))
    )


def _tool_ctx() -> ChatContext:
    return (
        ChatContext()
        .add(Message("user", "use a tool"))
        .add(
            Message(
                "assistant",
                "calling tool",
                thinking="tool-turn reasoning",
                tool_calls=_TOOL_CALLS,
            )
        )
        .add(Message("tool", "tool output"))
    )


async def _capture_ollama_conversation(ctx: ChatContext) -> list[dict]:
    """Return the messages array Ollama would send, without a live server."""
    from unittest.mock import MagicMock

    from mellea.backends.ollama import OllamaModelBackend

    captured: dict = {}

    def _record(*args, **kwargs):
        captured["messages"] = kwargs["messages"]
        raise _StopBeforeSend

    with (
        patch.object(OllamaModelBackend, "_check_ollama_server", return_value=True),
        patch.object(OllamaModelBackend, "_pull_ollama_model", return_value=True),
        patch("mellea.backends.ollama.ollama.Client", return_value=MagicMock()),
        patch("mellea.backends.ollama.ollama.AsyncClient", return_value=MagicMock()),
    ):
        backend = OllamaModelBackend(model_id="granite3.3:8b")
        with patch.object(backend._async_client, "chat", side_effect=_record):
            try:
                await backend.generate_from_chat_context(
                    CBlock(value="follow up"),
                    ctx,
                    model_options={ModelOption.STREAM: False},
                )
            except _StopBeforeSend:
                pass
    assert "messages" in captured, "Ollama never built the conversation"
    return captured["messages"]


async def test_ollama_strips_reasoning_on_plain_turn():
    """Ollama plain follow-up: no `thinking` key round-tripped."""
    conversation = await _capture_ollama_conversation(_plain_ctx())
    assistants = [m for m in conversation if m.get("role") == "assistant"]
    assert assistants, "expected the prior assistant turn in the payload"
    assert all("thinking" not in m for m in assistants)


async def test_ollama_round_trips_reasoning_after_tool_call():
    """Ollama tool turn: reasoning round-tripped under the native `thinking` key."""
    conversation = await _capture_ollama_conversation(_tool_ctx())
    tool_turn = next(m for m in conversation if m.get("thinking") is not None)
    assert tool_turn["thinking"] == "tool-turn reasoning"
    # And it must NOT be emitted under the OpenAI-compat key.
    assert all("reasoning_content" not in m for m in conversation)


async def _capture_watsonx_conversation(ctx: ChatContext) -> list[dict]:
    """Return the messages array WatsonX would send, without a live endpoint."""
    pytest.importorskip("ibm_watsonx_ai")
    from unittest.mock import MagicMock

    from mellea.backends.watsonx import WatsonxAIBackend

    captured: dict = {}

    def _record(*args, **kwargs):
        captured["messages"] = kwargs["messages"]
        raise _StopBeforeSend

    with (
        patch("mellea.backends.watsonx.Credentials", return_value=MagicMock()),
        patch("mellea.backends.watsonx.APIClient", return_value=MagicMock()),
        patch("mellea.backends.watsonx.ModelInference", return_value=MagicMock()),
    ):
        backend = WatsonxAIBackend(
            model_id="ibm/granite-3-8b-instruct",
            api_key="fake-key",
            base_url="https://example.invalid",
            project_id="fake-project",
        )
        with patch.object(backend._model, "achat", side_effect=_record):
            try:
                await backend.generate_from_chat_context(
                    CBlock(value="follow up"),
                    ctx,
                    model_options={ModelOption.STREAM: False},
                )
            except _StopBeforeSend:
                pass
    assert "messages" in captured, "WatsonX never built the conversation"
    return captured["messages"]


async def test_watsonx_strips_reasoning_on_plain_turn():
    """WatsonX plain follow-up: no `reasoning_content` round-tripped."""
    conversation = await _capture_watsonx_conversation(_plain_ctx())
    assistants = [m for m in conversation if m.get("role") == "assistant"]
    assert assistants, "expected the prior assistant turn in the payload"
    assert all("reasoning_content" not in m for m in assistants)


async def test_watsonx_round_trips_reasoning_after_tool_call():
    """WatsonX tool turn: reasoning round-tripped under `reasoning_content`."""
    conversation = await _capture_watsonx_conversation(_tool_ctx())
    tool_turn = next(m for m in conversation if m.get("reasoning_content") is not None)
    assert tool_turn["reasoning_content"] == "tool-turn reasoning"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
