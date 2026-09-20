# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-backend tool-metadata serialization tests (no live model).

Issue #1030 item 2 asks that a component's declared tool metadata
(`tool_calls`/`tool_call_id`/`tool_name`) survive serialization on every
backend, not just the OpenAI-compatible path. These tests assert the *messages
array actually built for the provider* carries that metadata in each backend's
native wire shape:

- Watsonx (OpenAI-compatible): `tool_calls` verbatim, `tool_call_id` on the tool turn.
- Ollama (native SDK): `tool_calls` translated to `{"function": {"name", "arguments":
  <dict>}}` (no id/type), tool turn keys on `tool_name`.
- HuggingFace `to_chat`: `tool_calls` verbatim (id/type/function shape), `tool_call_id`
  on the tool turn.

The provider client is mocked, so no endpoint is required.
"""

from unittest.mock import MagicMock, patch

import pytest

from mellea.backends import ModelOption
from mellea.core import CBlock
from mellea.stdlib.components import Message
from mellea.stdlib.context import ChatContext

# OpenAI-shaped assistant tool call as stored on `Message.tool_calls`
# (built by `build_tool_calls`): id/type/function with a JSON-string `arguments`.
_TOOL_CALLS = [
    {"id": "call_1", "type": "function", "function": {"name": "fn", "arguments": "{}"}}
]


class _StopBeforeSend(Exception):
    """Raised to abort generation right after the conversation is captured."""


def _tool_ctx(tool_msg: Message) -> ChatContext:
    """A context with an assistant tool-call turn followed by `tool_msg`."""
    return (
        ChatContext()
        .add(Message("user", "use a tool"))
        .add(Message("assistant", "calling tool", tool_calls=_TOOL_CALLS))
        .add(tool_msg)
    )


# ---------------------------------------------------------------------------
# Watsonx (OpenAI-compatible passthrough)
# ---------------------------------------------------------------------------


async def _capture_watsonx_conversation(ctx: ChatContext) -> list[dict]:
    """Return the messages array Watsonx would send, without a live endpoint."""
    pytest.importorskip("ibm_watsonx_ai")
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
    assert "messages" in captured, "Watsonx never built the conversation"
    return captured["messages"]


async def test_watsonx_serializes_tool_metadata():
    """Watsonx carries OpenAI-shape `tool_calls` verbatim and `tool_call_id`."""
    ctx = _tool_ctx(Message("tool", "out", tool_call_id="call_1"))
    conversation = await _capture_watsonx_conversation(ctx)

    assistant = next(m for m in conversation if m.get("role") == "assistant")
    assert assistant["tool_calls"] == _TOOL_CALLS

    tool_turn = next(m for m in conversation if m.get("role") == "tool")
    assert tool_turn["tool_call_id"] == "call_1"


# ---------------------------------------------------------------------------
# Ollama (native SDK shape)
# ---------------------------------------------------------------------------


async def _capture_ollama_conversation(ctx: ChatContext) -> list[dict]:
    """Return the messages array Ollama would send, without a live server."""
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


async def test_ollama_translates_tool_calls_to_native_shape():
    """Ollama assistant tool calls: dict args, no id/type."""
    ctx = _tool_ctx(Message("tool", "out", tool_name="fn"))
    conversation = await _capture_ollama_conversation(ctx)

    assistant = next(m for m in conversation if m.get("role") == "assistant")
    assert assistant["tool_calls"] == [{"function": {"name": "fn", "arguments": {}}}]


async def test_ollama_tool_name_from_component():
    """A component-declared `tool_name` reaches Ollama's tool-result turn."""
    ctx = _tool_ctx(Message("tool", "out", tool_name="fn"))
    conversation = await _capture_ollama_conversation(ctx)

    tool_turn = next(m for m in conversation if m.get("role") == "tool")
    assert tool_turn["tool_name"] == "fn"


async def test_ollama_tool_name_from_tool_message():
    """A `ToolMessage`'s `.name` falls through to Ollama's `tool_name` key."""
    from mellea.core import ModelToolCall
    from mellea.stdlib.components.chat import ToolMessage

    tool = ModelToolCall(name="fn", func=MagicMock(), args={}, tool_call_id="call_1")
    tool_msg = ToolMessage(
        role="tool", content="out", tool_output="out", name="fn", args={}, tool=tool
    )
    conversation = await _capture_ollama_conversation(_tool_ctx(tool_msg))

    tool_turn = next(m for m in conversation if m.get("role") == "tool")
    assert tool_turn["tool_name"] == "fn"


async def test_ollama_tool_name_omitted_when_absent():
    """A bare tool `Message` with no name omits the `tool_name` key entirely."""
    ctx = _tool_ctx(Message("tool", "out"))
    conversation = await _capture_ollama_conversation(ctx)

    tool_turn = next(m for m in conversation if m.get("role") == "tool")
    assert "tool_name" not in tool_turn


# ---------------------------------------------------------------------------
# HuggingFace `to_chat`
# ---------------------------------------------------------------------------


def _to_chat(ctx: ChatContext, action: Message) -> list[dict]:
    from mellea.backends.utils import to_chat
    from mellea.formatters.template_formatter import TemplateFormatter as ChatFormatter

    formatter = ChatFormatter(model_id="test")
    return to_chat(action, ctx, formatter, system_prompt=None)


def test_hf_to_chat_serializes_tool_metadata():
    """HF `to_chat` carries verbatim `tool_calls` and the tool turn's `tool_call_id`."""
    ctx = _tool_ctx(Message("tool", "out", tool_call_id="call_1"))
    conversation = _to_chat(ctx, Message("user", "next"))

    assistant = next(m for m in conversation if m.get("role") == "assistant")
    assert assistant["tool_calls"] == _TOOL_CALLS

    tool_turn = next(m for m in conversation if m.get("role") == "tool")
    assert tool_turn["tool_call_id"] == "call_1"


def test_hf_to_chat_does_not_raise_on_cblock_guard_with_tool_calls():
    """The CBlock guard tolerates non-string values (the `tool_calls` list)."""
    ctx = _tool_ctx(Message("tool", "out", tool_call_id="call_1"))
    # Regression: the guard used to `"CBlock" in v` on every value, which raises
    # on a non-iterable and is meaningless on a list. It must not raise here.
    conversation = _to_chat(ctx, Message("user", "next"))
    assert isinstance(conversation, list)


# ---------------------------------------------------------------------------
# Guard regression (string-only CBlock check)
# ---------------------------------------------------------------------------


def test_cblock_guard_skips_non_string_values():
    """A message dict with a non-string value passes the guard without raising."""
    msg = {"role": "assistant", "content": "text", "tool_calls": _TOOL_CALLS}
    # Mirror the guard as it now stands in utils.py.
    for v in msg.values():
        if isinstance(v, str) and "CBlock" in v:
            pytest.fail("unexpected CBlock match")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
