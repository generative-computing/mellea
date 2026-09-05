# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for issue #1597: empty user content must not silently reach the model.

`SimpleContext.view_for_generation()` returns `[]` by design (stateless context).
A caller can still `.add()` a user message and chain that to a generation; if the
action they pass is empty/whitespace, the assembled conversation sent to the
model contains no user-role content at all. The OpenAI/LiteLLM/Ollama chat
backends used to ship this empty payload, which makes some models (e.g. Granite
4.2, see issue #1587) spin on an empty prompt and burn tokens.

The expected behaviour is a `ValueError` raised before any HTTP/SDK request is
issued. These tests pin that contract for the three concrete backends covered
by the fix; WatsonxAIBackend and LocalHFBackend are tracked as remaining work
in the commit message.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch

import pytest
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice


def _ok_chat_completion(model: str = "gpt-4o") -> ChatCompletion:
    return ChatCompletion(
        id="test",
        choices=[
            Choice(
                finish_reason="stop",
                index=0,
                message=ChatCompletionMessage(role="assistant", content="ok"),
            )
        ],
        created=0,
        model=model,
        object="chat.completion",
    )


def _ok_litellm_response():
    """Return a non-streaming litellm ModelResponse that survives post_processing."""
    pytest.importorskip("litellm", reason="litellm not installed")
    from litellm.types.utils import Choices, Message, ModelResponse

    msg = Message(content="ok", role="assistant")
    choice = Choices(finish_reason="stop", index=0, message=msg)
    return ModelResponse(
        id="test",
        choices=[choice],
        created=0,
        model="hosted_vllm/qwen3",
        object="chat.completion",
        usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    )


def _ok_ollama_response(content: str = "ok") -> "ollama.ChatResponse":
    import ollama

    return ollama.ChatResponse(
        model="granite4.2:3b",
        created_at=None,
        message=ollama.Message(role="assistant", content=content),
        done=True,
    )


def _make_openai_backend():
    from mellea.backends.openai import OpenAIBackend

    return OpenAIBackend(
        model_id="gpt-4o",
        api_key="test-key",
        base_url="http://localhost:9999/v1",
    )


def _make_litellm_backend():
    pytest.importorskip("litellm", reason="litellm not installed")
    from mellea.backends.litellm import LiteLLMBackend

    return LiteLLMBackend(
        model_id="hosted_vllm/qwen3",
        base_url="http://localhost:9997",
    )


def _make_ollama_backend():
    from mellea.backends.ollama import OllamaModelBackend

    with (
        patch.object(OllamaModelBackend, "_check_ollama_server", return_value=True),
        patch.object(OllamaModelBackend, "_pull_ollama_model", return_value=True),
        patch("mellea.backends.ollama.ollama.Client", return_value=MagicMock()),
        patch("mellea.backends.ollama.ollama.AsyncClient", return_value=MagicMock()),
    ):
        return OllamaModelBackend(
            model_id="granite4.2:3b",
            model_options=None,
        )


# ---------------------------------------------------------------------------
# OpenAI backend
# ---------------------------------------------------------------------------


async def test_simple_context_with_empty_action_raises_on_openai():
    """OpenAI: SimpleContext + empty CBlock must raise before HTTP request."""
    from mellea.core import CBlock
    from mellea.stdlib.context import SimpleContext

    backend = _make_openai_backend()
    ctx = SimpleContext().add(CBlock("recorded-only"))

    with patch.object(
        backend._async_client.chat.completions, "create", new_callable=AsyncMock
    ) as mock_create:
        mock_create.return_value = _ok_chat_completion()

        with pytest.raises(ValueError, match="user"):
            await backend.generate_from_chat_context(CBlock(value=""), ctx)

    assert not mock_create.called, (
        "Empty-user-content request must be rejected before any HTTP call "
        "is issued. See issue #1597."
    )


async def test_simple_context_with_whitespace_action_raises_on_openai():
    """OpenAI: whitespace-only user content must also raise (issue #1587)."""
    from mellea.core import CBlock
    from mellea.stdlib.context import SimpleContext

    backend = _make_openai_backend()
    ctx = SimpleContext()

    with patch.object(
        backend._async_client.chat.completions, "create", new_callable=AsyncMock
    ) as mock_create:
        mock_create.return_value = _ok_chat_completion()

        with pytest.raises(ValueError, match="user"):
            await backend.generate_from_chat_context(CBlock(value="   \n\t"), ctx)

    assert not mock_create.called


async def test_nonempty_user_content_still_sends_request_on_openai():
    """Sanity check: a non-empty user action must still hit the network."""
    from mellea.core import CBlock
    from mellea.stdlib.context import SimpleContext

    backend = _make_openai_backend()
    ctx = SimpleContext()

    with patch.object(
        backend._async_client.chat.completions, "create", new_callable=AsyncMock
    ) as mock_create:
        mock_create.return_value = _ok_chat_completion()

        mot, _ = await backend.generate_from_chat_context(
            CBlock(value="real question"), ctx
        )
        await mot.avalue()

    assert mock_create.called, "Real user content must reach the model."


async def test_user_message_with_image_passes_guard_on_openai():
    """P3 passthrough: a user message with empty text but images must NOT raise.

    Vision/RAG workflows rely on attaching image or document blocks to a
    message whose text content is short (e.g. "Caption this:"). The guard
    must accept these as valid user-role content.
    """
    from mellea.core import CBlock, ImageBlock
    from mellea.stdlib.components import Message as MelleaMessage
    from mellea.stdlib.context import SimpleContext

    backend = _make_openai_backend()
    # SimpleContext discards prior turns, but the image-bearing action is the
    # one that drives the request.
    ctx = SimpleContext().add(CBlock("recorded-only"))
    # 1x1 transparent PNG, base64-encoded as a string.
    image = ImageBlock(
        value="iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="
    )
    action = MelleaMessage("user", "", images=[image])

    with patch.object(
        backend._async_client.chat.completions, "create", new_callable=AsyncMock
    ) as mock_create:
        mock_create.return_value = _ok_chat_completion()

        mot, _ = await backend.generate_from_chat_context(action, ctx)
        await mot.avalue()

    assert mock_create.called, (
        "User message with images must reach the model even when its text "
        "content is empty."
    )


# ---------------------------------------------------------------------------
# LiteLLM backend
# ---------------------------------------------------------------------------


async def test_simple_context_with_empty_action_raises_on_litellm():
    """LiteLLM: SimpleContext + empty CBlock must raise before litellm.acompletion."""
    pytest.importorskip("litellm", reason="litellm not installed")
    from mellea.core import CBlock
    from mellea.stdlib.context import SimpleContext

    backend = _make_litellm_backend()
    ctx = SimpleContext().add(CBlock("recorded-only"))

    with patch("litellm.acompletion", new_callable=AsyncMock) as mock_acomplete:
        mock_acomplete.return_value = _ok_litellm_response()

        with pytest.raises(ValueError, match="user"):
            await backend.generate_from_context(CBlock(value=""), ctx)

    assert not mock_acomplete.called


async def test_nonempty_user_content_still_sends_request_on_litellm():
    """LiteLLM counterpart to the OpenAI sanity check above."""
    pytest.importorskip("litellm", reason="litellm not installed")
    from mellea.core import CBlock
    from mellea.stdlib.context import SimpleContext

    backend = _make_litellm_backend()
    ctx = SimpleContext()

    with patch("litellm.acompletion", new_callable=AsyncMock) as mock_acomplete:
        mock_acomplete.return_value = _ok_litellm_response()

        mot, _ = await backend.generate_from_context(
            CBlock(value="real question"), ctx
        )
        await mot.avalue()

    assert mock_acomplete.called


# ---------------------------------------------------------------------------
# Ollama backend (issue #1587's real culprit)
# ---------------------------------------------------------------------------


def _patch_ollama_chat(backend, canned):
    """Return `(context_manager, mock_chat_call)` for `_async_client.chat`.

    `_async_client` is an event-loop-keyed property, so we patch it at the
    class level. The returned mock object's `.chat` is the AsyncMock
    asserting whether the request was issued.
    """
    mock_async = MagicMock()
    mock_chat = AsyncMock(return_value=canned)
    mock_async.chat = mock_chat
    cm = patch.object(
        type(backend),
        "_async_client",
        new_callable=PropertyMock,
        return_value=mock_async,
    )
    return cm, mock_chat


async def test_simple_context_with_empty_action_raises_on_ollama():
    """Ollama: SimpleContext + empty CBlock must raise before client.chat.

    Regression for the Granite 4.2 case in #1587. Without the guard the
    empty conversation reaches the local Ollama server, which burns tokens
    while the model spins on an empty prompt.
    """
    from mellea.core import CBlock
    from mellea.stdlib.context import SimpleContext

    backend = _make_ollama_backend()
    ctx = SimpleContext().add(CBlock("recorded-only"))

    cm, mock_chat = _patch_ollama_chat(backend, _ok_ollama_response())
    with cm:
        with pytest.raises(ValueError, match="user"):
            await backend.generate_from_chat_context(CBlock(value=""), ctx)

    assert not mock_chat.called, (
        "Empty-user-content request must be rejected before any SDK call "
        "is issued. See issue #1597."
    )


async def test_simple_context_with_whitespace_action_raises_on_ollama():
    """Ollama: whitespace-only user content must also raise."""
    from mellea.core import CBlock
    from mellea.stdlib.context import SimpleContext

    backend = _make_ollama_backend()
    ctx = SimpleContext()

    cm, mock_chat = _patch_ollama_chat(backend, _ok_ollama_response())
    with cm:
        with pytest.raises(ValueError, match="user"):
            await backend.generate_from_chat_context(CBlock(value="   \n\t"), ctx)

    assert not mock_chat.called


async def test_nonempty_user_content_still_sends_request_on_ollama():
    """Ollama sanity check: non-empty user action reaches client.chat."""
    from mellea.core import CBlock
    from mellea.stdlib.context import SimpleContext

    backend = _make_ollama_backend()
    ctx = SimpleContext()

    cm, mock_chat = _patch_ollama_chat(backend, _ok_ollama_response())
    with cm:
        mot = await backend.generate_from_chat_context(
            CBlock(value="real question"), ctx
        )
        await mot.avalue()

    assert mock_chat.called


async def test_ollama_guard_does_not_trigger_when_ctx_has_real_user_message():
    """Guardian-style regression: a chat context with real history is left alone.

    Mirrors the call pattern in `mellea/stdlib/components/guardian.py`, where
    the guardian calls the LLM with the *current* chat context (which
    already contains real user turns) plus an empty/instructional action.
    The guard must NOT fire here — the conversation has a non-empty user
    message even if the action itself is empty.
    """
    from mellea.core import CBlock
    from mellea.stdlib.components import Message as MelleaMessage
    from mellea.stdlib.context import ChatContext

    backend = _make_ollama_backend()
    gctx = ChatContext()
    gctx = gctx.add(MelleaMessage("user", "Is this code safe?"))
    gctx = gctx.add(MelleaMessage("assistant", "It looks fine."))
    # The "action" is an empty CBlock (a structural placeholder), but the
    # existing context already carries a real user turn, so the guard must
    # NOT fire.

    cm, mock_chat = _patch_ollama_chat(backend, _ok_ollama_response("looks safe"))
    with cm:
        mot = await backend.generate_from_chat_context(CBlock(value=""), gctx)
        await mot.avalue()

    assert mock_chat.called, (
        "ChatContext with real user history must reach the model even when "
        "the action is empty — see guardian-style usage pattern."
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
