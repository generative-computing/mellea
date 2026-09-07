# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the three silent-failure guards on token-id retention.

No server required: `_tokenize_chat` is mocked, so these exercise the guards
themselves rather than a transport.

Each guard closes a hole where the prompt actually sent diverges from the
conversation the caller described, with no error and no symptom beyond a fallen
prefix-cache hit rate:

* template kwargs that re-render the already-sent region (`documents=[...]`
  introduced mid-conversation) cancelling out of the delta;
* a preserved prefix growing past the count of control tokens the coded switch
  can address exactly;
* a turn terminator derived under different template kwargs than the turn it
  closes.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mellea.backends.openai import (
    MAX_RETAINED_CONTROL_TOKENS,
    DeltaNotDerivable,
    OpenAIBackend,
    _prompt_digest,
)
from mellea.stdlib.components import Message
from mellea.stdlib.context.chat import ChatContext


def _make_backend() -> OpenAIBackend:
    """Return an OpenAIBackend with a fake key, no server contacted."""
    return OpenAIBackend(
        model_id="gpt-4o", api_key="fake-key", base_url="http://localhost:9999/v1"
    )


def _ctx_holding(backend, ids, messages, template_kwargs=None):
    """A retaining context holding `ids` for `messages`."""
    return ChatContext(retain_token_ids=True).with_sent_token_ids(
        ids,
        model_id=backend._model_id,
        message_count=len(messages) + 1,
        prompt_digest=_prompt_digest(messages),
        template_kwargs=template_kwargs,
    )


# --- (a) template-kwargs drift ----------------------------------------------


def test_with_sent_token_ids_records_template_kwargs_without_adapter_name():
    """`adapter_name` is per-turn, so it is not part of the prefix's kwargs.

    Keeping it would make every adapter turn look like drift against the next one.
    """
    ctx = ChatContext(retain_token_ids=True).with_sent_token_ids(
        [1, 2],
        model_id="gpt-4o",
        message_count=2,
        template_kwargs={"adapter_name": "uncertainty", "enable_thinking": True},
    )
    assert ctx.sent_template_kwargs == {"enable_thinking": True}


def test_root_reset_clears_sent_template_kwargs():
    ctx = ChatContext(retain_token_ids=True).with_sent_token_ids(
        [1, 2], model_id="gpt-4o", message_count=2, template_kwargs={"a": 1}
    )
    assert ctx._make_root(None).sent_template_kwargs == {}


async def test_build_prompt_ids_renders_prev_under_the_kwargs_it_was_sent_with():
    """The already-sent side is re-rendered with the kwargs that produced it.

    Here that means the sent kwargs WITHOUT this turn's `adapter_name`: rendering the
    subtracted region with the adapter would put its control token inside the region
    that cancels out, so the token would reach neither side of the splice.
    """
    backend = _make_backend()
    sent = [{"role": "user", "content": "U1"}, {"role": "assistant", "content": "A1"}]
    ctx = _ctx_holding(
        backend, [10, 11], sent[:1], template_kwargs={"enable_thinking": True}
    )
    conversation = [*sent, {"role": "user", "content": "U2"}]

    prompt_renders: dict[bool, dict | None] = {}

    async def fake_tokenize(messages, *, add_generation_prompt=True, **kw):
        # Ignore the control-token learning probes (single-message conversations).
        if len(messages) > 1:
            prompt_renders[add_generation_prompt] = kw.get("chat_template_kwargs")
        return [1, 2, 3, 4, 40] if add_generation_prompt else [1, 2, 3, 4]

    backend._tokenize_chat = AsyncMock(side_effect=fake_tokenize)

    await backend._build_prompt_ids(
        ctx, conversation, {"enable_thinking": True, "adapter_name": "uncertainty"}
    )

    assert prompt_renders[False] == {"enable_thinking": True}
    assert prompt_renders[True] == {
        "enable_thinking": True,
        "adapter_name": "uncertainty",
    }


async def test_build_prompt_ids_refuses_when_template_kwargs_changed_mid_conversation():
    """A kwarg that re-renders history is refused by name, not silently dropped."""
    backend = _make_backend()
    sent = [{"role": "user", "content": "U1"}, {"role": "assistant", "content": "A1"}]
    ctx = _ctx_holding(backend, [10, 11], sent[:1], template_kwargs={})
    conversation = [*sent, {"role": "user", "content": "U2"}]
    backend._tokenize_chat = AsyncMock(return_value=[1, 2, 3])

    with pytest.raises(DeltaNotDerivable, match="enable_thinking"):
        await backend._build_prompt_ids(ctx, conversation, {"enable_thinking": True})


async def test_build_prompt_ids_ignores_adapter_name_when_comparing_kwargs():
    """An adapter on the new turn is not drift: its token belongs to the delta."""
    backend = _make_backend()
    sent = [{"role": "user", "content": "U1"}, {"role": "assistant", "content": "A1"}]
    ctx = _ctx_holding(backend, [10, 11], sent[:1], template_kwargs={})
    conversation = [*sent, {"role": "user", "content": "U2"}]

    async def fake_tokenize(messages, *, add_generation_prompt=True, **_):
        return [1, 2, 3, 4, 40] if add_generation_prompt else [1, 2, 3, 4]

    backend._tokenize_chat = AsyncMock(side_effect=fake_tokenize)
    result = await backend._build_prompt_ids(
        ctx, conversation, {"adapter_name": "uncertainty"}
    )
    assert result == [10, 11, 40]


# --- (b) control-token ceiling ----------------------------------------------


async def test_control_token_ids_learned_by_diffing_an_adapter_render():
    """The control token is found where an adapter render differs from a plain one.

    It SUBSTITUTES for the role marker rather than being inserted, so the two
    renders are the same length and only a positional diff finds it.
    """
    backend = _make_backend()

    async def fake_tokenize(messages, *, add_generation_prompt=True, **kw):
        kwargs = kw.get("chat_template_kwargs") or {}
        # Role marker 100264 replaced by control token 100356 at index 1.
        return [1, 100356, 2] if kwargs.get("adapter_name") else [1, 100264, 2]

    backend._tokenize_chat = AsyncMock(side_effect=fake_tokenize)
    assert await backend._control_token_ids("cti-technique-mapping") == [100356]


async def test_control_token_ids_are_cached_per_adapter():
    backend = _make_backend()

    async def fake_tokenize(messages, *, add_generation_prompt=True, **kw):
        kwargs = kw.get("chat_template_kwargs") or {}
        return [1, 100356, 2] if kwargs.get("adapter_name") else [1, 100264, 2]

    backend._tokenize_chat = AsyncMock(side_effect=fake_tokenize)
    await backend._control_token_ids("a")
    calls = backend._tokenize_chat.await_count
    await backend._control_token_ids("a")
    assert backend._tokenize_chat.await_count == calls


async def test_over_budget_prefix_is_rebaselined_instead_of_spliced():
    """Past the addressable count the prefix is dropped and the turn full-renders.

    Splicing on would slide the write addresses past the range the coded switch
    inverts exactly, where two control tokens key one codeword and the memory head
    returns the mean of two expert ids -- an arbitrary adapter, with no error.
    """
    backend = _make_backend()
    backend._control_token_id_set = {100356}
    sent = [{"role": "user", "content": "U1"}, {"role": "assistant", "content": "A1"}]
    over = [100356] * (MAX_RETAINED_CONTROL_TOKENS + 1)
    ctx = _ctx_holding(backend, over, sent[:1], template_kwargs={})
    conversation = [*sent, {"role": "user", "content": "U2"}]

    async def fake_tokenize(messages, *, add_generation_prompt=True, **_):
        return [1, 2, 3, 4, 40] if add_generation_prompt else [1, 2, 3, 4]

    backend._tokenize_chat = AsyncMock(side_effect=fake_tokenize)
    result = await backend._build_prompt_ids(ctx, conversation, None)

    assert result == [1, 2, 3, 4, 40], "expected a fresh full render, not a splice"
    assert backend.token_id_reprefills == 1


async def test_a_full_render_that_is_itself_over_budget_raises():
    """Re-prefilling cannot help when the count comes from the current turn."""
    backend = _make_backend()
    backend._control_token_id_set = {100356}
    conversation = [{"role": "user", "content": "U1"}]
    backend._tokenize_chat = AsyncMock(
        return_value=[100356] * (MAX_RETAINED_CONTROL_TOKENS + 1)
    )
    with pytest.raises(RuntimeError, match="control tokens"):
        await backend._build_prompt_ids(
            ctx=ChatContext(retain_token_ids=True),
            conversation=conversation,
            chat_template_kwargs=None,
        )


async def test_within_budget_prefix_is_still_spliced():
    backend = _make_backend()
    backend._control_token_id_set = {100356}
    sent = [{"role": "user", "content": "U1"}, {"role": "assistant", "content": "A1"}]
    ctx = _ctx_holding(backend, [100356, 11], sent[:1], template_kwargs={})
    conversation = [*sent, {"role": "user", "content": "U2"}]

    async def fake_tokenize(messages, *, add_generation_prompt=True, **_):
        return [1, 2, 3, 4, 40] if add_generation_prompt else [1, 2, 3, 4]

    backend._tokenize_chat = AsyncMock(side_effect=fake_tokenize)
    assert await backend._build_prompt_ids(ctx, conversation, None) == [100356, 11, 40]
    assert backend.token_id_reprefills == 0


# --- (c) turn terminator ----------------------------------------------------


async def test_turn_terminator_probes_with_the_turns_template_kwargs():
    """Granite 4.2 closes an assistant turn differently under `enable_thinking`."""
    backend = _make_backend()
    seen: list[dict | None] = []

    async def fake_tokenize(messages, *, add_generation_prompt=True, **kw):
        seen.append(kw.get("chat_template_kwargs"))
        return [1, 2] if add_generation_prompt else [1, 2, 77]

    backend._tokenize_chat = AsyncMock(side_effect=fake_tokenize)
    await backend._turn_terminator({"enable_thinking": True})
    assert all(k == {"enable_thinking": True} for k in seen), seen


async def test_turn_terminator_cached_per_template_kwargs():
    """Two kwarg sets get two terminators, not one cached under the first."""
    backend = _make_backend()

    async def fake_tokenize(messages, *, add_generation_prompt=True, **kw):
        thinking = (kw.get("chat_template_kwargs") or {}).get("enable_thinking")
        tail = [88] if thinking else [77]
        return [1, 2] if add_generation_prompt else [1, 2, *tail]

    backend._tokenize_chat = AsyncMock(side_effect=fake_tokenize)
    assert await backend._turn_terminator({"enable_thinking": False}) == [77]
    assert await backend._turn_terminator({"enable_thinking": True}) == [88]


async def test_turn_terminator_reuses_the_cache_for_identical_kwargs():
    backend = _make_backend()

    async def fake_tokenize(messages, *, add_generation_prompt=True, **_):
        return [1, 2] if add_generation_prompt else [1, 2, 77]

    backend._tokenize_chat = AsyncMock(side_effect=fake_tokenize)
    await backend._turn_terminator({"a": 1})
    calls = backend._tokenize_chat.await_count
    await backend._turn_terminator({"a": 1})
    assert backend._tokenize_chat.await_count == calls


async def test_terminator_cache_is_not_shared_between_backend_instances():
    """A per-class cache would leak one server's template onto another's."""
    a, b = _make_backend(), _make_backend()
    a._tokenize_chat = AsyncMock(
        side_effect=lambda m, *, add_generation_prompt=True, **_: (
            [1, 2] if add_generation_prompt else [1, 2, 77]
        )
    )
    await a._turn_terminator(None)
    assert b._turn_terminator_ids == {}


# --- (a) documents on the chat path -----------------------------------------


async def test_chat_path_declines_reuse_when_documents_are_supplied():
    """`documents` are not a `/tokenize` parameter, so ids would omit them entirely.

    The chat template renders them into the system block server-side. A pre-tokenized
    prompt therefore answers a RAG question against no context at all, with every other
    guard passing. The intrinsic path already declines for this reason; so must this one.
    """
    backend = _make_backend()
    sent = [{"role": "user", "content": "U1"}]
    ctx = _ctx_holding(backend, [10, 11], sent, template_kwargs={}).add(
        Message("user", "U1")
    )

    create = MagicMock(return_value=MagicMock(name="chat_request"))
    backend._async_client.chat.completions.create = create
    backend._tokenize_chat = AsyncMock(
        side_effect=AssertionError("documents must be refused before tokenizing")
    )
    backend._async_client.completions.create = AsyncMock(
        side_effect=AssertionError("the id path cannot carry documents")
    )

    with patch("mellea.backends.openai.send_to_queue", new=AsyncMock()):
        await backend._generate_from_context(
            Message("user", "U2"),
            ctx,
            model_options={"extra_body": {"documents": [{"text": "doc"}]}},
        )

    kwargs = create.call_args.kwargs
    assert "prompt" not in kwargs
    assert kwargs["extra_body"]["documents"] == [{"text": "doc"}]
