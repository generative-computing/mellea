# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for token-id history preservation on OpenAIBackend.

No server required: the one network boundary, `_tokenize_chat`, is mocked so
these tests exercise the pure prefix-subtraction and prefix-integrity logic that
`ChatContext(retain_token_ids=True)` relies on. The point of these tests is
semantics -- that a retained prefix is spliced verbatim rather than re-derived,
and that a prefix which can no longer be proven to match the conversation is
refused rather than silently corrupted -- not merely token counts.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mellea.backends.openai import (
    DeltaNotDerivable,
    OpenAIBackend,
    TokenizeUnavailable,
    _prompt_digest,
    derive_delta,
)
from mellea.core.base import PreTokenizedCBlock
from mellea.stdlib.components import Message
from mellea.stdlib.context.chat import ChatContext


def _make_backend() -> OpenAIBackend:
    """Return an OpenAIBackend with a fake key, no server contacted."""
    return OpenAIBackend(
        model_id="gpt-4o", api_key="fake-key", base_url="http://localhost:9999/v1"
    )


# --- PreTokenizedCBlock ------------------------------------------------------


def test_pretokenized_block_has_no_string_value():
    block = PreTokenizedCBlock([1, 2, 3])
    assert block.value is None
    assert block.token_ids == [1, 2, 3]


def test_pretokenized_block_copies_ids_so_caller_mutation_is_isolated():
    ids = [1, 2, 3]
    block = PreTokenizedCBlock(ids)
    ids.append(4)
    assert block.token_ids == [1, 2, 3]


def test_pretokenized_block_rejects_empty():
    with pytest.raises(ValueError):
        PreTokenizedCBlock([])


def test_pretokenized_block_rejects_bool_disguised_as_int():
    with pytest.raises(TypeError):
        PreTokenizedCBlock([1, True, 3])


def test_pretokenized_block_rejects_non_int():
    with pytest.raises(TypeError):
        PreTokenizedCBlock([1, "2", 3])


# --- derive_delta ------------------------------------------------------------


def test_derive_delta_returns_suffix_added_by_new_turn():
    assert derive_delta([1, 2, 3], [1, 2, 3, 4, 5]) == [4, 5]


def test_derive_delta_empty_when_identical():
    assert derive_delta([1, 2, 3], [1, 2, 3]) == []


def test_derive_delta_whole_when_prev_empty():
    assert derive_delta([], [1, 2, 3]) == [1, 2, 3]


def test_derive_delta_raises_when_full_is_shorter():
    with pytest.raises(DeltaNotDerivable):
        derive_delta([1, 2, 3], [1, 2])


def test_derive_delta_raises_when_prefix_diverges():
    # A control token substituting for the role marker mid-prefix: same length,
    # divergence at the substituted position.
    with pytest.raises(DeltaNotDerivable):
        derive_delta([1, 2, 99, 4], [1, 2, 3, 4])


# --- prompt digest: prefix integrity fingerprint ----------------------------


def test_prompt_digest_is_stable_for_identical_messages():
    msgs = [{"role": "user", "content": "hi"}]
    assert _prompt_digest(msgs) == _prompt_digest([{"role": "user", "content": "hi"}])


def test_prompt_digest_differs_when_content_changes():
    a = _prompt_digest([{"role": "user", "content": "hi"}])
    b = _prompt_digest([{"role": "user", "content": "bye"}])
    assert a != b


def test_prompt_digest_is_per_message():
    two = _prompt_digest(
        [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}]
    )
    assert len(two) == 2


# --- ChatContext retained state plumbing ------------------------------------


def test_with_sent_token_ids_records_prompt_digest():
    digest = _prompt_digest([{"role": "user", "content": "U1"}])
    ctx = ChatContext(retain_token_ids=True).with_sent_token_ids(
        [10, 11], model_id="gpt-4o", message_count=2, prompt_digest=digest
    )
    assert ctx.sent_prompt_digest == digest


def test_retained_state_propagates_across_add():
    digest = _prompt_digest([{"role": "user", "content": "U1"}])
    ctx = ChatContext(retain_token_ids=True).with_sent_token_ids(
        [10, 11], model_id="gpt-4o", message_count=2, prompt_digest=digest
    )
    from mellea.stdlib.components import Message

    child = ctx.add(Message("user", "U2"))
    assert child.sent_prompt_digest == digest
    assert child.retains_token_ids is True


def test_root_reset_clears_prompt_digest_but_keeps_policy():
    digest = _prompt_digest([{"role": "user", "content": "U1"}])
    ctx = ChatContext(retain_token_ids=True).with_sent_token_ids(
        [10, 11], model_id="gpt-4o", message_count=2, prompt_digest=digest
    )
    root = ctx._make_root(model_id="gpt-4o")
    assert root.sent_prompt_digest == ()
    assert root.sent_token_ids == ()
    assert root.retains_token_ids is True


# --- _build_prompt_ids: prefix integrity guard ------------------------------


async def test_build_prompt_ids_splices_retained_prefix_verbatim():
    """The retained prefix is appended as-is; only the new turn is subtracted.

    Retained ids differ in length from a fresh re-render (that is the round-trip
    loss the policy exists for), so the delta must come from subtracting two
    fresh renders and be spliced onto the retained ids -- never compared to them.
    """
    backend = _make_backend()
    # The ids cover [U1, A1], so the digest fingerprints both: a digest that stopped
    # short of the count is refused (see the digest-coverage test below).
    prev_msgs = [
        {"role": "user", "content": "U1"},
        {"role": "assistant", "content": "A1"},
    ]
    # Retained prefix carries a control token (999) that a fresh render would lose.
    ctx = ChatContext(retain_token_ids=True).with_sent_token_ids(
        [10, 11, 999],
        model_id=backend._model_id,
        message_count=len(prev_msgs),
        prompt_digest=_prompt_digest(prev_msgs),
    )
    conversation = [
        {"role": "user", "content": "U1"},
        {"role": "assistant", "content": "A1"},
        {"role": "user", "content": "U2"},
    ]

    async def fake_tokenize(messages, *, add_generation_prompt=True, **_):
        # Fresh renders (canonical); shorter than the retained prefix on purpose.
        if len(messages) == 3:  # full conversation + generation prompt
            return [1, 2, 3, 4, 40]
        return [1, 2, 3, 4]  # already-sent side [U1, A1]

    backend._tokenize_chat = AsyncMock(side_effect=fake_tokenize)

    result = await backend._build_prompt_ids(ctx, conversation, None)

    # Retained prefix (with 999) preserved verbatim, new turn's [40] appended.
    assert result == [10, 11, 999, 40]


async def test_build_prompt_ids_refuses_front_dropped_prefix():
    """A prefix rewritten while message count stayed >= already must be refused.

    The token budget drops oldest-first; if new turns keep the count from
    shrinking below `already`, the count guard passes but the retained prefix no
    longer describes the leading messages. The digest catches it.
    """
    backend = _make_backend()
    sent_prev = [{"role": "user", "content": "U1"}]
    ctx = ChatContext(retain_token_ids=True).with_sent_token_ids(
        [10, 11, 12, 13],
        model_id=backend._model_id,
        message_count=2,  # covers [U1, A1]
        prompt_digest=_prompt_digest(sent_prev),
    )
    # U1/A1 dropped from the front, three newer messages appended: len 3 >= 2.
    conversation = [
        {"role": "user", "content": "U2"},
        {"role": "assistant", "content": "A2"},
        {"role": "user", "content": "U3"},
    ]
    backend._tokenize_chat = AsyncMock(side_effect=AssertionError("must not tokenize"))

    with pytest.raises(DeltaNotDerivable):
        await backend._build_prompt_ids(ctx, conversation, None)


async def test_build_prompt_ids_refuses_in_place_modified_history():
    """A historical message edited in place (count unchanged) must be refused."""
    backend = _make_backend()
    sent_prev = [{"role": "user", "content": "U1"}]
    ctx = ChatContext(retain_token_ids=True).with_sent_token_ids(
        [10, 11],
        model_id=backend._model_id,
        message_count=2,
        prompt_digest=_prompt_digest(sent_prev),
    )
    # Same count, same positions, but U1's content was rewritten.
    conversation = [
        {"role": "user", "content": "U1-EDITED"},
        {"role": "assistant", "content": "A1"},
        {"role": "user", "content": "U2"},
    ]
    backend._tokenize_chat = AsyncMock(side_effect=AssertionError("must not tokenize"))

    with pytest.raises(DeltaNotDerivable):
        await backend._build_prompt_ids(ctx, conversation, None)


async def test_build_prompt_ids_refuses_ids_from_a_different_model():
    backend = _make_backend()
    ctx = ChatContext(retain_token_ids=True).with_sent_token_ids(
        [10, 11],
        model_id="some-other-model",
        message_count=2,
        prompt_digest=_prompt_digest([{"role": "user", "content": "U1"}]),
    )
    conversation = [
        {"role": "user", "content": "U1"},
        {"role": "assistant", "content": "A1"},
        {"role": "user", "content": "U2"},
    ]
    backend._tokenize_chat = AsyncMock(side_effect=AssertionError("must not tokenize"))

    with pytest.raises(DeltaNotDerivable):
        await backend._build_prompt_ids(ctx, conversation, None)


async def test_build_prompt_ids_refuses_shrunk_history():
    backend = _make_backend()
    ctx = ChatContext(retain_token_ids=True).with_sent_token_ids(
        [10, 11, 12, 13],
        model_id=backend._model_id,
        message_count=5,  # claims more messages than the conversation now has
        prompt_digest=_prompt_digest(
            [{"role": "user", "content": f"m{i}"} for i in range(4)]
        ),
    )
    conversation = [
        {"role": "user", "content": "U1"},
        {"role": "user", "content": "U2"},
    ]
    backend._tokenize_chat = AsyncMock(side_effect=AssertionError("must not tokenize"))

    with pytest.raises(DeltaNotDerivable):
        await backend._build_prompt_ids(ctx, conversation, None)


# --- fallback: an unusable prefix degrades to the chat send, it does not fail --


def _unmatched_prefix_ctx(backend: OpenAIBackend) -> ChatContext:
    """Return a retaining context whose recorded digest cannot match any conversation.

    The digest is over a message the conversation does not begin with, so
    `_build_prompt_ids` refuses this prefix before it issues any request.
    """
    return ChatContext(retain_token_ids=True).with_sent_token_ids(
        [10, 11, 999],
        model_id=backend._model_id,
        message_count=1,
        prompt_digest=_prompt_digest([{"role": "user", "content": "NEVER-SENT"}]),
    )


async def test_unusable_prefix_falls_back_to_chat_send_instead_of_raising():
    """A prefix that cannot be extended sends the turn as messages, not as ids.

    The turn must still succeed: `retain_token_ids` is an optimization, so losing
    the prefix cache is not a reason to fail a generation. Asserted by the chat
    endpoint being the one that receives the request, and by the completions
    endpoint receiving nothing.
    """
    backend = _make_backend()
    ctx = _unmatched_prefix_ctx(backend).add(Message("user", "U1"))

    create = MagicMock(return_value=MagicMock(name="chat_request"))
    backend._async_client.chat.completions.create = create
    backend._async_client.completions.create = AsyncMock(
        side_effect=AssertionError("the id path must not be used for a refused prefix")
    )
    # Would be the second round trip of a successful splice; a refused prefix is
    # rejected before any tokenize call, so reaching this is a regression.
    backend._tokenize_chat = AsyncMock(
        side_effect=AssertionError(
            "a refused prefix must not pay a tokenize round trip"
        )
    )

    with patch("mellea.backends.openai.send_to_queue", new=AsyncMock()):
        output, _ = await backend._generate_from_context(
            Message("user", "U2"), ctx, model_options={}
        )

    assert create.call_count == 1
    assert output is not None


async def test_fallback_send_carries_messages_not_token_ids():
    """The fallback request is an ordinary chat render, with no pre-tokenized prompt."""
    backend = _make_backend()
    ctx = _unmatched_prefix_ctx(backend).add(Message("user", "U1"))

    create = MagicMock(return_value=MagicMock(name="chat_request"))
    backend._async_client.chat.completions.create = create
    backend._tokenize_chat = AsyncMock(side_effect=AssertionError("must not tokenize"))

    with patch("mellea.backends.openai.send_to_queue", new=AsyncMock()):
        await backend._generate_from_context(
            Message("user", "U2"), ctx, model_options={}
        )

    kwargs = create.call_args.kwargs
    assert "prompt" not in kwargs
    assert [m["content"] for m in kwargs["messages"]] == ["U1", "U2"]


async def test_fallback_leaves_retained_ids_in_place_for_a_later_turn():
    """Falling back must not clear the prefix: a later matching turn can reuse it.

    A one-off divergence (documents added for a single turn) would otherwise
    permanently forfeit the cache for the rest of the conversation.
    """
    backend = _make_backend()
    ctx = _unmatched_prefix_ctx(backend).add(Message("user", "U1"))
    backend._async_client.chat.completions.create = MagicMock(
        return_value=MagicMock(name="chat_request")
    )
    backend._tokenize_chat = AsyncMock(side_effect=AssertionError("must not tokenize"))

    with patch("mellea.backends.openai.send_to_queue", new=AsyncMock()):
        _, new_ctx = await backend._generate_from_context(
            Message("user", "U2"), ctx, model_options={}
        )

    assert new_ctx.sent_token_ids == (10, 11, 999)
    assert new_ctx.retains_token_ids is True


async def test_tokenize_unavailable_is_not_swallowed_by_the_fallback():
    """A server with no /tokenize route must surface, not silently disable the policy."""
    backend = _make_backend()
    ctx = ChatContext(retain_token_ids=True).add(Message("user", "U1"))
    backend._tokenize_chat = AsyncMock(
        side_effect=TokenizeUnavailable("no /tokenize route")
    )
    backend._async_client.chat.completions.create = MagicMock(
        side_effect=AssertionError("must not fall back on TokenizeUnavailable")
    )

    with pytest.raises(TokenizeUnavailable):
        await backend._generate_from_context(
            Message("user", "U2"), ctx, model_options={}
        )


async def test_build_prompt_ids_refuses_a_digest_that_does_not_cover_the_count():
    """A non-empty digest shorter than the count proves less than the count claims.

    `sent_message_count` says which messages the ids cover; the digest is the proof they
    are unchanged. When the proof stops short, the uncovered tail messages -- the newest,
    most likely to be edited -- would be reused on trust alone. Refused before any round
    trip rather than verified only as far as the proof reaches.
    """
    backend = _make_backend()
    ctx = ChatContext(retain_token_ids=True).with_sent_token_ids(
        [10, 11],
        model_id=backend._model_id,
        message_count=2,  # claims [U1, A1]
        prompt_digest=_prompt_digest(
            [{"role": "user", "content": "U1"}]
        ),  # proves [U1]
    )
    conversation = [
        {"role": "user", "content": "U1"},
        {"role": "assistant", "content": "A1-EDITED"},
        {"role": "user", "content": "U2"},
    ]
    backend._tokenize_chat = AsyncMock(side_effect=AssertionError("must not tokenize"))

    with pytest.raises(DeltaNotDerivable):
        await backend._build_prompt_ids(ctx, conversation, None)


# --- /tokenize reply validation ----------------------------------------------
#
# These ids become the prompt verbatim, so the reader must reject anything it cannot
# trust rather than coerce it: a coerced id is a DIFFERENT prompt, and since both sides
# of `derive_delta` come through this same reader a consistent corruption cancels out
# and every downstream guard still passes.


def _backend_with_tokenize_payload(payload: object) -> OpenAIBackend:
    """A backend whose `/tokenize` route replies with `payload`."""
    backend = _make_backend()
    response = MagicMock()
    response.json = MagicMock(return_value=payload)
    # setattr so mypy does not flag mock-over-method assignment.
    setattr(backend._async_client, "post", AsyncMock(return_value=response))
    return backend


async def test_tokenize_chat_returns_ids_for_a_well_formed_reply():
    backend = _backend_with_tokenize_payload({"tokens": [1, 2, 3]})
    assert await backend._tokenize_chat([{"role": "user", "content": "U1"}]) == [
        1,
        2,
        3,
    ]


@pytest.mark.parametrize(
    ("tokens", "why"),
    [
        ([1, True, 3], "bool coerces to 1 even though it is not a vocabulary id"),
        ([1, 2.9, 3], "float would be silently truncated to a different id"),
        (["12", "13"], "numeric strings are not ids the server assigned"),
        (["abc"], "unparsable entry raised a bare ValueError, not TokenizeUnavailable"),
        ([None], "null entry"),
        ([[1], 2], "nested list"),
    ],
)
async def test_tokenize_chat_refuses_entries_that_are_not_exact_ids(tokens, why):
    """Any non-int entry fails as `TokenizeUnavailable`, the one documented type.

    Callers catch `TokenizeUnavailable` to fall back to a chat send, so a reply the
    reader cannot trust must cost a cache hit -- not the turn (a bare `ValueError`
    escapes the fallback) and not the prompt's integrity (a coerced id would be sent).
    """
    backend = _backend_with_tokenize_payload({"tokens": tokens})
    with pytest.raises(TokenizeUnavailable):
        await backend._tokenize_chat([{"role": "user", "content": "U1"}])


# --- digest coverage: every field that reaches the wire ----------------------
#
# The fingerprint's contract is "if the message changed, the digest changes". It is
# compared against messages about to be sent, so anything the SERVER sees must be in
# it -- an unfingerprinted field that the chat template renders is a prompt change the
# guard cannot see, which is the whole failure class this policy exists to prevent.


def test_prompt_digest_notices_a_changed_reasoning_trace():
    """`reasoning_content` reaches the wire, so it must be fingerprinted.

    The chat path emits it for a turn that carried tool calls (`should_replay_reasoning`),
    and the chat template renders it. Two messages differing only there are two different
    prompts, so they must not share a digest.
    """
    call = [
        {"id": "c1", "type": "function", "function": {"name": "f", "arguments": "{}"}}
    ]
    a = {
        "role": "assistant",
        "content": "Blue.",
        "tool_calls": call,
        "reasoning_content": "I will call the tool.",
    }
    b = dict(a, reasoning_content="Actually I will not call it.")
    assert _prompt_digest([a]) != _prompt_digest([b])


def test_prompt_digest_notices_a_changed_tool_call_id():
    """`tool_call_id` reaches the wire on a tool-result turn, so it must be fingerprinted."""
    a = {"role": "tool", "content": "42", "tool_call_id": "call_AAA"}
    b = dict(a, tool_call_id="call_ZZZ")
    assert _prompt_digest([a]) != _prompt_digest([b])


def test_prompt_digest_ignores_absent_versus_null_keys():
    """A key left out and the same key set to `None` are the same message.

    Path independence: the chat serializer omits keys it has no value for, while the
    intrinsic path's `model_dump()` can carry them as `None`. Fingerprinting that
    difference would refuse a valid `Chat -> Intrinsic` reuse.
    """
    lean = {"role": "user", "content": "U1"}
    padded = {
        "role": "user",
        "content": "U1",
        "tool_calls": None,
        "reasoning_content": "",
    }
    assert _prompt_digest([lean]) == _prompt_digest([padded])


async def test_build_prompt_ids_refuses_a_digest_longer_than_the_count():
    """A digest claiming MORE messages than the ids cover is refused, not worked around.

    Comparing over the digest's own length would verify messages beyond the retained
    boundary, so an edit to a message the ids never covered would refuse reuse. The two
    facts must agree exactly; a mismatch means the retained state was not recorded by
    this backend and cannot be trusted either way.
    """
    backend = _make_backend()
    msgs = [
        {"role": "user", "content": "U1"},
        {"role": "assistant", "content": "A1"},
        {"role": "user", "content": "U2"},
    ]
    ctx = ChatContext(retain_token_ids=True).with_sent_token_ids(
        [10, 11],
        model_id=backend._model_id,
        message_count=2,  # ids cover [U1, A1]
        prompt_digest=_prompt_digest(msgs),  # but three fingerprints
    )
    backend._tokenize_chat = AsyncMock(side_effect=AssertionError("must not tokenize"))

    with pytest.raises(DeltaNotDerivable):
        await backend._build_prompt_ids(ctx, msgs, None)
