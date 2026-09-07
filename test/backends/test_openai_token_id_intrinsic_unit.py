# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for token-id prefix REUSE across request paths — no server required.

Companion to `test_openai_token_id_history_unit.py`, which covers the `Chat -> Chat`
splice in isolation. This file covers the transitions the retain-ids policy has to hold
across once intrinsics are involved:

    Chat -> Intrinsic       (append-only rewrite reuses the exact prefix)
    Chat -> Intrinsic       (prefix-changing rewrite must NOT reuse)
    Intrinsic -> Chat       (an intrinsic turn does not corrupt the retained prefix)
    Intrinsic -> Intrinsic  (a second intrinsic reuses a genuinely-shared prefix)

Every reuse assertion is over the EXACT token ids the request would carry, mocking the
one network boundary (`_tokenize_chat`) so the pure splice/gate logic is what is under
test -- not message counts, and not `ChatContext` contents. The invariant: a genuinely
unchanged model-input prefix keeps its exact previously-sent ids (control tokens and
non-canonical BPE splits intact); a genuinely changed prefix is refused and re-rendered.
"""

from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch

from mellea.backends.openai import OpenAIBackend, _prompt_digest
from mellea.stdlib.context.chat import ChatContext


def _backend(model_id: str = "granite-switch") -> OpenAIBackend:
    """Return an OpenAIBackend with a fake key; no server is contacted."""
    return OpenAIBackend(
        model_id=model_id, api_key="fake-key", base_url="http://localhost:9999/v1"
    )


def _retaining_ctx(
    *,
    ids: list[int],
    prefix_messages: list[dict],
    model_id: str = "granite-switch",
    message_count: int | None = None,
) -> ChatContext:
    """Return a retaining ChatContext holding `ids` over `prefix_messages`."""
    return ChatContext(retain_token_ids=True).with_sent_token_ids(
        ids,
        model_id=model_id,
        message_count=message_count
        if message_count is not None
        else len(prefix_messages),
        prompt_digest=_prompt_digest(prefix_messages),
    )


# --- Chat -> Intrinsic: append-only rewrite reuses the exact prefix ----------


async def test_chat_to_intrinsic_appends_and_reuses_exact_prefix():
    """An intrinsic that only APPENDS an instruction reuses the exact retained ids.

    The retained prefix carries a Granite Switch control token (999) that a fresh
    render would lose; the appended instruction becomes the freshly-derived suffix.
    The result is the retained ids verbatim + the suffix -- never a re-tokenization
    of the prefix.
    """
    backend = _backend()
    prefix = [{"role": "user", "content": "U1"}, {"role": "assistant", "content": "A1"}]
    ctx = _retaining_ctx(ids=[10, 11, 999], prefix_messages=prefix)
    # The rewriter appended an instruction message; the prefix is byte-identical.
    rewritten = [*prefix, {"role": "user", "content": "Judge the answer."}]

    async def fake_tokenize(messages, *, add_generation_prompt=True, **_):
        if len(messages) == 3:  # full rewritten conversation + generation prompt
            return [1, 2, 3, 4, 70, 71, 72]
        return [1, 2, 3, 4]  # already-sent side [U1, A1]

    backend._tokenize_chat = AsyncMock(side_effect=fake_tokenize)

    ids = await backend._reuse_intrinsic_prefix_ids(
        ctx, rewritten, extra_body={}, api_params={}, use_tools=False
    )
    # Retained [10, 11, 999] verbatim, appended instruction's [70, 71, 72] spliced on.
    assert ids == [10, 11, 999, 70, 71, 72]


async def test_chat_to_intrinsic_preserves_multiple_control_tokens():
    """Multiple switch control-token regions in the retained prefix survive verbatim."""
    backend = _backend()
    prefix = [{"role": "user", "content": "U1"}, {"role": "assistant", "content": "A1"}]
    ctx = _retaining_ctx(ids=[10, 900, 11, 901, 12], prefix_messages=prefix)
    rewritten = [*prefix, {"role": "user", "content": "Check."}]

    async def fake_tokenize(messages, *, add_generation_prompt=True, **_):
        return [1, 2, 88] if len(messages) == 3 else [1, 2]

    backend._tokenize_chat = AsyncMock(side_effect=fake_tokenize)

    ids = await backend._reuse_intrinsic_prefix_ids(
        ctx, rewritten, extra_body={}, api_params={}, use_tools=False
    )
    assert ids == [10, 900, 11, 901, 12, 88]


async def test_reuse_does_not_reconstruct_prefix_through_text():
    """A non-round-tripping prefix ([71, 4896] -> 'hello' -> [15339]) is kept verbatim.

    The retained ids are `[71, 4896]` (a non-canonical split of "hello"); a fresh
    render canonicalizes to `[15339]`. Reuse must splice the retained `[71, 4896]`,
    not the reconstructed `[15339]`, or the server's prefix cache is destroyed.
    """
    backend = _backend()
    prefix = [{"role": "user", "content": "hello"}]
    ctx = _retaining_ctx(ids=[71, 4896], prefix_messages=prefix)
    rewritten = [*prefix, {"role": "user", "content": "Rate it."}]

    async def fake_tokenize(messages, *, add_generation_prompt=True, **_):
        # Fresh (canonical) renders: "hello" is a single id 15339 here.
        return [15339, 200, 201] if len(messages) == 2 else [15339]

    backend._tokenize_chat = AsyncMock(side_effect=fake_tokenize)

    ids = await backend._reuse_intrinsic_prefix_ids(
        ctx, rewritten, extra_body={}, api_params={}, use_tools=False
    )
    # [71, 4896] preserved — NOT collapsed to the canonical [15339].
    assert ids == [71, 4896, 200, 201]


# --- Chat -> Intrinsic: prefix-changing rewrite must NOT reuse ---------------


async def test_chat_to_intrinsic_rewritten_history_is_refused():
    """An io.yaml that edits earlier turns in place changes the prefix -> no reuse."""
    backend = _backend()
    ctx = _retaining_ctx(
        ids=[10, 11, 999],
        prefix_messages=[
            {"role": "user", "content": "U1"},
            {"role": "assistant", "content": "A1"},
        ],
    )
    # sentence_boundaries="all_but_last_message" tagged the earlier turns.
    rewritten = [
        {"role": "user", "content": "<c0>U1"},
        {"role": "assistant", "content": "<c1>A1"},
        {"role": "user", "content": "Cite sources."},
    ]
    backend._tokenize_chat = AsyncMock(side_effect=AssertionError("must not tokenize"))

    ids = await backend._reuse_intrinsic_prefix_ids(
        ctx, rewritten, extra_body={}, api_params={}, use_tools=False
    )
    assert ids is None  # digest mismatch -> fall back to the chat endpoint


async def test_chat_to_intrinsic_prepended_message_is_refused():
    """A docs-as-message intrinsic prepends at position 0 -> prefix diverges from token 0."""
    backend = _backend()
    ctx = _retaining_ctx(
        ids=[10, 11, 999],
        prefix_messages=[
            {"role": "user", "content": "U1"},
            {"role": "assistant", "content": "A1"},
        ],
    )
    rewritten = [
        {"role": "user", "content": "Documents: ..."},  # prepended
        {"role": "user", "content": "U1"},
        {"role": "assistant", "content": "A1"},
    ]
    backend._tokenize_chat = AsyncMock(side_effect=AssertionError("must not tokenize"))

    ids = await backend._reuse_intrinsic_prefix_ids(
        ctx, rewritten, extra_body={}, api_params={}, use_tools=False
    )
    assert ids is None


# --- gates: requests the completions transport cannot honour fall back -------


async def test_reuse_gated_off_when_logprobs_required():
    backend = _backend()
    ctx = _retaining_ctx(
        ids=[10, 11], prefix_messages=[{"role": "user", "content": "U1"}]
    )
    rewritten = [{"role": "user", "content": "U1"}, {"role": "user", "content": "x"}]
    backend._tokenize_chat = AsyncMock(side_effect=AssertionError("must not tokenize"))

    ids = await backend._reuse_intrinsic_prefix_ids(
        ctx, rewritten, extra_body={}, api_params={"logprobs": True}, use_tools=False
    )
    assert ids is None


async def test_reuse_gated_off_when_documents_present():
    backend = _backend()
    ctx = _retaining_ctx(
        ids=[10, 11], prefix_messages=[{"role": "user", "content": "U1"}]
    )
    rewritten = [{"role": "user", "content": "U1"}, {"role": "user", "content": "x"}]
    backend._tokenize_chat = AsyncMock(side_effect=AssertionError("must not tokenize"))

    ids = await backend._reuse_intrinsic_prefix_ids(
        ctx,
        rewritten,
        extra_body={"documents": [{"text": "doc"}]},
        api_params={},
        use_tools=False,
    )
    assert ids is None


async def test_reuse_gated_off_when_tools_used():
    backend = _backend()
    ctx = _retaining_ctx(
        ids=[10, 11], prefix_messages=[{"role": "user", "content": "U1"}]
    )
    rewritten = [{"role": "user", "content": "U1"}, {"role": "user", "content": "x"}]
    backend._tokenize_chat = AsyncMock(side_effect=AssertionError("must not tokenize"))

    ids = await backend._reuse_intrinsic_prefix_ids(
        ctx, rewritten, extra_body={}, api_params={}, use_tools=True
    )
    assert ids is None


async def test_reuse_gated_off_when_reasoning_effort_string():
    backend = _backend()
    ctx = _retaining_ctx(
        ids=[10, 11], prefix_messages=[{"role": "user", "content": "U1"}]
    )
    rewritten = [{"role": "user", "content": "U1"}, {"role": "user", "content": "x"}]
    backend._tokenize_chat = AsyncMock(side_effect=AssertionError("must not tokenize"))

    ids = await backend._reuse_intrinsic_prefix_ids(
        ctx,
        rewritten,
        extra_body={},
        api_params={"reasoning_effort": "low"},
        use_tools=False,
    )
    assert ids is None


async def test_no_reuse_without_retained_ids():
    """A retaining context that has sent nothing yet has no prefix to force."""
    backend = _backend()
    ctx = ChatContext(retain_token_ids=True)
    backend._tokenize_chat = AsyncMock(side_effect=AssertionError("must not tokenize"))

    ids = await backend._reuse_intrinsic_prefix_ids(
        ctx,
        [{"role": "user", "content": "U1"}],
        extra_body={},
        api_params={},
        use_tools=False,
    )
    assert ids is None


async def test_no_reuse_for_non_retaining_context():
    backend = _backend()
    ctx = ChatContext()  # retain_token_ids defaults to False
    backend._tokenize_chat = AsyncMock(side_effect=AssertionError("must not tokenize"))

    ids = await backend._reuse_intrinsic_prefix_ids(
        ctx,
        [{"role": "user", "content": "U1"}],
        extra_body={},
        api_params={},
        use_tools=False,
    )
    assert ids is None


async def test_reuse_refused_across_models():
    """Retained ids from another model are not reinterpreted; reuse falls back."""
    backend = _backend(model_id="granite-switch")
    ctx = _retaining_ctx(
        ids=[10, 11],
        prefix_messages=[{"role": "user", "content": "U1"}],
        model_id="some-other-model",
    )
    rewritten = [{"role": "user", "content": "U1"}, {"role": "user", "content": "x"}]
    backend._tokenize_chat = AsyncMock(side_effect=AssertionError("must not tokenize"))

    ids = await backend._reuse_intrinsic_prefix_ids(
        ctx, rewritten, extra_body={}, api_params={}, use_tools=False
    )
    assert ids is None


# --- Intrinsic -> Chat / Intrinsic -> Intrinsic ------------------------------


async def test_intrinsic_does_not_commit_ids_so_next_turn_keeps_chat_prefix():
    """An intrinsic reuse computes ids to SEND but records nothing on the context.

    `_reuse_intrinsic_prefix_ids` returns the spliced ids for the request; it must not
    mutate the context's retained state. The next turn (chat or intrinsic) therefore
    still branches from the original chat prefix, not from the intrinsic's request.
    """
    backend = _backend()
    prefix = [{"role": "user", "content": "U1"}, {"role": "assistant", "content": "A1"}]
    ctx = _retaining_ctx(ids=[10, 11, 999], prefix_messages=prefix)
    before = (
        ctx.sent_token_ids,
        ctx.sent_message_count,
        ctx.sent_prompt_digest,
        ctx.sent_model_id,
    )
    rewritten = [*prefix, {"role": "user", "content": "Judge."}]
    backend._tokenize_chat = AsyncMock(
        side_effect=lambda m, **_: [1, 2, 3, 4, 5] if len(m) == 3 else [1, 2, 3, 4]
    )

    await backend._reuse_intrinsic_prefix_ids(
        ctx, rewritten, extra_body={}, api_params={}, use_tools=False
    )

    assert (
        ctx.sent_token_ids,
        ctx.sent_message_count,
        ctx.sent_prompt_digest,
        ctx.sent_model_id,
    ) == before


async def test_intrinsic_to_intrinsic_reuses_shared_prefix():
    """A second intrinsic sharing the same retained prefix reuses the exact ids."""
    backend = _backend()
    prefix = [{"role": "user", "content": "U1"}, {"role": "assistant", "content": "A1"}]
    ctx = _retaining_ctx(ids=[10, 11, 999], prefix_messages=prefix)
    # Second intrinsic appends a different instruction over the same prefix.
    rewritten = [*prefix, {"role": "user", "content": "Different check."}]
    backend._tokenize_chat = AsyncMock(
        side_effect=lambda m, **_: [1, 2, 3, 4, 61, 62] if len(m) == 3 else [1, 2, 3, 4]
    )

    ids = await backend._reuse_intrinsic_prefix_ids(
        ctx, rewritten, extra_body={}, api_params={}, use_tools=False
    )
    assert ids == [10, 11, 999, 61, 62]


# --- transport: completions reply adapted to a ChatCompletion ----------------


async def test_intrinsic_completion_as_chat_sends_exact_ids_and_adapts_reply():
    """The spliced ids reach /v1/completions verbatim; the reply becomes a ChatCompletion.

    Also verifies the two param/body translations: `max_completion_tokens` -> `max_tokens`,
    and `chat_template_kwargs`/`documents` dropped from `extra_body` (a pre-tokenized
    prompt is not run through the chat template, and the control token is already inside
    the ids).
    """
    backend = _backend()

    fake_completion = MagicMock()
    fake_completion.id = "cmpl-1"
    fake_completion.created = 0
    fake_completion.model = "granite-switch"
    fake_completion.usage = None
    choice = MagicMock()
    choice.index = 0
    choice.text = '{"result": "ok"}'
    choice.finish_reason = "stop"
    fake_completion.choices = [choice]

    create = AsyncMock(return_value=fake_completion)
    mock_client = MagicMock()
    mock_client.completions.create = create

    with patch.object(
        OpenAIBackend,
        "_async_client",
        new_callable=PropertyMock,
        return_value=mock_client,
    ):
        chat = await backend._intrinsic_completion_as_chat(
            [10, 11, 999, 70, 71],
            api_params={"max_completion_tokens": 6, "temperature": 0.0},
            extra_body={
                "chat_template_kwargs": {"adapter_name": "x"},
                "documents": [{"text": "d"}],
                "structured_outputs": {"json": {"type": "object"}},
            },
        )

    kwargs = create.call_args.kwargs
    assert kwargs["prompt"] == [[10, 11, 999, 70, 71]]
    assert kwargs["max_tokens"] == 6  # translated from max_completion_tokens
    assert "max_completion_tokens" not in kwargs
    assert kwargs["temperature"] == 0.0
    # chat-template-only keys dropped; guided decoding forwarded.
    body = kwargs["extra_body"]
    assert "chat_template_kwargs" not in body
    assert "documents" not in body
    assert body["structured_outputs"] == {"json": {"type": "object"}}
    # The completion text is surfaced as the assistant message content.
    assert chat.choices[0].message.content == '{"result": "ok"}'


# --- Chat -> Chat: the digest recorded end-to-end proves the prefix ----------


async def test_generate_from_chat_context_records_prompt_digest(monkeypatch):
    """`generate_from_chat_context` moves the recorded digest onto the next context."""
    from mellea.core.base import ModelOutputThunk
    from mellea.stdlib.components import Message

    backend = _backend()
    ctx = ChatContext(retain_token_ids=True)
    action = Message("user", "U1")

    digest = _prompt_digest([{"role": "user", "content": "U1"}])
    mot = ModelOutputThunk("A1")
    mot._meta["retained_token_ids"] = [10, 11, 12]
    mot._meta["retained_model_id"] = "granite-switch"
    mot._meta["retained_message_count"] = 2
    mot._meta["retained_prompt_digest"] = digest

    async def fake_standard(*a, **k):
        return mot

    monkeypatch.setattr(backend, "_generate_from_chat_context_standard", fake_standard)
    monkeypatch.setattr(backend, "do_generate_walk", AsyncMock())

    _, new_ctx = await backend.generate_from_chat_context(action, ctx)
    assert new_ctx.sent_prompt_digest == digest
    assert new_ctx.sent_token_ids == (10, 11, 12)
    assert new_ctx.sent_message_count == 2


# --- integration: the reuse wiring inside _generate_from_intrinsic -----------

_APPEND_CONFIG = {
    "model": None,
    "response_format": None,
    "transformations": None,
    "instruction": "Judge: {last_message}",
    "parameters": {"max_completion_tokens": 8},
    "sentence_boundaries": None,
}


def _adapter_backend(config: dict) -> OpenAIBackend:
    from mellea.backends.adapters.adapter import EmbeddedIntrinsicAdapter

    backend = _backend()
    backend.add_adapter(
        EmbeddedIntrinsicAdapter(
            intrinsic_name="answerability", config=config, technology="alora"
        )
    )
    return backend


def _fake_completion(text: str = '{"result": "ok"}'):
    completion = MagicMock()
    completion.id = "cmpl-1"
    completion.created = 0
    completion.model = "granite-switch"
    completion.usage = None
    choice = MagicMock()
    choice.index = 0
    choice.text = text
    choice.finish_reason = "stop"
    completion.choices = [choice]
    return completion


async def test_intrinsic_path_sends_reused_ids_to_completions_endpoint():
    """End-to-end: an append-only intrinsic over a retaining context hits /v1/completions.

    The retained chat prefix's exact ids ([10, 11, 999], control token intact) are
    spliced with the appended instruction's suffix and sent as a pre-tokenized prompt;
    the chat endpoint is not touched.
    """
    from mellea.stdlib import functional as mfuncs
    from mellea.stdlib.components import Intrinsic, Message

    backend = _adapter_backend(_APPEND_CONFIG)
    # Retaining context holding ids over its single [U1] turn. Empty digest isolates
    # the transport wiring; the digest gate itself is covered by the unit tests above.
    ctx = ChatContext(retain_token_ids=True).add(Message("user", "U1"))
    ctx = ctx.with_sent_token_ids(
        [10, 11, 999], model_id="granite-switch", message_count=1, prompt_digest=()
    )

    backend._tokenize_chat = AsyncMock(
        side_effect=lambda m, **_: [1, 2, 50, 51] if len(m) == 2 else [1, 2]
    )
    completions_create = AsyncMock(return_value=_fake_completion())
    chat_create = AsyncMock()
    mock_client = MagicMock()
    mock_client.completions.create = completions_create
    mock_client.chat.completions.create = chat_create

    with patch.object(
        OpenAIBackend,
        "_async_client",
        new_callable=PropertyMock,
        return_value=mock_client,
    ):
        mot, _ = await mfuncs.aact(
            Intrinsic("answerability"), ctx, backend, strategy=None
        )
        await mot.avalue()

    completions_create.assert_called_once()
    assert completions_create.call_args.kwargs["prompt"] == [[10, 11, 999, 50, 51]]
    chat_create.assert_not_called()


async def test_intrinsic_path_falls_back_to_chat_when_prefix_changed():
    """A retaining context whose prefix no longer matches re-renders via the chat endpoint."""
    from openai.types.chat import ChatCompletion, ChatCompletionMessage
    from openai.types.chat.chat_completion import Choice
    from openai.types.completion_usage import CompletionUsage

    from mellea.stdlib import functional as mfuncs
    from mellea.stdlib.components import Intrinsic, Message

    backend = _adapter_backend(_APPEND_CONFIG)
    ctx = ChatContext(retain_token_ids=True).add(Message("user", "U1"))
    # Digest records a DIFFERENT leading message, so reuse is refused.
    ctx = ctx.with_sent_token_ids(
        [10, 11, 999],
        model_id="granite-switch",
        message_count=1,
        prompt_digest=_prompt_digest([{"role": "user", "content": "SOMETHING ELSE"}]),
    )

    backend._tokenize_chat = AsyncMock(
        side_effect=AssertionError("digest gate must refuse before tokenizing")
    )
    chat_create = AsyncMock(
        return_value=ChatCompletion(
            id="c",
            created=0,
            model="granite-switch",
            object="chat.completion",
            choices=[
                Choice(
                    index=0,
                    finish_reason="stop",
                    message=ChatCompletionMessage(
                        role="assistant", content='{"result": "ok"}'
                    ),
                )
            ],
            usage=CompletionUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        )
    )
    completions_create = AsyncMock()
    mock_client = MagicMock()
    mock_client.chat.completions.create = chat_create
    mock_client.completions.create = completions_create

    with patch.object(
        OpenAIBackend,
        "_async_client",
        new_callable=PropertyMock,
        return_value=mock_client,
    ):
        mot, _ = await mfuncs.aact(
            Intrinsic("answerability"), ctx, backend, strategy=None
        )
        await mot.avalue()

    chat_create.assert_called_once()
    completions_create.assert_not_called()
