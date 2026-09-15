# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""End-to-end token-id-history tests against a live vLLM server.

These assert the EXACT token ids the retain-ids policy sends, not output quality, so
they are not `qualitative`. They need a vLLM >= 0.10.2 server (for `return_token_ids`
and `/tokenize`); point `VLLM_TEST_BASE_URL` at it (see
`scratchpad/vela-tokenid-vllm.yaml`). The whole module skips when it is unset.

Serving `ibm-granite/granite-switch-4.1-3b-preview` (or the `gs_4.1_3b` build) also
exercises adapter control-token preservation; a plain Granite model still exercises
the BPE round-trip half.
"""

import logging
import os
import re
import uuid

import httpx
import pytest

from mellea import MelleaSession
from mellea.backends import ModelOption
from mellea.backends.openai import OpenAIBackend
from mellea.backends.tools import MelleaTool
from mellea.formatters import TemplateFormatter
from mellea.stdlib.context import ChatContext
from test.predicates import require_gpu

pytestmark = [
    pytest.mark.openai,
    pytest.mark.e2e,
    pytest.mark.vllm,
    # A full pass is a dozen multi-turn generations plus their /tokenize round trips
    # against a live server, which is past the one-minute bar test/README.md sets for
    # `slow`. Excluded from the default run by pyproject's addopts.
    pytest.mark.slow,
    # Same gate as `test_openai_intrinsics.py`, which serves the same
    # `granite-switch-4.1-3b-preview` over vLLM. The endpoint is remote here, so the
    # `VLLM_TEST_BASE_URL` skip below is what actually decides whether these run; this
    # keeps the resource declaration consistent with the rest of the vLLM suite for a
    # runner that serves the model itself.
    require_gpu(min_vram_gb=12),
    pytest.mark.skipif(
        int(os.environ.get("CICD", 0)) == 1,
        reason="needs a vLLM endpoint that CI does not provide",
    ),
    pytest.mark.skipif(
        not os.environ.get("VLLM_TEST_BASE_URL"),
        reason="set VLLM_TEST_BASE_URL to a live vLLM >= 0.10.2 endpoint",
    ),
]

_MODEL = os.environ.get(
    "VLLM_TEST_MODEL", "/danieloh_cos/granite-switch/hf_models/gs_4.1_3b"
)


_ADAPTER_SOURCE = os.environ.get("VLLM_TEST_ADAPTER_SOURCE")

# Adapter discovery is CLIENT-side: it reads `adapter_index.json` and `io_configs/` to
# map adapter names to control-token ids. When the served model name is a path inside the
# server's container (a locally-served checkpoint), the client cannot read it and falls
# back to treating it as a Hub repo id, which raises `HFValidationError` on any path with
# more than one "/". Skip rather than fail: the rest of the suite is still meaningful.
requires_adapter_source = pytest.mark.skipif(
    _ADAPTER_SOURCE is None,
    reason=(
        "set VLLM_TEST_ADAPTER_SOURCE to a local directory holding the served "
        "checkpoint's adapter metadata (adapter_index.json, io_configs/, config.json; "
        "no weights needed) to run the adapter cases"
    ),
)


@pytest.fixture(scope="module")
def backend() -> OpenAIBackend:
    """A backend pointed at the live server, with adapter metadata resolvable locally.

    `_MODEL` must match the server's served model name, which for a locally-served
    checkpoint is a filesystem path INSIDE the server's container. The client cannot
    read that path, and adapter discovery is client-side (it reads `adapter_index.json`
    and `io_configs/` to map adapter names and control-token ids), so
    `VLLM_TEST_ADAPTER_SOURCE` lets the metadata come from a local copy while the
    request still carries the served name. Point it at a directory holding that
    checkpoint's metadata -- no weights needed. Without it, adapter discovery falls
    back to treating the served name as a Hub repo id and raises
    `HFValidationError` on any path with more than one `/`.
    """
    base_url = os.environ["VLLM_TEST_BASE_URL"].rstrip("/")
    if not base_url.endswith("/v1"):
        base_url += "/v1"
    adapter_source = _ADAPTER_SOURCE
    return OpenAIBackend(
        model_id=_MODEL,
        formatter=TemplateFormatter(model_id=_MODEL),
        base_url=base_url,
        api_key="EMPTY",
        load_embedded_adapters=adapter_source is not None,
        adapter_source=adapter_source,
    )


@pytest.fixture(scope="function")
def session(backend: OpenAIBackend):
    s = MelleaSession(backend, ctx=ChatContext(retain_token_ids=True, model_id=_MODEL))
    yield s
    s.reset()


def test_chat_to_chat_reuses_exact_prefix(session: MelleaSession) -> None:
    """Turn 2's prompt begins with turn 1's exact sent ids, spliced verbatim.

    If the policy re-tokenized the history instead of splicing, the retained prefix
    could differ (BPE non-round-trip / lost control tokens) -- so an exact-prefix
    match is the observable proof the server saw an unchanged, cache-hittable prefix.
    """
    session.chat("Name one primary color.")
    prev = _chat_ctx(session).sent_token_ids
    assert prev, (
        "no ids retained: server did not report token_ids (need vLLM >= 0.10.2)"
    )
    assert _chat_ctx(session).sent_model_id == _MODEL
    prev_count = _chat_ctx(session).sent_message_count
    assert prev_count > 0

    session.chat("Name another one.")
    now = _chat_ctx(session).sent_token_ids
    assert now, "turn 2 retained no ids"
    # The whole of turn 1's sent sequence is the verbatim prefix of turn 2's.
    assert now[: len(prev)] == prev
    assert len(now) > len(prev)
    assert _chat_ctx(session).sent_message_count > prev_count


@requires_adapter_source
def test_intrinsic_reuses_but_does_not_commit(
    session: MelleaSession, backend: OpenAIBackend
) -> None:
    """An intrinsic between two chats reuses the prefix but records no ids of its own.

    After the intrinsic the retained state must be UNCHANGED (intrinsics never commit
    their rewritten request as history), and the next chat must still splice turn 1's
    exact prefix -- branching from the chat history, not the intrinsic scaffolding.
    """
    from mellea.stdlib.components.intrinsic import core

    session.chat("The capital of France is Paris.")
    after_chat = _chat_ctx(session).sent_token_ids
    after_chat_count = _chat_ctx(session).sent_message_count
    assert after_chat, "no ids retained after the first chat"

    # Intrinsic evaluates the conversation; it must not alter the retained prefix.
    score = core.check_certainty(_chat_ctx(session), backend)
    assert 0.0 <= float(score) <= 1.0
    assert _chat_ctx(session).sent_token_ids == after_chat, "intrinsic committed ids"
    assert _chat_ctx(session).sent_message_count == after_chat_count

    # The following chat branches from the chat history, reusing turn 1's exact ids.
    session.chat("And its most famous landmark?")
    now = _chat_ctx(session).sent_token_ids
    assert now[: len(after_chat)] == after_chat
    assert len(now) > len(after_chat)


def test_tool_turn_refused_under_retain(session: MelleaSession) -> None:
    """Tools are unsupported on the id transport and must raise, not silently drop."""

    def noop() -> str:
        """Return a fixed string; exists only to be a tool this turn may not use."""
        return "ok"

    # A bare callable is rejected by `add_tools_from_model_options` (tools.py:350)
    # before the backend is reached, which would make this pass for the wrong reason.
    with pytest.raises(NotImplementedError):
        session.instruct(
            "Call a tool.",
            model_options={ModelOption.TOOLS: [MelleaTool.from_callable(noop)]},
            tool_calls=True,
        )


def _chat_ctx(session: MelleaSession) -> ChatContext:
    """Narrow `session.ctx` (typed `Context`) to the ChatContext the fixture binds."""
    ctx = session.ctx
    assert isinstance(ctx, ChatContext)
    return ctx


def test_thinking_true_reuses_exact_prefix_across_turns(session: MelleaSession) -> None:
    """`THINKING=True` is a supported toggle on the id transport and still splices exactly.

    A boolean thinking toggle rides `chat_template_kwargs.enable_thinking`, which
    `_build_prompt_ids` must pass to /tokenize so the pre-tokenized prompt matches the
    server's render. If it did not, turn 2's retained prefix would diverge from turn 1's
    sent ids; an exact-prefix match is the proof the reasoning render round-tripped.
    (A STRING reasoning level would instead be refused -- that path is covered by the
    unit tests; only the supported boolean toggle is exercised end to end here.)
    """
    session.chat("Name one primary color.", model_options={ModelOption.THINKING: True})
    prev = _chat_ctx(session).sent_token_ids
    assert prev, (
        "no ids retained: server did not report token_ids (need vLLM >= 0.10.2)"
    )

    session.chat("Name another one.", model_options={ModelOption.THINKING: True})
    now = _chat_ctx(session).sent_token_ids
    assert now[: len(prev)] == prev
    assert len(now) > len(prev)


def test_unextendable_prefix_falls_back_and_the_turn_still_succeeds(
    backend: OpenAIBackend, caplog: pytest.LogCaptureFixture
) -> None:
    """A retained prefix that can no longer be extended costs the cache, not the turn.

    The unit tests prove the dispatch falls through; only this one proves the fallback
    request is one a real server accepts. A windowed context is the realistic trigger:
    the window drops the oldest turns, so the conversation shrinks below the message
    count the retained ids cover, `_build_prompt_ids` refuses the prefix, and
    `_generate_from_chat_context_standard` must send the turn as chat messages instead.

    Asserted on three things, because a turn that merely succeeds proves nothing -- it
    could have succeeded by never retaining ids at all: ids ARE retained first (so there
    is a prefix to refuse), the refusal IS logged (so the fallback ran rather than the
    splice), and every answer is non-empty (so the server accepted the fallback
    request).
    """
    windowed = MelleaSession(
        backend, ctx=ChatContext(retain_token_ids=True, window_size=2, model_id=_MODEL)
    )
    try:
        windowed.chat("Name one primary color.")
        assert _chat_ctx(windowed).sent_token_ids, (
            "no ids retained on turn 1: the server did not report token_ids "
            "(need vLLM >= 0.10.2), so the fallback is not under test here"
        )

        # Each turn adds two messages (user + assistant) while the window keeps two, so
        # by the end the retained count exceeds what the conversation still holds.
        with caplog.at_level(logging.WARNING, logger="mellea"):
            for prompt in ("Name another one.", "And a third?", "One more."):
                answer = windowed.chat(prompt)
                assert str(answer.content).strip(), f"empty answer for {prompt!r}"

        assert any(
            "token-id history could not be extended" in r.message
            for r in caplog.records
        ), (
            "the windowed conversation never refused its prefix, so this test did not "
            "exercise the fallback -- the shrink guard in `_build_prompt_ids` may have "
            "stopped firing"
        )
    finally:
        windowed.reset()


# --- Server-side proof: the prefix cache is actually HIT ----------------------
#
# Every assertion above compares ids on the CLIENT: it proves the prompt was eligible
# for a cache hit (its leading tokens are byte-identical to what the server already
# saw), not that the server hit its cache. Only the server's own counters show that.

# vLLM's prefix cache is keyed per BLOCK, 16 tokens by default; reuse is therefore
# reported in 16-token steps and two prompts differing by one token can differ by one
# whole block.
_CACHE_BLOCK_TOKENS = 16

_CACHE_METRIC = re.compile(
    r"^vllm:(?:gpu_)?prefix_cache_(queries|hits)_total(?:\{[^}]*\})?\s+([0-9.e+-]+)$",
    re.MULTILINE,
)


def _server_root() -> str:
    """The server root (no `/v1`), where vLLM serves `/metrics`."""
    return os.environ["VLLM_TEST_BASE_URL"].rstrip("/").removesuffix("/v1")


def _prefix_cache_counters() -> tuple[float, float] | None:
    """Return `(queries, hits)` summed over label sets, or `None` if unavailable.

    `None` covers both a missing `/metrics` route and a build that reports neither
    counter (the v0 engine exports a hit-RATE gauge instead), so a server that cannot
    answer the question makes the test skip rather than fail.
    """
    try:
        body = httpx.get(f"{_server_root()}/metrics", timeout=10.0).text
    except httpx.HTTPError:
        return None
    totals = {"queries": 0.0, "hits": 0.0}
    found = False
    for kind, value in _CACHE_METRIC.findall(body):
        totals[kind] += float(value)
        found = True
    return (totals["queries"], totals["hits"]) if found else None


def _turn_two_cache_delta(
    ctx: ChatContext, backend: OpenAIBackend, *, nonce: str = ""
) -> tuple[float, float]:
    """Run a two-turn conversation on `ctx`; return turn 2's `(queries, hits)` delta.

    Turn 1 is outside the measurement: it populates the cache, and what is under test is
    whether turn 2 finds it.

    `nonce` goes into turn 1's text so that two arms of a comparison occupy DIFFERENT
    cache blocks. Without it the arms cross-warm: both send the same transcript, so
    whichever runs second measures blocks the first one just cached, and the ordering
    advantage swamps the effect under test. Measured on a live server: with a shared
    transcript the second (non-retaining) arm read 96.97% against the first arm's
    94.12%; with per-arm nonces both reused exactly 32 tokens.
    """
    session = MelleaSession(backend, ctx=ctx)
    # Long enough that vLLM's 16-token block granularity is a small fraction of the
    # prefix: at a 2-3 block prompt a one-token shift moves a whole block, which is
    # larger than the effect under test (see the caller's docstring).
    filler = " ".join(f"Fact {i} is unremarkable." for i in range(40))
    suffix = f" Context note {nonce}. {filler}" if nonce else ""
    try:
        session.chat(f"Name one primary color.{suffix}")
        before = _prefix_cache_counters()
        assert before is not None
        session.chat("Name another one.")
        after = _prefix_cache_counters()
        assert after is not None
    finally:
        session.reset()
    return after[0] - before[0], after[1] - before[1]


def test_retained_turn_hits_the_server_prefix_cache(backend: OpenAIBackend) -> None:
    """Turn 2 of a retaining conversation is HIT by the server's prefix cache.

    The client-side exact-prefix assertions prove eligibility; this reads the server's
    own `prefix_cache_hits_total` and so proves reuse. Measured as a delta around turn 2
    only, and the module runs single-threaded against a dedicated server, so no other
    traffic contributes to the window.
    """
    if _prefix_cache_counters() is None:
        pytest.skip("server exports no prefix_cache_{queries,hits}_total counters")

    queries, hits = _turn_two_cache_delta(
        ChatContext(retain_token_ids=True, model_id=_MODEL), backend
    )

    assert queries > 0, "turn 2 queried no cache blocks; the counters are not moving"
    # Turn 2 re-sends turn 1's whole transcript and adds one short user turn, so the
    # overwhelming majority of its blocks must already be resident.
    assert hits > 0, "the retained prefix was queried but never hit"
    assert hits / queries > 0.5, f"only {hits}/{queries} blocks hit"


def test_retaining_ids_is_never_worse_than_re_rendering(backend: OpenAIBackend) -> None:
    """A retaining context reuses at least as many cached tokens as a non-retaining one.

    The control matters because a high hit rate alone does not implicate the policy: vLLM
    also hits on a plain chat turn whenever re-rendering the transcript happens to
    reproduce the same tokens.

    Compared as absolute reused TOKENS, not as a hit rate. The retained prompt can be a
    token or two longer than the re-rendered one (it carries the turn terminator the
    template would re-emit), which changes the denominator without changing what was
    reused: measured live, the two arms reused exactly 32 tokens each while querying 49
    and 48, so a rate comparison reported the retaining arm as WORSE (65.31% vs 66.67%)
    on identical reuse.

    `>=` rather than `>`, deliberately: this transcript contains no adapter turn, so
    nothing in the history carries a control token and re-rendering reproduces the same
    ids. Equality is the correct outcome here, and it is what the live run shows. The
    strict gap only appears once an adapter turn is part of the retained prefix -- see
    `test_adapter_to_base_transition_hits_the_server_prefix_cache`, which measures that
    case directly.
    """
    if _prefix_cache_counters() is None:
        pytest.skip("server exports no prefix_cache_{queries,hits}_total counters")

    # Fresh nonces per RUN, not fixed strings: a fixed transcript is still resident
    # from the previous invocation against a long-lived server, so the arms get warmed
    # by history rather than by each other. Observed with fixed nonces: alternating
    # 32-vs-48 reuse on consecutive runs of this one test.
    run_id = uuid.uuid4().hex[:8]
    retained_queries, retained_hits = _turn_two_cache_delta(
        ChatContext(retain_token_ids=True, model_id=_MODEL),
        backend,
        nonce=f"retain-{run_id}",
    )
    plain_queries, plain_hits = _turn_two_cache_delta(
        ChatContext(model_id=_MODEL), backend, nonce=f"plain-{run_id}"
    )

    assert retained_queries > 0 and plain_queries > 0, (
        "neither arm queried the cache; the counters are not moving"
    )
    # One block of tolerance. vLLM caches in 16-token blocks, and the retained prompt
    # legitimately differs from the re-rendered one by a token or two (it carries the
    # turn terminator the template would re-emit), so a boundary shift can move exactly
    # one block in either direction. Observed while building this test: consecutive runs
    # reporting 32-vs-48 and 16-vs-32, always a single block apart, on a transcript
    # where the true difference is zero. The tolerance is meaningful only because the
    # transcript above is long enough that one block is a small fraction of the prefix.
    assert retained_hits >= plain_hits - _CACHE_BLOCK_TOKENS, (
        f"retaining ids reused {retained_hits:.0f} cached tokens but re-rendering reused "
        f"{plain_hits:.0f}, a gap wider than one {_CACHE_BLOCK_TOKENS}-token block -- "
        "the policy is costing reuse rather than preserving it"
    )


def test_documents_turn_reuses_its_prefix_and_hits_the_cache(
    backend: OpenAIBackend,
) -> None:
    """A RAG turn keeps its prefix: `/tokenize` renders `documents` the way the server does.

    `documents` are a chat-template variable, so they are forwarded into the
    `chat_template_kwargs` sent to `/tokenize` rather than declined. That is only correct
    if the tokenize render matches the render the chat template would produce for the same
    documents -- and nothing client-side can verify it, because the assembled prompt is
    never returned. The server's prefix cache is the check: a render that differed would
    put the retained ids out of step with what the server saw, and turn 2 would miss.

    Documents are supplied from the FIRST turn, since introducing them later re-renders
    the already-sent region and is refused as template-kwargs drift by design.
    """
    if _prefix_cache_counters() is None:
        pytest.skip("server exports no prefix_cache_{queries,hits}_total counters")

    docs = [
        {"doc_id": "1", "text": "Mellea retains token ids to keep prefixes cached."}
    ]
    opts = {"extra_body": {"documents": docs}}
    session = MelleaSession(
        backend, ctx=ChatContext(retain_token_ids=True, model_id=_MODEL)
    )
    try:
        session.chat("What does the document say?", model_options=opts)
        prev = _chat_ctx(session).sent_token_ids
        assert prev, (
            "no ids retained on a documents turn: the server did not report token_ids "
            "(need vLLM >= 0.10.2), or the turn did not take the id transport"
        )

        before = _prefix_cache_counters()
        assert before is not None
        session.chat("And what else?", model_options=opts)
        after = _prefix_cache_counters()
        assert after is not None

        now = _chat_ctx(session).sent_token_ids
        assert now[: len(prev)] == prev, "turn 2 did not splice turn 1's exact ids"
        assert len(now) > len(prev)

        queries, hits = after[0] - before[0], after[1] - before[1]
        assert queries > 0, "turn 2 queried no cache blocks"
        assert hits / queries > 0.5, (
            f"only {hits}/{queries} blocks hit -- the /tokenize render of `documents` "
            "probably differs from the chat template's, so the retained ids describe a "
            "prompt the server never cached"
        )
    finally:
        session.reset()


@requires_adapter_source
def test_adapter_to_base_transition_keeps_the_retained_prefix(
    session: MelleaSession, backend: OpenAIBackend
) -> None:
    """A base turn after an adapter turn still splices the exact retained prefix.

    The transition is where the two divergence causes meet. Turn 1's render carries the
    adapter's control token in place of a role marker; the base turn that follows passes
    no `adapter_name`, so a re-render would emit `<|start_of_role|>` there instead and
    invalidate every block from that position on. Splicing the retained ids is what keeps
    the earlier turn byte-identical, control token included, across the transition.

    Asserted in both directions: the retained prefix grows verbatim, AND the ids the
    adapter turn contributed are still present after the base turn -- so a silent
    fallback to a chat render (which would drop them) fails here rather than passing as
    a plain cache miss.
    """
    from mellea.stdlib.components.intrinsic import core

    session.chat("The capital of France is Paris.")
    after_chat = _chat_ctx(session).sent_token_ids
    assert after_chat, (
        "no ids retained: server did not report token_ids (need vLLM >= 0.10.2)"
    )

    # Adapter turn: reuses the prefix, commits nothing of its own.
    score = core.check_certainty(_chat_ctx(session), backend)
    assert 0.0 <= float(score) <= 1.0
    assert _chat_ctx(session).sent_token_ids == after_chat

    # Base turn immediately after: must extend, not re-render.
    session.chat("And the capital of Germany?")
    after_base = _chat_ctx(session).sent_token_ids
    assert after_base[: len(after_chat)] == after_chat, (
        "the base turn re-rendered history instead of splicing the retained prefix"
    )
    assert len(after_base) > len(after_chat)

    # One more adapter call on the grown history, to prove the transition is not
    # one-directional: base -> adapter must reuse the base turn's ids too.
    score_again = core.check_certainty(_chat_ctx(session), backend)
    assert 0.0 <= float(score_again) <= 1.0
    assert _chat_ctx(session).sent_token_ids == after_base


@requires_adapter_source
def test_adapter_to_base_transition_hits_the_server_prefix_cache(
    backend: OpenAIBackend,
) -> None:
    """The adapter-to-base transition is a server cache HIT, not merely eligible.

    Client-side splicing can be exact and still miss if the server never cached the
    blocks the adapter turn produced. Measured as a delta around the base turn that
    follows the adapter call.
    """
    if _prefix_cache_counters() is None:
        pytest.skip("server exports no prefix_cache_{queries,hits}_total counters")

    from mellea.stdlib.components.intrinsic import core

    session = MelleaSession(
        backend, ctx=ChatContext(retain_token_ids=True, model_id=_MODEL)
    )
    try:
        session.chat("The capital of France is Paris.")
        core.check_certainty(_chat_ctx(session), backend)
        before = _prefix_cache_counters()
        assert before is not None
        session.chat("And the capital of Germany?")
        after = _prefix_cache_counters()
        assert after is not None
    finally:
        session.reset()

    queries = after[0] - before[0]
    hits = after[1] - before[1]
    assert queries > 0, "the base turn queried no cache blocks; counters are not moving"
    assert hits / queries > 0.5, (
        f"only {hits}/{queries} blocks hit across the adapter-to-base transition; "
        "the retained prefix is not being reused"
    )


_GREEDY = {ModelOption.TEMPERATURE: 0.0, ModelOption.MAX_NEW_TOKENS: 200}
"""Deterministic and LONG. Both properties are load-bearing for the test below."""

_UNDER_ADAPTER = {
    **_GREEDY,
    "extra_body": {"chat_template_kwargs": {"adapter_name": "uncertainty"}},
}
"""Generate the turn under an adapter, so its rendered prompt carries a control token."""


def _adapter_conversation_cache_delta(
    ctx: ChatContext, backend: OpenAIBackend, nonce: str, turns: int = 5
) -> tuple[float, float]:
    """Run `turns` adapter turns on `ctx`; return `(queries, hits)` summed over turns 2..n.

    Turn 1 is outside the measurement (it populates the cache). Every turn is generated
    under an adapter, so each one's prompt ends with a control token that a later
    re-render replaces with the plain role marker.
    """
    session = MelleaSession(backend, ctx=ctx)
    total_queries = total_hits = 0.0
    try:
        session.chat(
            f"Seed {nonce}: explain in detail why the sky appears blue, at length.",
            model_options=_UNDER_ADAPTER,
        )
        for i in range(turns - 1):
            before = _prefix_cache_counters()
            assert before is not None
            session.chat(
                f"Turn {i}: now explain sunsets, at length.",
                model_options=_UNDER_ADAPTER,
            )
            after = _prefix_cache_counters()
            assert after is not None
            total_queries += after[0] - before[0]
            total_hits += after[1] - before[1]
    finally:
        session.reset()
    return total_queries, total_hits


def test_retaining_ids_reuses_more_cache_than_re_rendering(
    backend: OpenAIBackend,
) -> None:
    """Over multi-turn ADAPTER conversation, retaining ids reuses strictly more cache.

    This is the outcome the policy exists for, measured on the server's own counters
    rather than inferred from client-side ids.

    Three things have to be true at once for the gap to be visible, and each was
    established by measurement:

    1. Every turn is generated under an adapter. Control tokens accumulate one per turn
       in the RETAINED sequence (1, 2, 3 measured over three turns) while a re-render
       carries exactly one, at the current generation prompt. Without adapter turns the
       two transports send the same ids and reuse is legitimately equal.
    2. Replies must be LONG. vLLM caches whole 16-token blocks and never a trailing
       partial one. The divergent control token sits at the end of each turn's prompt,
       so with short replies it lands in a partial block that was never cached and the
       re-rendering arm loses nothing it could have hit: at `MAX_NEW_TOKENS=16` both arms
       reused exactly 224 tokens. At 200 the divergence falls inside complete blocks and
       the gap appears (272 vs 240 hits on equal queries, repeatably).
    3. Decoding must be greedy, or the arms' transcripts differ in length and the
       comparison is meaningless -- an early version measured 376 vs 465 queried tokens
       purely from reply-length variance.

    Asserted on absolute reused tokens, not a rate: the retained prompt can differ from
    the re-rendered one by a token or two, which moves the denominator without changing
    what was reused.
    """
    if _prefix_cache_counters() is None:
        pytest.skip("server exports no prefix_cache_{queries,hits}_total counters")

    run_id = uuid.uuid4().hex[:8]
    retained_queries, retained_hits = _adapter_conversation_cache_delta(
        ChatContext(retain_token_ids=True, model_id=_MODEL), backend, f"ret-{run_id}"
    )
    plain_queries, plain_hits = _adapter_conversation_cache_delta(
        ChatContext(model_id=_MODEL), backend, f"plain-{run_id}"
    )

    assert retained_queries > 0 and plain_queries > 0, (
        "neither arm queried the cache; the counters are not moving"
    )
    assert retained_hits > plain_hits, (
        f"retaining ids reused {retained_hits:.0f} cached tokens over the adapter "
        f"conversation but re-rendering reused {plain_hits:.0f} -- the retained control "
        "tokens are not buying any prefix-cache reuse"
    )
