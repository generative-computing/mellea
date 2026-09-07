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

import pytest

from mellea import MelleaSession
from mellea.backends import ModelOption
from mellea.backends.openai import OpenAIBackend
from mellea.backends.tools import MelleaTool
from mellea.formatters import TemplateFormatter
from mellea.stdlib.context import ChatContext

pytestmark = [
    pytest.mark.openai,
    pytest.mark.e2e,
    pytest.mark.vllm,
    pytest.mark.skipif(
        not os.environ.get("VLLM_TEST_BASE_URL"),
        reason="set VLLM_TEST_BASE_URL to a live vLLM >= 0.10.2 endpoint",
    ),
]

_MODEL = os.environ.get(
    "VLLM_TEST_MODEL", "/danieloh_cos/granite-switch/hf_models/gs_4.1_3b"
)


@pytest.fixture(scope="module")
def backend() -> OpenAIBackend:
    base_url = os.environ["VLLM_TEST_BASE_URL"].rstrip("/")
    if not base_url.endswith("/v1"):
        base_url += "/v1"
    return OpenAIBackend(
        model_id=_MODEL,
        formatter=TemplateFormatter(model_id=_MODEL),
        base_url=base_url,
        api_key="EMPTY",
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
