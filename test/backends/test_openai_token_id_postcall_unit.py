# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the token-id retention path's thunk contract -- no server required.

When a chat turn is sent as a pre-tokenized completion instead of a chat request, the
produced thunk must still satisfy the same contract as a normal chat turn's. Two parts
of that contract are specific to this transport, plus two template-input invariants:

    1. The completions reply is text-shaped (`text`), but every consumer of
       `raw.response` on a chat turn expects chat shape -- `Message._parse` reads
       `response["choices"][0]["message"]`. The path normalizes the reply to chat shape
       so that read succeeds instead of raising `KeyError: "choices"`.
    2. The reply is materialized eagerly (the ids are derived from it), so the thunk is
       already computed and `astream()` never fires `generation_post_call`. The path
       stamps `_gen.start` (so latency is real, not -1) and flags the thunk so the
       public wrapper fires the post-call hook with the assigned `generation_id`.
    3. Template kwargs the turn is generated under (`enable_thinking`, an adapter name)
       must reach `_build_prompt_ids`, or the pre-tokenized prompt stops matching what
       the chat template would render and the server's prefix cache is corrupted.

The single network boundary (`_build_prompt_ids` / `_turn_terminator` / the completions
create call) is mocked; the normalization, parse, timing, and flagging under test are
the real code.
"""

from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch

import pytest
from openai.types import Completion
from openai.types.completion_choice import CompletionChoice
from openai.types.completion_usage import CompletionUsage

from mellea.backends import ModelOption
from mellea.backends.openai import OpenAIBackend
from mellea.core.base import ModelOutputThunk
from mellea.stdlib.components import Message
from mellea.stdlib.context.chat import ChatContext


def _backend(model_id: str = "granite-switch") -> OpenAIBackend:
    """Return an OpenAIBackend with a fake key; no server is contacted."""
    return OpenAIBackend(
        model_id=model_id, api_key="fake-key", base_url="http://localhost:9999/v1"
    )


def _completion(text: str = "Blue.", token_ids: list[int] | None = None) -> Completion:
    """A real `/v1/completions` reply carrying vLLM's on-choice `token_ids`.

    Real pydantic (not a MagicMock) so `.model_dump()` yields the dict the path
    normalizes and `_retained_ids` reads -- a MagicMock dump would skip normalization
    and defeat the checks below.
    """
    return Completion(
        id="cmpl-1",
        created=0,
        model="granite-switch",
        object="text_completion",
        choices=[
            CompletionChoice(
                index=0,
                text=text,
                finish_reason="stop",
                # vLLM extra field, preserved by pydantic (extra="allow"); not in the
                # stubbed CompletionChoice signature.
                token_ids=token_ids if token_ids is not None else [501, 502],  # type: ignore[call-arg]
            )
        ],
        usage=CompletionUsage(prompt_tokens=3, completion_tokens=2, total_tokens=5),
    )


async def _run_token_id_turn(
    backend: OpenAIBackend,
    ctx: ChatContext,
    *,
    prompt_ids: list[int] | None = None,
    terminator: list[int] | None = None,
    completion: Completion | None = None,
    model_options: dict | None = None,
):
    """Drive one chat turn through the token-id path with network boundaries mocked.

    Returns `(mot, new_ctx, build_prompt_ids_mock)` so callers can both inspect the
    produced thunk and assert what template kwargs reached `_build_prompt_ids`.
    """
    action = Message("user", "Name a primary color.")
    ctx = ctx.add(action)

    build_ids = AsyncMock(
        return_value=prompt_ids if prompt_ids is not None else [1, 2, 3]
    )
    create = AsyncMock(
        return_value=completion if completion is not None else _completion()
    )
    mock_client = MagicMock()
    mock_client.completions.create = create

    with patch.object(
        OpenAIBackend,
        "_async_client",
        new_callable=PropertyMock,
        return_value=mock_client,
    ):
        # setattr (not `backend.x = ...`) so mypy does not flag mock-over-method assignment.
        setattr(backend, "do_generate_walk", AsyncMock())
        setattr(backend, "_build_prompt_ids", build_ids)
        setattr(
            backend,
            "_turn_terminator",
            AsyncMock(return_value=terminator if terminator is not None else [999]),
        )
        mot, new_ctx = await backend.generate_from_chat_context(
            action, ctx, model_options=model_options
        )
    return mot, new_ctx, build_ids


async def test_token_id_turn_retains_ids_from_normalized_choice():
    """Retention reads `token_ids` off the normalized choice, proving one shape downstream.

    Retained sequence = sent prompt ids + emitted ids + turn terminator, so the next
    turn can splice this exact prefix. If normalization moved `token_ids` somewhere
    `_retained_ids` no longer reads, retention would silently stop.
    """
    mot, new_ctx, _ = await _run_token_id_turn(
        _backend(),
        ChatContext(retain_token_ids=True),
        prompt_ids=[1, 2, 3],
        terminator=[999],
        completion=_completion(token_ids=[501, 502]),
    )

    assert mot._meta["retained_token_ids"] == [1, 2, 3, 501, 502, 999]
    assert new_ctx.sent_token_ids == (1, 2, 3, 501, 502, 999)


# --- Already-computed thunk: timing stamped, post-call flagged ---------------


async def test_token_id_turn_stamps_gen_start_for_real_latency():
    """`_gen.start` is set on this path so `_elapsed_ms()` is real, not -1.

    `_generate_from_raw` never stamps it; without the path doing so, LatencyMetricsPlugin
    records a negative duration (`latency_ms=-1` -> -0.001 s) for every retained turn.
    """
    mot, _, _ = await _run_token_id_turn(_backend(), ChatContext(retain_token_ids=True))

    assert mot._gen.start is not None
    assert mot._elapsed_ms() >= 0


async def test_token_id_turn_flags_thunk_for_wrapper_post_call():
    """The computed thunk is flagged so the public wrapper fires the post-call hook.

    The path cannot fire it itself -- the wrapper assigns `generation_id` only after
    `_generate_from_context` returns, so an in-path fire would carry `None`. The flag
    defers it to where the id exists. See `_CallInfo.fire_post_call_on_return`.
    """
    mot, _, _ = await _run_token_id_turn(_backend(), ChatContext(retain_token_ids=True))

    assert mot._call.fire_post_call_on_return is True


# --- Template-input paths: kwargs must reach _build_prompt_ids ---------------


async def test_bool_thinking_reaches_prompt_ids_and_is_not_refused():
    """A boolean `THINKING` toggle is supported and travels via `chat_template_kwargs`.

    Only a STRING reasoning level is refused on this transport (it would route to
    `reasoning_effort`, which /v1/completions has no parameter for). `True` is supported
    -- it becomes `chat_template_kwargs.enable_thinking`, which must reach
    `_build_prompt_ids` so /tokenize renders the same prompt the server would.
    """
    _, _, build_ids = await _run_token_id_turn(
        _backend(),
        ChatContext(retain_token_ids=True),
        model_options={ModelOption.THINKING: True},
    )

    # third positional arg to _build_prompt_ids(ctx, conversation, chat_template_kwargs)
    ctk = build_ids.await_args.args[2]
    assert ctk is not None
    assert ctk.get("enable_thinking") is True


async def test_adapter_chat_template_kwargs_reach_prompt_ids():
    """An adapter's `chat_template_kwargs` (e.g. `adapter_name`) reach `_build_prompt_ids`.

    Template kwargs the turn is generated under alter what the chat template renders, so
    the pre-tokenized prompt must be built with the same kwargs or it stops matching the
    server's render -- corrupting the prefix cache. Passed here through model options'
    extra_body, as an activated embedded adapter would set them.
    """
    _, _, build_ids = await _run_token_id_turn(
        _backend(),
        ChatContext(retain_token_ids=True),
        model_options={
            "extra_body": {"chat_template_kwargs": {"adapter_name": "answerability"}}
        },
    )

    ctk = build_ids.await_args.args[2]
    assert ctk is not None
    assert ctk.get("adapter_name") == "answerability"


# --- Emitted-id parsing: both vLLM shapes, and every reject path -------------
#
# `_retained_ids` reads the ids the server emitted off the (normalized) choice. The
# rest of the retention machinery is only correct if this reader is: one wrong or
# missing id silently poisons every cache block after it. It is a pure function, so
# exercise every branch directly rather than through a full turn.


def _thunk_with_choice(
    choice: dict | None, *, response: object = "sentinel"
) -> ModelOutputThunk:
    """A thunk whose `raw.response` is a chat-shaped dict wrapping `choice`.

    Pass `response=<non-dict>` (or `choice=None`) to build the malformed shapes the
    reader must reject. The default `response` sentinel is only used when a caller
    overrides it to a non-dict on purpose.
    """
    mot = ModelOutputThunk("Blue.")
    if choice is not None:
        mot.raw.response = {"choices": [choice]}
    else:
        mot.raw.response = response
    return mot


def test_retained_ids_reads_plain_int_shape():
    """`return_token_ids=True` yields plain ints; they are appended verbatim.

    Result = prompt ids + emitted ids + terminator, which is the exact sequence the
    next turn splices as its prefix.
    """
    backend = _backend()
    mot = _thunk_with_choice({"token_ids": [501, 502]})
    assert backend._retained_ids([1, 2, 3], mot, [999]) == [1, 2, 3, 501, 502, 999]


def test_retained_ids_reads_token_id_string_shape():
    """`--return-tokens-as-token-ids` yields `"token_id:NNNN"` strings; parsed to ints.

    The second of the two documented vLLM id-reporting shapes. Only the plain-int
    shape was covered before, so a break in this parser would have gone unnoticed on
    any server started with the CLI flag instead of the request body flag.
    """
    backend = _backend()
    mot = _thunk_with_choice({"token_ids": ["token_id:501", "token_id:502"]})
    assert backend._retained_ids([1, 2, 3], mot, [999]) == [1, 2, 3, 501, 502, 999]


@pytest.mark.parametrize(
    ("emitted", "why"),
    [
        (["token_id:notanumber"], "non-digit suffix on the string shape"),
        (["token_id:"], "empty suffix on the string shape"),
        ([True], "bool is never a valid token id even though it subclasses int"),
        ([1.5], "float is not an int"),
        (["12"], "bare numeric string without the token_id: prefix"),
        ([None], "null entry"),
    ],
)
def test_retained_ids_rejects_unparsable_entry(emitted, why):
    """One unparsable id -> `None` for the whole turn, not a partial/guessed sequence.

    Retaining a truncated or coerced sequence would splice a prefix that no longer
    matches what the server saw, so any entry the reader cannot trust verbatim must
    abandon retention for the turn (the caller warns and still succeeds).
    """
    backend = _backend()
    mot = _thunk_with_choice({"token_ids": emitted})
    assert backend._retained_ids([1, 2, 3], mot, [999]) is None, why


@pytest.mark.parametrize(
    ("response", "why"),
    [
        (None, "no response recorded"),
        ("not-a-dict", "response is not a dict"),
        ({"choices": []}, "empty choices list"),
        ({"choices": "nope"}, "choices is not a list"),
        ({"choices": [None]}, "choice is not a dict"),
        ({"choices": [{"token_ids": None}]}, "token_ids null (return_token_ids off)"),
        ({"choices": [{"token_ids": []}]}, "token_ids empty"),
        ({"choices": [{}]}, "token_ids key absent"),
    ],
)
def test_retained_ids_returns_none_for_unreadable_response(response, why):
    """A response with no usable ids -> `None` (the vLLM < 0.10.2 / flag-off path).

    This is the branch that lets an older or misconfigured server degrade cleanly to
    a chat send instead of raising: the reader reports "no ids" and the turn proceeds.
    """
    backend = _backend()
    mot = ModelOutputThunk("Blue.")
    mot.raw.response = response
    assert backend._retained_ids([1, 2, 3], mot, [999]) is None, why


# --- End to end through mfuncs.achat -----------------------------------------


async def test_chat_through_achat_returns_message_not_keyerror():
    """`mfuncs.achat` returns a `Message` on a retaining turn instead of raising KeyError.

    `achat` computes the thunk and then asserts `isinstance(result.parsed_repr, Message)`
    (functional.py) -- the exact assertion the completions reply used to fail, because
    `Message._parse` read `["choices"][0]["message"]` on a text-shaped reply and raised
    `KeyError: "choices"`. Driving the real `achat` entry point is the end-to-end form of
    the reply-shape guarantee the unit tests above cover in isolation.
    """
    from mellea.stdlib import functional as mfuncs

    backend = _backend()
    ctx = ChatContext(retain_token_ids=True)

    build_ids = AsyncMock(return_value=[1, 2, 3])
    create = AsyncMock(return_value=_completion())
    mock_client = MagicMock()
    mock_client.completions.create = create

    with patch.object(
        OpenAIBackend,
        "_async_client",
        new_callable=PropertyMock,
        return_value=mock_client,
    ):
        setattr(backend, "do_generate_walk", AsyncMock())
        setattr(backend, "_build_prompt_ids", build_ids)
        setattr(backend, "_turn_terminator", AsyncMock(return_value=[999]))
        reply, _ = await mfuncs.achat("Name a primary color.", ctx, backend)

    assert isinstance(reply, Message)
    assert reply.content == "Blue."
