# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for LocalHFBackend's <think>...</think> tag splitting.

No GPU or real model is needed — _split_think_tags is a pure string function.

torch must be importable because importing huggingface.py triggers the top-level
`import torch`.  Install mellea[hf] to satisfy this requirement.
"""

import pytest

torch = pytest.importorskip("torch", reason="torch not installed — install mellea[hf]")

import mellea.backends.huggingface as hf_backend
from mellea.backends import ModelOption
from mellea.backends.huggingface import LocalHFBackend, _split_think_tags
from mellea.core.base import CBlock, ModelOutputThunk
from test.backends.test_huggingface_filter_options import (
    _GRANITE_THINKING_MODEL_ID,
    _try_load_granite_tokenizer,
)


def test_split_think_tags_with_both_tags() -> None:
    """Both tags present: reasoning and answer are split and stripped."""
    thinking, answer = _split_think_tags("<think>reasoning here</think>the answer")
    assert thinking == "reasoning here"
    assert answer == "the answer"


def test_split_think_tags_missing_opening_tag() -> None:
    """Only the closing tag present (e.g. granite-4.2's prompt-baked opening tag)."""
    thinking, answer = _split_think_tags("reasoning here</think>the answer")
    assert thinking == "reasoning here"
    assert answer == "the answer"


def test_split_think_tags_no_closing_tag() -> None:
    """No </think> anywhere: text is returned unchanged, thinking is None."""
    thinking, answer = _split_think_tags("just a plain answer, no tags")
    assert thinking is None
    assert answer == "just a plain answer, no tags"


def test_split_think_tags_strips_surrounding_whitespace() -> None:
    """Whitespace/newlines around tags (matches granite's <think>\\n form) are stripped."""
    thinking, answer = _split_think_tags(
        "<think>\n  reasoning here  \n</think>\n  the answer  \n"
    )
    assert thinking == "reasoning here"
    assert answer == "the answer"


def test_split_think_tags_empty_reasoning() -> None:
    """Empty think block (e.g. granite's thinking-disabled <think></think> form)."""
    thinking, answer = _split_think_tags("<think></think>the answer")
    assert thinking == ""
    assert answer == "the answer"


def test_split_think_tags_leading_whitespace_before_open_tag() -> None:
    """A newline/whitespace before <think> must not defeat removeprefix.

    Regression test: reasoning.removeprefix(_THINK_OPEN_TAG) only strips the tag
    when it is the literal first character(s) of the string. Without stripping
    first, "\n<think>\n..." still starts with "\n", removeprefix is a no-op, and
    the literal "<think>" tag leaks into mot.thinking.
    """
    thinking, answer = _split_think_tags("\n<think>\nreasoning\n</think>\nanswer")
    assert thinking == "reasoning"
    assert answer == "answer"


def test_split_think_tags_multiple_close_tags_uses_first() -> None:
    """Multiple </think> occurrences: only the first is treated as the boundary.

    Deliberate: Granite's own template uses the *last* occurrence when
    truncating old reasoning out of replayed history, where dropping too much
    is the safe direction. Here we're extracting a clean answer from a fresh
    completion, where the safe direction is the opposite: if the model's
    answer itself mentions the literal text "</think>", splitting on the
    first occurrence keeps the answer intact instead of corrupting it.
    """
    thinking, answer = _split_think_tags("<think>a</think>b</think>c")
    assert thinking == "a"
    assert answer == "b</think>c"


def _make_backend(
    *, thinking_template_var: str | None = "think", use_caches: bool = False
) -> LocalHFBackend:
    """Return a LocalHFBackend with __init__ bypassed, wired with the minimum
    state post_processing needs when there is no real GenerateDecoderOnlyOutput
    (i.e. every isinstance(hf_output, GenerateDecoderOnlyOutput) branch is skipped
    unless use_caches=True and the caller also sets mot.raw.response).

    Args:
        thinking_template_var: name of a thinking-related Jinja variable to bake
            into the fake chat template (gates the think-split in post_processing),
            or None for a template that does not reference any of them.
        use_caches: whether to wire up a real `SimpleLRUCache` so the KV-cache
            branch in post_processing actually runs.
    """
    b: LocalHFBackend = LocalHFBackend.__new__(LocalHFBackend)
    b._model_id = "test-org/test-model"
    b.model_id = "test-org/test-model"
    b._provider = "huggingface"
    b._use_caches = use_caches
    if use_caches:
        from mellea.backends.cache import SimpleLRUCache

        object.__setattr__(b, "_cache", SimpleLRUCache(5))
        object.__setattr__(b, "_device", torch.device("cpu"))

    template = (
        f"{{{{ {thinking_template_var} }}}}"
        if thinking_template_var
        else "{{ messages }}"
    )

    class _FakeTokenizer:
        chat_template = template

    object.__setattr__(b, "_tokenizer", _FakeTokenizer())
    return b


async def test_post_processing_splits_thinking_before_tool_scan(monkeypatch) -> None:
    """post_processing must strip <think>...</think> into mot.thinking and pass
    only the post-split answer text to the tool-call scan, not the raw combined
    output — verifies the ordering the inline comment in post_processing promises.
    """
    recorded_text: list[str] = []

    def fake_to_tool_calls(tools, text):
        recorded_text.append(text)
        return None

    monkeypatch.setattr(hf_backend, "to_tool_calls", fake_to_tool_calls)

    backend = _make_backend()
    mot = ModelOutputThunk(
        value="<think>reasoning here</think>call get_weather(city='Boston')"
    )
    mot._call.action = CBlock("What's the weather?")
    mot._call.model_options = {}

    await backend.post_processing(
        mot,
        conversation=[],
        _format=None,
        tool_calls=True,
        tools={},
        seed=None,
        input_ids=None,
    )

    assert mot.thinking == "reasoning here"
    assert mot.value == "call get_weather(city='Boston')"
    assert recorded_text == ["call get_weather(city='Boston')"]


async def test_post_processing_skips_split_when_streaming() -> None:
    """Streaming generations must not have mot.value shrunk by post_processing.

    Regression test for a real bug: ModelOutputThunk.astream() (mellea/core/base.py)
    computes each delta from an offset captured before post_processing runs, assuming
    mot._underlying_value only ever grows during streaming. If post_processing
    replaces mot.value with the shorter post-split answer, that offset goes stale and
    the final astream() delta is corrupted (truncated or empty) — verified separately
    by simulating astream()'s delta math against this exact before/after state.
    """
    backend = _make_backend()
    mot = ModelOutputThunk(value="<think>reasoning here</think>the answer")
    mot.generation.streaming = True
    mot._call.action = CBlock("test")
    mot._call.model_options = {}

    await backend.post_processing(
        mot,
        conversation=[],
        _format=None,
        tool_calls=False,
        tools={},
        seed=None,
        input_ids=None,
    )

    assert mot.thinking is None
    assert mot.value == "<think>reasoning here</think>the answer"


async def test_post_processing_does_not_split_without_thinking_template_var() -> None:
    """A chat template with no thinking-related variable must not trigger the split.

    Regression test: without this gate, any answer that happens to contain the
    literal substring "</think>" (e.g. a question about chat templates) would have
    legitimate answer text incorrectly moved into mot.thinking.
    """
    backend = _make_backend(thinking_template_var=None)
    mot = ModelOutputThunk(value="Use the </think> tag to close a reasoning block.")
    mot._call.action = CBlock("How do reasoning tags work?")
    mot._call.model_options = {}

    await backend.post_processing(
        mot,
        conversation=[],
        _format=None,
        tool_calls=False,
        tools={},
        seed=None,
        input_ids=None,
    )

    assert mot.thinking is None
    assert mot.value == "Use the </think> tag to close a reasoning block."


async def test_post_processing_does_not_split_when_thinking_explicitly_false() -> None:
    """Regression test: a template declaring a thinking var is not proof thinking
    was requested on this specific call. With ModelOption.THINKING explicitly
    False, a literal "</think>" in the answer must survive untouched, even though
    the template declares "think" (the gate must check the resolved per-call value,
    not just whether the template mentions the variable name at all).
    """
    backend = _make_backend(thinking_template_var="think")
    mot = ModelOutputThunk(value="Use the </think> tag to close a reasoning block.")
    mot._call.action = CBlock("How do reasoning tags work?")
    mot._call.model_options = {ModelOption.THINKING: False}

    await backend.post_processing(
        mot,
        conversation=[],
        _format=None,
        tool_calls=False,
        tools={},
        seed=None,
        input_ids=None,
    )

    assert mot.thinking is None
    assert mot.value == "Use the </think> tag to close a reasoning block."


async def test_post_processing_splits_when_thinking_unset() -> None:
    """Regression test: an unset/None ModelOption.THINKING must still allow the
    split when the template declares a thinking var, since Granite and Qwen3 both
    default thinking to True in their own template source — treating "unset" as
    "off" would under-split the common case.
    """
    backend = _make_backend(thinking_template_var="think")
    mot = ModelOutputThunk(value="<think>reasoning here</think>the answer")
    mot._call.action = CBlock("test")
    mot._call.model_options = {}

    await backend.post_processing(
        mot,
        conversation=[],
        _format=None,
        tool_calls=False,
        tools={},
        seed=None,
        input_ids=None,
    )

    assert mot.thinking == "reasoning here"
    assert mot.value == "the answer"


async def test_post_processing_preserves_answer_mentioning_think_tag() -> None:
    """False-positive risk test: thinking genuinely on, and the model's answer
    itself explains what the </think> tag does. The gate correctly fires (a
    real reasoning block exists), and first-occurrence splitting keeps the
    full answer intact rather than truncating it at the second, unrelated
    occurrence — the risk jakelorocco raised in the original PR review.
    """
    backend = _make_backend(thinking_template_var="think")
    mot = ModelOutputThunk(
        value=(
            "<think>the user wants an explanation</think>"
            "The </think> tag marks the end of a reasoning block."
        )
    )
    mot._call.action = CBlock("What does the </think> tag do?")
    mot._call.model_options = {}

    await backend.post_processing(
        mot,
        conversation=[],
        _format=None,
        tool_calls=False,
        tools={},
        seed=None,
        input_ids=None,
    )

    assert mot.thinking == "the user wants an explanation"
    assert mot.value == "The </think> tag marks the end of a reasoning block."


async def test_post_processing_cache_key_findable_after_split() -> None:
    """Regression test: the LRU cache key computed in post_processing must be
    derived from mot.value *after* the split has run, so a later cache_get() call
    against the same (now-split) thunk's value can find the entry. Before the
    fix, the key was computed from the pre-split string's object identity, then
    mot.value was reassigned to a new string a few lines later — orphaning the
    cache entry under a key nothing would ever look up again.
    """
    from transformers.generation.utils import GenerateDecoderOnlyOutput

    backend = _make_backend(thinking_template_var="think", use_caches=True)
    sequences = torch.tensor([[1, 2, 3, 4]])
    hf_output = GenerateDecoderOnlyOutput(
        sequences=sequences, scores=(torch.zeros(1, 10),)
    )
    mot = ModelOutputThunk(value="<think>reasoning here</think>the answer")
    mot.raw.response = hf_output
    mot._call.action = CBlock("test")
    mot._call.model_options = {}

    await backend.post_processing(
        mot,
        conversation=[],
        _format=None,
        tool_calls=False,
        tools={},
        seed=None,
        input_ids=torch.tensor([[1, 2]]),
    )

    assert mot.thinking == "reasoning here"
    assert mot.value == "the answer"
    cached = backend.cache_get(id(mot.value))
    assert cached is not None


def _render_history(messages: list) -> str:
    """Render a message list through the real granite-4.2-3b chat template.

    Loads the template only (no GPU, no model weights) via
    `_try_load_granite_tokenizer`; skips if not locally cached.
    """
    tok = _try_load_granite_tokenizer(_GRANITE_THINKING_MODEL_ID)
    if tok is None:
        pytest.skip(
            f"{_GRANITE_THINKING_MODEL_ID} not in local HF cache — "
            "run qualitative tests first"
        )
    return tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )


@pytest.mark.integration
def test_rendered_prompt_preserves_reasoning_on_tool_call_turn() -> None:
    """On a tool-call turn, the `reasoning_content` forward in `to_chat()`
    restores reasoning to the rendered prompt against the real template
    (not a synthetic one). The plain multi-turn shape (next test) isn't
    restored the same way.
    """
    messages = [
        {"role": "user", "content": "What's the weather in Boston?"},
        {
            "role": "assistant",
            "content": "",
            "reasoning_content": "I should call the weather tool.",
            "tool_calls": [
                {
                    "id": "1",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": {"city": "Boston"},
                    },
                }
            ],
        },
        {"role": "tool", "content": "72F, sunny", "tool_call_id": "1"},
    ]
    rendered = _render_history(messages)
    assert "I should call the weather tool." in rendered
    # The empty <think></think> pair is what a dropped/truncated reasoning_content
    # would leave behind; its absence here is the direct evidence reasoning survived.
    assert "<think></think>" not in rendered


@pytest.mark.integration
def test_rendered_prompt_drops_reasoning_on_plain_multi_turn() -> None:
    """Deliberate limitation: on a plain multi-turn shape (assistant turn
    with no tool call, followed by another user turn), Granite's own
    `truncate_history_thinking` gate strips reasoning even though `to_chat()`
    attaches `reasoning_content` — the reconstructed content carries both
    tags, which is exactly what that gate matches on. Decided: keep HF
    consistent with the #1201 cross-backend consensus (replay on tool-call
    turns only) rather than extend replay to plain turns.
    """
    messages = [
        {"role": "user", "content": "What is 2 + 2?"},
        {
            "role": "assistant",
            "content": "4",
            "reasoning_content": "Two plus two equals four.",
        },
        {"role": "user", "content": "And 3 + 3?"},
    ]
    rendered = _render_history(messages)
    assert "Two plus two equals four." not in rendered


@pytest.mark.parametrize("with_tool_call", [True, False])
def test_parse_then_to_chat_round_trips_thinking_for_hf(with_tool_call: bool) -> None:
    """Seam test: HF's `Message._parse()` (chat.py) carries `.thinking` onto the
    parsed assistant Message in both branches (tool-call and plain), and that
    Message then round-trips through `to_chat()` (utils.py) as `reasoning_content`.

    Closes the gap between the e2e tests (stop at `output.thinking`) and the
    `to_chat` unit tests (hand-build `Message(..., thinking=...)` directly,
    never exercising `_parse`) — this is the only test proving the link between
    them for HF specifically.
    """
    from typing import cast

    from mellea.backends.utils import to_chat
    from mellea.core import ModelToolCall
    from mellea.formatters.template_formatter import TemplateFormatter as ChatFormatter
    from mellea.stdlib.components import Message
    from mellea.stdlib.context import ChatContext

    mot = ModelOutputThunk(value="the answer")
    mot.thinking = "reasoning trace"
    mot.raw.provider = "huggingface"
    mot.raw.response = None
    if with_tool_call:
        # Only non-None matters for _parse's branch selection; the placeholder is
        # never read as a real ModelToolCall.
        mot.tool_calls = [cast(ModelToolCall, None)]

    parsed = Message(role="assistant", content="placeholder")._parse(mot)
    assert parsed.thinking == "reasoning trace"

    ctx = ChatContext()
    ctx = ctx.add(Message("user", "hello"))
    ctx = ctx.add(parsed)
    action = Message("user", "next question")
    formatter = ChatFormatter(model_id="test")

    result = to_chat(action, ctx, formatter, system_prompt=None)
    assistant_msg = next(m for m in result if m["role"] == "assistant")
    assert assistant_msg["reasoning_content"] == "reasoning trace"
