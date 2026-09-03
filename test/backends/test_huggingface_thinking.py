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
from mellea.backends.huggingface import LocalHFBackend, _split_think_tags
from mellea.core.base import CBlock, ModelOutputThunk


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

    Matches the first-match-wins convention used by transformers' own serving
    utilities (cli/serving/utils.py) for the same start/end tag pattern.
    """
    thinking, answer = _split_think_tags("<think>a</think>b</think>c")
    assert thinking == "a"
    assert answer == "b</think>c"


def _make_backend(*, thinking_template_var: str | None = "think") -> LocalHFBackend:
    """Return a LocalHFBackend with __init__ bypassed, wired with the minimum
    state post_processing needs when there is no real GenerateDecoderOnlyOutput
    (i.e. every isinstance(hf_output, GenerateDecoderOnlyOutput) branch is skipped).

    Args:
        thinking_template_var: name of a thinking-related Jinja variable to bake
            into the fake chat template (gates the think-split in post_processing),
            or None for a template that does not reference any of them.
    """
    b: LocalHFBackend = LocalHFBackend.__new__(LocalHFBackend)
    b._model_id = "test-org/test-model"
    b.model_id = "test-org/test-model"
    b._provider = "huggingface"
    b._use_caches = False

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
