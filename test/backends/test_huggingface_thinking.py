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
from mellea.backends.huggingface import (
    LocalHFBackend,
    _native_parse_thinking,
    _prompt_opens_think,
    _resolve_generation_caps,
    _split_think_tags,
)
from mellea.core.base import CBlock, ModelOutputThunk
from test.backends._granite_tokenizer import (
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


def test_split_think_tags_truncated_reasoning_output_opens_think() -> None:
    """No </think>, output itself starts with <think>, and generation was cut off:
    truncated reasoning. Without the truncation signal the text is left untouched."""
    thinking, answer = _split_think_tags("\n<think>\nstill working", truncated=True)
    assert thinking == "still working"
    assert answer == ""

    thinking, answer = _split_think_tags("\n<think>\nstill working")
    assert thinking is None
    assert answer == "\n<think>\nstill working"


def test_split_think_tags_truncated_with_signal() -> None:
    """Prompt-opened think block plus a genuine truncation signal: whole text is thinking."""
    thinking, answer = _split_think_tags(
        "step one, step two, and then I",
        prompt_opens_think=lambda: True,
        truncated=True,
    )
    assert thinking == "step one, step two, and then I"
    assert answer == ""


def test_split_think_tags_plain_answer_not_truncated_without_signal() -> None:
    """A prompt-opened think block alone is not evidence of truncation: the model may
    have answered directly, and that answer must stay intact."""
    thinking, answer = _split_think_tags(
        "The answer is 408.", prompt_opens_think=lambda: True, truncated=False
    )
    assert thinking is None
    assert answer == "The answer is 408."


def test_split_think_tags_close_tag_wins_over_prompt_opens_think() -> None:
    """A complete block still splits normally when the prompt opened it."""
    thinking, answer = _split_think_tags(
        "reasoning</think>answer", prompt_opens_think=lambda: True
    )
    assert thinking == "reasoning"
    assert answer == "answer"


@pytest.mark.parametrize(
    ("text", "truncated", "expected_calls", "expected_thinking", "expected_answer"),
    [
        # No truncation signal: the callable is not consulted.
        ("The answer is 408.", False, [], None, "The answer is 408."),
        # The output opens its own think tag: short-circuits before the callable.
        (
            f"\n{hf_backend._THINK_OPEN_TAG}\nstill working",
            True,
            [],
            "still working",
            "",
        ),
        # All preconditions hold: the callable decides the split.
        ("step one, and then I", True, [1], "step one, and then I", ""),
    ],
)
def test_split_think_tags_lazy_prompt_check(
    text: str,
    truncated: bool,
    expected_calls: list[int],
    expected_thinking: str | None,
    expected_answer: str,
) -> None:
    """The prompt-open callable runs only when truncation is suspected and the
    output neither opens nor closes a think block itself."""
    calls: list[int] = []

    def _opens() -> bool:
        calls.append(1)
        return True

    thinking, answer = _split_think_tags(
        text, prompt_opens_think=_opens, truncated=truncated
    )
    assert calls == expected_calls
    assert thinking == expected_thinking
    assert answer == expected_answer


@pytest.mark.parametrize(
    ("tail", "expected"),
    [
        ("<|im_start|>assistant\n<think>\n", True),
        ("<|im_start|>assistant\n<think>", True),
        ("<|im_start|>assistant\n<think></think>", False),
        ("<|im_start|>assistant\n", False),
    ],
)
def test_prompt_opens_think(tail: str, expected: bool) -> None:
    """Only a prompt ending in an unclosed <think> counts as opened."""

    class _FakeTok:
        def decode(self, ids, skip_special_tokens=True):
            return tail

    ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9]])
    assert _prompt_opens_think(_FakeTok(), ids) is expected
    assert _prompt_opens_think(_FakeTok(), {"input_ids": ids}) is expected


def test_prompt_opens_think_without_input_ids_or_on_decode_failure() -> None:
    """Missing ids or a tokenizer that cannot decode must fall back to False."""

    class _FakeTok:
        def decode(self, ids, skip_special_tokens=True):
            return "<think>"

    class _Broken:
        def decode(self, *_a, **_k):
            raise RuntimeError("boom")

    assert _prompt_opens_think(_FakeTok(), None) is False
    assert _prompt_opens_think(_Broken(), torch.tensor([[1, 2]])) is False
    assert _prompt_opens_think(None, torch.tensor([[1, 2]])) is False
    assert _prompt_opens_think(_FakeTok(), {"not_input_ids": 1}) is False


def _make_backend(
    *,
    thinking_template_var: str | None = "think",
    use_caches: bool = False,
    generation_config=None,
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
        generation_config: real `GenerationConfig` for the fake model's
            persistent config, or None for no `_model`.
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
    if generation_config is not None:

        class _FakeModel:
            def __init__(self) -> None:
                self.generation_config = generation_config

        object.__setattr__(b, "_model", _FakeModel())
    return b


def _generation_config(**kwargs):
    """A real transformers `GenerationConfig`, lazily imported, so fakes carry
    the true attribute shape and `None` defaults for unset caps."""
    from transformers import GenerationConfig

    return GenerationConfig(**kwargs)


# transformers' built-in default cap, applied when nothing is set anywhere
# (its `max_length` default counts new tokens: generate() adds the prompt).
# Read from the library via the backend's own import so tests track the
# installed version rather than a hardcoded constant.
_DEFAULT_CAP = hf_backend.GenerationConfig._get_default_generation_params()[
    "max_length"
]


async def _post_process_fake_generation(
    *,
    value: str,
    gen_ids: tuple[int, ...] = (7, 8, 9),
    prompt_len: int = 3,
    eos_token_id: int | None = 2,
    model_options: dict | None = None,
    generation_config=None,
    prompt_tail: str | None = None,
) -> ModelOutputThunk:
    """Run post_processing over a fabricated single-sequence generation.

    The prompt decodes to `prompt_tail` (default: an open think block, as
    granite-4.2 renders), the completion tokens are `gen_ids`, and the
    tokenizer's EOS id is `eos_token_id`, so the caller controls exactly
    which finish reason the derivation should reach.
    """
    if prompt_tail is None:
        prompt_tail = f"assistant\n{hf_backend._THINK_OPEN_TAG}\n"
    backend = _make_backend(generation_config=generation_config)
    object.__setattr__(backend._tokenizer, "decode", lambda ids, **_kw: prompt_tail)
    if eos_token_id is not None:
        object.__setattr__(backend._tokenizer, "eos_token_id", eos_token_id)
    mot = ModelOutputThunk(value=value)
    mot.raw.response = _hf_output(prompt_len=prompt_len, gen_ids=gen_ids)
    mot._call.action = CBlock("test")
    mot._call.model_options = model_options if model_options is not None else {}

    await backend.post_processing(
        mot,
        conversation=[],
        _format=None,
        tool_calls=False,
        tools={},
        seed=None,
        input_ids=torch.tensor([list(range(1, prompt_len + 1))]),
    )
    return mot


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


async def test_post_processing_registers_granite_xml_tool_call() -> None:
    """A Granite 4.2 XML tool call after a thinking block becomes a ModelToolCall (#1689)."""
    from mellea.backends.tools import MelleaTool

    def get_weather(city: str) -> str:
        """Get the weather.

        Args:
            city: the city name
        """
        return f"sunny in {city}"

    backend = _make_backend()
    mot = ModelOutputThunk(
        value=(
            "<think>I should call the weather tool.</think>"
            "<tool_call>\n<function=get_weather>\n<parameter=city>\nBoston\n"
            "</parameter>\n</function>\n</tool_call>"
        )
    )
    mot._call.action = CBlock("What's the weather in Boston?")
    mot._call.model_options = {}

    await backend.post_processing(
        mot,
        conversation=[],
        _format=None,
        tool_calls=True,
        tools={"get_weather": MelleaTool.from_callable(get_weather)},
        seed=None,
        input_ids=None,
    )

    assert mot.thinking == "I should call the weather tool."
    assert mot.tool_calls is not None
    assert [(c.name, c.args) for c in mot.tool_calls] == [
        ("get_weather", {"city": "Boston"})
    ]
    assert mot.tool_calls[0].call_func() == "sunny in Boston"


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


async def test_post_processing_skips_prompt_decode_when_close_tag_present() -> None:
    """The prompt-open decode must not run when the close tag is present, even
    on a length-truncated generation (a counter, not a raising stub, because
    `_prompt_opens_think` swallows decode exceptions)."""
    decode_calls: list[int] = []

    def _record_decode(ids, **_kwargs):
        decode_calls.append(1)
        return f"assistant\n{hf_backend._THINK_OPEN_TAG}\n"

    backend = _make_backend()
    object.__setattr__(backend._tokenizer, "decode", _record_decode)
    object.__setattr__(backend._tokenizer, "eos_token_id", 2)
    mot = ModelOutputThunk(value=f"reasoning{hf_backend._THINK_CLOSE_TAG}answer")
    mot.raw.response = _hf_output(prompt_len=3, gen_ids=(7, 8, 9))
    mot._call.action = CBlock("test")
    mot._call.model_options = {ModelOption.MAX_NEW_TOKENS: 3}

    await backend.post_processing(
        mot,
        conversation=[],
        _format=None,
        tool_calls=False,
        tools={},
        seed=None,
        input_ids=torch.tensor([[1, 2, 3]]),
    )

    assert mot.generation.finish_reasons == ["length"]
    assert decode_calls == []
    assert mot.thinking == "reasoning"
    assert mot.value == "answer"


async def test_post_processing_truncated_reasoning_goes_to_thinking() -> None:
    """End-to-end through post_processing: the prompt ends inside an open <think>
    block and generation stopped on length before </think>, so mot.value must not
    carry the unfinished reasoning as if it were the answer."""
    backend = _make_backend(thinking_template_var="think")
    object.__setattr__(
        backend._tokenizer, "decode", lambda ids, **_kw: "assistant\n<think>\n"
    )
    object.__setattr__(backend._tokenizer, "eos_token_id", 2)
    mot = ModelOutputThunk(value="step one, step two, and then I")
    mot.raw.response = _hf_output(prompt_len=3, gen_ids=(7, 8, 9))
    mot._call.action = CBlock("test")
    # 3 completion tokens, last token is not EOS -> finish_reasons == ["length"].
    mot._call.model_options = {ModelOption.MAX_NEW_TOKENS: 3}

    await backend.post_processing(
        mot,
        conversation=[],
        _format=None,
        tool_calls=False,
        tools={},
        seed=None,
        input_ids=torch.tensor([[1, 2, 3]]),
    )

    assert mot.generation.finish_reasons == ["length"]
    assert mot.thinking == "step one, step two, and then I"
    assert mot.value == ""


async def test_post_processing_invokes_lazy_prompt_check_when_truncated() -> None:
    """Laziness must not become never: on a genuinely truncated generation the
    prompt-open check (and its tokenizer decode) does run, exactly once."""
    decode_calls: list[int] = []

    def _record_decode(ids, **_kwargs):
        decode_calls.append(1)
        return f"assistant\n{hf_backend._THINK_OPEN_TAG}\n"

    backend = _make_backend()
    object.__setattr__(backend._tokenizer, "decode", _record_decode)
    object.__setattr__(backend._tokenizer, "eos_token_id", 2)
    mot = ModelOutputThunk(value="step one, and then I")
    mot.raw.response = _hf_output(prompt_len=3, gen_ids=(7, 8, 9))
    mot._call.action = CBlock("test")
    # 3 completion tokens, last token is not EOS -> finish_reasons == ["length"].
    mot._call.model_options = {ModelOption.MAX_NEW_TOKENS: 3}

    await backend.post_processing(
        mot,
        conversation=[],
        _format=None,
        tool_calls=False,
        tools={},
        seed=None,
        input_ids=torch.tensor([[1, 2, 3]]),
    )

    assert mot.generation.finish_reasons == ["length"]
    assert decode_calls == [1]
    assert mot.thinking == "step one, and then I"
    assert mot.value == ""


@pytest.mark.parametrize(
    ("config", "n_generated", "expected_finish"),
    [
        # Cap from the model config's max_new_tokens, reached exactly.
        ({"max_new_tokens": 3}, 3, "length"),
        # max_length counts the prompt: 3 prompt + 3 completion == 6.
        ({"max_length": 6}, 3, "length"),
        # Nothing configured: transformers' built-in default cap applies.
        ({}, _DEFAULT_CAP, "length"),
        # Below a configured cap.
        ({"max_new_tokens": 5}, 3, None),
        # Total length below a configured max_length.
        ({"max_length": 10}, 3, None),
        # Below the built-in default cap.
        ({}, 3, None),
    ],
)
async def test_post_processing_length_stop_by_cap_source(
    config: dict, n_generated: int, expected_finish: str | None
) -> None:
    """A length stop is derived (or not) from each cap source: the model's
    `generation_config` or transformers' built-in default."""
    mot = await _post_process_fake_generation(
        value="step one, and then I",
        gen_ids=tuple(range(7, 7 + n_generated)),
        generation_config=_generation_config(**config),
    )
    assert mot.generation.finish_reasons == (
        [expected_finish] if expected_finish is not None else None
    )
    if expected_finish == "length":
        assert mot.thinking == "step one, and then I"
        assert mot.value == ""
    else:
        assert mot.thinking is None
        assert mot.value == "step one, and then I"


async def test_post_processing_eos_at_cap_reports_stop() -> None:
    """EOS reached exactly at the cap reports stop, not length."""
    mot = await _post_process_fake_generation(
        value="all done",
        eos_token_id=9,
        generation_config=_generation_config(max_new_tokens=3),
    )
    assert mot.generation.finish_reasons == ["stop"]
    assert mot.thinking is None
    assert mot.value == "all done"


async def test_post_processing_stop_string_at_cap_reports_stop() -> None:
    """A stop string landing at the cap wins over the length report."""
    mot = await _post_process_fake_generation(
        value="the answer<STOP>",
        model_options={
            ModelOption.MAX_NEW_TOKENS: 3,
            ModelOption.STOP_SEQUENCES: ["<STOP>"],
        },
    )
    assert mot.generation.finish_reasons == ["stop"]
    assert mot.value == "the answer<STOP>"


@pytest.mark.parametrize(
    ("model_options", "model_config", "expected"),
    [
        # Per-call option wins over everything.
        ({ModelOption.MAX_NEW_TOKENS: 7}, {"max_new_tokens": 3}, (7, None)),
        # Per-call generation_config kwarg shadows the model's persistent config.
        (
            {"generation_config": _generation_config(max_new_tokens=7)},
            {"max_new_tokens": 3},
            (7, None),
        ),
        # Within a config, max_new_tokens ignores max_length.
        ({}, {"max_new_tokens": 7, "max_length": 4}, (7, None)),
        # max_length alone is resolved as the total-length cap.
        ({}, {"max_length": 9}, (None, 9)),
        # A free per-call max_length kwarg overrides the config's max_length.
        ({"max_length": 9}, {"max_length": 4}, (None, 9)),
        # But not a config's max_new_tokens.
        ({"max_length": 9}, {"max_new_tokens": 4}, (4, None)),
    ],
)
def test_resolve_generation_caps(model_options, model_config, expected) -> None:
    """Cap resolution mirrors generate()'s merge: per-call kwargs over config,
    max_new_tokens over max_length."""

    class _FakeModel:
        def __init__(self) -> None:
            self.generation_config = _generation_config(**model_config)

    assert _resolve_generation_caps(model_options, _FakeModel()) == expected


def test_resolve_generation_caps_transformers_default_when_unset() -> None:
    """Nothing set anywhere: transformers' built-in default cap applies as a
    new-token cap (`max_length` stays `None`), read from the library so it
    tracks the installed version."""

    class _FakeModel:
        def __init__(self) -> None:
            self.generation_config = _generation_config()

    assert _resolve_generation_caps({}, _FakeModel()) == (_DEFAULT_CAP, None)


async def test_post_processing_prompt_opened_think_without_length_stop_stays_value() -> (
    None
):
    """Prompt opened a think block but generation ended on EOS without `</think>`:
    the model answered directly, so the answer must stay in mot.value."""
    backend = _make_backend(thinking_template_var="think")
    object.__setattr__(
        backend._tokenizer, "decode", lambda ids, **_kw: "assistant\n<think>\n"
    )
    object.__setattr__(backend._tokenizer, "eos_token_id", 9)
    mot = ModelOutputThunk(value="The answer is 408.")
    mot.raw.response = _hf_output(prompt_len=3, gen_ids=(7, 8, 9))
    mot._call.action = CBlock("test")
    mot._call.model_options = {ModelOption.MAX_NEW_TOKENS: 100}

    await backend.post_processing(
        mot,
        conversation=[],
        _format=None,
        tool_calls=False,
        tools={},
        seed=None,
        input_ids=torch.tensor([[1, 2, 3]]),
    )

    assert mot.generation.finish_reasons == ["stop"]
    assert mot.thinking is None
    assert mot.value == "The answer is 408."


async def test_post_processing_structured_output_stays_in_value() -> None:
    """Structured output (`_format` set) must never be split into thinking, even if
    the prompt opened a think block and the generation looks length-truncated."""
    from pydantic import BaseModel

    class _Answer(BaseModel):
        answer: int

    backend = _make_backend(thinking_template_var="think")
    object.__setattr__(
        backend._tokenizer, "decode", lambda ids, **_kw: "assistant\n<think>\n"
    )
    object.__setattr__(backend._tokenizer, "eos_token_id", 2)
    mot = ModelOutputThunk(value='{"answer": 408}')
    mot.raw.response = _hf_output(prompt_len=3, gen_ids=(7, 8, 9))
    mot._call.action = CBlock("test")
    mot._call.model_options = {ModelOption.MAX_NEW_TOKENS: 3}

    await backend.post_processing(
        mot,
        conversation=[],
        _format=_Answer,
        tool_calls=False,
        tools={},
        seed=None,
        input_ids=torch.tensor([[1, 2, 3]]),
    )

    assert mot.thinking is None
    assert mot.value == '{"answer": 408}'


async def test_post_processing_plain_answer_not_misfiled_when_prompt_closed() -> None:
    """A closed prompt block (thinking off: `<think></think>`) with no tags in the
    output is a normal answer and must stay in mot.value."""
    backend = _make_backend(thinking_template_var="think")
    object.__setattr__(
        backend._tokenizer, "decode", lambda ids, **_kw: "assistant\n<think></think>"
    )
    mot = ModelOutputThunk(value="just the answer")
    mot._call.action = CBlock("test")
    mot._call.model_options = {}

    await backend.post_processing(
        mot,
        conversation=[],
        _format=None,
        tool_calls=False,
        tools={},
        seed=None,
        input_ids=torch.tensor([[1, 2, 3]]),
    )

    assert mot.thinking is None
    assert mot.value == "just the answer"


def _hf_output(prompt_len: int = 3, gen_ids=(7, 8, 9)):
    from transformers.generation.utils import GenerateDecoderOnlyOutput

    seq = torch.tensor([list(range(1, prompt_len + 1)) + list(gen_ids)])
    return GenerateDecoderOnlyOutput(sequences=seq)


def test_native_parse_thinking_returns_thinking_and_content() -> None:
    """A template-driven parse yields (thinking, content), reading the generated ids
    only (prompt sliced off) and passing the prompt ids as `prefix`."""

    class _TemplateTokenizer:
        response_template: dict = {"fields": {}}

        def __init__(self):
            self.calls = []

        def parse_response(self, ids, **kwargs):
            self.calls.append((ids, kwargs))
            return {"thinking": "plan", "content": "answer"}

    tok = _TemplateTokenizer()
    prompt = torch.tensor([[1, 2, 3]])

    result = _native_parse_thinking(tok, _hf_output(), prompt)

    assert result == ("plan", "answer")
    ids, kwargs = tok.calls[0]
    assert ids.tolist() == [7, 8, 9]
    assert kwargs["prefix"].tolist() == [1, 2, 3]


def test_native_parse_thinking_accepts_reasoning_content_key() -> None:
    """Templates may name the field `reasoning_content` instead of `thinking`."""

    class _TemplateTokenizer:
        response_template: dict = {"fields": {}}

        def parse_response(self, ids, **kwargs):
            return {"reasoning_content": "plan"}

    result = _native_parse_thinking(
        _TemplateTokenizer(), _hf_output(), torch.tensor([[1, 2, 3]])
    )
    assert result == ("plan", "")


@pytest.mark.parametrize(
    "parsed",
    [
        {"role": "assistant", "content": "just an answer"},
        {"role": "assistant"},
        {"thinking": "", "content": "x"},
        {"thinking": "plan", "tool_calls": [{"type": "function"}]},
    ],
)
def test_native_parse_thinking_defers_when_nothing_to_split(parsed: dict) -> None:
    """No reasoning found, or tool calls present, returns None so the string splitter
    (and the later tool-call scan over the raw text) run exactly as before."""

    class _TemplateTokenizer:
        response_template: dict = {"fields": {}}

        def parse_response(self, ids, **kwargs):
            return parsed

    assert (
        _native_parse_thinking(
            _TemplateTokenizer(), _hf_output(), torch.tensor([[1, 2, 3]])
        )
        is None
    )


def test_native_parse_thinking_skipped_without_template_or_inputs() -> None:
    """No template, no prompt ids, or no generation output: returns None."""

    class _NoTemplate:
        pass

    class _WithTemplate:
        response_template: dict = {"fields": {}}

        def parse_response(self, ids, **kwargs):
            return {"thinking": "plan"}

    prompt = torch.tensor([[1, 2, 3]])
    assert _native_parse_thinking(_NoTemplate(), _hf_output(), prompt) is None
    assert _native_parse_thinking(None, _hf_output(), prompt) is None
    assert _native_parse_thinking(_WithTemplate(), _hf_output(), None) is None
    assert _native_parse_thinking(_WithTemplate(), None, prompt) is None
    assert (
        _native_parse_thinking(_WithTemplate(), _hf_output(), {"not_input_ids": 1})
        is None
    )


@pytest.mark.parametrize("error", [ValueError("bad"), TypeError("old API")])
def test_native_parse_thinking_falls_back_on_parse_error(error: Exception) -> None:
    """A parse error must not break generation; it returns None so string splitting runs."""

    class _TemplateTokenizer:
        response_template: dict = {"fields": {}}

        def parse_response(self, ids, **kwargs):
            raise error

    assert (
        _native_parse_thinking(
            _TemplateTokenizer(), _hf_output(), torch.tensor([[1, 2, 3]])
        )
        is None
    )


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


@pytest.mark.integration
@pytest.mark.parametrize(
    ("template_kwargs", "expected"),
    [
        ({"enable_thinking": True}, True),
        ({"enable_thinking": False}, False),
        ({}, True),
    ],
)
def test_prompt_opens_think_against_real_granite_template(
    template_kwargs: dict, expected: bool
) -> None:
    """The generation prompt tail from the real granite-4.2 template is classified
    correctly, so the decode-and-suffix check is not only verified on fake strings."""
    tok = _try_load_granite_tokenizer(_GRANITE_THINKING_MODEL_ID)
    if tok is None:
        pytest.skip(
            f"{_GRANITE_THINKING_MODEL_ID} not in local HF cache — "
            "run qualitative tests first"
        )
    prompt = tok.apply_chat_template(
        [{"role": "user", "content": "hi"}],
        tokenize=False,
        add_generation_prompt=True,
        **template_kwargs,
    )
    ids = tok(prompt, add_special_tokens=False, return_tensors="pt")["input_ids"]
    assert _prompt_opens_think(tok, ids) is expected


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
