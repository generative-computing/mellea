# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for mellea.stdlib.frameworks.react.

Uses a ScriptedBackend (fake) so that real aact() and _call_tools() run
end-to-end — only LLM inference is faked. This makes the tests robust to
internal refactors of react() while still verifying observable behaviour.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from unittest.mock import Mock

import pytest

pytestmark = [pytest.mark.integration]

import pydantic

from mellea.backends.model_options import ModelOption
from mellea.backends.tools import MelleaTool
from mellea.core.backend import Backend, BaseModelSubclass
from mellea.core.base import (
    C,
    CBlock,
    Component,
    Context,
    GenerateLog,
    ModelOutputThunk,
    ModelToolCall,
)
from mellea.stdlib.components.react import (
    MELLEA_FINALIZER_TOOL,
    ReactInitiator,
    _mellea_finalize_tool,
)
from mellea.stdlib.context import ChatContext
from mellea.stdlib.frameworks.react import react

# --- fake backend ---


@dataclass
class _ScriptedTurn:
    """A single scripted backend response."""

    value: str
    tool_calls: dict[str, ModelToolCall] | None = None


class ScriptedBackend(Backend):
    """Fake backend returning pre-scripted responses.

    Each call to _generate_from_context pops the next response from the
    script. Raises StopIteration if the script runs out (test bug).
    """

    def __init__(self, script: list[_ScriptedTurn]) -> None:
        self._script = iter(script)
        self._model_id: str = "scripted-mock"
        self._provider: str = "scripted-mock"

    async def _generate_from_context(
        self,
        action: Component[C] | CBlock | ModelOutputThunk,
        ctx: Context,
        *,
        format: type[BaseModelSubclass] | None = None,
        model_options: dict | None = None,
        tool_calls: bool = False,
    ) -> tuple[ModelOutputThunk[C], Context]:
        turn = next(self._script)
        mot: ModelOutputThunk = ModelOutputThunk(
            value=turn.value, tool_calls=turn.tool_calls
        )
        mot._generate_log = GenerateLog(is_final_result=True)
        return mot, ctx.add(action).add(mot)

    async def _generate_from_raw(
        self,
        actions: Sequence[Component[C] | CBlock],
        ctx: Context,
        *,
        format: type[BaseModelSubclass] | None = None,
        model_options: dict | None = None,
        tool_calls: bool = False,
    ) -> tuple[list[ModelOutputThunk], dict | None]:
        raise NotImplementedError("react does not use generate_from_raw")


# --- helpers ---


def _make_tool(name: str, return_value: str = "tool_result") -> MelleaTool:
    """Create a real MelleaTool that returns a fixed string."""

    def _fn() -> str:
        return return_value

    return MelleaTool.from_callable(_fn, name=name)


def _final_answer_call(answer: str = "42") -> _ScriptedTurn:
    """Script a turn where the model calls final_answer with real arg flow."""
    tool = MelleaTool.from_callable(_mellea_finalize_tool, MELLEA_FINALIZER_TOOL)
    tc = ModelToolCall(name=MELLEA_FINALIZER_TOOL, func=tool, args={"answer": answer})
    return _ScriptedTurn(value="", tool_calls={MELLEA_FINALIZER_TOOL: tc})


def _tool_call_turn(
    tool_name: str, tool: MelleaTool, thought: str = "thinking..."
) -> _ScriptedTurn:
    """Script a turn where the model calls a non-final tool."""
    tc = ModelToolCall(name=tool_name, func=tool, args={})
    return _ScriptedTurn(value=thought, tool_calls={tool_name: tc})


# --- react loop termination ---


@pytest.mark.asyncio
async def test_react_final_answer_terminates():
    """Loop terminates when model calls final_answer tool."""
    backend = ScriptedBackend([_final_answer_call("42")])
    result, _ = await react(
        goal="answer", context=ChatContext(), backend=backend, tools=None, loop_budget=5
    )
    assert result.value == "42"


@pytest.mark.asyncio
async def test_react_budget_exhaustion():
    """RuntimeError raised when budget is exhausted without final answer."""
    # Script turns with no tool calls — loop spins until budget hit
    no_tools = [_ScriptedTurn(value="thinking...") for _ in range(2)]
    backend = ScriptedBackend(no_tools)

    with pytest.raises(RuntimeError, match="could not complete react loop in 2"):
        await react(
            goal="answer",
            context=ChatContext(),
            backend=backend,
            tools=None,
            loop_budget=2,
        )


@pytest.mark.asyncio
async def test_react_non_final_tool_continues():
    """Non-finalizer tool calls don't terminate the loop."""
    search = _make_tool("search", "found it")
    backend = ScriptedBackend(
        [_tool_call_turn("search", search), _final_answer_call("done")]
    )

    result, _ = await react(
        goal="g", context=ChatContext(), backend=backend, tools=[search], loop_budget=5
    )
    assert result.value == "done"


@pytest.mark.asyncio
async def test_react_tools_from_model_options_merged():
    """Tools provided via model_options[TOOLS] are merged with explicit tools."""
    extra = _make_tool("extra_tool")
    backend = ScriptedBackend([_final_answer_call("ok")])

    _, ctx = await react(
        goal="g",
        context=ChatContext(),
        backend=backend,
        tools=[],
        model_options={ModelOption.TOOLS: [extra]},
        loop_budget=5,
    )
    # The ReactInitiator in the context should contain the merged tool
    lin = ctx.view_for_generation()
    assert lin is not None
    initiators = [c for c in lin if isinstance(c, ReactInitiator)]
    assert len(initiators) == 1
    assert extra in initiators[0].tools


@pytest.mark.asyncio
async def test_react_format_triggers_second_generation():
    """When format is set, a second generation call is made after final_answer."""
    backend = ScriptedBackend(
        [
            _final_answer_call("raw"),
            _ScriptedTurn(value="formatted"),  # second call for format
        ]
    )

    result, _ = await react(
        goal="g",
        context=ChatContext(),
        backend=backend,
        tools=None,
        format=type(
            "FakeModel", (pydantic.BaseModel,), {}
        ),  # triggers the format branch
        loop_budget=5,
    )
    # The second aact call produces the final result
    assert result.value == "formatted"


@pytest.mark.asyncio
async def test_react_final_answer_with_extra_tool_rejected():
    """final_answer alongside another tool in the same turn triggers assertion."""
    search = _make_tool("search", "found it")
    finalizer = MelleaTool.from_callable(_mellea_finalize_tool, MELLEA_FINALIZER_TOOL)
    both = {
        "search": ModelToolCall(name="search", func=search, args={}),
        MELLEA_FINALIZER_TOOL: ModelToolCall(
            name=MELLEA_FINALIZER_TOOL, func=finalizer, args={"answer": "done"}
        ),
    }
    backend = ScriptedBackend([_ScriptedTurn(value="", tool_calls=both)])

    with pytest.raises(AssertionError, match="multiple tools were called with 'final'"):
        await react(
            goal="g",
            context=ChatContext(),
            backend=backend,
            tools=[search],
            loop_budget=5,
        )


@pytest.mark.asyncio
async def test_react_rejects_non_chat_context():
    """react() requires a ChatContext instance."""
    with pytest.raises(AssertionError, match="type of chat context"):
        await react(goal="g", context=Mock(), backend=Mock(), tools=None)


# --- compaction integration ---


def test_pin_react_initiator_finds_initiator():
    from mellea.stdlib.components.chat import Message
    from mellea.stdlib.components.react import pin_react_initiator

    components = [
        Message("system", "sys"),
        ReactInitiator("solve x", []),
        Message("user", "step 1"),
    ]
    # Pinned prefix = system + initiator = first two indices.
    assert pin_react_initiator(components) == 2


def test_pin_react_initiator_returns_zero_when_absent():
    from mellea.stdlib.components.chat import Message
    from mellea.stdlib.components.react import pin_react_initiator

    components = [Message("user", "a"), Message("assistant", "b")]
    assert pin_react_initiator(components) == 0


def test_react_summary_prompt_default():
    """Without a goal the prompt has no GOAL: line and contains {conversation}."""
    from mellea.stdlib.components.react import react_summary_prompt

    prompt = react_summary_prompt()
    assert "{conversation}" in prompt
    assert "GOAL:" not in prompt
    assert "research progress" in prompt
    assert "search queries" in prompt
    assert "dead ends" in prompt


def test_react_summary_prompt_with_goal():
    """Goal is interpolated and the prompt still has the {conversation} placeholder."""
    from mellea.stdlib.components.react import react_summary_prompt

    prompt = react_summary_prompt(goal="find papers on context compaction")
    assert "GOAL: find papers on context compaction" in prompt
    assert "{conversation}" in prompt


def test_react_summary_prompt_escapes_braces_in_goal():
    """Braces in the goal must survive str.format() in LLMSummarizeCompactor."""
    from mellea.stdlib.components.react import react_summary_prompt

    prompt = react_summary_prompt(goal="solve {x: 1, y: 2}")
    # After str.format(conversation=...), the goal should appear with literal braces.
    rendered = prompt.format(conversation="<chat>")
    assert "GOAL: solve {x: 1, y: 2}" in rendered
    assert "<chat>" in rendered


def test_react_summary_prompt_works_with_llm_summarize_compactor():
    """The factory's output passes LLMSummarizeCompactor's template validation."""
    from mellea.stdlib.components.react import react_summary_prompt
    from mellea.stdlib.context import LLMSummarizeCompactor

    # Should not raise on construction (template contains {conversation}).
    # Backend value is unused in this validation-only test; any non-None object
    # satisfies the required default_backend kwarg.
    backend = object()
    LLMSummarizeCompactor(
        default_backend=backend,  # type: ignore[arg-type]
        prompt_template=react_summary_prompt(goal="g"),
    )
    LLMSummarizeCompactor(
        default_backend=backend,  # type: ignore[arg-type]
        prompt_template=react_summary_prompt(),
    )
    LLMSummarizeCompactor(
        default_backend=backend,  # type: ignore[arg-type]
        prompt_template=react_summary_prompt(goal="g", max_tokens_hint=2000),
    )


def test_react_summary_prompt_max_tokens_hint_omitted_by_default():
    """Without a hint, the prompt is byte-identical to the un-hinted form."""
    from mellea.stdlib.components.react import react_summary_prompt

    prompt = react_summary_prompt(goal="g")
    prompt_explicit_none = react_summary_prompt(goal="g", max_tokens_hint=None)
    assert prompt == prompt_explicit_none
    assert "Be at most" not in prompt
    assert "tokens (roughly" not in prompt


def test_react_summary_prompt_max_tokens_hint_injects_bullet():
    """Positive hint adds a bullet with token + word estimates."""
    from mellea.stdlib.components.react import react_summary_prompt

    prompt = react_summary_prompt(goal="g", max_tokens_hint=2000)
    # The bullet sits after "structured clearly" and before "Context to summarize:".
    assert "- Be at most ~2000 tokens (roughly 1500 words)" in prompt
    assert "Prioritize density" in prompt
    # Ordering: structured-clearly bullet comes before the length bullet,
    # length bullet comes before the conversation marker.
    sc_idx = prompt.index("structured clearly")
    bullet_idx = prompt.index("Be at most ~2000")
    conv_idx = prompt.index("Context to summarize:")
    assert sc_idx < bullet_idx < conv_idx


def test_react_summary_prompt_max_tokens_hint_zero_or_negative_omits_bullet():
    """Non-positive hint values are treated as no hint."""
    from mellea.stdlib.components.react import react_summary_prompt

    base = react_summary_prompt()
    assert react_summary_prompt(max_tokens_hint=0) == base
    assert react_summary_prompt(max_tokens_hint=-1) == base


def test_react_summary_prompt_max_tokens_hint_word_estimate_scales():
    """Word estimate uses the ~0.75 words/token heuristic (int truncation)."""
    from mellea.stdlib.components.react import react_summary_prompt

    # 1000 tokens → 750 words; 4000 → 3000.
    assert "~1000 tokens (roughly 750 words)" in react_summary_prompt(
        max_tokens_hint=1000
    )
    assert "~4000 tokens (roughly 3000 words)" in react_summary_prompt(
        max_tokens_hint=4000
    )


@pytest.mark.asyncio
async def test_react_invokes_per_turn_compactor():
    """The ``compactor=`` hook runs once per turn after the tool observation."""
    search = _make_tool("search", "found it")
    backend = ScriptedBackend(
        [
            _tool_call_turn("search", search, "step 1"),
            _tool_call_turn("search", search, "step 2"),
            _final_answer_call("done"),
        ]
    )

    calls = []

    class RecordingCompactor:
        def compact(self, ctx, *, backend=None):
            calls.append(len(ctx.as_list()))
            return ctx  # no-op compaction; we just observe

    result, _ctx = await react(
        goal="find info",
        context=ChatContext(),
        backend=backend,
        tools=[search],
        loop_budget=10,
        compactor=RecordingCompactor(),
    )

    # Two non-terminal turns each invoke the compactor; the final turn skips it.
    assert result.value == "done"
    assert len(calls) == 2
    # Per-turn context monotonically grows in this trace.
    assert calls[0] < calls[1]


@pytest.mark.asyncio
async def test_react_runs_llm_summarize_compactor():
    """LLMSummarizeCompactor.compact is sync (hides async internally), so react()
    just calls it like any other sync Compactor.
    """
    from mellea.stdlib.components.react import pin_react_initiator
    from mellea.stdlib.context import LLMSummarizeCompactor

    search = _make_tool("search", "found it")
    backend = ScriptedBackend(
        [_tool_call_turn("search", search, "step 1"), _final_answer_call("done")]
    )

    # keep_n large → no actual summarisation fires; the test verifies that
    # the sync compact() method is callable from inside the async react()
    # loop without exception.
    result, ctx = await react(
        goal="find info",
        context=ChatContext(window_size=10_000),
        backend=backend,
        tools=[search],
        loop_budget=10,
        compactor=LLMSummarizeCompactor(
            default_backend=backend, keep_n=1000, pin_predicate=pin_react_initiator
        ),
    )
    assert result.value == "done"
    assert any(isinstance(c, ReactInitiator) for c in ctx.as_list())


@pytest.mark.asyncio
async def test_react_compactor_can_actually_compact():
    """A real WindowCompactor wired in via the per-turn hook truncates context."""
    from mellea.stdlib.components.react import pin_react_initiator
    from mellea.stdlib.context import WindowCompactor

    search = _make_tool("search", "found it")
    backend = ScriptedBackend(
        [
            _tool_call_turn("search", search, "step 1"),
            _tool_call_turn("search", search, "step 2"),
            _tool_call_turn("search", search, "step 3"),
            _final_answer_call("done"),
        ]
    )

    result, ctx = await react(
        goal="find info",
        # Permissive per-add window so we isolate the per-turn compactor's effect.
        context=ChatContext(window_size=10_000),
        backend=backend,
        tools=[search],
        loop_budget=10,
        compactor=WindowCompactor(size=2, pin_predicate=pin_react_initiator),
    )

    # The ReactInitiator must survive thanks to pin_react_initiator.
    assert any(isinstance(c, ReactInitiator) for c in ctx.as_list())
    assert result.value == "done"


if __name__ == "__main__":
    pytest.main([__file__])
