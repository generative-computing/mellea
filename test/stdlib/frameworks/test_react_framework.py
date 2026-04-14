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

    async def _generate_from_context(
        self,
        action: Component[C] | CBlock,
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

    async def generate_from_raw(
        self,
        actions: Sequence[Component[C] | CBlock],
        ctx: Context,
        *,
        format: type[BaseModelSubclass] | None = None,
        model_options: dict | None = None,
        tool_calls: bool = False,
    ) -> list[ModelOutputThunk]:
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


if __name__ == "__main__":
    pytest.main([__file__])
