# pytest: unit
"""Compose the ReACT loop with a sync `Compactor`.

Two integration points are available, and they're complementary:

1. **Per-add** — the `ChatContext`'s own compactor runs every time the
   ReACT loop appends a Message, ToolMessage, or thunk. This is fine
   for cheap strategies like `WindowCompactor`.
2. **Per-turn** — pass `compactor=` to ``react(...)`` to invoke a
   compactor once per ReACT iteration after the tool observation. Use
   it for heavier strategies that should fire at turn boundaries
   instead of on every component append.

In both cases use ``pin_react_initiator`` (from
``mellea.stdlib.components.react``) so the goal and tool registration
survive compaction.

This example exercises the wiring end-to-end against a fake backend so
no LLM is required.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

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
    pin_react_initiator,
)
from mellea.stdlib.context import ChatContext, WindowCompactor
from mellea.stdlib.frameworks.react import react

# --------------------------------------------------------------------------- #
# Fake backend so the example runs without an LLM                             #
# --------------------------------------------------------------------------- #


@dataclass
class _ScriptedTurn:
    value: str
    tool_calls: list[ModelToolCall] | None = None


class ScriptedBackend(Backend):
    """Returns pre-scripted responses; no real model is called."""

    def __init__(self, script: list[_ScriptedTurn]) -> None:
        self._script = iter(script)

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
    ) -> tuple[list[ModelOutputThunk], dict[str, Any] | None]:
        raise NotImplementedError


def _tool(name: str, return_value: str = "ok") -> MelleaTool:
    def _fn() -> str:
        return return_value

    return MelleaTool.from_callable(_fn, name=name)


def _tool_call(tool_name: str, tool: MelleaTool, thought: str) -> _ScriptedTurn:
    tc = ModelToolCall(name=tool_name, func=tool, args={})
    return _ScriptedTurn(value=thought, tool_calls=[tc])


def _final(answer: str) -> _ScriptedTurn:
    finalizer = MelleaTool.from_callable(_mellea_finalize_tool, MELLEA_FINALIZER_TOOL)
    tc = ModelToolCall(
        name=MELLEA_FINALIZER_TOOL, func=finalizer, args={"answer": answer}
    )
    return _ScriptedTurn(value="", tool_calls=[tc])


# --------------------------------------------------------------------------- #
# Pattern A — per-add compaction wired into the ChatContext                   #
# --------------------------------------------------------------------------- #


async def per_add_compaction():
    """A `WindowCompactor(pin_react_initiator)` on the ChatContext compacts
    on every ``add()`` — Messages, ToolMessages, thunks. The ReactInitiator
    stays pinned across the whole loop.
    """
    search = _tool("search")
    backend = ScriptedBackend(
        [
            _tool_call("search", search, "step 1"),
            _tool_call("search", search, "step 2"),
            _tool_call("search", search, "step 3"),
            _final("done"),
        ]
    )
    ctx = ChatContext(
        compactor=WindowCompactor(size=3, pin_predicate=pin_react_initiator)
    )
    result, ctx = await react(
        goal="find info", context=ctx, backend=backend, tools=[search], loop_budget=10
    )
    return (
        result.value,
        any(isinstance(c, ReactInitiator) for c in ctx.as_list()),
        len(ctx.as_list()),
    )


# --------------------------------------------------------------------------- #
# Pattern B — per-turn compaction passed to react()                           #
# --------------------------------------------------------------------------- #


async def per_turn_compaction():
    """Pass ``compactor=`` to ``react`` for once-per-turn invocation.

    Use a permissive ``ChatContext`` (large window) so the per-add path is
    effectively disabled — only the per-turn hook drives compaction.
    """
    search = _tool("search")
    backend = ScriptedBackend(
        [
            _tool_call("search", search, "step 1"),
            _tool_call("search", search, "step 2"),
            _tool_call("search", search, "step 3"),
            _final("done"),
        ]
    )
    result, ctx = await react(
        goal="find info",
        context=ChatContext(window_size=10_000),
        backend=backend,
        tools=[search],
        loop_budget=10,
        compactor=WindowCompactor(size=2, pin_predicate=pin_react_initiator),
    )
    return (result.value, any(isinstance(c, ReactInitiator) for c in ctx.as_list()))


# --------------------------------------------------------------------------- #
# Pattern C — LLM-driven summarisation                                        #
# --------------------------------------------------------------------------- #


async def llm_summarize_compaction():
    """Wire :class:`LLMSummarizeCompactor` into ``react()``.

    ``LLMSummarizeCompactor`` implements the sync :class:`Compactor`
    protocol — its ``compact`` method internally orchestrates the async
    backend call (running it on a worker thread when invoked from inside
    an event loop). From ``react()``'s perspective it's just another
    sync compactor.

    To keep the scripted backend simple, this example sets ``keep_n``
    large enough that summarisation never fires (no LLM call is needed).
    Real usage would pair it with ``ThresholdCompactor`` so it only
    activates once the conversation crosses a token budget. See
    ``TestLLMSummarizeCompactor`` in ``test/stdlib/test_compactor.py`` for
    unit tests that exercise the actual summary path.
    """
    from mellea.stdlib.context import LLMSummarizeCompactor

    search = _tool("search")
    backend = ScriptedBackend([_tool_call("search", search, "step 1"), _final("done")])
    result, ctx = await react(
        goal="find info",
        context=ChatContext(window_size=10_000),
        backend=backend,
        tools=[search],
        loop_budget=10,
        # keep_n=1000 → no summarisation triggers in this short script;
        # the example just shows the async compactor is wired correctly.
        compactor=LLMSummarizeCompactor(
            default_backend=backend, keep_n=1000, pin_predicate=pin_react_initiator
        ),
    )
    return (result.value, any(isinstance(c, ReactInitiator) for c in ctx.as_list()))


if __name__ == "__main__":
    print(f"per_add_compaction:    {asyncio.run(per_add_compaction())}")
    print(f"per_turn_compaction:   {asyncio.run(per_turn_compaction())}")
    print(f"llm_summarize_compact: {asyncio.run(llm_summarize_compaction())}")


def test_per_add_compaction():
    answer, has_initiator, _length = asyncio.run(per_add_compaction())
    assert answer == "done"
    assert has_initiator


def test_per_turn_compaction():
    answer, has_initiator = asyncio.run(per_turn_compaction())
    assert answer == "done"
    assert has_initiator


def test_llm_summarize_compaction():
    answer, has_initiator = asyncio.run(llm_summarize_compaction())
    assert answer == "done"
    assert has_initiator
