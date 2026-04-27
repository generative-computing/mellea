"""Unit and integration tests for mellea.stdlib.compaction."""

from collections.abc import Sequence
from dataclasses import dataclass

import pytest

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
from mellea.stdlib.compaction import (
    ClearAll,
    KeepLastN,
    LLMSummarize,
    _find_prefix_end,
    rebuild_chat_context,
)
from mellea.stdlib.components.chat import Message
from mellea.stdlib.components.react import (
    MELLEA_FINALIZER_TOOL,
    ReactInitiator,
    _mellea_finalize_tool,
)
from mellea.stdlib.context import ChatContext
from mellea.stdlib.frameworks.react import react

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_context(components: list[Component | CBlock]) -> ChatContext:
    """Build a ChatContext from a list of components."""
    ctx = ChatContext()
    for c in components:
        ctx = ctx.add(c)
    return ctx


def _msg(role: Message.Role, content: str) -> Message:
    return Message(role=role, content=content)


# ---------------------------------------------------------------------------
# rebuild_chat_context
# ---------------------------------------------------------------------------


class TestRebuildChatContext:
    def test_empty(self):
        ctx = rebuild_chat_context([])
        assert ctx.as_list() == []

    def test_round_trip(self):
        components = [_msg("user", "hello"), _msg("assistant", "hi")]
        ctx = rebuild_chat_context(components)
        result = ctx.as_list()
        assert len(result) == 2
        assert all(isinstance(c, Message) for c in result)

    def test_preserves_window_size(self):
        ctx = rebuild_chat_context([_msg("user", "a")], window_size=3)
        assert ctx._window_size == 3


# ---------------------------------------------------------------------------
# _find_prefix_end
# ---------------------------------------------------------------------------


class TestFindPrefixEnd:
    def test_no_initiator(self):
        components = [_msg("user", "a"), _msg("assistant", "b")]
        assert _find_prefix_end(components) == 0

    def test_initiator_at_start(self):
        components = [ReactInitiator("goal", []), _msg("user", "a")]
        assert _find_prefix_end(components) == 1

    def test_initiator_after_system_msg(self):
        components = [
            _msg("system", "sys"),
            ReactInitiator("goal", []),
            _msg("user", "a"),
        ]
        assert _find_prefix_end(components) == 2


# ---------------------------------------------------------------------------
# should_compact
# ---------------------------------------------------------------------------


class TestShouldCompact:
    def test_below_threshold(self):
        ctx = _build_context([_msg("user", "a"), _msg("assistant", "b")])
        strategy = KeepLastN(keep_n=1, threshold=5)
        assert strategy.should_compact(ctx) is False

    def test_above_threshold(self):
        ctx = _build_context([_msg("user", str(i)) for i in range(10)])
        strategy = KeepLastN(keep_n=1, threshold=5)
        assert strategy.should_compact(ctx) is True

    def test_zero_threshold_never_triggers(self):
        ctx = _build_context([_msg("user", str(i)) for i in range(10)])
        strategy = KeepLastN(keep_n=1, threshold=0)
        assert strategy.should_compact(ctx) is False


# ---------------------------------------------------------------------------
# ClearAll
# ---------------------------------------------------------------------------


class TestClearAll:
    @pytest.mark.asyncio
    async def test_keeps_only_prefix(self):
        initiator = ReactInitiator("find the answer", [])
        components = [initiator, _msg("user", "a"), _msg("assistant", "b")]
        ctx = _build_context(components)

        result = await ClearAll().compact(ctx)
        result_list = result.as_list()
        assert len(result_list) == 1
        assert isinstance(result_list[0], ReactInitiator)

    @pytest.mark.asyncio
    async def test_empty_body_is_noop(self):
        initiator = ReactInitiator("goal", [])
        ctx = _build_context([initiator])

        result = await ClearAll().compact(ctx)
        assert len(result.as_list()) == 1


# ---------------------------------------------------------------------------
# KeepLastN
# ---------------------------------------------------------------------------


class TestKeepLastN:
    @pytest.mark.asyncio
    async def test_keeps_prefix_and_last_n(self):
        initiator = ReactInitiator("goal", [])
        body = [_msg("user", str(i)) for i in range(10)]
        ctx = _build_context([initiator, *body])

        result = await KeepLastN(keep_n=3).compact(ctx)
        result_list = result.as_list()
        assert len(result_list) == 4  # 1 prefix + 3 body
        assert isinstance(result_list[0], ReactInitiator)
        # Last 3 body messages
        for i, c in enumerate(result_list[1:]):
            assert isinstance(c, Message)
            assert c.content == str(7 + i)

    @pytest.mark.asyncio
    async def test_fewer_than_n_is_noop(self):
        initiator = ReactInitiator("goal", [])
        body = [_msg("user", "a"), _msg("assistant", "b")]
        ctx = _build_context([initiator, *body])

        result = await KeepLastN(keep_n=5).compact(ctx)
        # Should return original context unchanged
        assert result is ctx

    @pytest.mark.asyncio
    async def test_preserves_window_size(self):
        initiator = ReactInitiator("goal", [])
        body = [_msg("user", str(i)) for i in range(10)]
        ctx = rebuild_chat_context([initiator, *body], window_size=7)

        result = await KeepLastN(keep_n=2).compact(ctx)
        assert result._window_size == 7


# ---------------------------------------------------------------------------
# LLMSummarize
# ---------------------------------------------------------------------------


@dataclass
class _ScriptedTurn:
    """A single scripted backend response."""

    value: str
    tool_calls: dict[str, ModelToolCall] | None = None


class ScriptedBackend(Backend):
    """Fake backend returning pre-scripted responses."""

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
        raise NotImplementedError


class TestLLMSummarize:
    @pytest.mark.asyncio
    async def test_raises_without_backend(self):
        ctx = _build_context([ReactInitiator("g", []), _msg("user", "a")])
        with pytest.raises(ValueError, match="backend"):
            await LLMSummarize(keep_n=0).compact(ctx)

    @pytest.mark.asyncio
    async def test_raises_without_goal(self):
        ctx = _build_context([ReactInitiator("g", []), _msg("user", "a")])
        backend = ScriptedBackend([])
        with pytest.raises(ValueError, match="goal"):
            await LLMSummarize(keep_n=0).compact(ctx, backend=backend)

    @pytest.mark.asyncio
    async def test_summarizes_old_keeps_recent(self):
        initiator = ReactInitiator("goal", [])
        body = [_msg("user", f"msg-{i}") for i in range(6)]
        ctx = _build_context([initiator, *body])

        # The backend will return one summary when the summarization prompt is sent
        backend = ScriptedBackend([_ScriptedTurn(value="Summary of old messages")])

        result = await LLMSummarize(keep_n=2).compact(ctx, backend=backend, goal="goal")
        result_list = result.as_list()

        # prefix (1) + summary message (1) + last 2 body = 4
        assert len(result_list) == 4
        assert isinstance(result_list[0], ReactInitiator)
        # Summary message
        assert isinstance(result_list[1], Message)
        assert "[CONTEXT SUMMARY]" in result_list[1].content
        # Recent messages preserved
        assert result_list[2].content == "msg-4"
        assert result_list[3].content == "msg-5"

    @pytest.mark.asyncio
    async def test_fewer_than_n_is_noop(self):
        initiator = ReactInitiator("goal", [])
        body = [_msg("user", "a")]
        ctx = _build_context([initiator, *body])
        backend = ScriptedBackend([])

        result = await LLMSummarize(keep_n=5).compact(ctx, backend=backend, goal="goal")
        assert result is ctx


# ---------------------------------------------------------------------------
# Integration: react() with compaction
# ---------------------------------------------------------------------------


from mellea.backends.tools import MelleaTool


def _make_tool(name: str, return_value: str = "tool_result") -> MelleaTool:
    def _fn() -> str:
        return return_value

    return MelleaTool.from_callable(_fn, name=name)


def _final_answer_call(answer: str = "42") -> _ScriptedTurn:
    tool = MelleaTool.from_callable(_mellea_finalize_tool, MELLEA_FINALIZER_TOOL)
    tc = ModelToolCall(name=MELLEA_FINALIZER_TOOL, func=tool, args={"answer": answer})
    return _ScriptedTurn(value="", tool_calls={MELLEA_FINALIZER_TOOL: tc})


def _tool_call_turn(
    tool_name: str, tool: MelleaTool, thought: str = "thinking..."
) -> _ScriptedTurn:
    tc = ModelToolCall(name=tool_name, func=tool, args={})
    return _ScriptedTurn(value=thought, tool_calls={tool_name: tc})


class TestReactWithCompaction:
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_compaction_triggers_during_react(self):
        """Compaction fires when context exceeds threshold, loop still completes."""
        search = _make_tool("search", "found it")
        backend = ScriptedBackend(
            [
                _tool_call_turn("search", search, "step 1"),
                _tool_call_turn("search", search, "step 2"),
                _tool_call_turn("search", search, "step 3"),
                _final_answer_call("done"),
            ]
        )

        result, _ctx = await react(
            goal="find info",
            context=ChatContext(),
            backend=backend,
            tools=[search],
            loop_budget=10,
            compaction=KeepLastN(keep_n=3, threshold=6),
        )
        assert result.value == "done"

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_no_compaction_when_disabled(self):
        """Without compaction params, react behaves identically to before."""
        backend = ScriptedBackend([_final_answer_call("42")])
        result, _ = await react(
            goal="answer",
            context=ChatContext(),
            backend=backend,
            tools=None,
            loop_budget=5,
        )
        assert result.value == "42"
