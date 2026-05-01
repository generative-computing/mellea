"""Tests for control-flow tool signalling on tool hook payloads.

Verifies that TOOL_PRE_INVOKE and TOOL_POST_INVOKE hooks always fire for all
tools (including framework-internal ones like ``final_answer``), and that the
``is_control_flow`` field is correctly populated so plugins can decide their
own policy.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

pytest.importorskip("cpex.framework")

from mellea.core.base import AbstractMelleaTool, ModelOutputThunk, ModelToolCall
from mellea.plugins import block, hook, is_internal_tool, register
from mellea.plugins.manager import shutdown_plugins
from mellea.plugins.types import HookType, PluginMode
from mellea.stdlib.functional import _acall_tools

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _RecordingTool(AbstractMelleaTool):
    """A tool that records invocations."""

    def __init__(self, name: str = "test_tool") -> None:
        self.name = name
        self.calls: list[dict[str, Any]] = []

    def run(self, **kwargs: Any) -> str:
        self.calls.append(dict(kwargs))
        return f"result from {self.name}"

    @property
    def as_json_tool(self) -> dict[str, Any]:
        return {"name": self.name, "description": "recording tool", "parameters": {}}


def _make_result(*tool_calls: ModelToolCall) -> ModelOutputThunk:
    """Wrap one or more ModelToolCalls in a minimal ModelOutputThunk."""
    mot = MagicMock(spec=ModelOutputThunk)
    mot.tool_calls = {tc.name: tc for tc in tool_calls}
    return mot


# ---------------------------------------------------------------------------
# Tests — is_internal_tool
# ---------------------------------------------------------------------------


class TestIsInternalTool:
    def test_recognizes_final_answer(self) -> None:
        assert is_internal_tool("final_answer") is True

    def test_rejects_user_tool(self) -> None:
        assert is_internal_tool("search") is False
        assert is_internal_tool("get_weather") is False


# ---------------------------------------------------------------------------
# Tests — hooks always fire, payload carries is_control_flow
# ---------------------------------------------------------------------------


class TestControlFlowPayloadField:
    async def test_pre_hook_fires_for_internal_tool(self) -> None:
        """TOOL_PRE_INVOKE fires for final_answer with is_control_flow=True."""
        tool = _RecordingTool("final_answer")
        tc = ModelToolCall(name="final_answer", func=tool, args={"answer": "42"})
        result = _make_result(tc)

        captured: list[Any] = []

        @hook(HookType.TOOL_PRE_INVOKE)
        async def spy(payload, *_):
            captured.append(payload)

        register(spy)

        await _acall_tools(result, MagicMock())

        assert len(captured) == 1
        assert captured[0].model_tool_call.name == "final_answer"
        assert captured[0].is_control_flow is True

    async def test_post_hook_fires_for_internal_tool(self) -> None:
        """TOOL_POST_INVOKE fires for final_answer with is_control_flow=True."""
        tool = _RecordingTool("final_answer")
        tc = ModelToolCall(name="final_answer", func=tool, args={"answer": "42"})
        result = _make_result(tc)

        captured: list[Any] = []

        @hook(HookType.TOOL_POST_INVOKE)
        async def spy(payload, *_):
            captured.append(payload)

        register(spy)

        await _acall_tools(result, MagicMock())

        assert len(captured) == 1
        assert captured[0].is_control_flow is True

    async def test_user_tool_has_control_flow_false(self) -> None:
        """User tools get is_control_flow=False."""
        tool = _RecordingTool("search")
        tc = ModelToolCall(name="search", func=tool, args={})
        result = _make_result(tc)

        captured: list[Any] = []

        @hook(HookType.TOOL_PRE_INVOKE)
        async def spy(payload, *_):
            captured.append(payload)

        register(spy)

        await _acall_tools(result, MagicMock())

        assert len(captured) == 1
        assert captured[0].is_control_flow is False

    async def test_mixed_batch_sets_flag_correctly(self) -> None:
        """In a batch, each tool gets the correct is_control_flow value."""
        internal_tool = _RecordingTool("final_answer")
        user_tool = _RecordingTool("search")
        tc_internal = ModelToolCall(
            name="final_answer", func=internal_tool, args={"answer": "done"}
        )
        tc_user = ModelToolCall(name="search", func=user_tool, args={})
        result = _make_result(tc_internal, tc_user)

        captured: list[Any] = []

        @hook(HookType.TOOL_PRE_INVOKE)
        async def spy(payload, *_):
            captured.append(payload)

        register(spy)

        await _acall_tools(result, MagicMock())

        assert len(captured) == 2
        by_name = {p.model_tool_call.name: p for p in captured}
        assert by_name["final_answer"].is_control_flow is True
        assert by_name["search"].is_control_flow is False


# ---------------------------------------------------------------------------
# Tests — plugin pattern: allowlist that skips control-flow tools
# ---------------------------------------------------------------------------


class TestAllowlistPluginPattern:
    async def test_allowlist_does_not_block_control_flow_tool(self) -> None:
        """An allowlist plugin using is_control_flow guard does not block final_answer."""
        allowed_tools = frozenset({"search"})

        @hook(HookType.TOOL_PRE_INVOKE, mode=PluginMode.CONCURRENT, priority=5)
        async def enforce_allowlist(payload, _):
            if payload.is_control_flow:
                return
            if payload.model_tool_call.name not in allowed_tools:
                return block(f"Tool '{payload.model_tool_call.name}' not permitted")

        register(enforce_allowlist)

        internal_tool = _RecordingTool("final_answer")
        tc = ModelToolCall(
            name="final_answer", func=internal_tool, args={"answer": "ok"}
        )
        result = _make_result(tc)

        msgs = await _acall_tools(result, MagicMock())
        assert len(msgs) == 1

    async def test_allowlist_still_blocks_unknown_user_tools(self) -> None:
        """The allowlist pattern still blocks non-allowed user tools."""
        from mellea.plugins.base import PluginViolationError

        allowed_tools = frozenset({"search"})

        @hook(HookType.TOOL_PRE_INVOKE, mode=PluginMode.CONCURRENT, priority=5)
        async def enforce_allowlist(payload, _):
            if payload.is_control_flow:
                return
            if payload.model_tool_call.name not in allowed_tools:
                return block(f"Tool '{payload.model_tool_call.name}' not permitted")

        register(enforce_allowlist)

        unknown_tool = _RecordingTool("hack_system")
        tc = ModelToolCall(name="hack_system", func=unknown_tool, args={})
        result = _make_result(tc)

        with pytest.raises(PluginViolationError, match="not permitted"):
            await _acall_tools(result, MagicMock())
