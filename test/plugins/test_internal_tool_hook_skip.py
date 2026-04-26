"""Tests for skipping tool hooks on framework-internal tools (e.g. final_answer).

Verifies that TOOL_PRE_INVOKE and TOOL_POST_INVOKE hooks are bypassed for
internal tools when the skip flag is enabled (default), and that user tools
are always subject to hooks regardless of the flag.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

pytest.importorskip("cpex.framework")

from mellea.core.base import AbstractMelleaTool, ModelOutputThunk, ModelToolCall
from mellea.plugins import PluginResult, hook, register
from mellea.plugins.manager import (
    is_internal_tool,
    set_skip_hooks_for_internal_tools,
    shutdown_plugins,
    skip_hooks_for_internal_tools,
)
from mellea.plugins.types import HookType
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
# Tests — hook skip behaviour
# ---------------------------------------------------------------------------


class TestInternalToolHookSkip:
    async def test_internal_tool_skips_pre_hook(self) -> None:
        """TOOL_PRE_INVOKE does not fire for final_answer when skip is enabled."""
        tool = _RecordingTool("final_answer")
        tc = ModelToolCall(name="final_answer", func=tool, args={"answer": "42"})
        result = _make_result(tc)

        fired: list[str] = []

        @hook(HookType.TOOL_PRE_INVOKE)
        async def spy(payload, *_):
            fired.append(payload.model_tool_call.name)

        register(spy)

        msgs = await _acall_tools(result, MagicMock())

        assert fired == []
        assert len(msgs) == 1
        assert "final_answer" in msgs[0].content or "result from" in msgs[0].content

    async def test_internal_tool_skips_post_hook(self) -> None:
        """TOOL_POST_INVOKE does not fire for final_answer when skip is enabled."""
        tool = _RecordingTool("final_answer")
        tc = ModelToolCall(name="final_answer", func=tool, args={"answer": "42"})
        result = _make_result(tc)

        fired: list[str] = []

        @hook(HookType.TOOL_POST_INVOKE)
        async def spy(payload, *_):
            fired.append(payload.model_tool_call.name)

        register(spy)

        await _acall_tools(result, MagicMock())

        assert fired == []

    async def test_internal_tool_hooks_fire_when_disabled(self) -> None:
        """Hooks fire for final_answer when skip is explicitly disabled."""
        set_skip_hooks_for_internal_tools(False)

        tool = _RecordingTool("final_answer")
        tc = ModelToolCall(name="final_answer", func=tool, args={"answer": "42"})
        result = _make_result(tc)

        fired: list[str] = []

        @hook(HookType.TOOL_PRE_INVOKE)
        async def spy(payload, *_):
            fired.append(payload.model_tool_call.name)

        register(spy)

        await _acall_tools(result, MagicMock())

        assert fired == ["final_answer"]

    async def test_user_tool_always_runs_hooks(self) -> None:
        """A non-internal tool always fires hooks regardless of skip config."""
        tool = _RecordingTool("search")
        tc = ModelToolCall(name="search", func=tool, args={})
        result = _make_result(tc)

        fired: list[str] = []

        @hook(HookType.TOOL_PRE_INVOKE)
        async def spy(payload, *_):
            fired.append(payload.model_tool_call.name)

        register(spy)

        await _acall_tools(result, MagicMock())

        assert fired == ["search"]

    async def test_mixed_calls_only_skip_internal(self) -> None:
        """In a batch with both internal and user tools, only user tool triggers hooks."""
        internal_tool = _RecordingTool("final_answer")
        user_tool = _RecordingTool("search")
        tc_internal = ModelToolCall(
            name="final_answer", func=internal_tool, args={"answer": "done"}
        )
        tc_user = ModelToolCall(name="search", func=user_tool, args={})
        result = _make_result(tc_internal, tc_user)

        fired: list[str] = []

        @hook(HookType.TOOL_PRE_INVOKE)
        async def spy(payload, *_):
            fired.append(payload.model_tool_call.name)

        register(spy)

        msgs = await _acall_tools(result, MagicMock())

        assert "search" in fired
        assert "final_answer" not in fired
        assert len(msgs) == 2


# ---------------------------------------------------------------------------
# Tests — shutdown reset
# ---------------------------------------------------------------------------


class TestShutdownResetsSkipFlag:
    async def test_shutdown_resets_skip_flag(self) -> None:
        set_skip_hooks_for_internal_tools(False)
        assert skip_hooks_for_internal_tools() is False

        await shutdown_plugins()

        assert skip_hooks_for_internal_tools() is True
