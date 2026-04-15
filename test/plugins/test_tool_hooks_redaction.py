"""Unit tests verifying that tool hooks can redact sensitive data:

- TOOL_PRE_INVOKE: replace model_tool_call.args before the tool runs
- TOOL_POST_INVOKE: replace tool_output after the tool runs

No LLM is required — the test constructs a ModelOutputThunk directly with a
pre-built tool_calls dict and exercises _acall_tools in isolation.
"""

from __future__ import annotations

import dataclasses
from typing import Any
from unittest.mock import MagicMock

import pytest

pytest.importorskip("cpex.framework")

from mellea.core.base import AbstractMelleaTool, ModelOutputThunk, ModelToolCall
from mellea.plugins import PluginResult, hook, register
from mellea.plugins.types import HookType
from mellea.stdlib.functional import _acall_tools

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _RecordingTool(AbstractMelleaTool):
    """A tool that records the kwargs it was invoked with."""

    name = "sensitive_tool"

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def run(self, **kwargs: Any) -> str:
        self.calls.append(dict(kwargs))
        return f"ran with {kwargs}"

    @property
    def as_json_tool(self) -> dict[str, Any]:
        return {"name": self.name, "description": "recording tool", "parameters": {}}


def _make_result(tool_call: ModelToolCall) -> ModelOutputThunk:
    """Wrap a single ModelToolCall in a minimal ModelOutputThunk."""
    mot = MagicMock(spec=ModelOutputThunk)
    mot.tool_calls = {tool_call.name: tool_call}
    return mot


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestToolArgRedactionViaHook:
    """A tool_pre_invoke hook that modifies model_tool_call.args causes the
    tool to be invoked with the redacted arguments."""

    async def test_hook_redacts_sensitive_arg(self) -> None:
        """Hook replaces a sensitive arg value; the ToolMessage reflects the redacted value."""
        recording_tool = _RecordingTool()
        original_call = ModelToolCall(
            name="sensitive_tool",
            func=recording_tool,
            args={"user_input": "my secret password", "limit": 10},
        )
        result = _make_result(original_call)

        @hook(HookType.TOOL_PRE_INVOKE)
        async def redact_password(payload, *_):
            mtc = payload.model_tool_call
            redacted_args = {**mtc.args, "user_input": "[REDACTED]"}
            new_call = dataclasses.replace(mtc, args=redacted_args)
            modified = payload.model_copy(update={"model_tool_call": new_call})
            return PluginResult(continue_processing=True, modified_payload=modified)

        register(redact_password)

        tool_messages = await _acall_tools(result, MagicMock())

        assert len(tool_messages) == 1
        assert "[REDACTED]" in tool_messages[0].content
        assert "my secret password" not in tool_messages[0].content

    async def test_hook_receives_original_args_in_payload(self) -> None:
        """The hook payload contains the original (un-redacted) args."""
        recording_tool = _RecordingTool()
        original_call = ModelToolCall(
            name="sensitive_tool", func=recording_tool, args={"token": "abc123"}
        )
        result = _make_result(original_call)

        observed_args: list[dict] = []

        @hook(HookType.TOOL_PRE_INVOKE)
        async def capture_payload(payload, *_):
            observed_args.append(dict(payload.model_tool_call.args))

        register(capture_payload)

        await _acall_tools(result, MagicMock())

        assert len(observed_args) == 1
        assert observed_args[0]["token"] == "abc123"

    async def test_no_hook_modification_leaves_args_unchanged(self) -> None:
        """When the hook returns None the original args are used unchanged."""
        recording_tool = _RecordingTool()
        original_call = ModelToolCall(
            name="sensitive_tool", func=recording_tool, args={"value": "keep-me"}
        )
        result = _make_result(original_call)

        @hook(HookType.TOOL_PRE_INVOKE)
        async def observe_only(*_):
            return None  # no modification

        register(observe_only)

        await _acall_tools(result, MagicMock())

        assert recording_tool.calls[0]["value"] == "keep-me"

    async def test_hook_can_redact_multiple_fields(self) -> None:
        """A hook that redacts several fields; all redacted values appear in the output."""
        recording_tool = _RecordingTool()
        original_call = ModelToolCall(
            name="sensitive_tool",
            func=recording_tool,
            args={"ssn": "123-45-6789", "email": "user@example.com", "count": 3},
        )
        result = _make_result(original_call)

        @hook(HookType.TOOL_PRE_INVOKE)
        async def redact_pii(payload, *_):
            mtc = payload.model_tool_call
            redacted_args = {**mtc.args, "ssn": "[REDACTED]", "email": "[REDACTED]"}
            new_call = dataclasses.replace(mtc, args=redacted_args)
            modified = payload.model_copy(update={"model_tool_call": new_call})
            return PluginResult(continue_processing=True, modified_payload=modified)

        register(redact_pii)

        tool_messages = await _acall_tools(result, MagicMock())

        assert len(tool_messages) == 1
        assert "123-45-6789" not in tool_messages[0].content
        assert "user@example.com" not in tool_messages[0].content
        assert "[REDACTED]" in tool_messages[0].content


class TestToolOutputRedactionViaHook:
    """A tool_post_invoke hook that modifies tool_output causes the ToolMessage
    content to reflect the redacted value."""

    async def test_hook_redacts_sensitive_output(self) -> None:
        """Hook replaces tool_output; ToolMessage content reflects the redacted value."""
        recording_tool = _RecordingTool()
        original_call = ModelToolCall(
            name="sensitive_tool", func=recording_tool, args={"query": "user profile"}
        )
        # The tool returns a string containing a secret token.
        recording_tool.run = lambda **_: "token=supersecret,role=admin"  # type: ignore[method-assign]
        result = _make_result(original_call)

        @hook(HookType.TOOL_POST_INVOKE)
        async def redact_token(payload, *_):
            redacted = str(payload.tool_output).replace("supersecret", "[REDACTED]")
            modified = payload.model_copy(update={"tool_output": redacted})
            return PluginResult(continue_processing=True, modified_payload=modified)

        register(redact_token)

        tool_messages = await _acall_tools(result, MagicMock())

        assert len(tool_messages) == 1
        assert "supersecret" not in tool_messages[0].content
        assert "[REDACTED]" in tool_messages[0].content

    async def test_hook_receives_raw_output_in_payload(self) -> None:
        """The hook payload contains the un-redacted tool output."""
        recording_tool = _RecordingTool()
        original_call = ModelToolCall(
            name="sensitive_tool", func=recording_tool, args={"query": "data"}
        )
        recording_tool.run = lambda **_: "password=hunter2"  # type: ignore[method-assign]
        result = _make_result(original_call)

        observed_outputs: list[str] = []

        @hook(HookType.TOOL_POST_INVOKE)
        async def capture_output(payload, *_):
            observed_outputs.append(str(payload.tool_output))

        register(capture_output)

        await _acall_tools(result, MagicMock())

        assert len(observed_outputs) == 1
        assert "password=hunter2" in observed_outputs[0]

    async def test_no_hook_modification_leaves_output_unchanged(self) -> None:
        """When the hook returns None the original tool output is used unchanged."""
        recording_tool = _RecordingTool()
        original_call = ModelToolCall(
            name="sensitive_tool", func=recording_tool, args={"query": "safe"}
        )
        recording_tool.run = lambda **_: "safe result"  # type: ignore[method-assign]
        result = _make_result(original_call)

        @hook(HookType.TOOL_POST_INVOKE)
        async def observe_only(*_):
            return None

        register(observe_only)

        tool_messages = await _acall_tools(result, MagicMock())

        assert tool_messages[0].content == "safe result"
