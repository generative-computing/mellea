"""Tests for blocking behavior and PluginViolationError.

Covers:
- ``block()`` helper: return shape and field population
- ``invoke_hook`` raising ``PluginViolationError`` when a plugin blocks (default behavior)
- Calling the underlying ``PluginManager.invoke_hook`` with ``violations_as_exceptions=False``
  to inspect the raw result without raising
- Priority ordering: a blocking plugin at priority=1 stops downstream plugins at priority=100
"""

from __future__ import annotations

import pytest

pytest.importorskip("mcpgateway.plugins.framework")

from mellea.plugins import block, hook, register
from mellea.plugins.base import PluginViolationError
from mellea.plugins.context import build_global_context
from mellea.plugins.hooks.session import SessionPreInitPayload
from mellea.plugins.manager import _ensure_plugin_manager, invoke_hook, shutdown_plugins
from mellea.plugins.types import HookType


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _payload(**kwargs) -> SessionPreInitPayload:
    defaults: dict = dict(backend_name="test-backend", model_id="test-model")
    defaults.update(kwargs)
    return SessionPreInitPayload(**defaults)


async def _invoke_no_raise(payload: SessionPreInitPayload):
    """Call the underlying ContextForge PluginManager directly with violations_as_exceptions=False.

    Mellea's ``invoke_hook`` wrapper always raises ``PluginViolationError`` on a block.
    To observe the raw ``PluginResult`` without raising, we bypass the wrapper and call
    ``PluginManager.invoke_hook`` directly.
    """
    pm = _ensure_plugin_manager()
    ctx = build_global_context(request_id="test-req")
    # Stamp the hook field the same way Mellea's wrapper would
    stamped = payload.model_copy(update={"hook": HookType.SESSION_PRE_INIT.value})
    result, _ = await pm.invoke_hook(
        hook_type=HookType.SESSION_PRE_INIT.value,
        payload=stamped,
        global_context=ctx,
        violations_as_exceptions=False,
    )
    return result


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
async def reset_plugins():
    """Shut down and reset the plugin manager after every test."""
    yield
    await shutdown_plugins()


# ---------------------------------------------------------------------------
# TestBlockHelper
# ---------------------------------------------------------------------------


class TestBlockHelper:
    """Unit tests for the ``block()`` convenience helper."""

    def test_block_basic_returns_non_continue_plugin_result(self):
        result = block("something went wrong")

        assert result.continue_processing is False
        assert result.violation is not None
        assert result.violation.reason == "something went wrong"

    def test_block_with_all_args_populates_violation_fields(self):
        extra = {"limit": 100, "used": 150}
        result = block(
            "Budget exceeded",
            code="BUDGET_001",
            description="The token budget for this request was exceeded.",
            details=extra,
        )

        assert result.continue_processing is False
        v = result.violation
        assert v.reason == "Budget exceeded"
        assert v.code == "BUDGET_001"
        assert v.description == "The token budget for this request was exceeded."
        assert v.details == extra

    def test_block_without_code_defaults_to_empty_string(self):
        result = block("No code provided")

        assert result.violation.code == ""


# ---------------------------------------------------------------------------
# TestViolationError
# ---------------------------------------------------------------------------


class TestViolationError:
    """Tests for PluginViolationError raised by Mellea's invoke_hook when a plugin blocks."""

    async def test_blocking_plugin_raises_plugin_violation_error_by_default(self):
        @hook("session_pre_init", priority=10)
        async def blocking_plugin(payload, ctx):
            return block("Access denied", code="AUTH_001")

        register(blocking_plugin)

        with pytest.raises(PluginViolationError):
            await invoke_hook(HookType.SESSION_PRE_INIT, _payload())

    async def test_violation_error_hook_type_matches_hook(self):
        @hook("session_pre_init", priority=10)
        async def blocking_plugin(payload, ctx):
            return block("blocked")

        register(blocking_plugin)

        with pytest.raises(PluginViolationError) as exc_info:
            await invoke_hook(HookType.SESSION_PRE_INIT, _payload())

        assert exc_info.value.hook_type == HookType.SESSION_PRE_INIT.value

    async def test_violation_error_reason_matches_block_reason(self):
        @hook("session_pre_init", priority=10)
        async def blocking_plugin(payload, ctx):
            return block("rate limit exceeded", code="RATE_001")

        register(blocking_plugin)

        with pytest.raises(PluginViolationError) as exc_info:
            await invoke_hook(HookType.SESSION_PRE_INIT, _payload())

        assert exc_info.value.reason == "rate limit exceeded"

    async def test_violation_error_str_contains_hook_type_and_reason(self):
        @hook("session_pre_init", priority=10)
        async def blocking_plugin(payload, ctx):
            return block("forbidden content detected")

        register(blocking_plugin)

        with pytest.raises(PluginViolationError) as exc_info:
            await invoke_hook(HookType.SESSION_PRE_INIT, _payload())

        error_str = str(exc_info.value)
        assert HookType.SESSION_PRE_INIT.value in error_str
        assert "forbidden content detected" in error_str

    async def test_violations_as_exceptions_false_returns_result_without_raising(self):
        @hook("session_pre_init", priority=10)
        async def blocking_plugin(payload, ctx):
            return block("blocked without raise", code="NO_RAISE_001")

        register(blocking_plugin)

        # Using the raw PluginManager call with violations_as_exceptions=False
        result = await _invoke_no_raise(_payload())

        assert result is not None
        assert result.continue_processing is False
        assert result.violation is not None
        assert result.violation.reason == "blocked without raise"


# ---------------------------------------------------------------------------
# TestBlockingPreventsDownstreamPlugins
# ---------------------------------------------------------------------------


class TestBlockingPreventsDownstreamPlugins:
    """A blocking plugin at lower priority must prevent higher-priority plugins from running."""

    async def test_blocker_at_priority_1_prevents_plugin_at_priority_100(self):
        downstream_fired: list[bool] = []

        @hook("session_pre_init", priority=1)
        async def early_blocker(payload, ctx):
            return block("blocked early", code="EARLY_BLOCK")

        @hook("session_pre_init", priority=100)
        async def late_recorder(payload, ctx):
            downstream_fired.append(True)
            return None

        register(early_blocker)
        register(late_recorder)

        # Use violations_as_exceptions=False so we can inspect the result
        result = await _invoke_no_raise(_payload())

        assert result.continue_processing is False
        # The downstream plugin must never have fired
        assert downstream_fired == []
