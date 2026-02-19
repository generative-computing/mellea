"""Tests for the MelleaPlugin base class typed accessors and lifecycle."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

pytest.importorskip("mcpgateway.plugins.framework")

from mellea.plugins import hook, plugin, register
from mellea.plugins.base import MelleaPlugin, PluginResult
from mellea.plugins.manager import invoke_hook, shutdown_plugins
from mellea.plugins.types import HookType
from mellea.plugins.hooks.session import SessionPreInitPayload


@pytest.fixture(autouse=True)
async def reset_plugins():
    yield
    await shutdown_plugins()


def _make_payload() -> SessionPreInitPayload:
    return SessionPreInitPayload(
        backend_name="test", model_id="test-model", model_options=None
    )


# ---------------------------------------------------------------------------
# Typed context accessors
# ---------------------------------------------------------------------------


class TestTypedContextAccessors:
    """get_backend(), get_mellea_context(), get_session() extract from GlobalContext.state."""

    async def test_get_backend_returns_backend(self):
        """get_backend() returns the object passed as backend= to invoke_hook."""
        received_backend = []

        @plugin("accessor-test-backend")
        class AccessorPlugin:
            @hook("session_pre_init")
            async def on_pre_init(self, payload: Any, ctx: Any) -> Any:
                # Access via GlobalContext.state directly
                received_backend.append(ctx.global_context.state.get("backend"))
                return None

        mock_backend = MagicMock()
        mock_backend.model_id = "mock-backend"
        register(AccessorPlugin())
        await invoke_hook(
            HookType.SESSION_PRE_INIT, _make_payload(), backend=mock_backend
        )
        assert len(received_backend) == 1
        assert received_backend[0] is mock_backend

    async def test_get_mellea_context_returns_context(self):
        """get_mellea_context() returns the object passed as context= to invoke_hook."""
        received_context = []

        @plugin("accessor-test-context")
        class AccessorPlugin:
            @hook("session_pre_init")
            async def on_pre_init(self, payload: Any, ctx: Any) -> Any:
                received_context.append(ctx.global_context.state.get("context"))
                return None

        mock_context = MagicMock()
        register(AccessorPlugin())
        await invoke_hook(
            HookType.SESSION_PRE_INIT, _make_payload(), context=mock_context
        )
        assert len(received_context) == 1
        assert received_context[0] is mock_context

    async def test_get_session_returns_session(self):
        """get_session() returns the object passed as session= to invoke_hook."""
        received_session = []

        @plugin("accessor-test-session")
        class AccessorPlugin:
            @hook("session_pre_init")
            async def on_pre_init(self, payload: Any, ctx: Any) -> Any:
                received_session.append(ctx.global_context.state.get("session"))
                return None

        mock_session = MagicMock()
        register(AccessorPlugin())
        await invoke_hook(
            HookType.SESSION_PRE_INIT, _make_payload(), session=mock_session
        )
        assert len(received_session) == 1
        assert received_session[0] is mock_session

    async def test_backend_absent_when_not_passed(self):
        """State key 'backend' is absent when backend is not passed."""
        received = []

        @plugin("accessor-absent-backend")
        class AccessorPlugin:
            @hook("session_pre_init")
            async def on_pre_init(self, payload: Any, ctx: Any) -> Any:
                received.append("backend" in ctx.global_context.state)
                return None

        register(AccessorPlugin())
        await invoke_hook(HookType.SESSION_PRE_INIT, _make_payload())
        assert received == [False]

    async def test_session_absent_when_not_passed(self):
        """State key 'session' is absent when session is not passed."""
        received = []

        @plugin("accessor-absent-session")
        class AccessorPlugin:
            @hook("session_pre_init")
            async def on_pre_init(self, payload: Any, ctx: Any) -> Any:
                received.append("session" in ctx.global_context.state)
                return None

        register(AccessorPlugin())
        await invoke_hook(HookType.SESSION_PRE_INIT, _make_payload())
        assert received == [False]


# ---------------------------------------------------------------------------
# MelleaPlugin as context manager (MelleaPlugin subclass)
# ---------------------------------------------------------------------------


class TestMelleaPluginContextManager:
    """MelleaPlugin (non-@plugin) instances can be used as context managers.

    The test here uses @plugin-decorated class, which inherits the same
    context manager contract as MelleaPlugin (see test_scoping.py for the
    full suite; these tests focus on the accessors).
    """

    async def test_mellea_plugin_fires_in_with_block(self):
        """@plugin instance used as context manager fires its hooks."""
        invocations: list = []

        @plugin("cm-accessor-plugin")
        class CmPlugin:
            @hook("session_pre_init")
            async def on_pre_init(self, payload: Any, ctx: Any) -> Any:
                invocations.append(payload)
                return None

        p = CmPlugin()
        with p:
            await invoke_hook(HookType.SESSION_PRE_INIT, _make_payload())

        assert len(invocations) == 1

    async def test_mellea_plugin_deregistered_after_with_block(self):
        """Hooks deregister on context manager exit."""
        invocations: list = []

        @plugin("cm-deregister-plugin")
        class CmPlugin:
            @hook("session_pre_init")
            async def on_pre_init(self, payload: Any, ctx: Any) -> Any:
                invocations.append(payload)
                return None

        p = CmPlugin()
        with p:
            await invoke_hook(HookType.SESSION_PRE_INIT, _make_payload())

        await invoke_hook(HookType.SESSION_PRE_INIT, _make_payload())
        assert len(invocations) == 1  # No new invocations outside block


# ---------------------------------------------------------------------------
# PluginViolationError attributes
# ---------------------------------------------------------------------------


class TestPluginViolationError:
    """PluginViolationError carries structured information about the violation."""

    async def test_violation_error_attributes(self):
        """PluginViolationError.hook_type, .reason, .code are set from the violation."""
        from mellea.plugins import block
        from mellea.plugins.base import PluginViolationError

        @hook("session_pre_init", priority=1)
        async def blocking(payload: Any, ctx: Any) -> Any:
            return block("Too expensive", code="BUDGET_001")

        register(blocking)
        with pytest.raises(PluginViolationError) as exc_info:
            await invoke_hook(HookType.SESSION_PRE_INIT, _make_payload())

        err = exc_info.value
        assert err.hook_type == "session_pre_init"
        assert err.reason == "Too expensive"
        assert err.code == "BUDGET_001"

    async def test_violation_error_message_contains_context(self):
        """str(PluginViolationError) includes hook type and reason."""
        from mellea.plugins import block
        from mellea.plugins.base import PluginViolationError

        @hook("session_pre_init")
        async def blocking(payload: Any, ctx: Any) -> Any:
            return block("Unauthorized access", code="AUTH_403")

        register(blocking)
        with pytest.raises(PluginViolationError) as exc_info:
            await invoke_hook(HookType.SESSION_PRE_INIT, _make_payload())

        msg = str(exc_info.value)
        assert "session_pre_init" in msg
        assert "Unauthorized access" in msg
