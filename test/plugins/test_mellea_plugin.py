"""Tests for Plugin base class typed accessors and lifecycle."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

pytest.importorskip("cpex.framework")

from mellea.plugins import Plugin, hook, register
from mellea.plugins.hooks.session import SessionPreInitPayload
from mellea.plugins.manager import invoke_hook, shutdown_plugins
from mellea.plugins.types import HookType


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


class TestGlobalContextAccessors:
    """GlobalContext.state carries lightweight ambient metadata (backend_name)."""

    async def test_backend_name_in_global_context(self) -> None:
        """backend_name is set in GlobalContext.state when backend= is passed."""
        received = []

        class AccessorPlugin(Plugin, name="accessor-test-backend-name"):
            @hook("session_pre_init")
            async def on_pre_init(self, payload: Any, ctx: Any) -> Any:
                received.append(ctx.global_context.state.get("backend_name"))
                return None

        mock_backend = MagicMock()
        mock_backend.model_id = "mock-backend"
        register(AccessorPlugin())
        await invoke_hook(
            HookType.SESSION_PRE_INIT, _make_payload(), backend=mock_backend
        )
        assert len(received) == 1
        assert received[0] == "mock-backend"

    async def test_backend_name_absent_when_not_passed(self) -> None:
        """'backend_name' is absent from state when backend is not passed."""
        received = []

        class AccessorPlugin(Plugin, name="accessor-absent-backend"):
            @hook("session_pre_init")
            async def on_pre_init(self, payload: Any, ctx: Any) -> Any:
                received.append("backend_name" in ctx.global_context.state)
                return None

        register(AccessorPlugin())
        await invoke_hook(HookType.SESSION_PRE_INIT, _make_payload())
        assert received == [False]

    async def test_full_backend_object_not_in_state(self) -> None:
        """The full backend object should NOT be stored in GlobalContext.state."""
        received = []

        class AccessorPlugin(Plugin, name="accessor-no-full-backend"):
            @hook("session_pre_init")
            async def on_pre_init(self, payload: Any, ctx: Any) -> Any:
                received.append("backend" in ctx.global_context.state)
                return None

        mock_backend = MagicMock()
        mock_backend.model_id = "test"
        register(AccessorPlugin())
        await invoke_hook(
            HookType.SESSION_PRE_INIT, _make_payload(), backend=mock_backend
        )
        assert received == [False]


# ---------------------------------------------------------------------------
# Plugin as context manager
# ---------------------------------------------------------------------------


class TestMelleaPluginContextManager:
    """Plugin subclass instances can be used as context managers."""

    async def test_mellea_plugin_fires_in_with_block(self) -> None:
        """Plugin instance used as context manager fires its hooks."""
        invocations: list = []

        class CmPlugin(Plugin, name="cm-accessor-plugin"):
            @hook("session_pre_init")
            async def on_pre_init(self, payload: Any, ctx: Any) -> Any:
                invocations.append(payload)
                return None

        p = CmPlugin()
        with p:
            await invoke_hook(HookType.SESSION_PRE_INIT, _make_payload())

        assert len(invocations) == 1

    async def test_mellea_plugin_deregistered_after_with_block(self) -> None:
        """Hooks deregister on context manager exit."""
        invocations: list = []

        class CmPlugin(Plugin, name="cm-deregister-plugin"):
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

    async def test_violation_error_attributes(self) -> None:
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

    async def test_violation_error_message_contains_context(self) -> None:
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
