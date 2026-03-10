"""Tests for the plugin manager."""

import pytest

from mellea.plugins.base import MelleaBasePayload
from mellea.plugins.manager import (
    ensure_plugin_manager,
    has_plugins,
    invoke_hook,
    shutdown_plugins,
)
from mellea.plugins.types import HookType, PluginMode

# These tests require the contextforge plugin framework
pytest.importorskip("cpex.framework")


@pytest.fixture(autouse=True)
async def cleanup_plugins():
    """Ensure plugin manager is shut down after each test."""
    yield
    await shutdown_plugins()


class TestNoOpGuards:
    @pytest.mark.asyncio
    async def test_invoke_hook_noop_when_no_plugins(self, request):
        """When no plugins are registered, invoke_hook returns (None, original_payload)."""
        if request.config.getoption("--with-plugins", default=False):
            pytest.skip("incompatible with --with-plugins")
        payload = MelleaBasePayload(request_id="test-123")
        result, returned_payload = await invoke_hook(HookType.SESSION_PRE_INIT, payload)
        assert result is None
        assert returned_payload is payload

    def test_has_plugins_false_by_default(self, request):
        """has_plugins() returns False when no plugins have been registered."""
        if request.config.getoption("--with-plugins", default=False):
            pytest.skip("incompatible with --with-plugins")
        # After shutdown, plugins should not be enabled
        assert not has_plugins()


class TestPluginManagerInit:
    def test_ensure_plugin_manager_creates_manager(self):
        pm = ensure_plugin_manager()
        assert pm is not None
        assert has_plugins()

    def test_ensure_plugin_manager_idempotent(self):
        pm1 = ensure_plugin_manager()
        pm2 = ensure_plugin_manager()
        # They may not be the same object due to Borg pattern, but should be functional
        assert pm1 is not None
        assert pm2 is not None


class TestHookRegistration:
    @pytest.mark.asyncio
    async def test_register_and_invoke_standalone_hook(self):
        from mellea.plugins import hook, register
        from mellea.plugins.hooks.session import SessionPreInitPayload

        invocations = []

        @hook("session_pre_init", mode=PluginMode.SEQUENTIAL, priority=10)
        async def test_hook(payload, ctx):
            invocations.append(payload)
            return None

        register(test_hook)

        payload = SessionPreInitPayload(
            backend_name="openai", model_id="gpt-4", model_options=None
        )
        _result, _returned_payload = await invoke_hook(
            HookType.SESSION_PRE_INIT, payload
        )
        assert len(invocations) == 1
        assert invocations[0].backend_name == "openai"

    @pytest.mark.asyncio
    async def test_register_class_plugin(self):
        from mellea.plugins import Plugin, hook, register
        from mellea.plugins.hooks.session import SessionPreInitPayload

        invocations = []

        class TestPlugin(Plugin, name="test-class-plugin", priority=5):
            @hook("session_pre_init")
            async def on_pre_init(self, payload, ctx):
                invocations.append(("pre_init", payload))
                return None

        register(TestPlugin())

        payload = SessionPreInitPayload(
            backend_name="openai", model_id="gpt-4", model_options=None
        )
        await invoke_hook(HookType.SESSION_PRE_INIT, payload)
        assert len(invocations) == 1
        assert invocations[0][0] == "pre_init"

    @pytest.mark.asyncio
    async def test_register_plugin_set(self):
        from mellea.plugins import PluginSet, hook, register
        from mellea.plugins.hooks.session import SessionPreInitPayload

        invocations = []

        @hook("session_pre_init")
        async def hook_a(payload, ctx):
            invocations.append("a")
            return None

        @hook("session_pre_init")
        async def hook_b(payload, ctx):
            invocations.append("b")
            return None

        ps = PluginSet("test-set", [hook_a, hook_b])
        register(ps)

        payload = SessionPreInitPayload(
            backend_name="openai", model_id="gpt-4", model_options=None
        )
        await invoke_hook(HookType.SESSION_PRE_INIT, payload)
        assert "a" in invocations
        assert "b" in invocations


class TestBlockHelper:
    def test_block_returns_plugin_result(self):
        from mellea.plugins import block

        result = block("Budget exceeded", code="BUDGET_001")
        assert result.continue_processing is False
        assert result.violation is not None
        assert result.violation.reason == "Budget exceeded"
        assert result.violation.code == "BUDGET_001"
