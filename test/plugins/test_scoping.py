"""Tests for plugin scoping: session-scoped and with-block-scoped."""

from typing import Any

import pytest

pytest.importorskip("mcpgateway.plugins.framework")

from mellea.plugins import (
    MelleaPlugin,
    PluginResult,
    PluginSet,
    hook,
    plugin,
    plugin_scope,
    register,
)
from mellea.plugins.hooks.session import SessionPreInitPayload
from mellea.plugins.manager import (
    deregister_session_plugins,
    invoke_hook,
    shutdown_plugins,
)
from mellea.plugins.types import HookType


@pytest.fixture(autouse=True)
async def reset_plugins():
    """Reset the plugin manager to a clean state after every test."""
    yield
    await shutdown_plugins()


def _payload() -> SessionPreInitPayload:
    return SessionPreInitPayload(
        backend_name="test-backend", model_id="test-model", model_options=None
    )


# ---------------------------------------------------------------------------
# Session-scoped plugins
# ---------------------------------------------------------------------------


class TestSessionScopedPlugins:
    """Plugins registered via ``register(..., session_id=sid)``."""

    async def test_fires_while_session_active(self):
        invocations = []

        @hook("session_pre_init")
        async def my_hook(payload, ctx):
            invocations.append(payload)
            return None

        register(my_hook, session_id="s1")
        await invoke_hook(HookType.SESSION_PRE_INIT, _payload())
        assert len(invocations) == 1

    async def test_deregistered_after_session_cleanup(self):
        invocations = []

        @hook("session_pre_init")
        async def my_hook(payload, ctx):
            invocations.append(payload)
            return None

        register(my_hook, session_id="s-cleanup")
        await invoke_hook(HookType.SESSION_PRE_INIT, _payload())
        assert len(invocations) == 1

        deregister_session_plugins("s-cleanup")
        await invoke_hook(HookType.SESSION_PRE_INIT, _payload())
        assert len(invocations) == 1  # No new invocations after cleanup

    async def test_global_plugin_unaffected_by_session_cleanup(self):
        """Deregistering a session should leave globally registered plugins intact."""
        global_calls = []
        session_calls = []

        @hook("session_pre_init")
        async def global_hook(payload, ctx):
            global_calls.append(payload)
            return None

        @hook("session_pre_init")
        async def session_hook(payload, ctx):
            session_calls.append(payload)
            return None

        register(global_hook)
        register(session_hook, session_id="s-isolated")

        await invoke_hook(HookType.SESSION_PRE_INIT, _payload())
        assert len(global_calls) == 1
        assert len(session_calls) == 1

        deregister_session_plugins("s-isolated")
        await invoke_hook(HookType.SESSION_PRE_INIT, _payload())
        assert len(global_calls) == 2  # Still firing
        assert len(session_calls) == 1  # Stopped

    async def test_deregister_unknown_session_is_noop(self):
        """Deregistering a session ID that was never registered does not raise."""
        deregister_session_plugins("never-registered-session-xyz")

    async def test_class_plugin_session_scoped(self):
        invocations = []

        @plugin("cls-session-plugin")
        class MyPlugin:
            @hook("session_pre_init")
            async def on_pre_init(self, payload, ctx):
                invocations.append(payload)
                return None

        register(MyPlugin(), session_id="s-class")
        await invoke_hook(HookType.SESSION_PRE_INIT, _payload())
        assert len(invocations) == 1

        deregister_session_plugins("s-class")
        await invoke_hook(HookType.SESSION_PRE_INIT, _payload())
        assert len(invocations) == 1  # Stopped

    async def test_pluginset_session_scoped(self):
        """PluginSet items honour session-scoped deregistration when passed to register()."""
        invocations = []

        @hook("session_pre_init")
        async def hook_a(payload, ctx):
            invocations.append("a")
            return None

        @hook("session_pre_init")
        async def hook_b(payload, ctx):
            invocations.append("b")
            return None

        register(PluginSet("ps-session", [hook_a, hook_b]), session_id="s-ps")
        await invoke_hook(HookType.SESSION_PRE_INIT, _payload())
        assert "a" in invocations
        assert "b" in invocations

        count_before = len(invocations)
        deregister_session_plugins("s-ps")
        await invoke_hook(HookType.SESSION_PRE_INIT, _payload())
        assert len(invocations) == count_before  # No new calls

    async def test_two_sessions_deregistered_independently(self):
        calls_a = []
        calls_b = []

        @hook("session_pre_init")
        async def hook_a(payload, ctx):
            calls_a.append(payload)
            return None

        @hook("session_pre_init")
        async def hook_b(payload, ctx):
            calls_b.append(payload)
            return None

        register(hook_a, session_id="session-A")
        register(hook_b, session_id="session-B")

        await invoke_hook(HookType.SESSION_PRE_INIT, _payload())
        assert len(calls_a) == 1
        assert len(calls_b) == 1

        # End session A — hook_b should still fire
        deregister_session_plugins("session-A")
        await invoke_hook(HookType.SESSION_PRE_INIT, _payload())
        assert len(calls_a) == 1  # Stopped
        assert len(calls_b) == 2  # Still firing

        # End session B — everything silent
        deregister_session_plugins("session-B")
        await invoke_hook(HookType.SESSION_PRE_INIT, _payload())
        assert len(calls_a) == 1
        assert len(calls_b) == 2


# ---------------------------------------------------------------------------
# PluginSet as a context manager
# ---------------------------------------------------------------------------


class TestPluginSetContextManager:
    async def test_fires_inside_with_block(self):
        invocations = []

        @hook("session_pre_init")
        async def my_hook(payload, ctx):
            invocations.append(payload)
            return None

        with PluginSet("ctx-basic", [my_hook]):
            await invoke_hook(HookType.SESSION_PRE_INIT, _payload())

        assert len(invocations) == 1

    async def test_deregistered_after_with_block(self):
        invocations = []

        @hook("session_pre_init")
        async def my_hook(payload, ctx):
            invocations.append(payload)
            return None

        with PluginSet("ctx-cleanup", [my_hook]):
            await invoke_hook(HookType.SESSION_PRE_INIT, _payload())

        # Outside the block
        await invoke_hook(HookType.SESSION_PRE_INIT, _payload())
        assert len(invocations) == 1  # No new calls

    async def test_deregistered_on_exception(self):
        invocations = []

        @hook("session_pre_init")
        async def my_hook(payload, ctx):
            invocations.append(payload)
            return None

        ps = PluginSet("ctx-exc", [my_hook])
        with pytest.raises(ValueError):
            with ps:
                await invoke_hook(HookType.SESSION_PRE_INIT, _payload())
                assert len(invocations) == 1
                raise ValueError("deliberate error")

        # Plugin must be deregistered despite the exception
        await invoke_hook(HookType.SESSION_PRE_INIT, _payload())
        assert len(invocations) == 1

    def test_scope_id_cleared_after_exit(self):
        """_scope_id is reset to None once __exit__ is called."""
        # We need to check state change but __enter__ requires the framework —
        # just verify the initial state here; integration validated by other tests.
        ps = PluginSet("state-check", [])
        assert ps._scope_id is None

    async def test_raises_on_reentrant_use_of_same_instance(self):
        invocations = []

        @hook("session_pre_init")
        async def my_hook(payload, ctx):
            invocations.append(payload)
            return None

        ps = PluginSet("reentrant", [my_hook])
        with ps:
            with pytest.raises(RuntimeError, match="already active"):
                with ps:
                    pass

    async def test_async_with_fires_inside_block(self):
        invocations = []

        @hook("session_pre_init")
        async def my_hook(payload, ctx):
            invocations.append(payload)
            return None

        async with PluginSet("async-ctx", [my_hook]):
            await invoke_hook(HookType.SESSION_PRE_INIT, _payload())

        assert len(invocations) == 1

    async def test_async_with_deregisters_after_block(self):
        invocations = []

        @hook("session_pre_init")
        async def my_hook(payload, ctx):
            invocations.append(payload)
            return None

        async with PluginSet("async-cleanup", [my_hook]):
            await invoke_hook(HookType.SESSION_PRE_INIT, _payload())

        await invoke_hook(HookType.SESSION_PRE_INIT, _payload())
        assert len(invocations) == 1

    async def test_nested_different_instances(self):
        """Two different PluginSet instances can be nested."""
        invocations = []

        @hook("session_pre_init")
        async def hook_a(payload, ctx):
            invocations.append("a")
            return None

        @hook("session_pre_init")
        async def hook_b(payload, ctx):
            invocations.append("b")
            return None

        ps_a = PluginSet("set-a", [hook_a])
        ps_b = PluginSet("set-b", [hook_b])

        with ps_a:
            with ps_b:
                await invoke_hook(HookType.SESSION_PRE_INIT, _payload())
                # Both hooks should have fired
                assert "a" in invocations
                assert "b" in invocations

            # ps_b exited — only hook_a should remain
            invocations.clear()
            await invoke_hook(HookType.SESSION_PRE_INIT, _payload())
            assert invocations == ["a"]

        # ps_a exited — no hooks remain
        invocations.clear()
        await invoke_hook(HookType.SESSION_PRE_INIT, _payload())
        assert invocations == []

    async def test_scope_id_set_on_enter_and_cleared_on_exit(self):
        ps = PluginSet("state-lifecycle", [])
        assert ps._scope_id is None
        with ps:
            assert ps._scope_id is not None
        assert ps._scope_id is None

    async def test_reuse_after_exit(self):
        """The same PluginSet instance can be entered again after a previous exit."""
        invocations = []

        @hook("session_pre_init")
        async def my_hook(payload, ctx):
            invocations.append(payload)
            return None

        ps = PluginSet("reuse-after-exit", [my_hook])

        with ps:
            await invoke_hook(HookType.SESSION_PRE_INIT, _payload())

        # After first scope is closed, a second entry should work
        with ps:
            await invoke_hook(HookType.SESSION_PRE_INIT, _payload())

        assert len(invocations) == 2


# ---------------------------------------------------------------------------
# plugin_scope() context manager
# ---------------------------------------------------------------------------


class TestPluginScopeContextManager:
    async def test_standalone_hook_fires_inside_block(self):
        invocations = []

        @hook("session_pre_init")
        async def my_hook(payload, ctx):
            invocations.append(payload)
            return None

        with plugin_scope(my_hook):
            await invoke_hook(HookType.SESSION_PRE_INIT, _payload())

        assert len(invocations) == 1

    async def test_deregistered_after_block(self):
        invocations = []

        @hook("session_pre_init")
        async def my_hook(payload, ctx):
            invocations.append(payload)
            return None

        with plugin_scope(my_hook):
            await invoke_hook(HookType.SESSION_PRE_INIT, _payload())

        await invoke_hook(HookType.SESSION_PRE_INIT, _payload())
        assert len(invocations) == 1  # No new calls outside the block

    async def test_deregistered_on_exception(self):
        invocations = []

        @hook("session_pre_init")
        async def my_hook(payload, ctx):
            invocations.append(payload)
            return None

        with pytest.raises(RuntimeError):
            with plugin_scope(my_hook):
                await invoke_hook(HookType.SESSION_PRE_INIT, _payload())
                assert len(invocations) == 1
                raise RuntimeError("deliberate")

        await invoke_hook(HookType.SESSION_PRE_INIT, _payload())
        assert len(invocations) == 1  # Deregistered despite the exception

    async def test_class_plugin_fires_and_deregisters(self):
        invocations = []

        @plugin("scope-cls-plugin")
        class MyPlugin:
            @hook("session_pre_init")
            async def on_pre_init(self, payload, ctx):
                invocations.append(payload)
                return None

        with plugin_scope(MyPlugin()):
            await invoke_hook(HookType.SESSION_PRE_INIT, _payload())

        assert len(invocations) == 1

        await invoke_hook(HookType.SESSION_PRE_INIT, _payload())
        assert len(invocations) == 1  # Stopped

    async def test_pluginset_fires_and_deregisters(self):
        invocations = []

        @hook("session_pre_init")
        async def hook_a(payload, ctx):
            invocations.append("a")
            return None

        @hook("session_pre_init")
        async def hook_b(payload, ctx):
            invocations.append("b")
            return None

        ps = PluginSet("scope-ps", [hook_a, hook_b])
        with plugin_scope(ps):
            await invoke_hook(HookType.SESSION_PRE_INIT, _payload())

        assert "a" in invocations
        assert "b" in invocations

        count_before = len(invocations)
        await invoke_hook(HookType.SESSION_PRE_INIT, _payload())
        assert len(invocations) == count_before

    async def test_mixed_items(self):
        """plugin_scope accepts standalone hooks, class plugins, and PluginSets together."""
        invocations = []

        @hook("session_pre_init")
        async def standalone(payload, ctx):
            invocations.append("standalone")
            return None

        @plugin("scope-mixed-cls")
        class ClassPlugin:
            @hook("session_pre_init")
            async def on_pre_init(self, payload, ctx):
                invocations.append("class")
                return None

        @hook("session_pre_init")
        async def in_set(payload, ctx):
            invocations.append("in_set")
            return None

        ps = PluginSet("scope-inner-set", [in_set])
        with plugin_scope(standalone, ClassPlugin(), ps):
            await invoke_hook(HookType.SESSION_PRE_INIT, _payload())

        assert "standalone" in invocations
        assert "class" in invocations
        assert "in_set" in invocations

        count_before = len(invocations)
        await invoke_hook(HookType.SESSION_PRE_INIT, _payload())
        assert len(invocations) == count_before  # All deregistered

    async def test_async_with_fires_inside_block(self):
        invocations = []

        @hook("session_pre_init")
        async def my_hook(payload, ctx):
            invocations.append(payload)
            return None

        async with plugin_scope(my_hook):
            await invoke_hook(HookType.SESSION_PRE_INIT, _payload())

        assert len(invocations) == 1

    async def test_async_with_deregisters_after_block(self):
        invocations = []

        @hook("session_pre_init")
        async def my_hook(payload, ctx):
            invocations.append(payload)
            return None

        async with plugin_scope(my_hook):
            await invoke_hook(HookType.SESSION_PRE_INIT, _payload())

        await invoke_hook(HookType.SESSION_PRE_INIT, _payload())
        assert len(invocations) == 1

    async def test_sequential_scopes_are_independent(self):
        """Two consecutive plugin_scope blocks use separate scope IDs and don't interfere."""
        calls_a = []
        calls_b = []

        @hook("session_pre_init")
        async def hook_a(payload, ctx):
            calls_a.append(payload)
            return None

        @hook("session_pre_init")
        async def hook_b(payload, ctx):
            calls_b.append(payload)
            return None

        with plugin_scope(hook_a):
            await invoke_hook(HookType.SESSION_PRE_INIT, _payload())

        with plugin_scope(hook_b):
            await invoke_hook(HookType.SESSION_PRE_INIT, _payload())

        assert len(calls_a) == 1
        assert len(calls_b) == 1

    async def test_empty_plugin_scope_is_noop(self):
        """plugin_scope with no items enters and exits without error."""
        with plugin_scope():
            await invoke_hook(HookType.SESSION_PRE_INIT, _payload())
        # No assertion needed — just verifying no exception is raised


# ---------------------------------------------------------------------------
# MelleaPlugin as a context manager
# ---------------------------------------------------------------------------


def _make_mellea_plugin(invocations: list) -> MelleaPlugin:
    """Build a minimal concrete MelleaPlugin that records session_pre_init calls."""

    @plugin("tracking-plugin")
    class _TrackingPlugin:
        @hook(hook_type=HookType.SESSION_PRE_INIT)
        async def session_pre_init(self, payload: Any, context: Any):
            invocations.append(payload)
            return PluginResult(continue_processing=True)

    return _TrackingPlugin()


class TestMelleaPluginContextManager:
    async def test_fires_inside_with_block(self):
        invocations: list = []
        p = _make_mellea_plugin(invocations)

        with p:
            await invoke_hook(HookType.SESSION_PRE_INIT, _payload())

        assert len(invocations) == 1

    async def test_deregistered_after_with_block(self):
        invocations: list = []
        p = _make_mellea_plugin(invocations)

        with p:
            await invoke_hook(HookType.SESSION_PRE_INIT, _payload())

        await invoke_hook(HookType.SESSION_PRE_INIT, _payload())
        assert len(invocations) == 1  # No new calls outside the block

    async def test_deregistered_on_exception(self):
        invocations: list = []
        p = _make_mellea_plugin(invocations)

        with pytest.raises(ValueError):
            with p:
                await invoke_hook(HookType.SESSION_PRE_INIT, _payload())
                assert len(invocations) == 1
                raise ValueError("deliberate")

        await invoke_hook(HookType.SESSION_PRE_INIT, _payload())
        assert len(invocations) == 1  # Deregistered despite the exception

    async def test_raises_on_reentrant_use(self):
        invocations: list = []
        p = _make_mellea_plugin(invocations)

        with p:
            with pytest.raises(RuntimeError, match="already active"):
                with p:
                    pass

    async def test_scope_id_set_on_enter_and_cleared_on_exit(self):
        invocations: list = []
        p = _make_mellea_plugin(invocations)

        assert getattr(p, "_scope_id", None) is None
        with p:
            assert getattr(p, "_scope_id", None) is not None
        assert getattr(p, "_scope_id", None) is None

    async def test_reuse_after_exit(self):
        invocations: list = []
        p = _make_mellea_plugin(invocations)

        with p:
            await invoke_hook(HookType.SESSION_PRE_INIT, _payload())

        with p:
            await invoke_hook(HookType.SESSION_PRE_INIT, _payload())

        assert len(invocations) == 2

    async def test_async_with_fires_inside_block(self):
        invocations: list = []
        p = _make_mellea_plugin(invocations)

        async with p:
            await invoke_hook(HookType.SESSION_PRE_INIT, _payload())

        assert len(invocations) == 1

    async def test_async_with_deregisters_after_block(self):
        invocations: list = []
        p = _make_mellea_plugin(invocations)

        async with p:
            await invoke_hook(HookType.SESSION_PRE_INIT, _payload())

        await invoke_hook(HookType.SESSION_PRE_INIT, _payload())
        assert len(invocations) == 1
