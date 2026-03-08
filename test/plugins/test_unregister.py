"""Tests for the ``unregister()`` public API."""

from typing import Any

import pytest

pytest.importorskip("cpex.framework")

from mellea.plugins import Plugin, PluginSet, hook, register, unregister
from mellea.plugins.hooks.session import SessionPreInitPayload
from mellea.plugins.manager import invoke_hook, shutdown_plugins
from mellea.plugins.types import HookType


@pytest.fixture(autouse=True)
async def reset_plugins():
    yield
    await shutdown_plugins()


def _payload() -> SessionPreInitPayload:
    return SessionPreInitPayload(
        backend_name="test-backend", model_id="test-model", model_options=None
    )


# ---------------------------------------------------------------------------
# Standalone @hook functions
# ---------------------------------------------------------------------------


class TestUnregisterStandaloneHook:
    async def test_hook_stops_firing_after_unregister(self) -> None:
        invocations: list[Any] = []

        @hook("session_pre_init")
        async def my_hook(payload, ctx):
            invocations.append(payload)

        register(my_hook)
        await invoke_hook(HookType.SESSION_PRE_INIT, _payload())
        assert len(invocations) == 1

        unregister(my_hook)
        await invoke_hook(HookType.SESSION_PRE_INIT, _payload())
        assert len(invocations) == 1  # No new calls after unregister

    async def test_other_hooks_unaffected(self) -> None:
        """Unregistering one hook leaves other registered hooks intact."""
        calls_a: list[Any] = []
        calls_b: list[Any] = []

        @hook("session_pre_init")
        async def hook_a(payload, ctx):
            calls_a.append(payload)

        @hook("session_pre_init")
        async def hook_b(payload, ctx):
            calls_b.append(payload)

        register(hook_a)
        register(hook_b)
        await invoke_hook(HookType.SESSION_PRE_INIT, _payload())
        assert len(calls_a) == 1
        assert len(calls_b) == 1

        unregister(hook_a)
        await invoke_hook(HookType.SESSION_PRE_INIT, _payload())
        assert len(calls_a) == 1  # Stopped
        assert len(calls_b) == 2  # Still firing

    async def test_unregister_not_registered_is_noop(self) -> None:
        """Unregistering a hook that was never registered does not raise."""

        @hook("session_pre_init")
        async def never_registered(payload, ctx):
            pass

        unregister(never_registered)  # Should not raise

    async def test_unregister_list_of_hooks(self) -> None:
        calls_a: list[Any] = []
        calls_b: list[Any] = []

        @hook("session_pre_init")
        async def hook_a(payload, ctx):
            calls_a.append(payload)

        @hook("session_pre_init")
        async def hook_b(payload, ctx):
            calls_b.append(payload)

        register(hook_a)
        register(hook_b)
        await invoke_hook(HookType.SESSION_PRE_INIT, _payload())
        assert len(calls_a) == 1
        assert len(calls_b) == 1

        unregister([hook_a, hook_b])
        await invoke_hook(HookType.SESSION_PRE_INIT, _payload())
        assert len(calls_a) == 1
        assert len(calls_b) == 1


# ---------------------------------------------------------------------------
# Plugin subclass instances
# ---------------------------------------------------------------------------


class TestUnregisterClassPlugin:
    async def test_class_plugin_stops_firing_after_unregister(self) -> None:
        invocations: list[Any] = []

        class MyPlugin(Plugin, name="unregister-cls-test"):
            @hook("session_pre_init")
            async def on_pre_init(self, payload, ctx):
                invocations.append(payload)

        instance = MyPlugin()
        register(instance)
        await invoke_hook(HookType.SESSION_PRE_INIT, _payload())
        assert len(invocations) == 1

        unregister(instance)
        await invoke_hook(HookType.SESSION_PRE_INIT, _payload())
        assert len(invocations) == 1

    async def test_unregister_class_plugin_leaves_standalone_hook_intact(self) -> None:
        standalone_calls: list[Any] = []
        class_calls: list[Any] = []

        @hook("session_pre_init")
        async def standalone(payload, ctx):
            standalone_calls.append(payload)

        class MyPlugin(Plugin, name="unregister-cls-isolation"):
            @hook("session_pre_init")
            async def on_pre_init(self, payload, ctx):
                class_calls.append(payload)

        instance = MyPlugin()
        register(standalone)
        register(instance)
        await invoke_hook(HookType.SESSION_PRE_INIT, _payload())
        assert len(standalone_calls) == 1
        assert len(class_calls) == 1

        unregister(instance)
        await invoke_hook(HookType.SESSION_PRE_INIT, _payload())
        assert len(standalone_calls) == 2  # Still firing
        assert len(class_calls) == 1  # Stopped


# ---------------------------------------------------------------------------
# PluginSet
# ---------------------------------------------------------------------------


class TestUnregisterPluginSet:
    async def test_pluginset_all_items_stop_firing(self) -> None:
        calls_a: list[Any] = []
        calls_b: list[Any] = []

        @hook("session_pre_init")
        async def hook_a(payload, ctx):
            calls_a.append(payload)

        @hook("session_pre_init")
        async def hook_b(payload, ctx):
            calls_b.append(payload)

        ps = PluginSet("unregister-ps-test", [hook_a, hook_b])
        register(ps)
        await invoke_hook(HookType.SESSION_PRE_INIT, _payload())
        assert len(calls_a) == 1
        assert len(calls_b) == 1

        unregister(ps)
        await invoke_hook(HookType.SESSION_PRE_INIT, _payload())
        assert len(calls_a) == 1
        assert len(calls_b) == 1

    async def test_unregister_pluginset_leaves_other_plugins_intact(self) -> None:
        ps_calls: list[Any] = []
        other_calls: list[Any] = []

        @hook("session_pre_init")
        async def in_set(payload, ctx):
            ps_calls.append(payload)

        @hook("session_pre_init")
        async def other(payload, ctx):
            other_calls.append(payload)

        ps = PluginSet("unregister-ps-isolation", [in_set])
        register(ps)
        register(other)
        await invoke_hook(HookType.SESSION_PRE_INIT, _payload())
        assert len(ps_calls) == 1
        assert len(other_calls) == 1

        unregister(ps)
        await invoke_hook(HookType.SESSION_PRE_INIT, _payload())
        assert len(ps_calls) == 1  # Stopped
        assert len(other_calls) == 2  # Still firing


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestUnregisterErrors:
    async def test_unregister_unrecognized_type_raises_type_error(self) -> None:
        # Initialize the plugin manager by registering something first.
        @hook("session_pre_init")
        async def dummy(payload, ctx):
            pass

        register(dummy)

        with pytest.raises(TypeError, match="Cannot unregister"):
            unregister(object())  # type: ignore[arg-type]
