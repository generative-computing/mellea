"""Tests for priority-based hook execution ordering.

Priority rules (actual framework behavior)
--------------------------------------------
- Lower priority numbers execute FIRST (priority=1 runs before priority=50 before priority=100).
- Default priority is 50 when not specified on @hook.
- @plugin class-level priority GOVERNS all methods; @hook(priority=N) on a method is NOT used.
- PluginSet.priority OVERRIDES the priority of all items in the set, including items with
  explicit @hook priorities.
"""

from __future__ import annotations

import pytest

pytest.importorskip("mcpgateway.plugins.framework")

from mellea.plugins import PluginResult, PluginSet, hook, plugin, register
from mellea.plugins.hooks.session import SessionPreInitPayload
from mellea.plugins.manager import invoke_hook, shutdown_plugins
from mellea.plugins.types import HookType


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _session_payload() -> SessionPreInitPayload:
    return SessionPreInitPayload(
        backend_name="test-backend", model_id="test-model", model_options=None
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
async def cleanup_plugins():
    """Reset plugin manager state after every test."""
    yield
    await shutdown_plugins()


# ---------------------------------------------------------------------------
# Priority ordering
# ---------------------------------------------------------------------------


class TestPriorityOrdering:
    """Lower numeric priority values run first."""

    @pytest.mark.asyncio
    async def test_three_hooks_fire_in_ascending_priority_order(self):
        """Hooks at priorities 30, 10, 50 execute in order [10, 30, 50]."""
        execution_order: list[int] = []

        @hook("session_pre_init", priority=30)
        async def hook_priority_30(payload, ctx):
            execution_order.append(30)
            return None

        @hook("session_pre_init", priority=10)
        async def hook_priority_10(payload, ctx):
            execution_order.append(10)
            return None

        @hook("session_pre_init", priority=50)
        async def hook_priority_50(payload, ctx):
            execution_order.append(50)
            return None

        register(hook_priority_30)
        register(hook_priority_10)
        register(hook_priority_50)

        await invoke_hook(HookType.SESSION_PRE_INIT, _session_payload())

        assert execution_order == [10, 30, 50]

    @pytest.mark.asyncio
    async def test_two_hooks_at_same_priority_both_fire(self):
        """Two hooks at the same priority both execute (order unspecified)."""
        fired: set[str] = set()

        @hook("session_pre_init", priority=50)
        async def hook_a(payload, ctx):
            fired.add("a")
            return None

        @hook("session_pre_init", priority=50)
        async def hook_b(payload, ctx):
            fired.add("b")
            return None

        register(hook_a)
        register(hook_b)

        await invoke_hook(HookType.SESSION_PRE_INIT, _session_payload())

        assert fired == {"a", "b"}

    @pytest.mark.asyncio
    async def test_default_priority_50_fires_after_priority_1_before_priority_100(self):
        """Default priority (50) fires after priority=1 and before priority=100."""
        execution_order: list[str] = []

        @hook("session_pre_init", priority=1)
        async def very_high_priority(payload, ctx):
            execution_order.append("priority_1")
            return None

        @hook("session_pre_init")  # default priority=50
        async def default_priority(payload, ctx):
            execution_order.append("priority_default")
            return None

        @hook("session_pre_init", priority=100)
        async def low_priority(payload, ctx):
            execution_order.append("priority_100")
            return None

        register(very_high_priority)
        register(default_priority)
        register(low_priority)

        await invoke_hook(HookType.SESSION_PRE_INIT, _session_payload())

        assert execution_order.index("priority_1") < execution_order.index(
            "priority_default"
        )
        assert execution_order.index("priority_default") < execution_order.index(
            "priority_100"
        )

    @pytest.mark.asyncio
    async def test_priority_ordering_is_independent_of_registration_order(self):
        """Execution order is determined by priority, not the order hooks are registered."""
        execution_order: list[str] = []

        # Register high number (low priority) first
        @hook("session_pre_init", priority=90)
        async def registered_first_low_priority(payload, ctx):
            execution_order.append("90")
            return None

        # Register low number (high priority) second
        @hook("session_pre_init", priority=10)
        async def registered_second_high_priority(payload, ctx):
            execution_order.append("10")
            return None

        register(registered_first_low_priority)
        register(registered_second_high_priority)

        await invoke_hook(HookType.SESSION_PRE_INIT, _session_payload())

        # priority=10 must run before priority=90 regardless of registration order
        assert execution_order == ["10", "90"]

    @pytest.mark.asyncio
    async def test_five_hooks_fire_in_strict_ascending_order(self):
        """Five hooks with distinct priorities fire in fully sorted order."""
        execution_order: list[int] = []

        # Note: each hook must have a unique function name because the plugin registry
        # uses the function name as an identifier and raises ValueError on duplicates.
        @hook("session_pre_init", priority=75)
        async def _h75(payload, ctx):
            execution_order.append(75)
            return None

        @hook("session_pre_init", priority=25)
        async def _h25(payload, ctx):
            execution_order.append(25)
            return None

        @hook("session_pre_init", priority=5)
        async def _h5(payload, ctx):
            execution_order.append(5)
            return None

        @hook("session_pre_init", priority=100)
        async def _h100(payload, ctx):
            execution_order.append(100)
            return None

        @hook("session_pre_init", priority=50)
        async def _h50(payload, ctx):
            execution_order.append(50)
            return None

        for h in (_h75, _h25, _h5, _h100, _h50):
            register(h)

        await invoke_hook(HookType.SESSION_PRE_INIT, _session_payload())

        assert execution_order == [5, 25, 50, 75, 100]


# ---------------------------------------------------------------------------
# Priority inheritance
# ---------------------------------------------------------------------------


class TestPriorityInheritance:
    """@plugin class-level priority governs all methods; @hook method priority is not used."""

    @pytest.mark.asyncio
    async def test_plugin_class_priority_applies_to_method_without_explicit_priority(
        self,
    ):
        """A @plugin with priority=5 makes its @hook method fire before a default-priority hook."""
        execution_order: list[str] = []

        @plugin("high-priority-class-plugin", priority=5)
        class EarlyPlugin:
            @hook(
                "session_pre_init"
            )  # no explicit priority — inherits class priority=5
            async def on_pre_init(self, payload, ctx):
                execution_order.append("class_plugin_p5")
                return None

        @hook("session_pre_init", priority=50)  # default priority
        async def default_hook(payload, ctx):
            execution_order.append("default_p50")
            return None

        register(EarlyPlugin())
        register(default_hook)

        await invoke_hook(HookType.SESSION_PRE_INIT, _session_payload())

        assert execution_order.index("class_plugin_p5") < execution_order.index(
            "default_p50"
        )

    @pytest.mark.asyncio
    async def test_hook_decorator_priority_does_not_override_plugin_class_priority(
        self,
    ):
        """@hook(priority=80) on a method does NOT override @plugin(priority=5) on the class.

        The @plugin class-level priority takes precedence over any method-level @hook priority.
        So a @plugin(priority=5) method fires at effective priority=5, regardless of any
        @hook(priority=80) on the method itself.
        """
        execution_order: list[str] = []

        @plugin("low-effective-priority-plugin", priority=5)
        class PluginWithOverriddenPriority:
            @hook(
                "session_pre_init", priority=80
            )  # @hook priority is ignored; class priority=5 wins
            async def on_pre_init(self, payload, ctx):
                execution_order.append("method_class_p5")
                return None

        @hook("session_pre_init", priority=50)
        async def mid_hook(payload, ctx):
            execution_order.append("standalone_p50")
            return None

        register(PluginWithOverriddenPriority())
        register(mid_hook)

        await invoke_hook(HookType.SESSION_PRE_INIT, _session_payload())

        # @plugin(priority=5) wins over @hook(priority=80): class plugin fires at priority=5 (before 50)
        assert execution_order.index("method_class_p5") < execution_order.index(
            "standalone_p50"
        )

    @pytest.mark.asyncio
    async def test_class_plugin_priority_lower_number_fires_before_higher_number(self):
        """Two @plugin classes with different priorities fire in the correct order."""
        execution_order: list[str] = []

        @plugin("plugin-priority-3", priority=3)
        class FirstPlugin:
            @hook("session_pre_init")
            async def on_pre_init(self, payload, ctx):
                execution_order.append("plugin_p3")
                return None

        @plugin("plugin-priority-99", priority=99)
        class SecondPlugin:
            @hook("session_pre_init")
            async def on_pre_init(self, payload, ctx):
                execution_order.append("plugin_p99")
                return None

        register(FirstPlugin())
        register(SecondPlugin())

        await invoke_hook(HookType.SESSION_PRE_INIT, _session_payload())

        assert execution_order == ["plugin_p3", "plugin_p99"]

    @pytest.mark.asyncio
    async def test_class_priority_governs_when_method_has_explicit_hook_priority(self):
        """Class priority governs execution order even when methods have explicit @hook priorities.

        @plugin(priority=1) fires before @plugin(priority=100) regardless of any
        @hook(priority=N) annotations on the methods. Class-level priority always wins.
        """
        execution_order: list[str] = []

        @plugin("multi-method-plugin-low-class", priority=100)
        class LowClassPriority:
            @hook(
                "session_pre_init", priority=10
            )  # @hook priority ignored; class=100 governs
            async def on_pre_init(self, payload, ctx):
                execution_order.append("class_p100")
                return None

        @plugin("multi-method-plugin-high-class", priority=1)
        class HighClassPriority:
            @hook(
                "session_pre_init", priority=90
            )  # @hook priority ignored; class=1 governs
            async def on_pre_init(self, payload, ctx):
                execution_order.append("class_p1")
                return None

        register(LowClassPriority())
        register(HighClassPriority())

        await invoke_hook(HookType.SESSION_PRE_INIT, _session_payload())

        # class priority=1 fires before class priority=100; @hook priorities are not used
        assert execution_order == ["class_p1", "class_p100"]


# ---------------------------------------------------------------------------
# PluginSet priority
# ---------------------------------------------------------------------------


class TestPluginSetPriority:
    """PluginSet.priority sets the default priority for items without their own @hook priority."""

    @pytest.mark.asyncio
    async def test_pluginset_priority_applied_to_items_without_own_priority(self):
        """Items in a PluginSet with priority=10 fire before an outside hook at priority=50."""
        execution_order: list[str] = []

        @hook(
            "session_pre_init"
        )  # @hook default priority=50; PluginSet will override to 10
        async def inside_set(payload, ctx):
            execution_order.append("inside_set")
            return None

        @hook("session_pre_init", priority=50)
        async def outside_hook(payload, ctx):
            execution_order.append("outside_hook")
            return None

        ps = PluginSet("high-priority-set", [inside_set], priority=10)
        register(ps)
        register(outside_hook)

        await invoke_hook(HookType.SESSION_PRE_INIT, _session_payload())

        assert execution_order.index("inside_set") < execution_order.index(
            "outside_hook"
        )

    @pytest.mark.asyncio
    async def test_pluginset_priority_overrides_per_item_hook_priority(self):
        """PluginSet priority overrides the item's @hook priority, even when the item has an explicit one.

        An item decorated with @hook(priority=80) placed in a PluginSet(priority=5)
        fires at effective priority=5, not 80. The PluginSet priority always wins.
        """
        execution_order: list[str] = []

        @hook(
            "session_pre_init", priority=80
        )  # @hook priority is overridden by PluginSet(priority=5)
        async def item_with_own_priority(payload, ctx):
            execution_order.append("item_effective_p5")
            return None

        @hook("session_pre_init", priority=20)
        async def standalone_hook(payload, ctx):
            execution_order.append("standalone_p20")
            return None

        # PluginSet(priority=5) overrides the item's @hook(priority=80)
        # so the item fires at effective priority=5 (before standalone_p20=20)
        ps = PluginSet("low-priority-set", [item_with_own_priority], priority=5)
        register(ps)
        register(standalone_hook)

        await invoke_hook(HookType.SESSION_PRE_INIT, _session_payload())

        # item fires at PluginSet priority=5, which is before standalone_p20=20
        assert execution_order.index("item_effective_p5") < execution_order.index(
            "standalone_p20"
        )

    @pytest.mark.asyncio
    async def test_pluginset_without_priority_uses_item_own_priority(self):
        """PluginSet with no priority (None) does not override the item's @hook priority."""
        execution_order: list[str] = []

        @hook("session_pre_init", priority=15)
        async def item_own_p15(payload, ctx):
            execution_order.append("item_p15")
            return None

        @hook("session_pre_init", priority=60)
        async def standalone_p60(payload, ctx):
            execution_order.append("standalone_p60")
            return None

        ps = PluginSet("no-priority-set", [item_own_p15])  # priority=None
        register(ps)
        register(standalone_p60)

        await invoke_hook(HookType.SESSION_PRE_INIT, _session_payload())

        assert execution_order.index("item_p15") < execution_order.index(
            "standalone_p60"
        )

    @pytest.mark.asyncio
    async def test_nested_pluginsets_honour_inner_set_priority(self):
        """In a nested PluginSet, the inner set's priority governs its items."""
        execution_order: list[str] = []

        @hook("session_pre_init")
        async def inner_item(payload, ctx):
            execution_order.append("inner")
            return None

        @hook("session_pre_init")
        async def outer_only_item(payload, ctx):
            execution_order.append("outer")
            return None

        inner_ps = PluginSet("inner", [inner_item], priority=5)
        outer_ps = PluginSet("outer", [inner_ps, outer_only_item], priority=70)

        register(outer_ps)

        await invoke_hook(HookType.SESSION_PRE_INIT, _session_payload())

        # inner_item gets priority=5 (inner set); outer_only_item gets priority=70 (outer set)
        assert execution_order.index("inner") < execution_order.index("outer")

    @pytest.mark.asyncio
    async def test_multiple_pluginsets_fire_in_set_priority_order(self):
        """Items from a lower-priority PluginSet fire before items from a higher-priority one."""
        execution_order: list[str] = []

        @hook("session_pre_init")
        async def alpha(payload, ctx):
            execution_order.append("alpha")
            return None

        @hook("session_pre_init")
        async def beta(payload, ctx):
            execution_order.append("beta")
            return None

        # alpha is in a set with priority=3 (fires first), beta in priority=80 (fires later)
        ps_early = PluginSet("early-set", [alpha], priority=3)
        ps_late = PluginSet("late-set", [beta], priority=80)

        register(ps_early)
        register(ps_late)

        await invoke_hook(HookType.SESSION_PRE_INIT, _session_payload())

        assert execution_order == ["alpha", "beta"]
