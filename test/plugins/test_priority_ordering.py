"""Tests for priority-based hook execution ordering.

Priority rules (actual framework behavior)
--------------------------------------------
- Lower priority numbers execute FIRST (priority=1 runs before priority=50 before priority=100).
- Resolution order: PluginSet override > @hook priority > Plugin class priority > 50.
- Plugin class-level priority is the DEFAULT for methods; @hook(priority=N) overrides it per method.
- PluginSet.priority OVERRIDES the priority of all items in the set, including items with
  explicit @hook priorities.
"""

from __future__ import annotations

import pytest

pytest.importorskip("cpex.framework")

from mellea.plugins import Plugin, PluginSet, hook, register
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
    async def test_three_hooks_fire_in_ascending_priority_order(self) -> None:
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
    async def test_two_hooks_at_same_priority_both_fire(self) -> None:
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
    async def test_default_priority_50_fires_after_priority_1_before_priority_100(
        self,
    ) -> None:
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
    async def test_priority_ordering_is_independent_of_registration_order(self) -> None:
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
    async def test_five_hooks_fire_in_strict_ascending_order(self) -> None:
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
    """Plugin class-level priority is the default; @hook method priority overrides it."""

    @pytest.mark.asyncio
    async def test_plugin_class_priority_applies_to_method_without_explicit_priority(
        self,
    ) -> None:
        """A Plugin with priority=5 makes its @hook method (no explicit priority) fire before a default-priority hook."""
        execution_order: list[str] = []

        class EarlyPlugin(Plugin, name="high-priority-class-plugin", priority=5):
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
    async def test_hook_decorator_priority_overrides_plugin_class_priority(
        self,
    ) -> None:
        """@hook(priority=80) on a method overrides Plugin(priority=5) on the class.

        The method-level @hook priority takes precedence over the class-level default.
        So a Plugin(priority=5) method with @hook(priority=80) fires at effective priority=80.
        """
        execution_order: list[str] = []

        class PluginWithMethodPriority(
            Plugin, name="low-effective-priority-plugin", priority=5
        ):
            @hook(
                "session_pre_init", priority=80
            )  # @hook priority=80 overrides class priority=5
            async def on_pre_init(self, payload, ctx):
                execution_order.append("method_p80")
                return None

        @hook("session_pre_init", priority=50)
        async def mid_hook(payload, ctx):
            execution_order.append("standalone_p50")
            return None

        register(PluginWithMethodPriority())
        register(mid_hook)

        await invoke_hook(HookType.SESSION_PRE_INIT, _session_payload())

        # @hook(priority=80) wins over Plugin(priority=5): method fires at priority=80 (after 50)
        assert execution_order.index("standalone_p50") < execution_order.index(
            "method_p80"
        )

    @pytest.mark.asyncio
    async def test_class_plugin_priority_lower_number_fires_before_higher_number(
        self,
    ) -> None:
        """Two @plugin classes with different priorities fire in the correct order."""
        execution_order: list[str] = []

        class FirstPlugin(Plugin, name="plugin-priority-3", priority=3):
            @hook("session_pre_init")
            async def on_pre_init(self, payload, ctx):
                execution_order.append("plugin_p3")
                return None

        class SecondPlugin(Plugin, name="plugin-priority-99", priority=99):
            @hook("session_pre_init")
            async def on_pre_init(self, payload, ctx):
                execution_order.append("plugin_p99")
                return None

        register(FirstPlugin())
        register(SecondPlugin())

        await invoke_hook(HookType.SESSION_PRE_INIT, _session_payload())

        assert execution_order == ["plugin_p3", "plugin_p99"]

    @pytest.mark.asyncio
    async def test_method_priority_overrides_class_priority(self) -> None:
        """@hook method priority overrides the class-level default.

        Plugin(priority=100) with @hook(priority=10) fires at effective priority=10.
        Plugin(priority=1) with @hook(priority=90) fires at effective priority=90.
        So the first plugin fires before the second, despite its class priority being higher.
        """
        execution_order: list[str] = []

        class PluginA(Plugin, name="multi-method-plugin-low-class", priority=100):
            @hook(
                "session_pre_init", priority=10
            )  # method priority=10 overrides class=100
            async def on_pre_init(self, payload, ctx):
                execution_order.append("plugin_a_effective_p10")
                return None

        class PluginB(Plugin, name="multi-method-plugin-high-class", priority=1):
            @hook(
                "session_pre_init", priority=90
            )  # method priority=90 overrides class=1
            async def on_pre_init(self, payload, ctx):
                execution_order.append("plugin_b_effective_p90")
                return None

        register(PluginA())
        register(PluginB())

        await invoke_hook(HookType.SESSION_PRE_INIT, _session_payload())

        # effective priorities are 10 and 90; PluginA fires first despite class priority=100
        assert execution_order == ["plugin_a_effective_p10", "plugin_b_effective_p90"]


# ---------------------------------------------------------------------------
# PluginSet priority
# ---------------------------------------------------------------------------


class TestPluginSetPriority:
    """PluginSet.priority sets the default priority for items without their own @hook priority."""

    @pytest.mark.asyncio
    async def test_pluginset_priority_applied_to_items_without_own_priority(
        self,
    ) -> None:
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
    async def test_pluginset_priority_overrides_per_item_hook_priority(self) -> None:
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
    async def test_pluginset_without_priority_uses_item_own_priority(self) -> None:
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
    async def test_nested_pluginsets_honour_inner_set_priority(self) -> None:
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
    async def test_multiple_pluginsets_fire_in_set_priority_order(self) -> None:
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
