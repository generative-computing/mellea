"""Plugin sets covering every HookType × PluginMode for acceptance tests.

Each set contains one hook per call site in a single execution mode:
  1. logging_plugin_set  — AUDIT (observe-only)
  2. sequential_plugin_set — SEQUENTIAL (serial, can block + modify)
  3. concurrent_plugin_set — CONCURRENT (parallel, can block only)
  4. fandf_plugin_set — FIRE_AND_FORGET (background, observe-only)
"""

from mellea.plugins import HookType, PluginMode, PluginSet, hook

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_ALL_HOOK_TYPES = list(HookType)


def _make_hooks(prefix: str, mode: PluginMode):
    """Generate one async hook function per HookType in the given mode."""
    hooks = []
    for ht in _ALL_HOOK_TYPES:

        @hook(ht, mode=mode)
        async def _hook(payload, ctx, _ht=ht, _prefix=prefix):
            print(f"[{_prefix}] {_ht.value}:", payload)

        # Give each closure a unique qualname so the adapter names don't clash.
        _hook.__qualname__ = f"{prefix}_{ht.value}"
        _hook.__name__ = f"{prefix}_{ht.value}"
        hooks.append(_hook)
    return hooks


# ---------------------------------------------------------------------------
# Plugin sets
# ---------------------------------------------------------------------------

logging_plugin_set = PluginSet("logging", _make_hooks("log", PluginMode.AUDIT))
sequential_plugin_set = PluginSet("sequential", _make_hooks("seq", PluginMode.SEQUENTIAL))
concurrent_plugin_set = PluginSet("concurrent", _make_hooks("concurrent", PluginMode.CONCURRENT))
fandf_plugin_set = PluginSet("fandf", _make_hooks("fandf", PluginMode.FIRE_AND_FORGET))

ALL_ACCEPTANCE_SETS = [
    logging_plugin_set,
    sequential_plugin_set,
    concurrent_plugin_set,
    fandf_plugin_set,
]
