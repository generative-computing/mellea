---
id: index
title: "mellea.plugins"
sidebar_label: "Overview"
sidebar_position: 0
description: "Mellea Plugin System — extension points for policy enforcement, observability, and customization."
# diataxis: reference
---

Source: [`mellea/plugins/__init__.py`](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/plugins/__init__.py) at commit `a535fc6345a0`.

Mellea Plugin System — extension points for policy enforcement, observability, and customization.

Declared exports (`__all__`): `HookType`, `Plugin`, `PluginMode`, `PluginResult`, `PluginSet`, `PluginViolationError`, `block`, `hook`, `is_internal_tool`, `modify`, `plugin_scope`, `register`, `unregister`

## Modules

- [`mellea.plugins.base`](base.md) — Base types for the Mellea plugin system.
- [`mellea.plugins.builtin_debug`](builtin_debug.md) — Built-in debug plugins for Mellea.
- [`mellea.plugins.context`](context.md) — Plugin context factory — maps Mellea domain objects to ContextForge GlobalContext.
- [`mellea.plugins.decorators`](decorators.md) — Mellea hook decorator.
- [`mellea.plugins.hooks`](hooks.md) — Hook payload classes for the Mellea plugin system.
- [`mellea.plugins.manager`](manager.md) — Singleton plugin manager wrapper with session-tag filtering.
- [`mellea.plugins.pluginset`](pluginset.md) — PluginSet — composable groups of hooks and plugins.
- [`mellea.plugins.policies`](policies.md) — Hook payload policies for Mellea hooks.
- [`mellea.plugins.registry`](registry.md) — Plugin registration and helpers.
- [`mellea.plugins.types`](types.md) — Mellea hook type enum and hook registration.

---

*Generated from source by `docs/scripts/generate_package_map.py` — do not edit by hand; regenerate instead. Symbols and annotations reflect the pinned commit exactly; where type annotations are absent in source, this page says so rather than guessing.*
