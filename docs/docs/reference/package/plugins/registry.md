---
id: registry
title: "mellea.plugins.registry"
sidebar_label: "registry"
sidebar_position: 9
description: "Plugin registration and helpers."
# diataxis: reference
---

Source: [`mellea/plugins/registry.py`](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/plugins/registry.py) at commit `a535fc6345a0`.

Plugin registration and helpers.

## `modify()`

*function* — [line 50](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/plugins/registry.py#L50)

`modify(payload: Any, **field_updates: Any) -> Any`

Convenience helper for returning a modifying `PluginResult`.

## `block()`

*function* — [line 88](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/plugins/registry.py#L88)

`block(reason: str, *, code: str = '', description: str = '', details: dict[str, Any] | None = None) -> Any`

Convenience helper for returning a blocking `PluginResult`.

## `register()`

*function* — [line 125](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/plugins/registry.py#L125)

`register(items: Callable | Any | PluginSet | list[Callable | Any | PluginSet], *, session_id: str | None = None) -> None`

Register plugins globally or for a specific session.

## `unregister()`

*function* — [line 476](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/plugins/registry.py#L476)

`unregister(items: Callable | Any | PluginSet | list[Callable | Any | PluginSet]) -> None`

Unregister globally-registered plugins.

## `plugin_scope()`

*function* — [line 517](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/plugins/registry.py#L517)

`plugin_scope(*items: Callable | Any | PluginSet | list[Callable | Any | PluginSet]) -> _PluginScope`

Return a context manager that temporarily registers plugins for a block of code.

---

*Generated from source by `docs/scripts/generate_package_map.py` — do not edit by hand; regenerate instead. Symbols and annotations reflect the pinned commit exactly; where type annotations are absent in source, this page says so rather than guessing.*
