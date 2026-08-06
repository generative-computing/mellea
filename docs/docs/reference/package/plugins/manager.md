---
id: manager
title: "mellea.plugins.manager"
sidebar_label: "manager"
sidebar_position: 6
description: "Singleton plugin manager wrapper with session-tag filtering."
# diataxis: reference
---

Source: [`mellea/plugins/manager.py`](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/plugins/manager.py) at commit `a535fc6345a0`.

Singleton plugin manager wrapper with session-tag filtering.

## `enable_background_collection()`

*function* — [line 43](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/plugins/manager.py#L43)

`enable_background_collection() -> None`

Enable fire-and-forget result collection. Call in test fixtures before each test.

## `disable_background_collection()`

*function* — [line 49](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/plugins/manager.py#L49)

`disable_background_collection() -> None`

Disable fire-and-forget result collection and clear any accumulated results.

## `drain_background_tasks()`

*async function* — [line 56](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/plugins/manager.py#L56)

`drain_background_tasks() -> None`

Await all accumulated FIRE_AND_FORGET tasks and clear the pending list.

## `discard_background_tasks()`

*function* — [line 68](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/plugins/manager.py#L68)

`discard_background_tasks() -> None`

Discard all accumulated FIRE_AND_FORGET tasks without awaiting them.

## `has_plugins()`

*function* — [line 77](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/plugins/manager.py#L77)

`has_plugins(hook_type: HookType | None = None) -> bool`

Fast check: are plugins configured and available for the given hook type.

## `is_internal_tool()`

*function* — [line 98](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/plugins/manager.py#L98)

`is_internal_tool(tool_name: str) -> bool`

Return whether the given tool name is a framework-internal tool.

## `get_plugin_manager()`

*function* — [line 110](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/plugins/manager.py#L110)

`get_plugin_manager() -> Any | None`

Return the initialized PluginManager, or `None` if plugins are not configured.

## `ensure_plugin_manager()`

*function* — [line 119](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/plugins/manager.py#L119)

`ensure_plugin_manager() -> Any`

Lazily initialize the PluginManager if not already created.

## `initialize_plugins()`

*async function* — [line 153](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/plugins/manager.py#L153)

`initialize_plugins(config_path: str | None = None, *, timeout: int = DEFAULT_PLUGIN_TIMEOUT) -> Any`

Initialize the PluginManager with Mellea hook registrations and optional YAML config.

## `shutdown_plugins()`

*async function* — [line 192](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/plugins/manager.py#L192)

`shutdown_plugins() -> None`

Shut down the PluginManager and reset all state.

## `track_session_plugin()`

*function* — [line 204](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/plugins/manager.py#L204)

`track_session_plugin(session_id: str, plugin_name: str) -> None`

Track a plugin as belonging to a session for later deregistration.

## `deregister_session_plugins()`

*function* — [line 214](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/plugins/manager.py#L214)

`deregister_session_plugins(session_id: str) -> None`

Deregister all plugins scoped to the given session.

## `invoke_hook()`

*async function* — [line 238](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/plugins/manager.py#L238)

`invoke_hook(hook_type: HookType, payload: _MelleaBasePayload, *, backend: Backend | None = None, **context_fields: Any) -> tuple[Any | None, _MelleaBasePayload]`

Invoke a hook if plugins are configured.

---

*Generated from source by `docs/scripts/generate_package_map.py` — do not edit by hand; regenerate instead. Symbols and annotations reflect the pinned commit exactly; where type annotations are absent in source, this page says so rather than guessing.*
