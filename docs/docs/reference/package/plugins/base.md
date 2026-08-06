---
id: base
title: "mellea.plugins.base"
sidebar_label: "base"
sidebar_position: 1
description: "Base types for the Mellea plugin system."
# diataxis: reference
---

Source: [`mellea/plugins/base.py`](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/plugins/base.py) at commit `a535fc6345a0`.

Base types for the Mellea plugin system.

## `PluginMeta`

*class* — [line 21](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/plugins/base.py#L21) 

Metadata attached to Plugin subclasses.

## `Plugin`

*class* — [line 60](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/plugins/base.py#L60) 

Base class for multi-hook Mellea plugins.

## `PluginViolationError`

*class* — [line 138](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/plugins/base.py#L138) (`Exception`)

Raised when a plugin blocks execution in enforce mode.

Constructor: `PluginViolationError(hook_type: str, reason: str, code: str = '', plugin_name: str = '')`

## `MelleaBasePayload`

*class* — [line 172](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/plugins/base.py#L172) (`_PayloadBase`)

Frozen base — all payloads are immutable by design.

## `MelleaPlugin`

*class* — [line 188](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/plugins/base.py#L188) (`_PluginBase`)

Base class for Mellea plugins with lifecycle hooks and typed accessors.

Properties:

- `plugin_config` → `dict[str, Any]` — Plugin-specific configuration from PluginConfig.config.

Methods (defined on this class; inherited members not listed):

- `get_backend(context: PluginContext) -> Backend | None` — [line 212](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/plugins/base.py#L212)
  Get the Backend from the plugin context.
- `get_mellea_context(context: PluginContext) -> Context | None` — [line 223](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/plugins/base.py#L223)
  Get the Mellea Context from the plugin context.
- `get_session(context: PluginContext) -> MelleaSession | None` — [line 234](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/plugins/base.py#L234)
  Get the MelleaSession from the plugin context.

---

*Generated from source by `docs/scripts/generate_package_map.py` — do not edit by hand; regenerate instead. Symbols and annotations reflect the pinned commit exactly; where type annotations are absent in source, this page says so rather than guessing.*
