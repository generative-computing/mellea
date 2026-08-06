---
id: pluginset
title: "mellea.plugins.pluginset"
sidebar_label: "pluginset"
sidebar_position: 7
description: "PluginSet — composable groups of hooks and plugins."
# diataxis: reference
---

Source: [`mellea/plugins/pluginset.py`](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/plugins/pluginset.py) at commit `a535fc6345a0`.

PluginSet — composable groups of hooks and plugins.

## `PluginSet`

*class* — [line 12](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/plugins/pluginset.py#L12) 

A named, composable group of hook functions and plugin instances.

Constructor: `PluginSet(name: str, items: list[Callable | Any | PluginSet], *, priority: int | None = None)`

Methods (defined on this class; inherited members not listed):

- `flatten() -> list[tuple[Callable | Any, int | None]]` — [line 48](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/plugins/pluginset.py#L48)
  Recursively flatten nested PluginSets into `(item, priority_override)` pairs.

---

*Generated from source by `docs/scripts/generate_package_map.py` — do not edit by hand; regenerate instead. Symbols and annotations reflect the pinned commit exactly; where type annotations are absent in source, this page says so rather than guessing.*
