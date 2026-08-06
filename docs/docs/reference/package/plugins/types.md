---
id: types
title: "mellea.plugins.types"
sidebar_label: "types"
sidebar_position: 10
description: "Mellea hook type enum and hook registration."
# diataxis: reference
---

Source: [`mellea/plugins/types.py`](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/plugins/types.py) at commit `a535fc6345a0`.

Mellea hook type enum and hook registration.

## `PluginMode`

*class* — [line 20](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/plugins/types.py#L20) (`StrEnum`)

Execution modes for Mellea plugins.

## `HookType`

*class* — [line 33](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/plugins/types.py#L33) (`StrEnum`)

All Mellea hook types.

## `register_mellea_hooks()`

*function* — [line 207](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/plugins/types.py#L207)

`register_mellea_hooks() -> None`

Register all Mellea hook types with the ContextForge HookRegistry.

---

*Generated from source by `docs/scripts/generate_package_map.py` — do not edit by hand; regenerate instead. Symbols and annotations reflect the pinned commit exactly; where type annotations are absent in source, this page says so rather than guessing.*
