---
id: decorators
title: "mellea.plugins.decorators"
sidebar_label: "decorators"
sidebar_position: 4
description: "Mellea hook decorator."
# diataxis: reference
---

Source: [`mellea/plugins/decorators.py`](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/plugins/decorators.py) at commit `a535fc6345a0`.

Mellea hook decorator.

## `HookMeta`

*class* — [line 15](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/plugins/decorators.py#L15) 

Metadata attached by the @hook decorator.

## `hook()`

*function* — [line 29](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/plugins/decorators.py#L29)

`hook(hook_type: str, *, mode: PluginMode = PluginMode.SEQUENTIAL, priority: int | None = None) -> Callable`

Register an async function or method as a hook handler.

---

*Generated from source by `docs/scripts/generate_package_map.py` — do not edit by hand; regenerate instead. Symbols and annotations reflect the pinned commit exactly; where type annotations are absent in source, this page says so rather than guessing.*
