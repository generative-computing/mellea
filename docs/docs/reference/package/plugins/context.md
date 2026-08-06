---
id: context
title: "mellea.plugins.context"
sidebar_label: "context"
sidebar_position: 3
description: "Plugin context factory — maps Mellea domain objects to ContextForge GlobalContext."
# diataxis: reference
---

Source: [`mellea/plugins/context.py`](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/plugins/context.py) at commit `a535fc6345a0`.

Plugin context factory — maps Mellea domain objects to ContextForge GlobalContext.

## `build_global_context()`

*function* — [line 21](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/plugins/context.py#L21)

`build_global_context(*, backend: Backend | None = None, **extra_fields: Any) -> Any`

Build a ContextForge `GlobalContext` from Mellea domain objects.

---

*Generated from source by `docs/scripts/generate_package_map.py` — do not edit by hand; regenerate instead. Symbols and annotations reflect the pinned commit exactly; where type annotations are absent in source, this page says so rather than guessing.*
