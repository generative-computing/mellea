---
id: backend
title: "mellea.backends.backend"
sidebar_label: "backend"
sidebar_position: 2
description: "`FormatterBackend`: base class for prompt-engineering backends."
# diataxis: reference
---

Source: [`mellea/backends/backend.py`](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/backends/backend.py) at commit `a535fc6345a0`.

`FormatterBackend`: base class for prompt-engineering backends.

## `FormatterBackend`

*class* — [line 22](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/backends/backend.py#L22) (`Backend`, `abc.ABC`)

`FormatterBackend`s support legacy model types.

Constructor: `FormatterBackend(model_id: str | ModelIdentifier, formatter: ChatFormatter, *, model_options: dict | None = None)`

---

*Generated from source by `docs/scripts/generate_package_map.py` — do not edit by hand; regenerate instead. Symbols and annotations reflect the pinned commit exactly; where type annotations are absent in source, this page says so rather than guessing.*
