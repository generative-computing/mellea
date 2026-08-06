---
id: formatter
title: "mellea.core.formatter"
sidebar_label: "formatter"
sidebar_position: 3
description: "Abstract `Formatter` interface for rendering components to strings."
# diataxis: reference
---

Source: [`mellea/core/formatter.py`](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/core/formatter.py) at commit `a535fc6345a0`.

Abstract `Formatter` interface for rendering components to strings.

## `Formatter`

*class* — [line 18](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/core/formatter.py#L18) (`abc.ABC`)

A Formatter converts `Component`s into strings and parses `ModelOutputThunk`s into `Component`s (or `CBlock`s).

Methods (defined on this class; inherited members not listed):

- `print(c: Span) -> str` — [line 22](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/core/formatter.py#L22)
  Renders a `Component`, `CBlock`, or `ModelOutputThunk` into a string suitable for use as model input.

---

*Generated from source by `docs/scripts/generate_package_map.py` — do not edit by hand; regenerate instead. Symbols and annotations reflect the pinned commit exactly; where type annotations are absent in source, this page says so rather than guessing.*
