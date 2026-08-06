---
id: context_lengths
title: "mellea.backends.context_lengths"
sidebar_label: "context_lengths"
sidebar_position: 5
description: "Model context-length lookup table."
# diataxis: reference
---

Source: [`mellea/backends/context_lengths.py`](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/backends/context_lengths.py) at commit `a535fc6345a0`.

Model context-length lookup table.

## `get_context_length()`

*function* — [line 30](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/backends/context_lengths.py#L30)

`get_context_length(model_id: str | ModelIdentifier) -> int | None`

Return the maximum context length in tokens for a known model, or `None`.

---

*Generated from source by `docs/scripts/generate_package_map.py` — do not edit by hand; regenerate instead. Symbols and annotations reflect the pinned commit exactly; where type annotations are absent in source, this page says so rather than guessing.*
