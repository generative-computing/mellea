---
id: model_options
title: "mellea.backends.model_options"
sidebar_label: "model_options"
sidebar_position: 11
description: "Common ModelOptions for Backend Generation."
# diataxis: reference
---

Source: [`mellea/backends/model_options.py`](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/backends/model_options.py) at commit `a535fc6345a0`.

Common ModelOptions for Backend Generation.

## `ModelOption`

*class* — [line 11](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/backends/model_options.py#L11) 

A type that wraps around model options.

Methods (defined on this class; inherited members not listed):

- `replace_keys(options: dict, from_to: dict[str, str]) -> dict[str, Any]` *(staticmethod)* — [line 120](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/backends/model_options.py#L120)
  Return a new dict with selected keys in `options` renamed according to `from_to`.
- `remove_special_keys(model_options: dict[str, Any]) -> dict[str, Any]` *(staticmethod)* — [line 192](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/backends/model_options.py#L192)
  Return a copy of `model_options` with all sentinel-valued keys removed.
- `merge_model_options(persistent_opts: dict[str, Any], overwrite_opts: dict[str, Any] | None) -> dict[str, Any]` *(staticmethod)* — [line 211](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/backends/model_options.py#L211)
  Merge two model-options dicts, with `overwrite_opts` taking precedence on conflicts.

---

*Generated from source by `docs/scripts/generate_package_map.py` — do not edit by hand; regenerate instead. Symbols and annotations reflect the pinned commit exactly; where type annotations are absent in source, this page says so rather than guessing.*
