---
id: kv_block_helpers
title: "mellea.backends.kv_block_helpers"
sidebar_label: "kv_block_helpers"
sidebar_position: 8
description: "Low-level utilities for concatenating transformer KV caches (KV smashing)."
# diataxis: reference
---

Source: [`mellea/backends/kv_block_helpers.py`](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/backends/kv_block_helpers.py) at commit `a535fc6345a0`.

Low-level utilities for concatenating transformer KV caches (KV smashing).

## `prefill_cache_v5()`

*function* — [line 34](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/backends/kv_block_helpers.py#L34)

`prefill_cache_v5(model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase, text: str, device: torch.device) -> tuple[dict, DynamicCache]`

Prefills cache for transformers v5.

## `merge_dynamic_caches_v5()`

*function* — [line 56](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/backends/kv_block_helpers.py#L56)

`merge_dynamic_caches_v5(caches: Iterable[DynamicCache]) -> DynamicCache`

Merge multiple v5 DynamicCache objects by concatenating KV states along the time axis.

## `merge_v5()`

*function* — [line 86](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/backends/kv_block_helpers.py#L86)

`merge_v5(model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase, strs: list[str], device: torch.device)`

Merges DynamicCache for transformers>=5.0.0.

*Annotation gaps in source: return type unannotated.*

---

*Generated from source by `docs/scripts/generate_package_map.py` — do not edit by hand; regenerate instead. Symbols and annotations reflect the pinned commit exactly; where type annotations are absent in source, this page says so rather than guessing.*
