---
id: pricing
title: "mellea.telemetry.pricing"
sidebar_label: "pricing"
sidebar_position: 5
description: "LLM pricing via litellm's pricing API."
# diataxis: reference
---

Source: [`mellea/telemetry/pricing.py`](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/telemetry/pricing.py) at commit `a535fc6345a0`.

LLM pricing via litellm's pricing API.

## `compute_cost()`

*function* — [line 117](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/telemetry/pricing.py#L117)

`compute_cost(model: str, provider: str | None, prompt_tokens: int | None, completion_tokens: int | None, cached_tokens: int | None = None, cache_creation_tokens: int | None = None) -> float | None`

Estimate request cost in USD using litellm's pricing data.

## `is_pricing_enabled()`

*function* — [line 163](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/telemetry/pricing.py#L163)

`is_pricing_enabled() -> bool`

Return True if pricing metrics are enabled.

---

*Generated from source by `docs/scripts/generate_package_map.py` — do not edit by hand; regenerate instead. Symbols and annotations reflect the pinned commit exactly; where type annotations are absent in source, this page says so rather than guessing.*
