---
id: builtin_debug
title: "mellea.plugins.builtin_debug"
sidebar_label: "builtin_debug"
sidebar_position: 2
description: "Built-in debug plugins for Mellea."
# diataxis: reference
---

Source: [`mellea/plugins/builtin_debug/__init__.py`](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/plugins/builtin_debug/__init__.py) at commit `a535fc6345a0`.

Built-in debug plugins for Mellea.

Declared exports (`__all__`): `log_generation_post_call`, `log_generation_pre_call`, `log_sampling_iteration`, `log_sampling_loop_end`, `log_sampling_loop_start`, `log_sampling_repair`, `log_validation_post_check`, `log_validation_pre_check`

---

## Module `mellea.plugins.builtin_debug.generation`

Source: [`mellea/plugins/builtin_debug/generation.py`](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/plugins/builtin_debug/generation.py) at commit `a535fc6345a0`.

Built-in debug plugin for generation pipeline (pre-call and post-call).

### `log_generation_pre_call()`

*async function* — [line 109](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/plugins/builtin_debug/generation.py#L109)

`log_generation_pre_call(payload: GenerationPreCallPayload, ctx: PluginContext) -> None`

Log request details before calling the LLM.

### `log_generation_post_call()`

*async function* — [line 163](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/plugins/builtin_debug/generation.py#L163)

`log_generation_post_call(payload: GenerationPostCallPayload, ctx: PluginContext) -> None`

Log response details after LLM returns.

---

## Module `mellea.plugins.builtin_debug.sampling`

Source: [`mellea/plugins/builtin_debug/sampling.py`](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/plugins/builtin_debug/sampling.py) at commit `a535fc6345a0`.

Built-in debug plugin for sampling pipeline.

### `log_sampling_loop_start()`

*async function* — [line 51](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/plugins/builtin_debug/sampling.py#L51)

`log_sampling_loop_start(payload: SamplingLoopStartPayload, ctx: PluginContext) -> None`

Log sampling strategy initialization with budget and requirement count.

### `log_sampling_iteration()`

*async function* — [line 76](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/plugins/builtin_debug/sampling.py#L76)

`log_sampling_iteration(payload: SamplingIterationPayload, ctx: PluginContext) -> None`

Log validation results for each sampling attempt.

### `log_sampling_repair()`

*async function* — [line 114](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/plugins/builtin_debug/sampling.py#L114)

`log_sampling_repair(payload: SamplingRepairPayload, ctx: PluginContext) -> None`

Log when repair is triggered during sampling iterations.

### `log_sampling_loop_end()`

*async function* — [line 137](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/plugins/builtin_debug/sampling.py#L137)

`log_sampling_loop_end(payload: SamplingLoopEndPayload, ctx: PluginContext) -> None`

Log sampling completion with success status and attempt statistics.

---

## Module `mellea.plugins.builtin_debug.validation`

Source: [`mellea/plugins/builtin_debug/validation.py`](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/plugins/builtin_debug/validation.py) at commit `a535fc6345a0`.

Built-in debug plugin for validation pipeline.

### `log_validation_pre_check()`

*async function* — [line 45](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/plugins/builtin_debug/validation.py#L45)

`log_validation_pre_check(payload: ValidationPreCheckPayload, ctx: PluginContext) -> None`

Log validation setup before requirements are checked.

### `log_validation_post_check()`

*async function* — [line 70](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/plugins/builtin_debug/validation.py#L70)

`log_validation_post_check(payload: ValidationPostCheckPayload, ctx: PluginContext) -> None`

Log validation results after requirements are checked.

---

*Generated from source by `docs/scripts/generate_package_map.py` — do not edit by hand; regenerate instead. Symbols and annotations reflect the pinned commit exactly; where type annotations are absent in source, this page says so rather than guessing.*
