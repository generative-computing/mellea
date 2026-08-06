---
id: sampling
title: "mellea.core.sampling"
sidebar_label: "sampling"
sidebar_position: 5
description: "Abstract interfaces for sampling strategies and their results."
# diataxis: reference
---

Source: [`mellea/core/sampling.py`](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/core/sampling.py) at commit `a535fc6345a0`.

Abstract interfaces for sampling strategies and their results.

## `SamplingResult`

*class* — [line 39](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/core/sampling.py#L39) (`CBlock`, `Generic[S]`)

Stores the results from a sampling operation. This includes successful and failed samplings.

Constructor: `SamplingResult(result_index: int, success: bool, *, sample_generations: list[ComputedModelOutputThunk[S]] | None = None, sample_validations: list[list[tuple[Requirement, ValidationResult]]] | None = None, sample_actions: Sequence[SampleActionType] | None = None, sample_contexts: list[Context] | None = None)`

Properties:

- `result` → `ComputedModelOutputThunk[S]` — The final output or result from applying the sampling strategy.
- `result_ctx` → `Context` — The context of the final output or result from applying the sampling strategy.
- `result_action` → `SampleActionType` — The action that generated the final output or result from applying the sampling strategy.
- `result_validations` → `list[tuple[Requirement, ValidationResult]]` — The validation results associated with the final output or result from applying the sampling strategy.

## `SamplingStrategy`

*class* — [line 122](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/core/sampling.py#L122) (`abc.ABC`)

A SamplingStrategy class defines an abstract base class for implementing various sampling strategies.

Methods (defined on this class; inherited members not listed):

- `sample(action: Component[S] | CBlock | ModelOutputThunk, context: Context, backend: Backend, requirements: list[Requirement] | None, *, validation_ctx: Context | None = None, format: type[BaseModelSubclass] | None = None, model_options: dict | None = None, tool_calls: bool = False) -> SamplingResult[S]` *(async)* — [line 130](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/core/sampling.py#L130)
  This method is the abstract method for sampling a given component.

---

*Generated from source by `docs/scripts/generate_package_map.py` — do not edit by hand; regenerate instead. Symbols and annotations reflect the pinned commit exactly; where type annotations are absent in source, this page says so rather than guessing.*
