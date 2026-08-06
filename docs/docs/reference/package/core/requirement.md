---
id: requirement
title: "mellea.core.requirement"
sidebar_label: "requirement"
sidebar_position: 4
description: "`Requirement` interface for constrained and validated generation."
# diataxis: reference
---

Source: [`mellea/core/requirement.py`](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/core/requirement.py) at commit `a535fc6345a0`.

`Requirement` interface for constrained and validated generation.

## `ValidationResult`

*class* — [line 32](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/core/requirement.py#L32) 

ValidationResults store the output of a Requirement's validation. They can be used to return additional info from validation functions, which is useful for sampling/repairing.

Constructor: `ValidationResult(result: bool, *, reason: str | None = None, score: float | None = None, thunk: ModelOutputThunk | None = None, context: Context | None = None)`

Properties:

- `reason` → `str | None` — Reason for the validation result.
- `score` → `float | None` — An optional score for the validation result.
- `thunk` → `ModelOutputThunk | None` — The ModelOutputThunk associated with the validation func if an llm was used to generate the final result.
- `context` → `Context | None` — The context associated with validation if a backend was used to generate the final result.

Methods (defined on this class; inherited members not listed):

- `as_bool() -> bool` — [line 80](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/core/requirement.py#L80)
  Return a boolean value based on the validation result.

## `PartialValidationResult`

*class* — [line 97](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/core/requirement.py#L97) 

Tri-state result from per-chunk streaming validation.

Constructor: `PartialValidationResult(success: Literal['pass', 'fail', 'unknown'], *, reason: str | None = None, score: float | None = None, thunk: ModelOutputThunk | None = None, context: Context | None = None)`

Properties:

- `success` → `Literal['pass', 'fail', 'unknown']` — The tri-state validation result.
- `reason` → `str | None` — Reason for the validation result.
- `score` → `float | None` — An optional score for the validation result.
- `thunk` → `ModelOutputThunk | None` — The ModelOutputThunk associated with the validation call, if any.
- `context` → `Context | None` — The context associated with the validation call, if any.

Methods (defined on this class; inherited members not listed):

- `as_bool() -> bool` — [line 162](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/core/requirement.py#L162)
  Return True for `"pass"`, False for `"fail"` or `"unknown"`.

## `Requirement`

*class* — [line 208](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/core/requirement.py#L208) (`Component[str]`)

Requirements are a special type of Component used as input to the Validate step in Instruct/Validate/Repair patterns.

Constructor: `Requirement(description: str | None = None, validation_fn: Callable[[Context], ValidationResult] | None = None, *, output_to_bool: Callable[[CBlock | ModelOutputThunk | str], bool] | None = default_output_to_bool, check_only: bool = False)`

Methods (defined on this class; inherited members not listed):

- `validate(backend: Backend, ctx: Context, *, format: type[BaseModelSubclass] | None = None, model_options: dict | None = None) -> ValidationResult` *(async)* — [line 249](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/core/requirement.py#L249)
  Chooses the appropriate validation strategy and applies it to the given context.
- `stream_validate(chunk: str, *, backend: Backend, ctx: Context) -> PartialValidationResult` *(async)* — [line 318](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/core/requirement.py#L318)
  Hook for per-chunk streaming validation.
- `parts() -> list[Span]` — [line 368](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/core/requirement.py#L368)
  Returns all of the constituent parts of a Requirement.
- `format_for_llm() -> TemplateRepresentation | str` — [line 377](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/core/requirement.py#L377)
  Returns a `TemplateRepresentation` for LLM-as-a-Judge evaluation of this requirement.

## `default_output_to_bool()`

*function* — [line 184](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/core/requirement.py#L184)

`default_output_to_bool(x: CBlock | ModelOutputThunk | str) -> bool`

Convert a model output string to a boolean by checking for a "yes" answer.

---

*Generated from source by `docs/scripts/generate_package_map.py` — do not edit by hand; regenerate instead. Symbols and annotations reflect the pinned commit exactly; where type annotations are absent in source, this page says so rather than guessing.*
