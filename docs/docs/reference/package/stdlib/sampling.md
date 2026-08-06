---
id: sampling
title: "mellea.stdlib.sampling"
sidebar_label: "sampling"
sidebar_position: 7
description: "sampling methods go here."
# diataxis: reference
---

Source: [`mellea/stdlib/sampling/__init__.py`](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/sampling/__init__.py) at commit `a535fc6345a0`.

sampling methods go here.

Declared exports (`__all__`): `BaseSamplingStrategy`, `BudgetForcingSamplingStrategy`, `MBRDRougeLStrategy`, `MajorityVotingStrategyForMath`, `ModelFriendlyFeedbackFormatter`, `ModelFriendlyRepairStrategy`, `MultiTurnStrategy`, `RejectionSamplingStrategy`, `RepairTemplateStrategy`, `SamplingPreset`, `SamplingResult`, `SamplingStrategy`, `python_code_generation_sampling`, `python_plotting_sampling`

---

## Module `mellea.stdlib.sampling.base`

Source: [`mellea/stdlib/sampling/base.py`](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/sampling/base.py) at commit `a535fc6345a0`.

Base Sampling Strategies.

### `BaseSamplingStrategy`

*class* — [line 116](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/sampling/base.py#L116) (`SamplingStrategy`)

Base class for multiple strategies that reject samples based on given instructions.

Constructor: `BaseSamplingStrategy(*, loop_budget: int = 1, concurrency_budget: int = 1, requirements: list[Requirement] | None = None)`

Methods (defined on this class; inherited members not listed):

- `repair(old_ctx: Context, new_ctx: Context, past_actions: Sequence[SampleActionType], past_results: list[ComputedModelOutputThunk], past_val: list[list[tuple[Requirement, ValidationResult]]]) -> tuple[SampleActionType, Context]` *(staticmethod)* — [line 163](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/sampling/base.py#L163)
  Repair function that is being invoked if not all requirements are fulfilled. It should return a next action component.
- `select_from_failure(sampled_actions: Sequence[SampleActionType], sampled_results: list[ComputedModelOutputThunk], sampled_val: list[list[tuple[Requirement, ValidationResult]]]) -> int` *(staticmethod)* — [line 188](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/sampling/base.py#L188)
  This function returns the index of the result that should be selected as `.value` iff the loop budget is exhausted and no success.
- `sample(action: Component[S] | CBlock | ModelOutputThunk, context: Context, backend: Backend, requirements: list[Requirement] | None, *, validation_ctx: Context | None = None, format: type[BaseModelSubclass] | None = None, model_options: dict | None = None, tool_calls: bool = False, show_progress: bool = True) -> SamplingResult[S]` *(async)* — [line 205](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/sampling/base.py#L205)
  This method performs a sampling operation based on the given instruction.

### `RejectionSamplingStrategy`

*class* — [line 634](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/sampling/base.py#L634) (`BaseSamplingStrategy`)

Simple rejection sampling strategy that just repeats the same call on failure.

Methods (defined on this class; inherited members not listed):

- `select_from_failure(sampled_actions: Sequence[SampleActionType], sampled_results: list[ComputedModelOutputThunk], sampled_val: list[list[tuple[Requirement, ValidationResult]]]) -> int` *(staticmethod)* — [line 638](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/sampling/base.py#L638)
  Always returns the 0th index.
- `repair(old_ctx: Context, new_ctx: Context, past_actions: Sequence[SampleActionType], past_results: list[ComputedModelOutputThunk], past_val: list[list[tuple[Requirement, ValidationResult]]]) -> tuple[SampleActionType, Context]` *(staticmethod)* — [line 656](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/sampling/base.py#L656)
  Always returns the unedited, last action.

### `RepairTemplateStrategy`

*class* — [line 678](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/sampling/base.py#L678) (`BaseSamplingStrategy`)

A sampling strategy that adds a repair string to the instruction object.

Methods (defined on this class; inherited members not listed):

- `select_from_failure(sampled_actions: Sequence[SampleActionType], sampled_results: list[ComputedModelOutputThunk], sampled_val: list[list[tuple[Requirement, ValidationResult]]]) -> int` *(staticmethod)* — [line 682](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/sampling/base.py#L682)
  Always returns the 0th index.
- `repair(old_ctx: Context, new_ctx: Context, past_actions: Sequence[SampleActionType], past_results: list[ComputedModelOutputThunk], past_val: list[list[tuple[Requirement, ValidationResult]]]) -> tuple[SampleActionType, Context]` *(staticmethod)* — [line 700](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/sampling/base.py#L700)
  Adds a description of the requirements that failed to a copy of the original instruction.

### `MultiTurnStrategy`

*class* — [line 743](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/sampling/base.py#L743) (`BaseSamplingStrategy`)

Rejection sampling strategy with (agentic) multi-turn repair.

Methods (defined on this class; inherited members not listed):

- `select_from_failure(sampled_actions: Sequence[SampleActionType], sampled_results: list[ComputedModelOutputThunk], sampled_val: list[list[tuple[Requirement, ValidationResult]]]) -> int` *(staticmethod)* — [line 747](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/sampling/base.py#L747)
  Always returns the last index. The last message from the model will always be returned if all results are failures.
- `repair(old_ctx: Context, new_ctx: Context, past_actions: Sequence[SampleActionType], past_results: list[ComputedModelOutputThunk], past_val: list[list[tuple[Requirement, ValidationResult]]]) -> tuple[SampleActionType, Context]` *(staticmethod)* — [line 768](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/sampling/base.py#L768)
  Returns a Message with a description (and validation reasons) of the failed requirements.

---

## Module `mellea.stdlib.sampling.budget_forcing`

Source: [`mellea/stdlib/sampling/budget_forcing.py`](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/sampling/budget_forcing.py) at commit `a535fc6345a0`.

Sampling Strategies for budget forcing generation.

### `BudgetForcingSamplingStrategy`

*class* — [line 32](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/sampling/budget_forcing.py#L32) (`RejectionSamplingStrategy`)

Sampling strategy that enforces a token budget for chain-of-thought reasoning.

Constructor: `BudgetForcingSamplingStrategy(*, think_max_tokens: int | None = 4096, answer_max_tokens: int | None = None, start_think_token: str | None = '<think>', end_think_token: str | None = '</think>', begin_response_token: str | None = '', end_response_token: str = '', think_more_suffix: str | None = '', answer_suffix: str | None = '', loop_budget: int = 1, requirements: list[Requirement] | None)`

Methods (defined on this class; inherited members not listed):

- `sample(action: Component[S] | CBlock | ModelOutputThunk, context: Context, backend: Backend, requirements: list[Requirement] | None, *, validation_ctx: Context | None = None, format: type[BaseModelSubclass] | None = None, model_options: dict | None = None, tool_calls: bool = False, show_progress: bool = True) -> SamplingResult[S]` *(async)* — [line 101](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/sampling/budget_forcing.py#L101)
  This method performs a sampling operation based on the given instruction.

---

## Module `mellea.stdlib.sampling.feedback`

Source: [`mellea/stdlib/sampling/feedback.py`](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/sampling/feedback.py) at commit `a535fc6345a0`.

Model-friendly feedback formatters for validation failures.

### `ModelFriendlyFeedbackFormatter`

*class* — [line 62](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/sampling/feedback.py#L62) 

Converts validation failures into model-friendly repair instructions.

Methods (defined on this class; inherited members not listed):

- `format_python_syntax_error(validation_result: ValidationResult) -> str` *(staticmethod)* — [line 74](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/sampling/feedback.py#L74)
  Format syntax errors into actionable guidance.
- `format_import_error(validation_result: ValidationResult) -> str` *(staticmethod)* — [line 112](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/sampling/feedback.py#L112)
  Format import restriction violations into actionable guidance.
- `format_execution_error(validation_result: ValidationResult) -> str` *(staticmethod)* — [line 144](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/sampling/feedback.py#L144)
  Format runtime/execution errors into actionable guidance.
- `format_output_size_error(validation_result: ValidationResult) -> str` *(staticmethod)* — [line 210](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/sampling/feedback.py#L210)
  Format output size limit violations into actionable guidance.
- `format_matplotlib_error(validation_result: ValidationResult) -> str` *(staticmethod)* — [line 241](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/sampling/feedback.py#L241)
  Format matplotlib-specific errors into actionable guidance.
- `format_extraction_error(validation_result: ValidationResult) -> str` *(staticmethod)* — [line 283](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/sampling/feedback.py#L283)
  Format code extraction errors into actionable guidance.
- `format_requirement_reason(requirement: Requirement, validation_result: ValidationResult) -> str` *(classmethod)* — [line 311](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/sampling/feedback.py#L311)
  Intelligently format feedback based on requirement type.

### `ModelFriendlyRepairStrategy`

*class* — [line 352](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/sampling/feedback.py#L352) (`RepairTemplateStrategy`)

RepairTemplateStrategy with model-friendly feedback formatting.

Methods (defined on this class; inherited members not listed):

- `repair(old_ctx: Context, new_ctx: Context, past_actions: Sequence[SampleActionType], past_results: list[Any], past_val: list[list[tuple[Requirement, ValidationResult]]]) -> tuple[SampleActionType, Context]` *(staticmethod)* — [line 369](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/sampling/feedback.py#L369)
  Repair with model-friendly feedback formatting.

---

## Module `mellea.stdlib.sampling.majority_voting`

Source: [`mellea/stdlib/sampling/majority_voting.py`](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/sampling/majority_voting.py) at commit `a535fc6345a0`.

Sampling Strategies for Minimum Bayes Risk Decoding (MBRD).

### `BaseMBRDSampling`

*class* — [line 27](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/sampling/majority_voting.py#L27) (`RejectionSamplingStrategy`)

Abstract Minimum Bayes Risk Decoding (MBRD) Sampling Strategy.

Constructor: `BaseMBRDSampling(*, number_of_samples: int = 8, weighted: bool = False, loop_budget: int = 1, requirements: list[Requirement] | None = None)`

Methods (defined on this class; inherited members not listed):

- `compare_strings(ref: str, pred: str) -> float` — [line 68](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/sampling/majority_voting.py#L68)
  Compute a similarity score between a reference and a predicted string.
- `maybe_apply_weighted(scr: np.ndarray) -> np.ndarray` — [line 82](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/sampling/majority_voting.py#L82)
  Apply per-sample weights to the score vector if `self.weighted` is `True`.
- `sample(action: Component[S] | CBlock | ModelOutputThunk, context: Context, backend: Backend, requirements: list[Requirement] | None, *, validation_ctx: Context | None = None, format: type[BaseModelSubclass] | None = None, model_options: dict | None = None, tool_calls: bool = False, show_progress: bool = True) -> SamplingResult[S]` *(async)* — [line 102](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/sampling/majority_voting.py#L102)
  Samples using majority voting.

### `MajorityVotingStrategyForMath`

*class* — [line 195](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/sampling/majority_voting.py#L195) (`BaseMBRDSampling`)

MajorityVoting Sampling Strategy for Math Expressions.

Constructor: `MajorityVotingStrategyForMath(*, number_of_samples: int = 8, float_rounding: int = 6, strict: bool = True, allow_set_relation_comp: bool = False, weighted: bool = False, loop_budget: int = 1, requirements: list[Requirement] | None = None)`

Methods (defined on this class; inherited members not listed):

- `compare_strings(ref: str, pred: str) -> float` — [line 259](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/sampling/majority_voting.py#L259)
  Compare two strings using math-aware extraction and verification.

### `MBRDRougeLStrategy`

*class* — [line 297](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/sampling/majority_voting.py#L297) (`BaseMBRDSampling`)

Sampling Strategy that uses RougeL to compute symbol-level distances for majority voting.

Constructor: `MBRDRougeLStrategy(*, number_of_samples: int = 8, weighted: bool = False, loop_budget: int = 1, requirements: list[Requirement] | None = None)`

Methods (defined on this class; inherited members not listed):

- `compare_strings(ref: str, pred: str) -> float` — [line 343](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/sampling/majority_voting.py#L343)
  Compare two strings using the RougeL F-measure.

---

## Module `mellea.stdlib.sampling.presets`

Source: [`mellea/stdlib/sampling/presets.py`](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/sampling/presets.py) at commit `a535fc6345a0`.

Pre-configured sampling presets bundling requirements and strategies.

### `SamplingPreset`

*class* — [line 41](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/sampling/presets.py#L41) (`Generic[S]`)

Bundle of requirements and strategy for a specific use case.

### `python_code_generation_sampling()`

*function* — [line 66](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/sampling/presets.py#L66)

`python_code_generation_sampling(loop_budget: int = 2, *, allowed_imports: list[str] | None = None, output_limit_chars: int = 10000, timeout_seconds: int = 5, use_sandbox: bool = False) -> SamplingPreset`

Pre-configured preset for Python code generation with repair feedback.

### `python_plotting_sampling()`

*function* — [line 150](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/sampling/presets.py#L150)

`python_plotting_sampling(output_path: str | None = None, loop_budget: int = 3, *, allowed_imports: list[str] | None = None, timeout_seconds: int = 10, use_sandbox: bool = True) -> SamplingPreset`

Pre-configured preset for matplotlib plotting with repair feedback.

---

## Module `mellea.stdlib.sampling.sampling_algos`

Source: [`mellea/stdlib/sampling/sampling_algos/__init__.py`](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/sampling/sampling_algos/__init__.py) at commit `a535fc6345a0`.

Module for Sampling Algorithms.

Declared exports (`__all__`): `think_budget_forcing`

---

## Module `mellea.stdlib.sampling.sampling_algos.budget_forcing_alg`

Source: [`mellea/stdlib/sampling/sampling_algos/budget_forcing_alg.py`](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/sampling/sampling_algos/budget_forcing_alg.py) at commit `a535fc6345a0`.

Budget-forcing generation algorithm for thinking models.

### `think_budget_forcing()`

*async function* — [line 26](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/sampling/sampling_algos/budget_forcing_alg.py#L26)

`think_budget_forcing(backend: OllamaModelBackend, action: Span, *, ctx: Context, format: type[BaseModelSubclass] | None = None, tool_calls: bool = False, think_max_tokens: int | None = 4096, answer_max_tokens: int | None = None, start_think_token: str | None = '<think>', end_think_token: str | None = '</think>', begin_response_token: str | None = '', think_more_suffix: str | None = '', answer_suffix: str | None = '', model_options: dict | None = None) -> ModelOutputThunk`

Generate with budget forcing using the completions APIs.

---

## Module `mellea.stdlib.sampling.sofai`

Source: [`mellea/stdlib/sampling/sofai.py`](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/sampling/sofai.py) at commit `a535fc6345a0`.

SOFAI (Slow and Fast AI) Sampling Strategy.

### `SOFAISamplingStrategy`

*class* — [line 45](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/sampling/sofai.py#L45) (`SamplingStrategy`)

SOFAI (Slow and Fast AI) two-solver sampling strategy.

Constructor: `SOFAISamplingStrategy(s1_solver_backend: Backend, s2_solver_backend: Backend, s2_solver_mode: Literal['fresh_start', 'continue_chat', 'best_attempt'] = 'fresh_start', *, loop_budget: int = 3, judge_backend: Backend | None = None, feedback_strategy: Literal['simple', 'first_error', 'all_errors'] = 'simple')`

Methods (defined on this class; inherited members not listed):

- `repair(old_ctx: Context, new_ctx: Context, past_actions: Sequence[SampleActionType], past_results: list[ComputedModelOutputThunk], past_val: list[list[tuple[Requirement, ValidationResult]]]) -> tuple[SampleActionType, Context]` *(staticmethod)* — [line 113](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/sampling/sofai.py#L113)
  Create targeted feedback message from validation results.
- `select_from_failure(sampled_actions: Sequence[SampleActionType], sampled_results: list[ComputedModelOutputThunk], sampled_val: list[list[tuple[Requirement, ValidationResult]]]) -> int` *(staticmethod)* — [line 161](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/sampling/sofai.py#L161)
  Select the most informed attempt (last) when all fail.
- `sample(action: Component[S] | CBlock | ModelOutputThunk, context: Context, backend: Backend, requirements: list[Requirement] | None, *, validation_ctx: Context | None = None, format: type[BaseModelSubclass] | None = None, model_options: dict | None = None, tool_calls: bool = False) -> SamplingResult[S]` *(async)* — [line 577](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/sampling/sofai.py#L577)
  Execute SOFAI two-solver sampling strategy.

---

*Generated from source by `docs/scripts/generate_package_map.py` — do not edit by hand; regenerate instead. Symbols and annotations reflect the pinned commit exactly; where type annotations are absent in source, this page says so rather than guessing.*
