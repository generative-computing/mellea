---
id: requirements
title: "mellea.stdlib.requirements"
sidebar_label: "requirements"
sidebar_position: 6
description: "Module for working with Requirements."
# diataxis: reference
---

Source: [`mellea/stdlib/requirements/__init__.py`](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/requirements/__init__.py) at commit `a535fc6345a0`.

Module for working with Requirements.

Declared exports (`__all__`): `ALoraRequirement`, `GroundednessRequirement`, `ImportRestrictions`, `LLMaJRequirement`, `MatplotlibHeadlessBackend`, `NoImportRestrictions`, `OutputSizeLimit`, `PlotDependenciesAvailable`, `PlotFileSaved`, `PythonCodeExtraction`, `PythonExecutionReq`, `PythonSyntaxValid`, `Requirement`, `ValidationResult`, `as_markdown_list`, `check`, `default_output_to_bool`, `is_markdown_list`, `is_markdown_table`, `python_code_generation_requirements`, `python_plotting_requirements`, `req`, `reqify`, `requirement_check_to_bool`, `simple_validate`, `tool_arg_validator`, `uses_tool`

---

## Module `mellea.stdlib.requirements.md`

Source: [`mellea/stdlib/requirements/md.py`](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/requirements/md.py) at commit `a535fc6345a0`.

This file contains various requirements for Markdown-formatted files.

### `as_markdown_list()`

*function* — [line 30](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/requirements/md.py#L30)

`as_markdown_list(ctx: Context) -> list[str] | None`

Attempts to format the last_output of the given context as a markdown list.

---

## Module `mellea.stdlib.requirements.plotting`

Source: [`mellea/stdlib/requirements/plotting/__init__.py`](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/requirements/plotting/__init__.py) at commit `a535fc6345a0`.

Matplotlib-specific requirements for validating plotting code.

Declared exports (`__all__`): `MatplotlibHeadlessBackend`, `PlotDependenciesAvailable`, `PlotFileSaved`, `python_plotting_requirements`

---

## Module `mellea.stdlib.requirements.plotting.matplotlib`

Source: [`mellea/stdlib/requirements/plotting/matplotlib.py`](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/requirements/plotting/matplotlib.py) at commit `a535fc6345a0`.

Matplotlib-specific code generation requirements.

### `MatplotlibHeadlessBackend`

*class* — [line 200](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/requirements/plotting/matplotlib.py#L200) (`Requirement`)

Validates that matplotlib is configured with a headless backend.

Constructor: `MatplotlibHeadlessBackend() -> None`

### `PlotFileSaved`

*class* — [line 257](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/requirements/plotting/matplotlib.py#L257) (`Requirement`)

Validates that a plot is explicitly saved to a file.

Constructor: `PlotFileSaved(output_path: str) -> None`

### `PlotDependenciesAvailable`

*class* — [line 304](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/requirements/plotting/matplotlib.py#L304) (`Requirement`)

Validates that matplotlib and numpy are importable.

Constructor: `PlotDependenciesAvailable() -> None`

### `python_plotting_requirements()`

*function* — [line 343](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/requirements/plotting/matplotlib.py#L343)

`python_plotting_requirements(output_path: str, allowed_imports: list[str] | None = None, output_limit_chars: int = 10000, timeout_seconds: int = 5, use_sandbox: bool = False) -> list[Requirement]`

Bundle matplotlib-specific requirements for plotting code validation.

---

## Module `mellea.stdlib.requirements.python_reqs`

Source: [`mellea/stdlib/requirements/python_reqs.py`](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/requirements/python_reqs.py) at commit `a535fc6345a0`.

Requirements for Python code generation validation.

### `PythonExecutionReq`

*class* — [line 266](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/requirements/python_reqs.py#L266) (`Requirement`)

Verifies that Python code runs without raising exceptions.

Constructor: `PythonExecutionReq(execution_tier: ExecutionTier = 'static', *, policy: CapabilityPolicy | None = None, allowed_imports: list[str] | None = None, max_output_chars: int | None = None, timeout: int | None = None, allow_unsafe_execution: bool = False, use_sandbox: bool = False)`

---

## Module `mellea.stdlib.requirements.python_tools`

Source: [`mellea/stdlib/requirements/python_tools.py`](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/requirements/python_tools.py) at commit `a535fc6345a0`.

Generic Python tool requirements for code generation validation.

### `PythonCodeExtraction`

*class* — [line 35](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/requirements/python_tools.py#L35) (`Requirement`)

Code blocks are present and extractable from model output.

Constructor: `PythonCodeExtraction() -> None`

### `PythonSyntaxValid`

*class* — [line 51](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/requirements/python_tools.py#L51) (`Requirement`)

Python code is syntactically valid (parses without AST errors).

Constructor: `PythonSyntaxValid() -> None`

### `OutputSizeLimit`

*class* — [line 98](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/requirements/python_tools.py#L98) (`Requirement`)

Captured output does not exceed size limit (in characters).

Constructor: `OutputSizeLimit(limit_chars: int = 10000, execution_tier: ExecutionTier = 'static', policy: CapabilityPolicy | None = None, allowed_imports: list[str] | None = None) -> None`

### `ImportRestrictions`

*class* — [line 226](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/requirements/python_tools.py#L226) (`Requirement`)

Only whitelisted modules are imported in the code.

Constructor: `ImportRestrictions(allowed_imports: list[str] | None = None) -> None`

### `NoImportRestrictions`

*class* — [line 332](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/requirements/python_tools.py#L332) (`Requirement`)

Explicit no-op requirement indicating no import checks are configured.

Constructor: `NoImportRestrictions() -> None`

### `python_code_generation_requirements()`

*function* — [line 353](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/requirements/python_tools.py#L353)

`python_code_generation_requirements(allowed_imports: list[str] | None = None, output_limit_chars: int = 10000, timeout_seconds: int = 5, use_sandbox: bool = False) -> list[Requirement]`

Bundle generic Python tool requirements with configurable parameters.

---

## Module `mellea.stdlib.requirements.rag`

Source: [`mellea/stdlib/requirements/rag.py`](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/requirements/rag.py) at commit `a535fc6345a0`.

Requirements for RAG (Retrieval-Augmented Generation) workflows.

### `GroundednessRequirement`

*class* — [line 25](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/requirements/rag.py#L25) (`Requirement`)

Requirement that validates LLM responses are grounded by citations.

Constructor: `GroundednessRequirement(documents: Iterable[Document] | Iterable[str] | None = None, allow_partial_support: bool = False, max_new_tokens: int = 500, description: str | None = None)`

Methods (defined on this class; inherited members not listed):

- `validate(backend: Backend, ctx: Context, *, format: type | None = None, model_options: dict | None = None) -> ValidationResult` *(async)* — [line 109](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/requirements/rag.py#L109)
  Validate groundedness of the response using the 4-step pipeline.

---

## Module `mellea.stdlib.requirements.requirement`

Source: [`mellea/stdlib/requirements/requirement.py`](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/requirements/requirement.py) at commit `a535fc6345a0`.

Requirements are a special type of Component used as input to the "validate" step in Instruct/Validate/Repair design patterns.

### `LLMaJRequirement`

*class* — [line 23](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/requirements/requirement.py#L23) (`Requirement`)

A requirement that always uses LLM-as-a-Judge. Any available constraint ALoRA will be ignored.

### `ALoraRequirement`

*class* — [line 83](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/requirements/requirement.py#L83) (`Requirement`, `Intrinsic`)

A requirement validated by an ALoRA adapter; falls back to LLM-as-a-Judge only on generation error.

Constructor: `ALoraRequirement(description: str, intrinsic_name: str | None = None)`

### `requirement_check_to_bool()`

*function* — [line 34](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/requirements/requirement.py#L34)

`requirement_check_to_bool(x: CBlock | ModelOutputThunk | str) -> bool`

Convert a `requirement-check` adapter output string to a boolean result.

### `reqify()`

*function* — [line 125](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/requirements/requirement.py#L125)

`reqify(r: str | Requirement) -> Requirement`

Map strings to Requirements.

### `req()`

*function* — [line 147](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/requirements/requirement.py#L147)

`req(*args, **kwargs) -> Requirement`

Shorthand for `Requirement.__init__`.

### `check()`

*function* — [line 160](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/requirements/requirement.py#L160)

`check(*args, **kwargs) -> Requirement`

Shorthand for `Requirement.__init__(..., check_only=True)`.

### `simple_validate()`

*function* — [line 185](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/requirements/requirement.py#L185)

`simple_validate(fn: Callable[[str], Any], *, reason: str | None = None) -> Callable[[Context], ValidationResult]`

Syntactic sugar for writing validation functions that only operate over the last output from the model (interpreted as a string).

---

## Module `mellea.stdlib.requirements.safety`

Source: [`mellea/stdlib/requirements/safety/__init__.py`](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/requirements/safety/__init__.py) at commit `a535fc6345a0`.

Utilities for safe/responsible AI live here.

---

## Module `mellea.stdlib.requirements.safety.guardian`

Source: [`mellea/stdlib/requirements/safety/guardian.py`](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/requirements/safety/guardian.py) at commit `a535fc6345a0`.

Risk checking with Granite Guardian models via existing backends.

### `GuardianRisk`

*class* — [line 24](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/requirements/safety/guardian.py#L24) (`Enum`)

Risk definitions for Granite Guardian models.

Methods (defined on this class; inherited members not listed):

- `get_available_risks() -> list[str]` *(classmethod)* — [line 54](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/requirements/safety/guardian.py#L54)
  Return a list of all available risk type identifiers.

### `GuardianCheck`

*class* — [line 90](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/requirements/safety/guardian.py#L90) (`Requirement`)

Enhanced risk checking using Granite Guardian 3.3 8B with multiple backend support.

Constructor: `GuardianCheck(risk: str | GuardianRisk | None = None, *, backend_type: BackendType = 'ollama', model_version: str | None = None, device: str | None = None, ollama_url: str | None = None, thinking: bool = False, custom_criteria: str | None = None, context_text: str | None = None, tools: list[dict] | None = None, backend: Backend | None = None)`

Methods (defined on this class; inherited members not listed):

- `get_effective_risk() -> str` — [line 208](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/requirements/safety/guardian.py#L208)
  Return the effective risk criteria to use for validation.
- `get_available_risks() -> list[str]` *(classmethod)* — [line 220](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/requirements/safety/guardian.py#L220)
  Return a list of all available standard risk type identifiers.
- `validate(backend: Backend, ctx: Context, *, format: type[BaseModelSubclass] | None = None, model_options: dict | None = None) -> ValidationResult` *(async)* — [line 248](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/requirements/safety/guardian.py#L248)
  Validate a conversation using Granite Guardian via the selected backend.

---

## Module `mellea.stdlib.requirements.tool_reqs`

Source: [`mellea/stdlib/requirements/tool_reqs.py`](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/requirements/tool_reqs.py) at commit `a535fc6345a0`.

`Requirement` factories for tool-use validation.

### `uses_tool()`

*function* — [line 29](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/requirements/tool_reqs.py#L29)

`uses_tool(tool_name: str | Callable, check_only: bool = False) -> Requirement`

Forces the model to call a given tool.

### `tool_arg_validator()`

*function* — [line 59](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/requirements/tool_reqs.py#L59)

`tool_arg_validator(description: str, tool_name: str | Callable | None, arg_name: str, validation_fn: Callable, check_only: bool = False) -> Requirement`

A requirement that passes only if `validation_fn` returns a True value for the *value* of the `arg_name` argument to `tool_name`.

---

*Generated from source by `docs/scripts/generate_package_map.py` — do not edit by hand; regenerate instead. Symbols and annotations reflect the pinned commit exactly; where type annotations are absent in source, this page says so rather than guessing.*
