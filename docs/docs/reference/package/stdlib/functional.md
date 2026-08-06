---
id: functional
title: "mellea.stdlib.functional"
sidebar_label: "functional"
sidebar_position: 5
description: "Low-level primitives for Mellea operations: Instruct, Chat, and friends."
# diataxis: reference
---

Source: [`mellea/stdlib/functional.py`](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/functional.py) at commit `a535fc6345a0`.

Low-level primitives for Mellea operations: Instruct, Chat, and friends.

## `act()`

*function* — [line 90](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/functional.py#L90)

`act(action: Component[S] | CBlock | ModelOutputThunk, context: Context, backend: Backend, *, requirements: list[Requirement] | None = None, strategy: SamplingStrategy | None = None, return_sampling_results: bool = False, format: type[BaseModelSubclass] | None = None, model_options: dict | None = None, tool_calls: bool = False) -> tuple[ComputedModelOutputThunk[S], Context] | SamplingResult[S]`

Runs a generic action, and adds both the action and the result to the context.

## `instruct()`

*function* — [line 196](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/functional.py#L196)

`instruct(description: str, context: Context, backend: Backend, *, images: list[ImageBlock | ImageUrlBlock] | list[PILImage.Image] | None = None, audio: list[AudioBlock | AudioUrlBlock] | None = None, requirements: list[Requirement | str] | None = None, icl_examples: list[str | CBlock] | None = None, grounding_context: dict[str, str | Span] | None = None, user_variables: dict[str, str] | None = None, prefix: str | CBlock | None = None, output_prefix: str | CBlock | None = None, strategy: SamplingStrategy | None = RejectionSamplingStrategy(loop_budget=2), return_sampling_results: bool = False, format: type[BaseModelSubclass] | None = None, model_options: dict | None = None, tool_calls: bool = False) -> tuple[ComputedModelOutputThunk[str], Context] | SamplingResult[str]`

Generates from an instruction.

## `chat()`

*function* — [line 277](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/functional.py#L277)

`chat(content: str, context: Context, backend: Backend, *, role: Message.Role = 'user', images: list[ImageBlock | ImageUrlBlock] | list[PILImage.Image] | None = None, audio: list[AudioBlock | AudioUrlBlock] | None = None, documents: Iterable[str | Document] | None = None, user_variables: dict[str, str] | None = None, format: type[BaseModelSubclass] | None = None, model_options: dict | None = None, tool_calls: bool = False) -> tuple[Message, Context]`

Sends a simple chat message and returns the response. Adds both messages to the Context.

## `validate()`

*function* — [line 341](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/functional.py#L341)

`validate(reqs: Requirement | list[Requirement], context: Context, backend: Backend, *, output: CBlock | ModelOutputThunk | None = None, format: type[BaseModelSubclass] | None = None, model_options: dict | None = None, generate_logs: list[GenerateLog] | None = None, input: CBlock | ModelOutputThunk | None = None) -> list[ValidationResult]`

Validates a set of requirements over the output (if provided) or the current context (if the output is not provided).

## `query()`

*function* — [line 387](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/functional.py#L387)

`query(obj: Any, query: str, context: Context, backend: Backend, *, format: type[BaseModelSubclass] | None = None, model_options: dict | None = None, tool_calls: bool = False) -> tuple[ComputedModelOutputThunk, Context]`

Query method for retrieving information from an object.

## `transform()`

*function* — [line 430](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/functional.py#L430)

`transform(obj: Any, transformation: str, context: Context, backend: Backend, *, format: type[BaseModelSubclass] | None = None, model_options: dict | None = None) -> tuple[ModelOutputThunk | Any, Context]`

Transform method for creating a new object with the transformation applied.

## `aact()`

*async function* — [line 580](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/functional.py#L580)

`aact(action: Component[S] | CBlock | ModelOutputThunk, context: Context, backend: Backend, *, requirements: list[Requirement] | None = None, strategy: SamplingStrategy | None = None, return_sampling_results: bool = False, format: type[BaseModelSubclass] | None = None, model_options: dict | None = None, tool_calls: bool = False, silence_context_type_warning: bool = False, await_result: bool = False) -> tuple[ModelOutputThunk[S], Context] | SamplingResult`

Asynchronous version of .act; runs a generic action, and adds both the action and the result to the context.

## `ainstruct()`

*async function* — [line 873](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/functional.py#L873)

`ainstruct(description: str, context: Context, backend: Backend, *, images: list[ImageBlock | ImageUrlBlock] | list[PILImage.Image] | None = None, audio: list[AudioBlock | AudioUrlBlock] | None = None, requirements: list[Requirement | str] | None = None, icl_examples: list[str | CBlock] | None = None, grounding_context: dict[str, str | Span] | None = None, user_variables: dict[str, str] | None = None, prefix: str | CBlock | None = None, output_prefix: str | CBlock | None = None, strategy: SamplingStrategy | None = RejectionSamplingStrategy(loop_budget=2), return_sampling_results: bool = False, format: type[BaseModelSubclass] | None = None, model_options: dict | None = None, tool_calls: bool = False, await_result: bool = False) -> tuple[ModelOutputThunk[str], Context] | SamplingResult`

Generates from an instruction.

## `achat()`

*async function* — [line 956](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/functional.py#L956)

`achat(content: str, context: Context, backend: Backend, *, role: Message.Role = 'user', images: list[ImageBlock | ImageUrlBlock] | list[PILImage.Image] | None = None, audio: list[AudioBlock | AudioUrlBlock] | None = None, documents: Iterable[str | Document] | None = None, user_variables: dict[str, str] | None = None, format: type[BaseModelSubclass] | None = None, model_options: dict | None = None, tool_calls: bool = False) -> tuple[Message, Context]`

Sends a simple chat message and returns the response. Adds both messages to the Context.

## `avalidate()`

*async function* — [line 1021](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/functional.py#L1021)

`avalidate(reqs: Requirement | list[Requirement], context: Context, backend: Backend, *, output: CBlock | ModelOutputThunk | None = None, format: type[BaseModelSubclass] | None = None, model_options: dict | None = None, generate_logs: list[GenerateLog] | None = None, input: CBlock | ModelOutputThunk | None = None) -> list[ValidationResult]`

Asynchronous version of .validate; validates a set of requirements over the output (if provided) or the current context (if the output is not provided).

## `aquery()`

*async function* — [line 1162](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/functional.py#L1162)

`aquery(obj: Any, query: str, context: Context, backend: Backend, *, format: type[BaseModelSubclass] | None = None, model_options: dict | None = None, tool_calls: bool = False, await_result: bool = False) -> tuple[ModelOutputThunk, Context]`

Query method for retrieving information from an object.

## `atransform()`

*async function* — [line 1208](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/functional.py#L1208)

`atransform(obj: Any, transformation: str, context: Context, backend: Backend, *, format: type[BaseModelSubclass] | None = None, model_options: dict | None = None) -> tuple[ModelOutputThunk | Any, Context]`

Transform method for creating a new object with the transformation applied.

---

*Generated from source by `docs/scripts/generate_package_map.py` — do not edit by hand; regenerate instead. Symbols and annotations reflect the pinned commit exactly; where type annotations are absent in source, this page says so rather than guessing.*
