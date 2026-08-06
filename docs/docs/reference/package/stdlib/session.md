---
id: session
title: "mellea.stdlib.session"
sidebar_label: "session"
sidebar_position: 8
description: "`MelleaSession`: the primary entry point for running generative programs."
# diataxis: reference
---

Source: [`mellea/stdlib/session.py`](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/session.py) at commit `a535fc6345a0`.

`MelleaSession`: the primary entry point for running generative programs.

## `MelleaSession`

*class* — [line 261](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/session.py#L261) 

Mellea sessions are a THIN wrapper around `m` convenience functions with NO special semantics.

Constructor: `MelleaSession(backend: Backend, ctx: Context | None = None, *, session_id: str | None = None)`

Properties:

- `ctx` → `Context` — The session's current conversation context.

Methods (defined on this class; inherited members not listed):

- `ctx(value: Context) -> None` — [line 332](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/session.py#L332)
  Replace the context and count this as one interaction.
- `clone() -> MelleaSession` — [line 408](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/session.py#L408)
  Useful for running multiple generation requests while keeping the context at a given point in time.
- `reset() -> None` — [line 433](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/session.py#L433)
  Reset the context state to a fresh, empty context of the same type.
- `cleanup(*, exception: BaseException | None = None) -> None` — [line 455](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/session.py#L455)
  Clean up session resources and deregister session-scoped plugins.
- `act(action: Component[S] | CBlock | ModelOutputThunk, *, requirements: list[Requirement] | None = None, strategy: SamplingStrategy | None = None, return_sampling_results: bool = False, format: type[BaseModelSubclass] | None = None, model_options: dict | None = None, tool_calls: bool = False) -> ModelOutputThunk[S] | SamplingResult` — [line 508](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/session.py#L508)
  Runs a generic action, and adds both the action and the result to the context.
- `instruct(description: str, *, images: list[ImageBlock | ImageUrlBlock] | list[PILImage.Image] | None = None, audio: list[AudioBlock | AudioUrlBlock] | None = None, requirements: list[Requirement | str] | None = None, icl_examples: list[str | CBlock] | None = None, grounding_context: dict[str, str | Span] | None = None, user_variables: dict[str, str] | None = None, prefix: str | CBlock | None = None, output_prefix: str | CBlock | None = None, strategy: SamplingStrategy | None = RejectionSamplingStrategy(loop_budget=2), return_sampling_results: bool = False, format: type[BaseModelSubclass] | None = None, model_options: dict | None = None, tool_calls: bool = False) -> ModelOutputThunk[str] | SamplingResult` — [line 599](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/session.py#L599)
  Generates from an instruction.
- `chat(content: str, role: Message.Role = 'user', *, images: list[ImageBlock | ImageUrlBlock] | list[PILImage.Image] | None = None, audio: list[AudioBlock | AudioUrlBlock] | None = None, documents: collections.abc.Iterable[str | Document] | None = None, user_variables: dict[str, str] | None = None, format: type[BaseModelSubclass] | None = None, model_options: dict | None = None, tool_calls: bool = False) -> Message` — [line 670](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/session.py#L670)
  Sends a simple chat message and returns the response. Adds both messages to the Context.
- `validate(reqs: Requirement | list[Requirement], *, output: CBlock | ModelOutputThunk | None = None, format: type[BaseModelSubclass] | None = None, model_options: dict | None = None, generate_logs: list[GenerateLog] | None = None, input: CBlock | ModelOutputThunk | None = None) -> list[ValidationResult]` — [line 717](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/session.py#L717)
  Validates a set of requirements over the output (if provided) or the current context (if the output is not provided).
- `query(obj: Any, query: str, *, format: type[BaseModelSubclass] | None = None, model_options: dict | None = None, tool_calls: bool = False) -> ComputedModelOutputThunk` — [line 751](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/session.py#L751)
  Query method for retrieving information from an object.
- `transform(obj: Any, transformation: str, *, format: type[BaseModelSubclass] | None = None, model_options: dict | None = None) -> ModelOutputThunk | Any` — [line 784](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/session.py#L784)
  Transform method for creating a new object with the transformation applied.
- `aact(action: Component[S] | CBlock | ModelOutputThunk, *, requirements: list[Requirement] | None = None, strategy: SamplingStrategy | None = None, return_sampling_results: bool = False, format: type[BaseModelSubclass] | None = None, model_options: dict | None = None, tool_calls: bool = False, await_result: bool = False) -> ModelOutputThunk[S] | SamplingResult` *(async)* — [line 872](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/session.py#L872)
  Runs a generic action, and adds both the action and the result to the context.
- `ainstruct(description: str, *, images: list[ImageBlock | ImageUrlBlock] | list[PILImage.Image] | None = None, audio: list[AudioBlock | AudioUrlBlock] | None = None, requirements: list[Requirement | str] | None = None, icl_examples: list[str | CBlock] | None = None, grounding_context: dict[str, str | Span] | None = None, user_variables: dict[str, str] | None = None, prefix: str | CBlock | None = None, output_prefix: str | CBlock | None = None, strategy: SamplingStrategy | None = RejectionSamplingStrategy(loop_budget=2), return_sampling_results: bool = False, format: type[BaseModelSubclass] | None = None, model_options: dict | None = None, tool_calls: bool = False, await_result: bool = False) -> ModelOutputThunk[str] | SamplingResult[str]` *(async)* — [line 1011](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/session.py#L1011)
  Generates from an instruction.
- `achat(content: str, role: Message.Role = 'user', *, images: list[ImageBlock | ImageUrlBlock] | list[PILImage.Image] | None = None, audio: list[AudioBlock | AudioUrlBlock] | None = None, documents: collections.abc.Iterable[str | Document] | None = None, user_variables: dict[str, str] | None = None, format: type[BaseModelSubclass] | None = None, model_options: dict | None = None, tool_calls: bool = False) -> Message` *(async)* — [line 1086](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/session.py#L1086)
  Sends a simple chat message and returns the response. Adds both messages to the Context.
- `avalidate(reqs: Requirement | list[Requirement], *, output: CBlock | ModelOutputThunk | None = None, format: type[BaseModelSubclass] | None = None, model_options: dict | None = None, generate_logs: list[GenerateLog] | None = None, input: CBlock | ModelOutputThunk | None = None) -> list[ValidationResult]` *(async)* — [line 1133](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/session.py#L1133)
  Validates a set of requirements over the output (if provided) or the current context (if the output is not provided).
- `aquery(obj: Any, query: str, *, format: type[BaseModelSubclass] | None = None, model_options: dict | None = None, tool_calls: bool = False, await_result: bool = False) -> ModelOutputThunk` *(async)* — [line 1191](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/session.py#L1191)
  Query method for retrieving information from an object.
- `atransform(obj: Any, transformation: str, *, format: type[BaseModelSubclass] | None = None, model_options: dict | None = None) -> ModelOutputThunk | Any` *(async)* — [line 1227](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/session.py#L1227)
  Transform method for creating a new object with the transformation applied.
- `powerup(powerup_cls: type) -> None` *(classmethod)* — [line 1260](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/session.py#L1260)
  Appends methods in a class object `powerup_cls` to MelleaSession.
- `last_prompt() -> str | list[dict] | None` — [line 1277](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/session.py#L1277)
  Returns the last prompt that has been called from the session context.

## `get_session()`

*function* — [line 78](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/session.py#L78)

`get_session() -> MelleaSession`

Get the current session from context.

## `start_session()`

*function* — [line 95](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/session.py#L95)

`start_session(backend_name: Literal['ollama', 'hf', 'openai', 'watsonx', 'litellm'] = 'ollama', model_id: str | ModelIdentifier = IBM_GRANITE_4_1_3B, ctx: Context | None = None, *, context_type: Literal['simple', 'chat'] | None = None, model_options: dict | None = None, plugins: list[Any] | None = None, **backend_kwargs: Any) -> MelleaSession`

Start a new Mellea session. Can be used as a context manager or called directly.

---

*Generated from source by `docs/scripts/generate_package_map.py` — do not edit by hand; regenerate instead. Symbols and annotations reflect the pinned commit exactly; where type annotations are absent in source, this page says so rather than guessing.*
