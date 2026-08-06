---
id: tools
title: "mellea.backends.tools"
sidebar_label: "tools"
sidebar_position: 14
description: "LLM tool definitions, parsing, and validation for mellea backends."
# diataxis: reference
---

Source: [`mellea/backends/tools.py`](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/backends/tools.py) at commit `a535fc6345a0`.

LLM tool definitions, parsing, and validation for mellea backends.

## `MelleaTool`

*class* — [line 39](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/backends/tools.py#L39) (`AbstractMelleaTool[P, R]`)

Tool class to represent a callable tool with an OpenAI-compatible JSON schema.

Constructor: `MelleaTool(name: str, tool_call: Callable[P, R], as_json_tool: dict[str, Any]) -> None`

Properties:

- `as_json_tool` → `dict[str, Any]` — Return the tool converted to a OpenAI compatible JSON object.

Methods (defined on this class; inherited members not listed):

- `run(*args: P.args, **kwargs: P.kwargs) -> R` — [line 72](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/backends/tools.py#L72)
  Run the tool with the given arguments.
- `from_langchain(tool: Any) -> 'MelleaTool[..., Any]'` *(classmethod)* — [line 90](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/backends/tools.py#L90)
  Create a MelleaTool from a LangChain tool object.
- `from_smolagents(tool: Any) -> 'MelleaTool[..., Any]'` *(classmethod)* — [line 137](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/backends/tools.py#L137)
  Create a Tool from a Hugging Face smolagents tool object.
- `from_callable(func: Callable[P, R] | Callable[P, Awaitable[R]], name: str | None = None) -> 'MelleaTool[P, R]'` *(classmethod)* — [line 204](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/backends/tools.py#L204)
  Create a MelleaTool from a plain Python callable.

## `SubscriptableBaseModel`

*class* — [line 797](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/backends/tools.py#L797) (`BaseModel`)

Pydantic `BaseModel` subclass that also supports subscript (`[]`) access.

Methods (defined on this class; inherited members not listed):

- `get(key: str, default: Any = None) -> Any` — [line 875](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/backends/tools.py#L875)
  Return the value of a field by name, or a default if the field does not exist.

## `OllamaTool`

*class* — [line 902](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/backends/tools.py#L902) (`SubscriptableBaseModel`)

Pydantic model for an Ollama-compatible tool schema, imported from the Ollama Python SDK.

## `tool()`

*function* — [line 254](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/backends/tools.py#L254)

`tool(func: Callable[P, R] | Callable[P, Awaitable[R]] | None = None, name: str | None = None) -> MelleaTool[P, R] | Callable[[Callable[P, R]], MelleaTool[P, R]]`

Decorator to mark a function as a Mellea tool with type-safe parameter and return types.

## `add_tools_from_model_options()`

*function* — [line 322](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/backends/tools.py#L322)

`add_tools_from_model_options(tools_dict: dict[str, AbstractMelleaTool], model_options: dict[str, Any])`

If model_options has tools, add those tools to the tools_dict.

*Annotation gaps in source: return type unannotated.*

## `add_tools_from_context_actions()`

*function* — [line 366](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/backends/tools.py#L366)

`add_tools_from_context_actions(tools_dict: dict[str, AbstractMelleaTool], ctx_actions: list[Span] | None)`

If any of the actions in ctx_actions have tools in their template_representation, add those to the tools_dict.

*Annotation gaps in source: return type unannotated.*

## `convert_tools_to_json()`

*function* — [line 393](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/backends/tools.py#L393)

`convert_tools_to_json(tools: dict[str, AbstractMelleaTool]) -> list[dict]`

Convert tools to json dict representation.

## `json_extraction()`

*function* — [line 410](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/backends/tools.py#L410)

`json_extraction(text: str) -> Generator[dict, None, None]`

Yield the next valid JSON object found in a given string.

## `find_func()`

*function* — [line 437](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/backends/tools.py#L437)

`find_func(d: object) -> tuple[str | None, Mapping | None]`

Find the first function in a json-like dictionary.

## `parse_tools()`

*function* — [line 473](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/backends/tools.py#L473)

`parse_tools(llm_response: str) -> list[tuple[str, Mapping]]`

A simple parser that will scan a string for tools and attempt to extract them; only works for json based outputs.

## `validate_tool_arguments()`

*function* — [line 492](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/backends/tools.py#L492)

`validate_tool_arguments(tool: AbstractMelleaTool, args: Mapping[str, Any], *, coerce_types: bool = True, strict: bool = False) -> dict[str, Any]`

Validate and optionally coerce tool arguments against tool's JSON schema.

## `get_code_field_from_schema()`

*function* — [line 976](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/backends/tools.py#L976)

`get_code_field_from_schema(tool_call: ModelToolCall) -> str | None`

Determine the executable content field name from a tool's JSON schema.

## `convert_function_to_ollama_tool()`

*function* — [line 1330](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/backends/tools.py#L1330)

`convert_function_to_ollama_tool(func: Callable, name: str | None = None) -> OllamaTool`

Convert a Python callable to an Ollama-compatible tool schema.

---

*Generated from source by `docs/scripts/generate_package_map.py` — do not edit by hand; regenerate instead. Symbols and annotations reflect the pinned commit exactly; where type annotations are absent in source, this page says so rather than guessing.*
