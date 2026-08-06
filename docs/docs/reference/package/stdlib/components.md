---
id: components
title: "mellea.stdlib.components"
sidebar_label: "components"
sidebar_position: 2
description: "Module for Components."
# diataxis: reference
---

Source: [`mellea/stdlib/components/__init__.py`](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/components/__init__.py) at commit `a535fc6345a0`.

Module for Components.

Declared exports (`__all__`): `AudioBlock`, `AudioUrlBlock`, `CBlock`, `Component`, `ComponentParseError`, `Document`, `ImageBlock`, `ImageUrlBlock`, `Instruction`, `Intrinsic`, `MObject`, `MObjectProtocol`, `Message`, `ModelOutputThunk`, `Query`, `SimpleComponent`, `TemplateRepresentation`, `ToolMessage`, `Transform`, `as_chat_history`, `as_generic_chat_history`, `blockify`, `mify`

---

## Module `mellea.stdlib.components.adapter_based_component`

Source: [`mellea/stdlib/components/adapter_based_component/__init__.py`](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/components/adapter_based_component/__init__.py) at commit `a535fc6345a0`.

`AdapterBasedComponent` — the user-facing component class for adapter-backed capabilities.

Declared exports (`__all__`): `AdapterBasedComponent`

---

## Module `mellea.stdlib.components.chat`

Source: [`mellea/stdlib/components/chat.py`](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/components/chat.py) at commit `a535fc6345a0`.

Chat primitives: the `Message` and `ToolMessage` components.

### `Message`

*class* — [line 37](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/components/chat.py#L37) (`Component['Message']`)

A single Message in a Chat history.

Constructor: `Message(role: 'Message.Role', content: str, *, images: None | list[ImageBlock | ImageUrlBlock] = None, audio: None | list[AudioBlock | AudioUrlBlock] = None, documents: None | Iterable[str | Document] = None, tool_calls: list[dict[str, Any]] | None = None, thinking: str | None = None)`

Properties:

- `images` → `None | list[ImageBlock | ImageUrlBlock]` — Returns the images associated with this message.
- `audio` → `None | list[AudioBlock | AudioUrlBlock]` — Returns the audio associated with this message.
- `tool_calls` → `list[dict[str, Any]] | None` — Returns the OpenAI-compatible tool calls associated with this message.

Methods (defined on this class; inherited members not listed):

- `parts() -> list[Span]` — [line 111](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/components/chat.py#L111)
  Return the constituent parts of this message, including content, documents, images, and audio.
- `format_for_llm() -> TemplateRepresentation` — [line 127](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/components/chat.py#L127)
  Formats the content for a Language Model.

### `ToolMessage`

*class* — [line 241](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/components/chat.py#L241) (`Message`)

Adds the name field for function name.

Constructor: `ToolMessage(role: Message.Role, content: str, tool_output: Any, name: str, args: Mapping[str, Any], tool: ModelToolCall)`

### `as_chat_history()`

*function* — [line 279](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/components/chat.py#L279)

`as_chat_history(ctx: Context) -> list[Message]`

Returns a list of Messages corresponding to a Context.

### `as_generic_chat_history()`

*function* — [line 330](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/components/chat.py#L330)

`as_generic_chat_history(ctx: Context, formatter: Callable[[object], str] | None = None) -> list[Message]`

Returns a list of Messages corresponding to a Context, with flexible type handling.

---

## Module `mellea.stdlib.components.docs`

Source: [`mellea/stdlib/components/docs/__init__.py`](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/components/docs/__init__.py) at commit `a535fc6345a0`.

Classes and functions for working with document-like objects.

Declared exports (`__all__`): `Document`

---

## Module `mellea.stdlib.components.docs.document`

Source: [`mellea/stdlib/components/docs/document.py`](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/components/docs/document.py) at commit `a535fc6345a0`.

`Document` component for grounding model inputs with text passages.

### `Document`

*class* — [line 18](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/components/docs/document.py#L18) (`Component[str]`)

A text passage with optional metadata for grounding model inputs.

Constructor: `Document(text: str, title: str | None = None, doc_id: str | None = None)`

Methods (defined on this class; inherited members not listed):

- `parts() -> list[Span]` — [line 37](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/components/docs/document.py#L37)
  Returns the constituent parts of this document.
- `format_for_llm() -> TemplateRepresentation` — [line 47](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/components/docs/document.py#L47)
  Formats the `Document` as a `TemplateRepresentation`.

---

## Module `mellea.stdlib.components.docs.richdocument`

Source: [`mellea/stdlib/components/docs/richdocument.py`](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/components/docs/richdocument.py) at commit `a535fc6345a0`.

`RichDocument`, `Table`, and related helpers backed by Docling.

### `RichDocument`

*class* — [line 40](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/components/docs/richdocument.py#L40) (`Component[str]`)

A document wrapper that exposes content to a language model as Markdown.

Constructor: `RichDocument(doc: DoclingDocument)`

Methods (defined on this class; inherited members not listed):

- `parts() -> list[Span]` — [line 74](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/components/docs/richdocument.py#L74)
  Return the constituent parts of this document.
- `format_for_llm() -> TemplateRepresentation | str` — [line 87](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/components/docs/richdocument.py#L87)
  Return the document content as a Markdown string.
- `docling() -> DoclingDocument` — [line 101](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/components/docs/richdocument.py#L101)
  Return the underlying `DoclingDocument`.
- `to_markdown() -> str` — [line 109](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/components/docs/richdocument.py#L109)
  Get the full text of the document as markdown.
- `get_tables() -> list[Table]` — [line 113](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/components/docs/richdocument.py#L113)
  Return all tables found in this document.
- `save(filename: str | Path) -> None` — [line 121](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/components/docs/richdocument.py#L121)
  Save the underlying `DoclingDocument` to a JSON file for later reuse.
- `load(filename: str | Path) -> RichDocument` *(classmethod)* — [line 133](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/components/docs/richdocument.py#L133)
  Load a `RichDocument` from a previously saved `DoclingDocument` JSON file.
- `from_document_file(source: str | Path | DocumentStream, do_ocr: bool = True) -> RichDocument` *(classmethod)* — [line 164](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/components/docs/richdocument.py#L164)
  Convert a document file to a `RichDocument` using Docling.

### `TableQuery`

*class* — [line 221](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/components/docs/richdocument.py#L221) (`Query`)

A `Query` component specialised for `Table` objects.

Constructor: `TableQuery(obj: Table, query: str) -> None`

Methods (defined on this class; inherited members not listed):

- `parts() -> list[Span]` — [line 236](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/components/docs/richdocument.py#L236)
  Return the constituent parts of this table query.
- `format_for_llm() -> TemplateRepresentation` — [line 246](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/components/docs/richdocument.py#L246)
  Format this table query for the language model.

### `TableTransform`

*class* — [line 268](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/components/docs/richdocument.py#L268) (`Transform`)

A `Transform` component specialised for `Table` objects.

Constructor: `TableTransform(obj: Table, transformation: str) -> None`

Methods (defined on this class; inherited members not listed):

- `parts() -> list[Span]` — [line 283](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/components/docs/richdocument.py#L283)
  Return the constituent parts of this table transform.
- `format_for_llm() -> TemplateRepresentation` — [line 293](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/components/docs/richdocument.py#L293)
  Format this table transform for the language model.

### `Table`

*class* — [line 318](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/components/docs/richdocument.py#L318) (`MObject`)

A `Table` represents a single table within a larger Docling Document.

Constructor: `Table(ti: TableItem, doc: DoclingDocument)`

Methods (defined on this class; inherited members not listed):

- `from_markdown(md: str) -> Table | None` *(classmethod)* — [line 334](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/components/docs/richdocument.py#L334)
  Create a `Table` from a Markdown string by round-tripping through Docling.
- `parts() -> list[Span]` — [line 355](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/components/docs/richdocument.py#L355)
  Return the constituent parts of this table component.
- `to_markdown() -> str` — [line 366](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/components/docs/richdocument.py#L366)
  Export this table as a Markdown string.
- `transpose() -> Table | None` — [line 374](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/components/docs/richdocument.py#L374)
  Transpose this table and return the result as a new `Table`.
- `format_for_llm() -> TemplateRepresentation | str` — [line 384](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/components/docs/richdocument.py#L384)
  Return the table representation for the Formatter.

---

## Module `mellea.stdlib.components.genslot`

Source: [`mellea/stdlib/components/genslot.py`](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/components/genslot.py) at commit `a535fc6345a0`.

Backward-compatibility shim — use `mellea.stdlib.components.genstub` instead.

---

## Module `mellea.stdlib.components.genstub`

Source: [`mellea/stdlib/components/genstub.py`](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/components/genstub.py) at commit `a535fc6345a0`.

A method to generate outputs based on python functions and a Generative Stub function.

Declared exports (`__all__`): `PreconditionException`, `generative`

### `FunctionResponse`

*class* — [line 38](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/components/genstub.py#L38) (`BaseModel`, `Generic[R]`)

Generic base class for function response formats.

### `FunctionDict`

*class* — [line 71](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/components/genstub.py#L71) (`TypedDict`)

Return Type for a Function Component.

### `ArgumentDict`

*class* — [line 85](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/components/genstub.py#L85) (`TypedDict`)

Return Type for an Argument Component.

### `Argument`

*class* — [line 99](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/components/genstub.py#L99) 

A single function argument with its name, type annotation, and value.

Constructor: `Argument(annotation: str | None = None, name: str | None = None, value: str | None = None)`

### `Arguments`

*class* — [line 122](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/components/genstub.py#L122) (`CBlock`)

A `CBlock` that renders a list of `Argument` objects as human-readable text.

Constructor: `Arguments(arguments: list[Argument])`

### `ArgPreconditionRequirement`

*class* — [line 147](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/components/genstub.py#L147) (`Requirement`)

Specific requirement with template for validating precondition requirements against a set of args.

Constructor: `ArgPreconditionRequirement(req: Requirement)`

### `PreconditionException`

*class* — [line 170](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/components/genstub.py#L170) (`Exception`)

Exception raised when validation fails for a generative stub's arguments.

Constructor: `PreconditionException(message: str, validation_results: list[ValidationResult]) -> None`

### `Function`

*class* — [line 191](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/components/genstub.py#L191) (`Generic[P, R]`)

Wraps a callable with its introspected `FunctionDict` metadata.

Constructor: `Function(func: Callable[P, R])`

### `ExtractedArgs`

*class* — [line 284](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/components/genstub.py#L284) 

Used to extract the mellea args and original function args. See @generative decorator for additional notes on these fields.

Constructor: `ExtractedArgs()`

### `GenerativeStub`

*class* — [line 324](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/components/genstub.py#L324) (`Component[R]`, `Generic[P, R]`)

Abstract base class for AI-powered function wrappers produced by `@generative`.

Constructor: `GenerativeStub(func: Callable[P, R])`

Methods (defined on this class; inherited members not listed):

- `extract_args_and_kwargs(*args, **kwargs) -> ExtractedArgs` *(staticmethod)* — [line 376](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/components/genstub.py#L376)
  Take a mix of args and kwargs for both the generative stub and the original function and extract them. Ensures the original function's args are all kwargs.
- `parts() -> list[Span]` — [line 474](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/components/genstub.py#L474)
  Return the constituent parts of this generative stub component.
- `format_for_llm() -> TemplateRepresentation` — [line 489](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/components/genstub.py#L489)
  Format this generative stub for the language model.

### `SyncGenerativeStub`

*class* — [line 528](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/components/genstub.py#L528) (`GenerativeStub`, `Generic[P, R]`)

A synchronous generative stub that blocks until the LLM response is ready.

### `AsyncGenerativeStub`

*class* — [line 673](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/components/genstub.py#L673) (`GenerativeStub`, `Generic[P, R]`)

A generative stub component that generates asynchronously and returns a coroutine.

### `create_response_format()`

*function* — [line 48](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/components/genstub.py#L48)

`create_response_format(func: Callable[..., R]) -> type[FunctionResponse[R]]`

Create a Pydantic response format class for a given function.

### `describe_function()`

*function* — [line 208](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/components/genstub.py#L208)

`describe_function(func: Callable) -> FunctionDict`

Generates a FunctionDict given a function.

### `get_argument()`

*function* — [line 224](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/components/genstub.py#L224)

`get_argument(func: Callable, key: str, val: Any) -> Argument`

Returns an argument given a parameter.

### `bind_function_arguments()`

*function* — [line 250](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/components/genstub.py#L250)

`bind_function_arguments(func: Callable[P, R], *args: P.args, **kwargs: P.kwargs) -> dict[str, Any]`

Bind arguments to function parameters and return as dictionary.

### `generative()`

*function* — [line 833](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/components/genstub.py#L833)

`generative(func: Callable[P, R]) -> GenerativeStub[P, R]`

Convert a function into an AI-powered function.

---

## Module `mellea.stdlib.components.instruction`

Source: [`mellea/stdlib/components/instruction.py`](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/components/instruction.py) at commit `a535fc6345a0`.

`Instruction` component for instruct/validate/repair loops.

### `Instruction`

*class* — [line 36](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/components/instruction.py#L36) (`Component[str]`)

The Instruction in an instruct/validate/repair loop.

Constructor: `Instruction(description: str | CBlock | None = None, requirements: list[Requirement | str] | None = None, icl_examples: list[str | CBlock] | None = None, grounding_context: dict[str, str | Span] | None = None, user_variables: dict[str, str] | None = None, prefix: str | CBlock | None = None, output_prefix: str | CBlock | None = None, images: list[ImageBlock | ImageUrlBlock] | None = None, audio: list[AudioBlock | AudioUrlBlock] | None = None)`

Properties:

- `images` → `list[ImageBlock | ImageUrlBlock] | None` — Returns the images associated with this instruction.
- `audio` → `list[AudioBlock | AudioUrlBlock] | None` — Returns the audio associated with this instruction.
- `requirements` → `list[Requirement]` — Returns a list of Requirement instances.

Methods (defined on this class; inherited members not listed):

- `parts() -> list[Span]` — [line 157](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/components/instruction.py#L157)
  Returns all of the constituent parts of an Instruction.
- `format_for_llm() -> TemplateRepresentation` — [line 175](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/components/instruction.py#L175)
  Format this instruction for the language model.
- `apply_user_dict_from_jinja(user_dict: dict[str, str], s: str) -> str` *(staticmethod)* — [line 209](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/components/instruction.py#L209)
  Render a Jinja2 template string using the provided variable dictionary.
- `copy_and_repair(repair_string: str) -> Instruction` — [line 238](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/components/instruction.py#L238)
  Create a deep copy of this instruction with the repair string set.

---

## Module `mellea.stdlib.components.intrinsic`

Source: [`mellea/stdlib/components/intrinsic/__init__.py`](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/components/intrinsic/__init__.py) at commit `a535fc6345a0`.

Module for working with intrinsics.

Declared exports (`__all__`): `Intrinsic`

---

## Module `mellea.stdlib.components.intrinsic.core`

Source: [`mellea/stdlib/components/intrinsic/core.py`](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/components/intrinsic/core.py) at commit `a535fc6345a0`.

Adapter functions for core model capabilities.

### `check_certainty()`

*function* — [line 17](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/components/intrinsic/core.py#L17)

`check_certainty(context: ChatContext, backend: AdapterMixin, model_options: dict | None = None) -> float`

Estimate the model's certainty about its last response.

### `requirement_check()`

*function* — [line 42](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/components/intrinsic/core.py#L42)

`requirement_check(context: ChatContext, backend: AdapterMixin, requirement: str, model_options: dict | None = None) -> float`

Detect if text adheres to provided requirements.

### `find_context_attributions()`

*function* — [line 101](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/components/intrinsic/core.py#L101)

`find_context_attributions(response: str | None, documents: collections.abc.Iterable[str | Document], context: ChatContext, backend: AdapterMixin, model_options: dict | None = None) -> list[dict]`

Find sentences in conversation history and documents that most influence an LLM's response.

---

## Module `mellea.stdlib.components.intrinsic.guardian`

Source: [`mellea/stdlib/components/intrinsic/guardian.py`](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/components/intrinsic/guardian.py) at commit `a535fc6345a0`.

Adapter functions for Guardian safety and hallucination detection.

### `policy_guardrails()`

*function* — [line 179](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/components/intrinsic/guardian.py#L179)

`policy_guardrails(context: ChatContext, backend: AdapterMixin, policy_text: str, *, model_options: dict | None = None) -> str`

Check whether the last context turn complies with a policy.

### `guardian_check()`

*function* — [line 327](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/components/intrinsic/guardian.py#L327)

`guardian_check(context: ChatContext, backend: AdapterMixin, criteria: str, scoring_schema: str | object = _UNSET, target_role: str | None = None, *, model_options: dict | None = None) -> float`

Check whether text meets specified safety/quality criteria.

### `factuality_detection()`

*function* — [line 420](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/components/intrinsic/guardian.py#L420)

`factuality_detection(context: ChatContext, backend: AdapterMixin, *, documents: collections.abc.Iterable[str | Document] | None = None, model_options: dict | None = None) -> str`

Determine whether the last assistant response is factually incorrect.

### `factuality_correction()`

*function* — [line 481](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/components/intrinsic/guardian.py#L481)

`factuality_correction(context: ChatContext, backend: AdapterMixin, *, documents: collections.abc.Iterable[str | Document] | None = None, model_options: dict | None = None) -> str`

Correct the last assistant response to make it factually accurate.

---

## Module `mellea.stdlib.components.intrinsic.intrinsic`

Source: [`mellea/stdlib/components/intrinsic/intrinsic.py`](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/components/intrinsic/intrinsic.py) at commit `a535fc6345a0`.

`Intrinsic` component for invoking fine-tuned adapter capabilities.

### `Intrinsic`

*class* — [line 17](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/components/intrinsic/intrinsic.py#L17) (`Component[str]`)

A component representing an intrinsic fine-tuned adapter capability.

Constructor: `Intrinsic(intrinsic_name: str, intrinsic_kwargs: dict | None = None, adapter_types: tuple[AdapterType, ...] | None = None) -> None`

Properties:

- `intrinsic_name` *(type not annotated in source)* — User-visible name of this intrinsic.
- `adapter_types` → `tuple[AdapterType, ...]` — Tuple of available adapter types that implement this intrinsic.

Methods (defined on this class; inherited members not listed):

- `parts() -> list[Span]` — [line 65](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/components/intrinsic/intrinsic.py#L65)
  Return the constituent parts of this intrinsic component.
- `format_for_llm() -> TemplateRepresentation | str` — [line 76](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/components/intrinsic/intrinsic.py#L76)
  Not implemented for the base `Intrinsic` class.

---

## Module `mellea.stdlib.components.intrinsic.rag`

Source: [`mellea/stdlib/components/intrinsic/rag.py`](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/components/intrinsic/rag.py) at commit `a535fc6345a0`.

Adapter functions related to retrieval-augmented generation.

### `check_answerability()`

*function* — [line 162](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/components/intrinsic/rag.py#L162)

`check_answerability(question: str | None, documents: collections.abc.Iterable[str | Document], context: ChatContext, backend: AdapterMixin, *, model_options: dict | None = None) -> str`

Test a user's question for answerability.

### `rewrite_question()`

*function* — [line 214](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/components/intrinsic/rag.py#L214)

`rewrite_question(question: str | None, context: ChatContext, backend: AdapterMixin, *, model_options: dict | None = None) -> str`

Rewrite a user's question for retrieval.

### `clarify_query()`

*function* — [line 259](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/components/intrinsic/rag.py#L259)

`clarify_query(question: str | None, documents: collections.abc.Iterable[str | Document], context: ChatContext, backend: AdapterMixin, *, model_options: dict | None = None) -> str`

Generate clarification for an ambiguous query.

### `find_citations()`

*function* — [line 312](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/components/intrinsic/rag.py#L312)

`find_citations(response: str | None, documents: collections.abc.Iterable[str | Document], context: ChatContext, backend: AdapterMixin, *, model_options: dict | None = None) -> list[dict]`

Find information in documents that supports an assistant response.

### `check_context_relevance()`

*function* — [line 376](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/components/intrinsic/rag.py#L376)

`check_context_relevance(question: str | None, document: str | Document, context: ChatContext, backend: AdapterMixin, *, model_options: dict | None = None) -> str`

Test whether a document is relevant to a user's question.

### `flag_hallucinated_content()`

*function* — [line 445](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/components/intrinsic/rag.py#L445)

`flag_hallucinated_content(response: str | None, documents: collections.abc.Iterable[str | Document], context: ChatContext, backend: AdapterMixin, *, model_options: dict | None = None) -> list[dict]`

Flag potentially-hallucinated sentences in an agent's response.

---

## Module `mellea.stdlib.components.mify`

Source: [`mellea/stdlib/components/mify.py`](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/components/mify.py) at commit `a535fc6345a0`.

The `@mify` decorator for turning Python objects into `Component`s.

### `MifiedProtocol`

*class* — [line 26](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/components/mify.py#L26) (`MObjectProtocol`, `Protocol`)

Adds additional functionality to the MObjectProtocol and modifies MObject functions so that mified objects can be more easily interacted with and modified.

Methods (defined on this class; inherited members not listed):

- `parts() -> list[Span]` — [line 43](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/components/mify.py#L43)
  Return the constituent sub-components of this mified object.
- `get_query_object(query: str) -> Query` — [line 57](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/components/mify.py#L57)
  Return the instantiated query object for this mified object.
- `get_transform_object(transformation: str) -> Transform` — [line 70](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/components/mify.py#L70)
  Return the instantiated transform object for this mified object.
- `content_as_string() -> str` — [line 84](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/components/mify.py#L84)
  Return the content of the mified object as a plain string.
- `format_for_llm() -> TemplateRepresentation` — [line 192](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/components/mify.py#L192)
  Return the `TemplateRepresentation` for this mified object.
- `parse(computed: ModelOutputThunk) -> str` — [line 238](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/components/mify.py#L238)
  Parse the model output into a string value.

### `mify()`

*function* — [line 298](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/components/mify.py#L298)

`mify(*args, **kwargs) -> object`

M-ify an object or class.

---

## Module `mellea.stdlib.components.mobject`

Source: [`mellea/stdlib/components/mobject.py`](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/components/mobject.py) at commit `a535fc6345a0`.

`MObject`, `Query`, `Transform`, and `MObjectProtocol` for query/transform workflows.

### `Query`

*class* — [line 24](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/components/mobject.py#L24) (`Component[str]`)

A `Component` that pairs an `MObject` with a natural-language question.

Constructor: `Query(obj: Component, query: str) -> None`

Methods (defined on this class; inherited members not listed):

- `parts() -> list[Span]` — [line 41](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/components/mobject.py#L41)
  Return the constituent parts of this query component.
- `format_for_llm() -> TemplateRepresentation | str` — [line 49](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/components/mobject.py#L49)
  Format this query for the language model.

### `Transform`

*class* — [line 82](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/components/mobject.py#L82) (`Component[str]`)

A `Component` that pairs an `MObject` with a natural-language mutation instruction.

Constructor: `Transform(obj: Component, transformation: str) -> None`

Methods (defined on this class; inherited members not listed):

- `parts() -> list[Span]` — [line 99](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/components/mobject.py#L99)
  Return the constituent parts of this transform component.
- `format_for_llm() -> TemplateRepresentation | str` — [line 107](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/components/mobject.py#L107)
  Format this transform for the language model.

### `MObjectProtocol`

*class* — [line 141](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/components/mobject.py#L141) (`Protocol`)

Protocol to describe the necessary functionality of a MObject. Implementers should prefer inheriting from MObject than MObjectProtocol.

Methods (defined on this class; inherited members not listed):

- `parts() -> list[Span]` — [line 144](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/components/mobject.py#L144)
  Return a list of parts for this MObject.
- `get_query_object(query: str) -> Query` — [line 152](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/components/mobject.py#L152)
  Return the instantiated query object.
- `get_transform_object(transformation: str) -> Transform` — [line 164](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/components/mobject.py#L164)
  Return the instantiated transform object.
- `content_as_string() -> str` — [line 176](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/components/mobject.py#L176)
  Return the content of this MObject as a plain string.
- `format_for_llm() -> TemplateRepresentation | str` — [line 195](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/components/mobject.py#L195)
  Return the template representation used by the formatter.

### `MObject`

*class* — [line 212](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/components/mobject.py#L212) (`Component[str]`)

An extension of `Component` for adding query and transform operations.

Constructor: `MObject(*, query_type: type = Query, transform_type: type = Transform) -> None`

Methods (defined on this class; inherited members not listed):

- `parts() -> list[Span]` — [line 229](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/components/mobject.py#L229)
  MObject has no parts because of how format_for_llm is defined.
- `get_query_object(query: str) -> Query` — [line 237](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/components/mobject.py#L237)
  Return the instantiated query object.
- `get_transform_object(transformation: str) -> Transform` — [line 249](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/components/mobject.py#L249)
  Return the instantiated transform object.
- `content_as_string() -> str` — [line 261](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/components/mobject.py#L261)
  Return the content of this MObject as a plain string.
- `format_for_llm() -> TemplateRepresentation | str` — [line 296](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/components/mobject.py#L296)
  Return the template representation used by the formatter.

---

## Module `mellea.stdlib.components.react`

Source: [`mellea/stdlib/components/react.py`](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/components/react.py) at commit `a535fc6345a0`.

Components that implement the ReACT (Reason + Act) agentic pattern.

### `ReactInitiator`

*class* — [line 135](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/components/react.py#L135) (`Component[str]`)

`ReactInitiator` is used at the start of the ReACT loop to prime the model.

Constructor: `ReactInitiator(goal: str, tools: list[AbstractMelleaTool] | None)`

Methods (defined on this class; inherited members not listed):

- `parts() -> list[Span]` — [line 153](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/components/react.py#L153)
  Return the constituent parts of this component.
- `format_for_llm() -> TemplateRepresentation` — [line 161](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/components/react.py#L161)
  Formats the `Component` into a `TemplateRepresentation` or string.

### `ReactThought`

*class* — [line 193](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/components/react.py#L193) (`Component[str]`)

ReactThought signals that a thinking step should be done.

Constructor: `ReactThought()`

Methods (defined on this class; inherited members not listed):

- `parts() -> list[Span]` — [line 199](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/components/react.py#L199)
  Return the constituent parts of this component.
- `format_for_llm() -> TemplateRepresentation` — [line 213](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/components/react.py#L213)
  Formats the `Component` into a `TemplateRepresentation` or string.

### `pin_react_initiator()`

*function* — [line 39](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/components/react.py#L39)

`pin_react_initiator(components: list[Component | CBlock]) -> int`

A `PinPredicate` that pins everything up to and including the first `ReactInitiator`.

### `react_summary_prompt()`

*function* — [line 65](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/components/react.py#L65)

`react_summary_prompt(goal: str | None = None, max_tokens_hint: int | None = None) -> str`

Build a research-flavoured summary prompt for :class:`LLMSummarizeCompactor`.

---

## Module `mellea.stdlib.components.simple`

Source: [`mellea/stdlib/components/simple.py`](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/components/simple.py) at commit `a535fc6345a0`.

`SimpleComponent`: a lightweight named-span component.

### `SimpleComponent`

*class* — [line 21](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/components/simple.py#L21) (`Component[str]`)

A Component that is make up of named spans.

Constructor: `SimpleComponent(**kwargs: Any) -> None`

Methods (defined on this class; inherited members not listed):

- `parts() -> list[Span]` — [line 32](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/components/simple.py#L32)
  Returns the values of the kwargs.
- `make_simple_string(kwargs: dict[str, Any]) -> str` *(staticmethod)* — [line 50](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/components/simple.py#L50)
  Render keyword arguments as `<|key|>value</|key|>` tagged strings.
- `make_json_string(kwargs: dict[str, Any]) -> str` *(staticmethod)* — [line 65](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/components/simple.py#L65)
  Serialize keyword arguments to a JSON string.
- `format_for_llm() -> str` — [line 90](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/components/simple.py#L90)
  Format this component as a JSON string representation for the language model.

---

## Module `mellea.stdlib.components.unit_test_eval`

Source: [`mellea/stdlib/components/unit_test_eval.py`](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/components/unit_test_eval.py) at commit `a535fc6345a0`.

LLM Evaluation with Unit Tests in Mellea.

### `Message`

*class* — [line 15](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/components/unit_test_eval.py#L15) (`BaseModel`)

Schema for a message in the test data.

### `Example`

*class* — [line 28](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/components/unit_test_eval.py#L28) (`BaseModel`)

Schema for an example in the test data.

### `TestData`

*class* — [line 42](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/components/unit_test_eval.py#L42) (`BaseModel`)

Schema for test data loaded from json.

Methods (defined on this class; inherited members not listed):

- `validate_examples(v: list[Example]) -> list[Example]` *(classmethod)* — [line 61](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/components/unit_test_eval.py#L61)
  Validate that the examples list is not empty.

### `TestBasedEval`

*class* — [line 79](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/components/unit_test_eval.py#L79) (`Component[str]`)

Each TestBasedEval represents a single unit test.

Constructor: `TestBasedEval(source: str, name: str, instructions: str, inputs: list[str], targets: list[list[str]] | None = None, test_id: str | None = None, input_ids: list[str] | None = None)`

Methods (defined on this class; inherited members not listed):

- `parts() -> list[Span]` — [line 113](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/components/unit_test_eval.py#L113)
  Return the constituent parts of this component.
- `format_for_llm() -> TemplateRepresentation` — [line 122](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/components/unit_test_eval.py#L122)
  Format this test for judge evaluation.
- `set_judge_context(input_text: str, prediction: str, targets_for_input: list[str]) -> None` — [line 141](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/components/unit_test_eval.py#L141)
  Set the context dictionary used when formatting this test for judge evaluation.
- `from_json_file(filepath: str) -> list['TestBasedEval']` *(classmethod)* — [line 169](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/components/unit_test_eval.py#L169)
  Load test evaluations from a JSON file, returning one `TestBasedEval` per unit test.

---

*Generated from source by `docs/scripts/generate_package_map.py` — do not edit by hand; regenerate instead. Symbols and annotations reflect the pinned commit exactly; where type annotations are absent in source, this page says so rather than guessing.*
