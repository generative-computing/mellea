---
id: base
title: "mellea.core.base"
sidebar_label: "base"
sidebar_position: 2
description: "Foundational data structures for mellea's generative programming model."
# diataxis: reference
---

Source: [`mellea/core/base.py`](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/core/base.py) at commit `a535fc6345a0`.

Foundational data structures for mellea's generative programming model.

## `CBlock`

*class* — [line 54](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/core/base.py#L54) 

A `CBlock` is a block of content that can serve as input to or output from an LLM.

Constructor: `CBlock(value: str | None, meta: dict[str, Any] | None = None, *, cache: bool = False)`

Properties:

- `value` → `str | None` — Gets the value of the block.

Methods (defined on this class; inherited members not listed):

- `value(v: str) -> None` — [line 87](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/core/base.py#L87)
  Sets the value of the block.

## `ImageBlock`

*class* — [line 100](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/core/base.py#L100) (`CBlock`)

A `ImageBlock` represents an image (as base64 PNG).

Constructor: `ImageBlock(value: str, meta: dict[str, Any] | None = None)`

Methods (defined on this class; inherited members not listed):

- `is_valid_base64_png(s: str) -> bool` *(staticmethod)* — [line 120](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/core/base.py#L120)
  Checks whether a string is a valid base64-encoded PNG image.
- `pil_to_base64(image: PILImage.Image) -> str` *(staticmethod)* — [line 159](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/core/base.py#L159)
  Converts a PIL image to a base64-encoded PNG string.
- `from_pil_image(image: PILImage.Image, meta: dict[str, Any] | None = None) -> ImageBlock` *(classmethod)* — [line 173](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/core/base.py#L173)
  Creates an `ImageBlock` from a PIL image object.

## `ImageUrlBlock`

*class* — [line 196](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/core/base.py#L196) (`CBlock`)

An `ImageUrlBlock` represents an image as a URL.

Constructor: `ImageUrlBlock(value: str, meta: dict[str, Any] | None = None)`

Methods (defined on this class; inherited members not listed):

- `resolve_base64() -> str` — [line 223](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/core/base.py#L223)
  Return the image as a base64-encoded PNG, downloading it once per URL.

## `AudioBlock`

*class* — [line 247](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/core/base.py#L247) (`CBlock`)

An `AudioBlock` represents audio as base64 data.

Constructor: `AudioBlock(value: str, format: str | None = None, meta: dict[str, Any] | None = None)`

Methods (defined on this class; inherited members not listed):

- `is_valid_base64_audio(s: str) -> bool` *(staticmethod)* — [line 306](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/core/base.py#L306)
  Checks whether a string is valid base64-encoded audio payload data.

## `AudioUrlBlock`

*class* — [line 337](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/core/base.py#L337) (`CBlock`)

An `AudioUrlBlock` represents audio as a URL.

Constructor: `AudioUrlBlock(value: str, format: str, meta: dict[str, Any] | None = None)`

## `ComponentParseError`

*class* — [line 530](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/core/base.py#L530) (`Exception`)

Raised by `Component.parse()` when the underlying parsing method throws an exception.

## `Component`

*class* — [line 535](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/core/base.py#L535) (`Protocol`, `Generic[S]`)

A `Component` is a composite data structure that is intended to be represented to an LLM.

Methods (defined on this class; inherited members not listed):

- `parts() -> list[Span]` — [line 538](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/core/base.py#L538)
  Returns the set of all constituent sub-components and content blocks of this `Component`.
- `format_for_llm() -> TemplateRepresentation | str` — [line 551](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/core/base.py#L551)
  Formats the `Component` into a `TemplateRepresentation` or plain string for LLM consumption.
- `parse(computed: ModelOutputThunk) -> S` — [line 563](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/core/base.py#L563)
  Parses the expected type `S` from a given `ModelOutputThunk`.

## `GenerateType`

*class* — [line 588](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/core/base.py#L588) (`enum.Enum`)

Used to track what functions can be used to extract a value from a ModelOutputThunk.

## `GenerationMetadata`

*class* — [line 603](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/core/base.py#L603) 

Backend execution metadata attached to every ModelOutputThunk.

## `RawProviderResponse`

*class* — [line 699](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/core/base.py#L699) 

Backend-native response payload from the provider's SDK.

## `ModelOutputThunk`

*class* — [line 783](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/core/base.py#L783) (`Generic[S]`)

A `ModelOutputThunk` represents a lazily-evaluated model response. It is possible to instantiate one without the output being computed yet.

Constructor: `ModelOutputThunk(value: str | None, meta: dict[str, Any] | None = None, parsed_repr: S | None = None, tool_calls: list[ModelToolCall] | None = None)`

Properties:

- `cancelled` → `bool` — `True` if :meth:`cancel_generation` ran to completion on this MOT.
- `error` → `Exception | None` — Soft-failure cause recorded by the backend, or `None` on success.
- `generate_log` → `GenerateLog | None` — The `GenerateLog` recorded for this generation.
- `value` → `str | None` — Gets the value of the block.

Methods (defined on this class; inherited members not listed):

- `cancel_generation(error: Exception | None = None) -> None` *(async)* — [line 856](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/core/base.py#L856)
  Cancel an in-progress streaming generation, drain the queue, and fire the `generation_error` hook.
- `is_computed() -> bool` — [line 1012](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/core/base.py#L1012)
  Returns true only if this Thunk has already been filled.
- `value(v: str) -> None` — [line 1028](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/core/base.py#L1028)
  Sets the value of the block.
- `avalue() -> str` *(async)* — [line 1032](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/core/base.py#L1032)
  Returns the fully resolved value of the ModelOutputThunk, awaiting generation if necessary.
- `astream() -> str` *(async)* — [line 1061](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/core/base.py#L1061)
  Returns only the NEW text fragment (delta) received since the last call.

## `ComputedModelOutputThunk`

*class* — [line 1315](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/core/base.py#L1315) (`ModelOutputThunk[S]`)

A `ComputedModelOutputThunk` is a `ModelOutputThunk` that is guaranteed to be computed.

Constructor: `ComputedModelOutputThunk(thunk: ModelOutputThunk[S]) -> None`

Properties:

- `value` → `str` — The raw string value produced by the model.

Methods (defined on this class; inherited members not listed):

- `avalue() -> str` *(async)* — [line 1348](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/core/base.py#L1348)
  Return the value of the thunk. Use .value instead.
- `astream() -> str` *(async)* — [line 1357](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/core/base.py#L1357)
  Cannot astream from ComputedModelOutputThunks. Use .value instead.
- `value(v: str)` — [line 1382](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/core/base.py#L1382)
  Sets the value of the block.
  *Annotation gaps in source: return type unannotated.*
- `is_computed() -> Literal[True]` — [line 1386](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/core/base.py#L1386)
  Returns `True` since thunk is always computed.

## `ContextTurn`

*class* — [line 1413](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/core/base.py#L1413) 

A turn of model input and model output.

## `Context`

*class* — [line 1434](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/core/base.py#L1434) (`abc.ABC`)

A `Context` is used to track the state of a `MelleaSession`.

Constructor: `Context() -> None`

Properties:

- `is_root_node` → `bool` — Returns whether this context is the root context node.
- `previous_node` → `Context | None` — Returns the context node from which this context node was created.
- `node_data` → `Span | None` — Returns the data associated with this context node.
- `is_chat_context` → `bool` — Returns whether this context is a chat context.

Methods (defined on this class; inherited members not listed):

- `from_previous(cls: type[ContextT], previous: Context, data: Span) -> ContextT` *(classmethod)* — [line 1462](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/core/base.py#L1462)
  Constructs a new context node linked to an existing context node.
- `reset_to_new(cls: type[ContextT]) -> ContextT` *(classmethod)* — [line 1485](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/core/base.py#L1485)
  Returns a new empty (root) context.
- `new_instance() -> Context` — [line 1493](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/core/base.py#L1493)
  Return a new empty root context, preserving any subclass configuration.
- `as_list(last_n_components: int | None = None) -> list[Span]` — [line 1536](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/core/base.py#L1536)
  Returns a list of context components sorted from earliest (first) to most recent (last).
- `actions_for_available_tools() -> list[Span] | None` — [line 1571](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/core/base.py#L1571)
  Provides a list of actions to extract tools from for use during generation.
- `last_output(check_last_n_components: int = 3) -> ModelOutputThunk | None` — [line 1584](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/core/base.py#L1584)
  Returns the most recent `ModelOutputThunk` found within the last N context components.
- `last_turn() -> ContextTurn | None` — [line 1600](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/core/base.py#L1600)
  The last input/output turn of the context.
- `add(c: Span) -> Context` — [line 1632](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/core/base.py#L1632)
  Returns a new context obtained by appending `c` to this context.
- `view_for_generation() -> list[Span] | None` — [line 1645](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/core/base.py#L1645)
  Provides a linear list of context components to use for generation.

## `AbstractMelleaTool`

*class* — [line 1662](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/core/base.py#L1662) (`abc.ABC`, `Generic[P, R]`)

Abstract base class for Mellea Tool with parameter and return type support.

Properties:

- `as_json_tool` → `dict[str, Any]` — Provides a JSON description for Mellea Tool.

Methods (defined on this class; inherited members not listed):

- `run(*args: P.args, **kwargs: P.kwargs) -> R` — [line 1679](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/core/base.py#L1679)
  Executes the tool with the provided arguments and returns the result.

## `TemplateRepresentation`

*class* — [line 1697](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/core/base.py#L1697) 

Representing a component as a set of important attributes that can be consumed by the formatter.

## `GenerateLog`

*class* — [line 1730](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/core/base.py#L1730) 

A dataclass for capturing log entries for a single generation call.

## `ModelToolCall`

*class* — [line 1761](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/core/base.py#L1761) 

A dataclass for capturing the tool calls a model wants to make.

Methods (defined on this class; inherited members not listed):

- `call_func() -> Any` — [line 1780](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/core/base.py#L1780)
  Invokes the tool represented by this object and returns the result.

## `make_image_block()`

*function* — [line 466](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/core/base.py#L466)

`make_image_block(src: str | PILImage.Image, *, convert_to_base64: bool = False, meta: dict[str, Any] | None = None) -> ImageBlock | ImageUrlBlock`

Create the appropriate image block from any supported image source.

## `blockify()`

*function* — [line 1789](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/core/base.py#L1789)

`blockify(s: str | Span) -> Span`

Turn a raw string into a `CBlock`, leaving `CBlock`, `Component`, and `ModelOutputThunk` objects unchanged.

## `get_images_from_component()`

*function* — [line 1815](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/core/base.py#L1815)

`get_images_from_component(c: Component) -> None | list[ImageBlock | ImageUrlBlock]`

Return the images attached to a `Component`, or `None` if absent or empty.

## `get_audio_from_component()`

*function* — [line 1843](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/core/base.py#L1843)

`get_audio_from_component(c: Component) -> None | list[AudioBlock | AudioUrlBlock]`

Return the audio attached to a `Component`, or `None` if absent or empty.

---

*Generated from source by `docs/scripts/generate_package_map.py` — do not edit by hand; regenerate instead. Symbols and annotations reflect the pinned commit exactly; where type annotations are absent in source, this page says so rather than guessing.*
