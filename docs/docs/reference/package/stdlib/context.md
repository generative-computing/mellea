---
id: context
title: "mellea.stdlib.context"
sidebar_label: "context"
sidebar_position: 3
description: "Concrete `Context` implementations and the `Compactor` protocol."
# diataxis: reference
---

Source: [`mellea/stdlib/context/__init__.py`](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/context/__init__.py) at commit `a535fc6345a0`.

Concrete `Context` implementations and the `Compactor` protocol.

Declared exports (`__all__`): `CBlock`, `ChatContext`, `Compactor`, `Component`, `Context`, `ContextTurn`, `InlineCompactor`, `LLMSummarizeCompactor`, `PinPredicate`, `SimpleContext`, `ThresholdCompactor`, `WindowCompactor`, `pin_nothing`, `pin_system`, `pin_system_and_initial_user`

---

## Module `mellea.stdlib.context.chat`

Source: [`mellea/stdlib/context/chat.py`](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/context/chat.py) at commit `a535fc6345a0`.

Chat-style context with pluggable compaction.

### `ChatContext`

*class* — [line 21](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/context/chat.py#L21) (`Context`)

Chat context that accumulates turns and optionally compacts on each `add`.

Constructor: `ChatContext(*, compactor: InlineCompactor | None = None, window_size: int | None = None, token_context_length_limit: int | None = None, model_id: str | ModelIdentifier | None = None) -> None`

Properties:

- `model_id` → `str | ModelIdentifier | None` — The model identifier bound to this context, or `None` if unbound.

Methods (defined on this class; inherited members not listed):

- `new_instance() -> ChatContext` — [line 156](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/context/chat.py#L156)
  Return a new empty root `ChatContext`, preserving compactor, token budget, and `model_id`.
- `add(c: Span) -> ChatContext` — [line 170](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/context/chat.py#L170)
  Append `c` and run the compactor; return the resulting context.
- `view_for_generation() -> list[Span] | None` — [line 187](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/context/chat.py#L187)
  Return the components to forward to the model.

---

## Module `mellea.stdlib.context.compactor`

Source: [`mellea/stdlib/context/compactor.py`](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/context/compactor.py) at commit `a535fc6345a0`.

Generic `Compactor` protocol for shrinking a `Context`.

### `Compactor`

*class* — [line 138](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/context/compactor.py#L138) (`Protocol`)

Protocol for objects that compact a `Context` into a smaller copy.

Methods (defined on this class; inherited members not listed):

- `compact(ctx: T, *, backend: Backend | None = None) -> T` — [line 156](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/context/compactor.py#L156)
  Return a compacted copy of `ctx`.

### `InlineCompactor`

*class* — [line 171](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/context/compactor.py#L171) 

Marker base for compactors safe to attach directly to `ChatContext`.

Methods (defined on this class; inherited members not listed):

- `compact(ctx: ChatContext, *, backend: Backend | None = None) -> ChatContext` — [line 192](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/context/compactor.py#L192)
  Subclasses must override this with their concrete strategy.

### `WindowCompactor`

*class* — [line 214](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/context/compactor.py#L214) (`InlineCompactor`)

Retains the last `size` body components of a `ChatContext`.

Constructor: `WindowCompactor(*, size: int, pin_predicate: PinPredicate = pin_system) -> None`

Methods (defined on this class; inherited members not listed):

- `compact(ctx: ChatContext, *, backend: Backend | None = None) -> ChatContext` — [line 248](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/context/compactor.py#L248)
  Return a copy of `ctx` truncated to the last `size` body components.

### `ThresholdCompactor`

*class* — [line 279](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/context/compactor.py#L279) (`InlineCompactor`)

Wraps an inner `Compactor`, gating it on the conversation's token size.

Constructor: `ThresholdCompactor(inner: Compactor, *, threshold: int) -> None`

Methods (defined on this class; inherited members not listed):

- `compact(ctx: ChatContext, *, backend: Backend | None = None) -> ChatContext` — [line 320](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/context/compactor.py#L320)
  Forward to `inner.compact` only when `ctx` exceeds the threshold.

### `LLMSummarizeCompactor`

*class* — [line 401](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/context/compactor.py#L401) 

Replace old body components with an LLM-generated summary, keep last `keep_n` verbatim.

Constructor: `LLMSummarizeCompactor(*, default_backend: Backend, keep_n: int = 5, pin_predicate: PinPredicate = pin_nothing, prompt_template: str | None = None, model_options: dict | None = None) -> None`

Methods (defined on this class; inherited members not listed):

- `compact(ctx: ChatContext, *, backend: Backend | None = None) -> ChatContext` — [line 483](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/context/compactor.py#L483)
  Return a context with the prefix, an LLM summary, and recent body components.

### `pin_nothing()`

*function* — [line 56](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/context/compactor.py#L56)

`pin_nothing(components: list[Span]) -> int`

A :class:`PinPredicate` that pins nothing — pure body, no protected prefix.

### `pin_system()`

*function* — [line 69](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/context/compactor.py#L69)

`pin_system(components: list[Span]) -> int`

Pin contiguous leading `Message(role="system")` components.

### `pin_system_and_initial_user()`

*function* — [line 89](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/context/compactor.py#L89)

`pin_system_and_initial_user(components: list[Span]) -> int`

Pin leading system messages PLUS the first user message that follows.

---

## Module `mellea.stdlib.context.simple`

Source: [`mellea/stdlib/context/simple.py`](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/context/simple.py) at commit `a535fc6345a0`.

Stateless single-turn context (no history is forwarded to the model).

### `SimpleContext`

*class* — [line 11](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/context/simple.py#L11) (`Context`)

A `SimpleContext` is a context in which each interaction is a separate and independent turn. The history of all previous turns is NOT saved..

Methods (defined on this class; inherited members not listed):

- `add(c: Span) -> SimpleContext` — [line 14](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/context/simple.py#L14)
  Add a new component or CBlock to the context and return the updated context.
- `view_for_generation() -> list[Span] | None` — [line 27](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/context/simple.py#L27)
  Return an empty list, since `SimpleContext` does not pass history to the model.

---

*Generated from source by `docs/scripts/generate_package_map.py` — do not edit by hand; regenerate instead. Symbols and annotations reflect the pinned commit exactly; where type annotations are absent in source, this page says so rather than guessing.*
