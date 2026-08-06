---
id: backend
title: "mellea.core.backend"
sidebar_label: "backend"
sidebar_position: 1
description: "Abstract `Backend` interface and generation-walk utilities."
# diataxis: reference
---

Source: [`mellea/core/backend.py`](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/core/backend.py) at commit `a535fc6345a0`.

Abstract `Backend` interface and generation-walk utilities.

## `Backend`

*class* — [line 48](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/core/backend.py#L48) (`abc.ABC`)

Abstract base class for all inference backends.

Methods (defined on this class; inherited members not listed):

- `generate_from_context(action: Component[C] | CBlock | ModelOutputThunk, ctx: Context, *, format: type[BaseModelSubclass] | None = None, model_options: dict | None = None, tool_calls: bool = False) -> tuple[ModelOutputThunk[C], Context]` *(async)* — [line 66](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/core/backend.py#L66)
  Generates a model output from a context. May not mutate the context. This must be called from a running event loop as it creates a task to run the generation request.
- `generate_from_raw(actions: Sequence[Component[C] | CBlock], ctx: Context, *, format: type[BaseModelSubclass] | None = None, model_options: dict | None = None, tool_calls: bool = False) -> list[ModelOutputThunk]` *(async)* — [line 186](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/core/backend.py#L186)
  Generates a model output from the provided input. Does not use context or templates.
- `do_generate_walk(action: Span) -> None` *(async)* — [line 305](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/core/backend.py#L305)
  Awaits all uncomputed `ModelOutputThunk` leaves reachable from `action`.
- `do_generate_walks(actions: list[Span]) -> None` *(async)* — [line 323](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/core/backend.py#L323)
  Awaits all uncomputed `ModelOutputThunk` leaves reachable from each action in `actions`.

## `generate_walk()`

*function* — [line 344](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/core/backend.py#L344)

`generate_walk(c: Span) -> list[ModelOutputThunk]`

Return all uncomputed `ModelOutputThunk` leaves reachable from `c`.

---

*Generated from source by `docs/scripts/generate_package_map.py` — do not edit by hand; regenerate instead. Symbols and annotations reflect the pinned commit exactly; where type annotations are absent in source, this page says so rather than guessing.*
