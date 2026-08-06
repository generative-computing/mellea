---
id: context
title: "mellea.telemetry.context"
sidebar_label: "context"
sidebar_position: 1
description: "Async-safe context propagation for Mellea telemetry."
# diataxis: reference
---

Source: [`mellea/telemetry/context.py`](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/telemetry/context.py) at commit `a535fc6345a0`.

Async-safe context propagation for Mellea telemetry.

## `MelleaContextFilter`

*class* — [line 213](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/telemetry/context.py#L213) (`logging.Filter`)

Logging filter that injects telemetry context fields into every log record.

Methods (defined on this class; inherited members not listed):

- `filter(record: logging.LogRecord) -> bool` — [line 222](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/telemetry/context.py#L222)
  Attach telemetry context fields to *record*.

## `get_session_id()`

*function* — [line 64](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/telemetry/context.py#L64)

`get_session_id() -> str | None`

Return the session_id for the current async context, or `None`.

## `get_request_id()`

*function* — [line 73](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/telemetry/context.py#L73)

`get_request_id() -> str | None`

Return the request_id for the current async context, or `None`.

## `get_model_id()`

*function* — [line 82](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/telemetry/context.py#L82)

`get_model_id() -> str | None`

Return the model_id for the current async context, or `None`.

## `get_sampling_iteration()`

*function* — [line 91](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/telemetry/context.py#L91)

`get_sampling_iteration() -> int | None`

Return the sampling_iteration for the current async context, or `None`.

## `get_current_context()`

*function* — [line 100](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/telemetry/context.py#L100)

`get_current_context() -> dict[str, Any]`

Return a snapshot of all non-`None` context values.

## `generate_request_id()`

*function* — [line 110](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/telemetry/context.py#L110)

`generate_request_id() -> str`

Generate a new unique request ID (UUID4 hex string).

## `with_context()`

*function* — [line 152](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/telemetry/context.py#L152)

`with_context(**kwargs: Any) -> Generator[None, None, None]`

Synchronous context manager that sets telemetry context for the block duration.

## `async_with_context()`

*async function* — [line 185](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/telemetry/context.py#L185)

`async_with_context(**kwargs: Any) -> AsyncGenerator[None, None]`

Async-with variant of :func:`with_context`.

---

*Generated from source by `docs/scripts/generate_package_map.py` — do not edit by hand; regenerate instead. Symbols and annotations reflect the pinned commit exactly; where type annotations are absent in source, this page says so rather than guessing.*
