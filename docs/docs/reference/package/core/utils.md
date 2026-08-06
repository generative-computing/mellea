---
id: utils
title: "mellea.core.utils"
sidebar_label: "utils"
sidebar_position: 6
description: "Logging utilities for the mellea core library."
# diataxis: reference
---

Source: [`mellea/core/utils.py`](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/core/utils.py) at commit `a535fc6345a0`.

Logging utilities for the mellea core library.

## `ContextFilter`

*class* — [line 187](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/core/utils.py#L187) (`logging.Filter`)

Logging filter that injects async-safe ContextVar fields into every record.

Methods (defined on this class; inherited members not listed):

- `filter(record: logging.LogRecord) -> bool` — [line 195](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/core/utils.py#L195)
  Attach async-safe ContextVar fields to *record* and allow it through.

## `OtelTraceFilter`

*class* — [line 210](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/core/utils.py#L210) (`logging.Filter`)

Logging filter that injects the current OpenTelemetry trace context into log records.

Methods (defined on this class; inherited members not listed):

- `filter(record: logging.LogRecord) -> bool` — [line 220](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/core/utils.py#L220)
  Adds trace_id and span_id to the log record from the current OTel span.

## `RESTHandler`

*class* — [line 239](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/core/utils.py#L239) (`logging.Handler`)

Logging handler that forwards records to an HTTP endpoint unconditionally.

Constructor: `RESTHandler(api_url: str, method: str = 'POST', headers: dict[str, str] | None = None) -> None`

Methods (defined on this class; inherited members not listed):

- `emit(record: logging.LogRecord) -> None` — [line 264](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/core/utils.py#L264)
  Forward *record* to the configured REST endpoint.

## `JsonFormatter`

*class* — [line 291](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/core/utils.py#L291) (`logging.Formatter`)

Logging formatter that serialises log records as structured JSON strings.

Constructor: `JsonFormatter(timestamp_format: str = '%Y-%m-%dT%H:%M:%S', include_fields: list[str] | None = None, exclude_fields: list[str] | None = None, extra_fields: dict[str, Any] | None = None, **kwargs: Any) -> None`

Methods (defined on this class; inherited members not listed):

- `format_as_dict(record: logging.LogRecord) -> dict[str, Any]` — [line 353](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/core/utils.py#L353)
  Return the log record as a dictionary (public API for external callers).
- `format(record: logging.LogRecord) -> str` — [line 436](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/core/utils.py#L436)
  Formats a log record as a JSON string.

## `CustomFormatter`

*class* — [line 452](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/core/utils.py#L452) (`logging.Formatter`)

A nice custom formatter copied from [Sergey Pleshakov's post on StackOverflow](https://stackoverflow.com/questions/384076/how-can-i-color-python-logging-output).

Methods (defined on this class; inherited members not listed):

- `format(record: logging.LogRecord) -> str` — [line 481](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/core/utils.py#L481)
  Formats a log record using a colour-coded ANSI format string based on the record's log level.

## `MelleaLogger`

*class* — [line 643](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/core/utils.py#L643) 

Singleton logger with colour-coded console output and configurable handlers.

Methods (defined on this class; inherited members not listed):

- `get_logger() -> logging.Logger` *(staticmethod)* — [line 690](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/core/utils.py#L690)
  Return the shared `logging.Logger`, creating it on first call.

## `set_log_context()`

*function* — [line 115](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/core/utils.py#L115)

`set_log_context(**fields: Any) -> None`

Inject extra fields into every log record emitted from this coroutine or thread.

## `clear_log_context()`

*function* — [line 141](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/core/utils.py#L141)

`clear_log_context() -> None`

Remove all context fields set by `set_log_context` for this coroutine/thread.

## `log_context()`

*function* — [line 147](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/core/utils.py#L147)

`log_context(**fields: Any) -> Generator[None, None, None]`

Context manager that injects *fields* for the duration of the block.

## `configure_logging()`

*function* — [line 559](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/core/utils.py#L559)

`configure_logging(logger: logging.Logger) -> None`

Attach log handlers to *logger* based on current environment variables.

---

*Generated from source by `docs/scripts/generate_package_map.py` — do not edit by hand; regenerate instead. Symbols and annotations reflect the pinned commit exactly; where type annotations are absent in source, this page says so rather than guessing.*
