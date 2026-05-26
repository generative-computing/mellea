---
name: mellea-logging
description: >
  Best-practices guide for adding or reviewing logging in the Mellea codebase.
  Covers when to use log_context() vs a dedicated logger call, canonical field
  names, reserved attribute constraints, async/thread safety, and what events
  deserve dedicated log lines.
  Use when: adding a new log call; reviewing a PR that touches MelleaLogger;
  deciding where to inject context fields; debugging why a field is missing from
  a log record; or ensuring consistency with the project logging conventions.
argument-hint: "[file-or-directory]"
compatibility: "Claude Code, IBM Bob"
metadata:
  version: "2026-04-15"
  capabilities: [read_file, grep, glob]
---

# Mellea Logging Best Practices

All logging in Mellea flows through `MelleaLogger.get_logger()`, defined in
`mellea/core/utils.py`. This skill documents the conventions for adding and
reviewing log instrumentation.

## Quick reference

```python
from mellea.core import MelleaLogger, log_context, set_log_context, clear_log_context

logger = MelleaLogger.get_logger()

# Dedicated log call — for a discrete event
logger.info("SUCCESS")

# Context injection — attach fields to every record in a scope
with log_context(request_id="req-abc", trace_id="t-1"):
    logger.info("Starting generation")  # includes request_id, trace_id
    # ... all nested calls inherit these fields automatically
```

## When to add a dedicated log call

Use `logger.info/warning/error(...)` for **discrete, named events**:

| Event type | Level | Example |
|------------|-------|---------|
| Phase transition | INFO | `"SUCCESS"`, `"FAILED"`, `"Starting session"` |
| Loop progress | INFO | `"Running loop 2 of 3"` |
| Recoverable issue | WARNING | `"Warmup failed for model: ..."` |
| Unexpected failure | ERROR | exception tracebacks, hard failures |
| Verbose diagnostics | DEBUG | token counts, prompt previews |

Do **not** add log calls for:

- Values already captured in a `log_context` field (redundant noise)
- Internal helper functions where the calling function already logs the event
- State that is already reflected in telemetry spans

## When to use log_context

Use `log_context` (or `set_log_context`) to attach **identifiers and metadata
that should appear on every log record within a scope** — without threading
them through every call.

Typical injection points:

| Scope | Where to inject | Fields |
|-------|----------------|--------|
| Session lifetime | `MelleaSession.__enter__` | `session_id`, `backend`, `model_id` |
| Sampling loop | `BaseSamplingStrategy.sample()` | `strategy`, `loop_budget` |
| HTTP request handler | entry point of the handler | `request_id`, `trace_id` |
| Background task | top of the task coroutine | `task_id`, `job_name` |

## Canonical field names

Use these names consistently. Do not invent synonyms.

| Field | Type | Description |
|-------|------|-------------|
| `session_id` | str (UUID) | Unique ID for a `MelleaSession` |
| `backend` | str | Backend class name, e.g. `"OllamaModelBackend"` |
| `model_id` | str | Model identifier string |
| `strategy` | str | Sampling strategy class name |
| `loop_budget` | int | Max generate/validate cycles for this sampling call |
| `request_id` | str | Caller-supplied request identifier |
| `trace_id` | str | Distributed trace ID (from OpenTelemetry or caller) |
| `span_id` | str | Span ID within a trace |
| `user_id` | str | End-user identifier (when applicable) |

## Reserved attribute names — do not use as context fields

The following names are standard `logging.LogRecord` attributes. Passing them
to `log_context()` or `set_log_context()` raises `ValueError`. See
`RESERVED_LOG_RECORD_ATTRS` in `mellea/core/utils.py` for the full set.

`args`, `created`, `exc_info`, `exc_text`, `filename`, `funcName`,
`levelname`, `levelno`, `lineno`, `message`, `module`, `msecs`, `msg`,
`name`, `pathname`, `process`, `processName`, `relativeCreated`,
`stack_info`, `thread`, `threadName`

## Prefer the context manager over set/clear

```python
# Preferred — guaranteed cleanup even on exceptions
with log_context(trace_id="abc"):
    do_work()

# Acceptable only when lifetime equals __enter__/__exit__
# (e.g. MelleaSession, where the CM already guarantees cleanup)
set_log_context(session_id=self.id)
# ... later in __exit__ ...
clear_log_context()
```

The context manager uses a `ContextVar` token to restore the previous state
on exit. This means **nesting works correctly** — inner calls can add fields
without clobbering the outer scope's values.

## Async and thread safety

`log_context` uses `contextvars.ContextVar`, which is safe for concurrent
asyncio tasks:

- Each `asyncio.Task` gets its own copy of the context.
- Fields set in one task do not bleed into sibling tasks.

**Plugin hooks**: Mellea hooks (`AUDIT`, `SEQUENTIAL`, `CONCURRENT`) are
`await`ed in the same asyncio task as the call site. `ContextVar` state IS
inherited — fields set around a `strategy.sample()` call will appear on
records emitted inside hook handlers automatically.

## Checklist before committing

1. New log calls use `MelleaLogger.get_logger()`, not `logging.getLogger(...)`.
2. Context fields use canonical names from the table above.
3. No reserved attribute names passed to `log_context`.
4. Scoped fields use `with log_context(...)`, not `set_log_context` (unless
   managing an `__enter__`/`__exit__` pair).
5. Hook handlers that need context set it internally — they do not inherit the
   caller's context.
6. New events that span multiple log records inject fields via context, not by
   repeating them on every `logger.info(...)` call.
