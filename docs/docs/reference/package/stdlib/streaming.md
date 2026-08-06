---
id: streaming
title: "mellea.stdlib.streaming"
sidebar_label: "streaming"
sidebar_position: 10
description: "Streaming generation with per-chunk validation."
# diataxis: reference
---

Source: [`mellea/stdlib/streaming.py`](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/streaming.py) at commit `a535fc6345a0`.

Streaming generation with per-chunk validation.

## `StreamEvent`

*class* — [line 46](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/streaming.py#L46) 

Base class for all streaming events emitted by :func:`stream_with_chunking`.

## `ChunkEvent`

*class* — [line 63](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/streaming.py#L63) (`StreamEvent`)

Emitted after each validated chunk is delivered to the consumer.

## `QuickCheckEvent`

*class* — [line 82](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/streaming.py#L82) (`StreamEvent`)

Emitted after each per-chunk streaming validation batch.

## `StreamingDoneEvent`

*class* — [line 105](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/streaming.py#L105) (`StreamEvent`)

Emitted after all chunks have been validated and delivered to the consumer.

## `FullValidationEvent`

*class* — [line 123](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/streaming.py#L123) (`StreamEvent`)

Emitted after the final :meth:`~mellea.core.requirement.Requirement.validate` calls complete.

## `RetryEvent`

*class* — [line 143](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/streaming.py#L143) (`StreamEvent`)

Reserved for future use.

## `CompletedEvent`

*class* — [line 161](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/streaming.py#L161) (`StreamEvent`)

Emitted when the orchestrator exits, including early-exit cases.

## `ErrorEvent`

*class* — [line 181](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/streaming.py#L181) (`StreamEvent`)

Emitted when an unhandled exception occurs in the orchestrator.

## `StreamChunkingResult`

*class* — [line 201](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/streaming.py#L201) 

Result of a :func:`stream_with_chunking` operation.

Constructor: `StreamChunkingResult(mot: ModelOutputThunk, ctx: Context, streaming_id: str) -> None`

Properties:

- `as_thunk` → `ModelOutputThunk[str]` — Wrap the output as a computed :class:`~mellea.core.base.ModelOutputThunk`.

Methods (defined on this class; inherited members not listed):

- `astream() -> AsyncIterator[str]` *(async)* — [line 276](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/streaming.py#L276)
  Yield validated text chunks as they complete.
- `events() -> AsyncIterator[StreamEvent]` *(async)* — [line 317](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/streaming.py#L317)
  Yield typed streaming events as they are emitted by the orchestrator.
- `acomplete() -> None` *(async)* — [line 377](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/streaming.py#L377)
  Await full completion, including final validation.

## `stream_with_chunking()`

*async function* — [line 734](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/streaming.py#L734)

`stream_with_chunking(action: Component[Any] | CBlock, backend: Backend, ctx: Context, *, requirements: Sequence[Requirement] | None = None, chunking: str | ChunkingStrategy = 'sentence', validation_backend: Backend | None = None) -> StreamChunkingResult`

Generate a streaming response with per-chunk validation.

---

*Generated from source by `docs/scripts/generate_package_map.py` — do not edit by hand; regenerate instead. Symbols and annotations reflect the pinned commit exactly; where type annotations are absent in source, this page says so rather than guessing.*
