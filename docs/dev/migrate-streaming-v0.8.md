# Migrating streaming from v0.7 to the single-task `stream()` API

v0.7 exposed streaming validation through `stream_with_chunking()`, which
returned a `StreamChunkingResult` driven by a background orchestration task. You
observed it through `result.events()` (typed events) or `result.astream()` (raw
chunks), and called `result.acomplete()` to wait for the background task to
finish.

That two-task model is replaced by a single-task primitive: `stream()` returns a
`Streamer` you consume directly with `async for`, on your own task. There is no
background orchestrator and no `acomplete()`. Typed events now come from an
`EventStreamer`, returned by `stream(as_events=True)` and iterated with `async
for`. This is a **breaking change with no deprecation shim** — call sites must be
updated.

## API mapping

| v0.7 | v0.8 | Notes |
| --- | --- | --- |
| `stream_with_chunking(...)` | `stream(...)` | Same arguments, except `chunking` now defaults to `None` (raw deltas) instead of `"sentence"`; pass `chunking="sentence"` to preserve v0.7 chunk boundaries |
| returns `StreamChunkingResult` | returns `Streamer` | Consume with `async for`, ideally inside `async with` |
| `async for chunk in result.astream()` | `async for chunk in streamer` | Iterate the `Streamer` directly |
| `async for event in result.events()` | `async for event in streamer` | Iterate the `EventStreamer` directly (`stream(as_events=True)`) |
| `await result.acomplete()` | *(removed)* | Consuming the stream drives it to completion |
| `STREAMING_ORCHESTRATION_START`/`_END` hooks | *(removed)* | No replacement — the whole run is on one task now, so there is nothing to reattach a span across |
| `result.completed` | `not streamer.failed_early` | |
| *(new)* | `streamer.completed_normally` | `True` only on natural completion; unlike `not failed_early`, it is `False` after an early `break` |
| `result.full_text` | `streamer.full_text` | Same |
| `result.streaming_failures` | `streamer.streaming_failures` | Same |
| `result.final_validations` | `streamer.final_validations` | Same |
| `result.as_thunk` | `streamer.mot` | Set on natural completion |
| `QuickCheckEvent.results` items | `PartialValidationSummary` | Was `PartialValidationResult`; `.success` reads the same, but `.reason` is only the failing chunk's reason (`None` if none failed) — read `.results` for non-failure reasons |
| `SentenceChunker` | `SentenceChunking` | Strategy classes renamed |
| `WordChunker` | `WordChunking` | |
| `ParagraphChunker` | `ParagraphChunking` | |

Wrap consumption in `async with` so the generation is cancelled on every exit
path — an early `break` or an exception — instead of leaking an abandoned
background stream.

## Chunking strategy renames and module move

The three built-in strategy classes were renamed from `...Chunker` to
`...Chunking`, freeing the `Chunker` name for the new stateful driver, and the
chunking module moved from `mellea.stdlib.chunking` to `mellea.core.chunking`:

```python
# v0.7
from mellea.stdlib.chunking import SentenceChunker, WordChunker, ParagraphChunker

# v0.8
from mellea.core.chunking import SentenceChunking, WordChunking, ParagraphChunking
```

This only affects code that imports a strategy class by name (for example, to
subclass it or pass an instance). Passing a string alias — `chunking="sentence"`,
`"word"`, or `"paragraph"` — is unchanged.

## Before and after: observing events

If you consumed typed events with `result.events()`, you now iterate an
`EventStreamer` (`stream(as_events=True)`). Both snippets below produce the same
output — they are the `main()` from
`docs/examples/streaming/validated_streaming.py`, v0.7 then v0.8, using the same
requirement, prompt, and chunking.

### v0.7

```python
result = await stream_with_chunking(
    action, backend, ctx, requirements=[req], chunking="sentence"
)

print("Streaming events as they arrive:")
async for event in result.events():
    match event:
        case ChunkEvent():
            print(f"  CHUNK[{event.chunk_index}]: {event.text!r}")
        case QuickCheckEvent(passed=False):
            print(
                f"  QUICK_CHECK[{event.chunk_index}]: FAIL — "
                f"{event.results[0].reason if event.results else 'unknown reason'}"
            )
        case QuickCheckEvent():
            print(f"  QUICK_CHECK[{event.chunk_index}]: pass")
        case StreamingDoneEvent():
            print(f"  STREAMING_DONE: {len(event.full_text)} chars accumulated")
        case FullValidationEvent():
            print(f"  FULL_VALIDATION: {'PASS' if event.passed else 'FAIL'}")
        case CompletedEvent():
            print(f"  COMPLETED: success={event.success}")
        case _:
            pass

await result.acomplete()

print(f"\nCompleted normally: {result.completed}")
print(f"Full text: {result.full_text!r}")
```

### v0.8

```python
print("Stream events as they arrive:")
async with await stream(
    action, backend, ctx, requirements=[req], chunking="sentence", as_events=True
) as streamer:
    async for event in streamer:
        match event:
            case ChunkEvent():
                print(f"  CHUNK[{event.chunk_index}]: {event.text!r}")
            case QuickCheckEvent(passed=False):
                print(
                    f"  QUICK_CHECK[{event.chunk_index}]: FAIL — "
                    f"{event.results[0].reason if event.results else 'unknown reason'}"
                )
            case QuickCheckEvent():
                print(f"  QUICK_CHECK[{event.chunk_index}]: pass")
            case StreamingDoneEvent():
                print(f"  STREAMING_DONE: {len(event.full_text)} chars accumulated")
            case FullValidationEvent():
                print(f"  FULL_VALIDATION: {'PASS' if event.passed else 'FAIL'}")
            case CompletedEvent():
                print(f"  COMPLETED: success={event.success}")
            case _:
                pass

print(f"\nCompleted normally: {streamer.completed_normally}")
print(f"Full text: {streamer.full_text!r}")
```

The three transformations to make:

1. **`stream_with_chunking(...)` → `async with await stream(..., as_events=True)
   as streamer:`**, and iterate `streamer` directly. The `async with` replaces
   `acomplete()` for cleanup.
2. **`async for event in result.events()` → `async for event in streamer`.** The
   event vocabulary and payloads are unchanged.
3. **`result.<attr>` → `streamer.<attr>`**. `result.completed` maps directly to
   `not streamer.failed_early`; prefer the new `streamer.completed_normally` if
   you need a signal that also excludes an early `break` (used above).

## Before and after: consuming raw chunks

If you never used `events()` and only consumed the validated chunk text with
`result.astream()`, the migration is smaller — iterate the `Streamer` directly
and drop `acomplete()`:

### v0.7

```python
result = await stream_with_chunking(
    action, backend, ctx, requirements=[req], chunking="sentence"
)
async for chunk in result.astream():
    print(chunk)
await result.acomplete()
```

### v0.8

```python
async with await stream(
    action, backend, ctx, requirements=[req], chunking="sentence"
) as streamer:
    async for chunk in streamer:
        print(chunk)
```

## See also

- [Streaming validation tutorial](../docs/tutorials/06-streaming-validation.md)
- [How-to: Async and streaming](../docs/how-to/use-async-and-streaming.md)
- [`docs/examples/streaming/`](../examples/streaming/)
