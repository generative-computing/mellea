# Migrating streaming from v0.7 to the single-task `stream()` API

v0.7 exposed streaming validation through `stream_with_chunking()`, which
returned a `StreamChunkingResult` driven by a background orchestration task. You
observed it through `result.events()` (typed events) or `result.astream()` (raw
chunks), and called `result.acomplete()` to wait for the background task to
finish.

That two-task model is replaced by a single-task primitive: `stream()` returns a
`Streamer` you consume directly with `async for`, on your own task. There is no
background orchestrator and no `acomplete()`. Typed events now come from the
`streaming_event` plugin hook rather than an iterator on the result. This is a
**breaking change with no deprecation shim** — call sites must be updated.

## API mapping

| v0.7 | v0.8 | Notes |
| --- | --- | --- |
| `stream_with_chunking(...)` | `stream(...)` | Same arguments, except `chunking` now defaults to `None` (raw deltas) instead of `"sentence"`; pass `chunking="sentence"` to preserve v0.7 chunk boundaries |
| returns `StreamChunkingResult` | returns `Streamer` | Consume with `async for`, ideally inside `async with` |
| `async for chunk in result.astream()` | `async for chunk in streamer` | Iterate the `Streamer` directly |
| `async for event in result.events()` | `@hook("streaming_event")` plugin | Events move to the hook (see below) |
| `await result.acomplete()` | *(removed)* | Consuming the stream drives it to completion |
| `result.completed` | `not streamer.failed_early` | |
| `result.full_text` | `streamer.full_text` | Same |
| `result.streaming_failures` | `streamer.streaming_failures` | Same |
| `result.final_validations` | `streamer.final_validations` | Same |
| `result.as_thunk` | `streamer.mot` | Set on natural completion |
| `SentenceChunker` | `SentenceChunking` | Strategy classes renamed |
| `WordChunker` | `WordChunking` | |
| `ParagraphChunker` | `ParagraphChunking` | |

Wrap consumption in `async with` so the generation is cancelled on every exit
path — an early `break` or an exception — instead of leaking an abandoned
background stream.

## Chunking strategy renames

The three built-in strategy classes were renamed from `...Chunker` to
`...Chunking`, freeing the `Chunker` name for the new stateful driver:

```python
# v0.7
from mellea.stdlib.chunking import SentenceChunker, WordChunker, ParagraphChunker

# v0.8
from mellea.stdlib.chunking import SentenceChunking, WordChunking, ParagraphChunking
```

This only affects code that imports a strategy class by name (for example, to
subclass it or pass an instance). Passing a string alias — `chunking="sentence"`,
`"word"`, or `"paragraph"` — is unchanged.

## Before and after: observing events

If you consumed typed events with `result.events()`, the events move to a
`streaming_event` plugin. Both snippets below produce the same output — they are
the `main()` from `docs/examples/streaming/validated_streaming.py`, v0.7 then
v0.8, using the same requirement, prompt, and chunking.

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
@hook("streaming_event")
async def print_events(payload, ctx) -> None:
    event = payload.event
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


register(print_events)

print("Stream events as they arrive:")
async with await stream(
    action, backend, ctx, requirements=[req], chunking="sentence"
) as streamer:
    # Draining the stream fires the events; the hook does the printing.
    async for _chunk in streamer:
        pass

print(f"\nCompleted normally: {not streamer.failed_early}")
print(f"Full text: {streamer.full_text!r}")
```

The three transformations to make:

1. **`stream_with_chunking(...)` → `async with await stream(...) as streamer:`**,
   and iterate `streamer` directly. The `async with` replaces `acomplete()` for
   cleanup.
2. **The `result.events()` match loop → a `@hook("streaming_event")` plugin.**
   The event vocabulary and payloads are unchanged; register the plugin (with
   `register()` or a `plugin_scope`) before consuming. Draining the stream is
   what fires the events.
3. **`result.<attr>` → `streamer.<attr>`**, with `result.completed` becoming
   `not streamer.failed_early`.

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
