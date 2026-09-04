---
title: "Async and Streaming"
description: "Use async methods, parallel generation, and streaming output with Mellea."
# diataxis: how-to
---

**Prerequisites:** [Quick Start](../getting-started/quickstart) complete,
`pip install mellea`, Ollama running locally.

## Async methods

Every sync method on `MelleaSession` has an `a`-prefixed async counterpart with the
same signature and return type:

| Sync | Async |
| ---- | ----- |
| `instruct()` | `ainstruct()` |
| `chat()` | `achat()` |
| `act()` | `aact()` |
| `validate()` | `avalidate()` |
| `query()` | `aquery()` |
| `transform()` | `atransform()` |

```python
# Requires: mellea
# Returns: None
import asyncio
import mellea

async def main():
    m = mellea.start_session()
    result = await m.ainstruct("Write a haiku about concurrency.")
    print(str(result))
    # Output will vary — LLM responses depend on model and temperature.

asyncio.run(main())
```

## Parallel generation

`ainstruct()` returns a `ModelOutputThunk` immediately — generation starts in the
background but the value is not resolved until you call `avalue()`. This lets you
fire multiple generations and resolve them all at once:

```python
# Requires: mellea
# Returns: None
import asyncio
import mellea

async def main():
    m = mellea.start_session()

    # Fire off all three — generation starts for each immediately
    thunk_a = await m.ainstruct("Write a poem about mountains.")
    thunk_b = await m.ainstruct("Write a poem about rivers.")
    thunk_c = await m.ainstruct("Write a poem about forests.")

    # None are resolved yet
    print(thunk_a.is_computed())  # False

    # Resolve all in parallel
    await asyncio.gather(
        thunk_a.avalue(),
        thunk_b.avalue(),
        thunk_c.avalue(),
    )

    print(thunk_a.value)
    print(thunk_b.value)
    print(thunk_c.value)
    # Output will vary — LLM responses depend on model and temperature.

asyncio.run(main())
```

For a list of thunks, `wait_for_all_mots` is a convenience wrapper:

```python
# Requires: mellea
# Returns: None
import asyncio
import mellea
from mellea.helpers.async_helpers import wait_for_all_mots

async def main():
    m = mellea.start_session()

    thunks = []
    for topic in ["mountains", "rivers", "forests"]:
        thunks.append(await m.ainstruct(f"Write a short poem about {topic}."))

    await wait_for_all_mots(thunks)

    for t in thunks:
        print(t.value)
    # Output will vary — LLM responses depend on model and temperature.

asyncio.run(main())
```

> **Note:** All thunks passed to `wait_for_all_mots` must belong to the same event
> loop, which is always the case when using `MelleaSession`.

## Streaming

Enable streaming by passing `ModelOption.STREAM: True` in `model_options`. Consume
incremental output chunks with `mot.astream()`:

```python
# Requires: mellea
# Returns: None
import asyncio
import mellea
from mellea.backends import ModelOption

async def main():
    m = mellea.start_session()
    mot = await m.ainstruct(
        "Write a short story about a robot learning to cook.",
        model_options={ModelOption.STREAM: True},
    )

    # Consume chunks as they arrive
    while not mot.is_computed():
        chunk = await mot.astream()
        print(chunk, end="", flush=True)

    print()  # newline after streaming completes

asyncio.run(main())
# Output will vary — LLM responses depend on model and temperature.
```

How `astream()` behaves:

- Each call returns only the **new content** since the previous call.
- When the thunk is fully computed (`is_computed()` returns `True`), the final
  `astream()` call returns the **complete value**.
- If the thunk is already computed, `astream()` returns the full value immediately.

> **Warning:** Do not call `astream()` from multiple coroutines simultaneously on
> the same thunk. Each thunk should have a single reader.

### Iterating a thunk with `async for`

`astream()` is the low-level primitive. To consume a thunk as an async iterator,
use `async for` — ideally inside `async with`, which cancels the generation if
you leave the loop early (an exception or `break`) so an abandoned stream does
not keep running:

```python
# Requires: mellea
# Returns: None
import asyncio
import mellea
from mellea.backends import ModelOption

async def main():
    m = mellea.start_session()
    mot = await m.ainstruct(
        "Write a short story about a robot learning to cook.",
        model_options={ModelOption.STREAM: True},
    )

    async with mot:
        async for delta in mot:
            print(delta, end="", flush=True)
    print()  # newline after streaming completes

asyncio.run(main())
# Output will vary — LLM responses depend on model and temperature.
```

Each iteration yields the same delta `astream()` would return; iteration ends
when the thunk is computed. Like `astream()`, a thunk has a single reader — a
second `async for` over the same thunk raises rather than splitting the stream.

### Streaming timeout

Mellea waits up to 120 seconds for each chunk by default, including the first
(time-to-first-token). If the backend stops sending without closing the connection
the stream aborts with a `TimeoutError` rather than hanging indefinitely.

For slow local inference or large models on CPU, increase the timeout or disable it:

```python
# Shorter timeout for a fast remote endpoint
mot = await m.ainstruct(
    "Summarise this document.",
    model_options={ModelOption.STREAM: True, ModelOption.STREAM_TIMEOUT: 10},
)

# Larger value for slow local inference
mot = await m.ainstruct(
    "Write a long analysis.",
    model_options={ModelOption.STREAM: True, ModelOption.STREAM_TIMEOUT: 300},
)

# Disable entirely — original unbounded behaviour
mot = await m.ainstruct(
    "Write a long analysis.",
    model_options={ModelOption.STREAM: True, ModelOption.STREAM_TIMEOUT: None},
)
```

See [Configure model options — Streaming timeout](../how-to/configure-model-options#streaming-timeout)
for the full reference.

## Async and context

Use `SimpleContext` (the default) with concurrent async requests. Using `ChatContext`
with concurrent requests can cause stale context issues — Mellea logs a warning
when this is detected:

```text
WARNING: Not using a SimpleContext with asynchronous requests could cause
unexpected results due to stale contexts. Ensure you await between requests.
```

If you need `ChatContext` with async, await each call before starting the next:

```python
# Requires: mellea
# Returns: None
import asyncio
import mellea
from mellea.stdlib.context import ChatContext

async def sequential_chat():
    m = mellea.start_session(ctx=ChatContext())
    r1 = await m.achat("Hello.")
    r2 = await m.achat("Tell me more.")  # safe — r1 is fully resolved
    print(str(r2))
    # Output will vary — LLM responses depend on model and temperature.

asyncio.run(sequential_chat())
```

For parallel generation, use `SimpleContext`.

## Streaming with per-chunk validation

`stream()` adds per-chunk validation to a streaming generation. It splits the
accumulated text into semantic units (sentences, words, or paragraphs), calls
`stream_validate()` on each chunk in parallel, and can exit early if any
requirement returns `"fail"` — preventing the consumer from seeing invalid
content mid-stream.

`stream()` returns a `Streamer` you consume with `async for`, ideally inside
`async with` so the generation is released on every exit path (including an
early `break` or exception):

```python
# Requires: mellea
# Returns: None
import asyncio

from mellea.core.backend import Backend
from mellea.core.base import Context
from mellea.core.requirement import PartialValidationResult, Requirement, ValidationResult
from mellea.stdlib.components import Instruction
from mellea.stdlib.streaming import stream


class MaxSentencesReq(Requirement):
    """Fails if the model generates more than *limit* sentences."""

    def __init__(self, limit: int) -> None:
        super().__init__()
        self._limit = limit
        self._count = 0

    def format_for_llm(self) -> str:
        return f"The response must be at most {self._limit} sentences."

    async def _stream_validate(
        self, chunk: str, *, backend: Backend, ctx: Context
    ) -> PartialValidationResult:
        self._count += 1
        if self._count > self._limit:
            return PartialValidationResult("fail", reason="Too many sentences")
        return PartialValidationResult("unknown")

    async def validate(
        self, backend: Backend, ctx: Context, *, format=None, model_options=None
    ) -> ValidationResult:
        return ValidationResult(result=True)


async def main() -> None:
    from mellea.stdlib.session import start_session

    m = start_session()
    action = Instruction("Write a two-sentence summary of the water cycle.")
    req = MaxSentencesReq(limit=3)

    async with await stream(
        action, m.backend, m.ctx, requirements=[req], chunking="sentence"
    ) as streamer:
        async for chunk in streamer:
            print(chunk)

    # Terminal state on the Streamer, after the loop.
    print(f"Completed normally: {streamer.completed_normally}")
    for _req, result in streamer.streaming_failures:
        print(f"Streaming failure: {result.reason}")


asyncio.run(main())
```

### Consuming events with `EventStreamer`

The `Streamer` above yields validated chunks. To consume the run's typed events
instead, pass `as_events=True` to `stream()`: it returns an `EventStreamer` you
iterate the same way, each step yielding an event rather than a chunk:

```python
from mellea.stdlib.streaming import (
    ChunkEvent,
    CompletedEvent,
    FullValidationEvent,
    QuickCheckEvent,
    StreamingDoneEvent,
)

async with await stream(
    action, m.backend, m.ctx, requirements=[req], chunking="sentence", as_events=True
) as streamer:
    async for event in streamer:
        match event:
            case ChunkEvent():
                print(f"  chunk[{event.chunk_index}]: {event.text!r}")
            case QuickCheckEvent(passed=False):
                print(f"  quick-check[{event.chunk_index}]: FAIL")
            case StreamingDoneEvent():
                print(f"  done — {len(event.full_text)} chars")
            case FullValidationEvent():
                print(f"  final validation: {'pass' if event.passed else 'fail'}")
            case CompletedEvent():
                print(f"  completed — success={event.success}")
            case _:
                pass  # ErrorEvent

print(f"Completed normally: {streamer.completed_normally}")
```

See the [Streaming Validation tutorial](../tutorials/06-streaming-validation.md)
for a full walkthrough.

### The `_stream_validate` tri-state

Each call to `_stream_validate` returns a `PartialValidationResult` with one of
three values:

| Value | Meaning |
| ----- | ------- |
| `"unknown"` | No conclusion yet — wait for the full output before judging. |
| `"pass"` | This chunk is valid so far (informational; does not skip final `validate()`). |
| `"fail"` | Invalid — cancel the stream immediately and record a streaming failure. |

After a natural stream end, `validate()` is called on every non-`"fail"`
requirement (both `"pass"` and `"unknown"`). This means `"pass"` from
`_stream_validate` does **not** replace the final `validate()` call.

### Requirement chunking

The `chunking=` on `stream()` sets what the *consumer* receives. A requirement
declares the granularity its own check needs, independent of the stream's: a
sentence-level check knows it wants sentences, so it sets `chunking="sentence"` in
its constructor and validates sentence by sentence regardless of the stream's
chunking. See
[`docs/examples/streaming/per_requirement_chunking.py`](https://github.com/generative-computing/mellea/blob/main/docs/examples/streaming/per_requirement_chunking.py)
for two requirements validating one stream at different granularities.

> **See also:** [The Requirements System — Streaming validation](../concepts/requirements-system#streaming-validation)

---

**See also:** [Tutorial 02: Streaming and Async](../tutorials/streaming-and-async) | [act() and aact()](../how-to/act-and-aact)
