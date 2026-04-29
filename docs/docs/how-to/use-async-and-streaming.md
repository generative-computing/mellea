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

`stream_with_chunking()` adds per-chunk validation to a streaming generation.
It splits the accumulated text into semantic units (sentences, words, or
paragraphs), calls `stream_validate()` on each chunk in parallel, and can
exit early if any requirement returns `"fail"` — preventing the consumer from
seeing invalid content mid-stream.

The primary way to observe a `stream_with_chunking()` run is via typed
`StreamEvent` objects from `result.events()`:

```python
# Requires: mellea
# Returns: None
import asyncio

from mellea.core.backend import Backend
from mellea.core.base import Context
from mellea.core.requirement import PartialValidationResult, Requirement, ValidationResult
from mellea.stdlib.components import Instruction
from mellea.stdlib.streaming import (
    ChunkEvent,
    CompletedEvent,
    FullValidationEvent,
    QuickCheckEvent,
    StreamingDoneEvent,
    stream_with_chunking,
)


class MaxSentencesReq(Requirement):
    """Fails if the model generates more than *limit* sentences."""

    def __init__(self, limit: int) -> None:
        super().__init__()
        self._limit = limit
        self._count = 0

    def format_for_llm(self) -> str:
        return f"The response must be at most {self._limit} sentences."

    async def stream_validate(
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

    result = await stream_with_chunking(
        action, m.backend, m.ctx, quick_check_requirements=[req], chunking="sentence"
    )

    async for event in result.events():
        match event:
            case ChunkEvent():
                print(f"  chunk[{event.chunk_index}]: {event.text!r}")
            case QuickCheckEvent(passed=False):
                print(f"  FAIL at chunk {event.chunk_index}: {event.results}")
            case StreamingDoneEvent():
                print(f"  stream done — {len(event.full_text)} chars")
            case FullValidationEvent():
                print(f"  final: {'pass' if event.passed else 'fail'}")
            case CompletedEvent():
                print(f"  completed — success={event.success}")
            case _:
                pass  # ErrorEvent and other future types

    await result.acomplete()
    print(f"completed={result.completed}, failures={len(result.streaming_failures)}")


asyncio.run(main())
```

If you only need the raw validated text without event metadata, use
`result.astream()` instead:

```python
result = await stream_with_chunking(
    action, m.backend, m.ctx, quick_check_requirements=[req], chunking="sentence"
)
async for chunk in result.astream():
    print(chunk)
await result.acomplete()
```

Both `astream()` (raw chunks) and `events()` are available on the same result
object. They use independent queues, so you can run them concurrently with
`asyncio.gather`. Both are **single-consumer** — a second iteration on either
will block indefinitely.

### The `stream_validate` tri-state

Each call to `stream_validate` returns a `PartialValidationResult` with one of
three values:

| Value | Meaning |
| ----- | ------- |
| `"unknown"` | No conclusion yet — wait for the full output before judging. |
| `"pass"` | This chunk is valid so far (informational; does not skip final `validate()`). |
| `"fail"` | Invalid — cancel the stream immediately and record a streaming failure. |

After a natural stream end, `validate()` is called on every non-`"fail"`
requirement (both `"pass"` and `"unknown"`). This means `"pass"` from
`stream_validate` does **not** replace the final `validate()` call.

> **See also:** [The Requirements System — Streaming validation](../concepts/requirements-system#streaming-validation)

---

**See also:** [Tutorial 02: Streaming and Async](../tutorials/02-streaming-and-async) | [act() and aact()](../how-to/act-and-aact)
