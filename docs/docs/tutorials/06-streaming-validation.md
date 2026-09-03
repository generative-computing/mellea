---
title: "Tutorial: Streaming Validation"
description: "Validate LLM output chunk by chunk as it streams — detect policy violations the moment they appear and cancel generation before invalid content reaches your users."
# diataxis: tutorial
---

Post-generation validation waits until the model has finished writing before
checking the output. That is fine for short responses, but wastes time and
compute when a violation appears in the first sentence of a ten-paragraph
reply. Streaming validation moves the check into the generation loop: each
chunk is validated as soon as it arrives, and generation is cancelled the
moment a requirement fails.

By the end you will have covered:

- `stream()` — the streaming validation entry point
- Consuming validated chunks with `async for` inside `async with`
- Early-exit cancellation and reading `streaming_failures`
- Choosing between `"word"`, `"sentence"`, and `"paragraph"` chunking
- Observing the typed event vocabulary (`ChunkEvent`, `QuickCheckEvent`, …)
  by iterating `stream(as_events=True)`
- Subclassing `ChunkingStrategy` to define a custom split boundary

**Prerequisites:** [Tutorial 02](./streaming-and-async) (async and streaming),
[Tutorial 04](./making-agents-reliable) (requirements and validation),
`pip install mellea`, Ollama running locally with `granite4.1:3b` downloaded.

---

## Step 1: Your first streaming validation call

`stream()` starts a streaming generation and returns a `Streamer`. Consume it
with `async for` inside `async with`: each iteration yields the next chunk,
already validated against every requirement, and the `async with` block cancels
the generation on every exit path — including an early `break` or an exception —
so an abandoned stream never keeps running in the background.

```python
# Requires: mellea
# Returns: None
import asyncio
import re

from mellea.core.backend import Backend
from mellea.core.base import Context
from mellea.core.requirement import PartialValidationResult, Requirement, ValidationResult
from mellea.stdlib.components import Instruction
from mellea.stdlib.streaming import stream

_SENTENCE_END = re.compile(r"[.!?]+")


class MaxSentencesReq(Requirement):
    """Fails the stream if the model writes more sentences than *limit*."""

    def __init__(self, limit: int) -> None:
        super().__init__()
        self._limit = limit
        self._count = 0

    def format_for_llm(self) -> str:
        return f"The response must be at most {self._limit} sentences."

    async def stream_validate(
        self, chunk: str, *, backend: Backend, ctx: Context
    ) -> PartialValidationResult:
        self._count += len(_SENTENCE_END.findall(chunk))
        if self._count > self._limit:
            return PartialValidationResult(
                "fail", reason=f"Exceeded {self._limit}-sentence limit"
            )
        return PartialValidationResult("unknown")

    async def validate(
        self, backend: Backend, ctx: Context, *, format=None, model_options=None
    ) -> ValidationResult:
        return ValidationResult(result=self._count <= self._limit)


async def main() -> None:
    from mellea.stdlib.session import start_session

    m = start_session()

    async with await stream(
        Instruction("Write a two-sentence summary of how photosynthesis works."),
        m.backend,
        m.ctx,
        requirements=[MaxSentencesReq(limit=3)],
        chunking="sentence",
    ) as streamer:
        async for chunk in streamer:
            print(f"  chunk: {chunk!r}")

    print(f"\nCompleted normally: {streamer.completed_normally}")
    print(f"Full text: {streamer.full_text!r}")


asyncio.run(main())
```

```text Sample output
  chunk: 'Photosynthesis is the process by which plants use sunlight, water, and carbon dioxide to produce glucose and oxygen.'
  chunk: 'This reaction takes place in the chloroplasts and is essential to nearly all life on Earth.'

Completed normally: True
Full text: 'Photosynthesis is the process by which plants use sunlight, water, and carbon dioxide to produce glucose and oxygen. This reaction takes place in the chloroplasts and is essential to nearly all life on Earth.'
```

> **Note:** LLM output is non-deterministic. Your result will vary in wording.

Two things to notice:

- `stream()` is awaited to obtain the `Streamer`, but generation begins eagerly —
  it is already running by the time you enter the `async for` loop.
- Terminal state is read from the `Streamer` **after** the loop: `failed_early`,
  `full_text`, `streaming_failures`, and `final_validations`.

---

## Step 2: Early exit on failure

When `stream_validate()` returns `"fail"`, the backend generation is cancelled
immediately and the loop ends. No further chunks are delivered, `failed_early`
becomes `True`, and the failure is recorded in `streamer.streaming_failures`.

Lower the sentence limit so the model is likely to exceed it:

```python
# Requires: mellea
# Returns: None
import asyncio
import re

from mellea.core.backend import Backend
from mellea.core.base import Context
from mellea.core.requirement import PartialValidationResult, Requirement, ValidationResult
from mellea.stdlib.components import Instruction
from mellea.stdlib.streaming import stream

_SENTENCE_END = re.compile(r"[.!?]+")


class MaxSentencesReq(Requirement):
    def __init__(self, limit: int) -> None:
        super().__init__()
        self._limit = limit
        self._count = 0

    def format_for_llm(self) -> str:
        return f"The response must be at most {self._limit} sentences."

    async def stream_validate(
        self, chunk: str, *, backend: Backend, ctx: Context
    ) -> PartialValidationResult:
        self._count += len(_SENTENCE_END.findall(chunk))
        if self._count > self._limit:
            return PartialValidationResult(
                "fail", reason=f"Exceeded {self._limit}-sentence limit"
            )
        return PartialValidationResult("unknown")

    async def validate(
        self, backend: Backend, ctx: Context, *, format=None, model_options=None
    ) -> ValidationResult:
        return ValidationResult(result=self._count <= self._limit)


async def main() -> None:
    from mellea.stdlib.session import start_session

    m = start_session()

    # Ask for five sentences but cap the requirement at two.
    # The stream should be cancelled after the third sentence arrives.
    async with await stream(
        Instruction("Write five sentences about the history of the internet."),
        m.backend,
        m.ctx,
        requirements=[MaxSentencesReq(limit=2)],
        chunking="sentence",
    ) as streamer:
        async for chunk in streamer:
            print(f"  chunk: {chunk[:60]!r}...")

    if streamer.streaming_failures:
        _req, pvr = streamer.streaming_failures[0]
        print(f"\nStreaming failure: {pvr.reason}")
        print(f"Text at cancellation:\n{streamer.full_text!r}")
    else:
        print(f"\nFull text: {streamer.full_text!r}")


asyncio.run(main())
```

```text Sample output
  chunk: 'The internet began as ARPANET, a U.S. Defense Department pr'...
  chunk: 'In the 1980s, the network expanded beyond government use an'...

Streaming failure: Exceeded 2-sentence limit
Text at cancellation:
'The internet began as ARPANET, a U.S. Defense Department project in the late 1960s. In the 1980s, the network expanded beyond government use and began connecting universities and research centres.'
```

> **Note:** Whether the stream is cancelled depends on whether the model
> exceeds the limit. If the model happens to comply, `streaming_failures` will
> be empty and `failed_early` will be `False`.

`streamer.full_text` always contains the text accumulated up to the point where
generation stopped — useful for debugging what the model produced before the
requirement failed.

---

## Step 3: Choosing a chunking strategy

The built-in strategies cover a coarse-to-fine spectrum:

| Alias | Splits on | Good for |
| --- | --- | --- |
| `"word"` | Whitespace | Token-local checks: forbidden words, numeric limits |
| `"sentence"` | `.`, `!`, `?` followed by whitespace | Grammar, coherence, per-sentence content rules |
| `"paragraph"` | Two or more consecutive newlines | Topic coherence, citation presence, heading structure |

The trade-off is **latency vs context**. Word chunking fires after every word —
maximum reaction speed, but each chunk carries only a single word. Paragraph
chunking waits for blank lines — full paragraph context for the validator, but
detection is later and may happen after the model has produced a large amount
of invalid content.

To see the granularity difference concretely, switch to word chunking and count
how many chunks arrive compared to Step 1's two sentences:

```python
# Requires: mellea
# Returns: None
import asyncio

from mellea.core.backend import Backend
from mellea.core.base import Context
from mellea.core.requirement import PartialValidationResult, Requirement, ValidationResult
from mellea.stdlib.components import Instruction
from mellea.stdlib.streaming import stream

_FORBIDDEN = {"deprecated", "legacy", "obsolete"}


class ForbiddenWordReq(Requirement):
    """Cancels the stream the moment any forbidden word appears."""

    def format_for_llm(self) -> str:
        return f"Do not use any of the following words: {', '.join(sorted(_FORBIDDEN))}."

    async def stream_validate(
        self, chunk: str, *, backend: Backend, ctx: Context
    ) -> PartialValidationResult:
        word = chunk.strip().lower().strip(".,!?;:\"'")
        if word in _FORBIDDEN:
            return PartialValidationResult("fail", reason=f"Forbidden word: {chunk.strip()!r}")
        return PartialValidationResult("unknown")

    async def validate(
        self, backend: Backend, ctx: Context, *, format=None, model_options=None
    ) -> ValidationResult:
        return ValidationResult(result=True)


async def main() -> None:
    from mellea.stdlib.session import start_session

    m = start_session()

    word_count = 0
    async with await stream(
        Instruction(
            "Describe three advantages of cloud-native development in two sentences."
        ),
        m.backend,
        m.ctx,
        requirements=[ForbiddenWordReq()],
        chunking="word",
    ) as streamer:
        async for word in streamer:
            word_count += 1
            # Print every fifth word to show how many chunks arrive.
            if word_count % 5 == 1:
                print(f"  word {word_count:>3}: {word!r}")

    if streamer.streaming_failures:
        print(f"Failure: {streamer.streaming_failures[0][1].reason}")
    else:
        print(f"{word_count} word chunks total")


asyncio.run(main())
```

```text Sample output
  word   1: 'Cloud-native'
  word   6: 'resilient'
  word  11: 'and'
  word  16: 'allows'
  word  21: 'horizontally,'
  word  26: 'costs,'
  word  31: 'deployments,'
  word  36: 'services.'
38 word chunks total
```

> **Note:** LLM output is non-deterministic. Your result will vary in wording.

The same two-sentence response that produced **2** chunks with sentence chunking
now produces **38**. The validator fires on every word — maximum reaction speed
at the cost of per-chunk context.

If a forbidden word appears, the stream stops at that word and no further chunks
are delivered. To see early exit in action, change `_FORBIDDEN` to include a
common English word like `"and"` or `"the"`.

---

## Step 4: Observing the event lifecycle

The `async for` loop above yields validated chunks. To observe the full lifecycle
instead — per-chunk validation results, stream completion, final validation,
errors — pass `as_events=True`: `stream()` then returns an `EventStreamer` that
yields one typed `StreamEvent` per lifecycle moment in place of chunks.

```python
# Requires: mellea
# Returns: None
import asyncio
import re

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
    stream,
)

_SENTENCE_END = re.compile(r"[.!?]+")


class MaxSentencesReq(Requirement):
    def __init__(self, limit: int) -> None:
        super().__init__()
        self._limit = limit
        self._count = 0

    def format_for_llm(self) -> str:
        return f"The response must be at most {self._limit} sentences."

    async def stream_validate(
        self, chunk: str, *, backend: Backend, ctx: Context
    ) -> PartialValidationResult:
        self._count += len(_SENTENCE_END.findall(chunk))
        if self._count > self._limit:
            return PartialValidationResult(
                "fail", reason=f"Exceeded {self._limit}-sentence limit"
            )
        return PartialValidationResult("unknown")

    async def validate(
        self, backend: Backend, ctx: Context, *, format=None, model_options=None
    ) -> ValidationResult:
        return ValidationResult(result=self._count <= self._limit)


async def main() -> None:
    from mellea.stdlib.session import start_session

    m = start_session()

    async with await stream(
        Instruction("Write a two-sentence summary of the water cycle."),
        m.backend,
        m.ctx,
        requirements=[MaxSentencesReq(limit=3)],
        chunking="sentence",
        as_events=True,
    ) as streamer:
        async for event in streamer:
            match event:
                case ChunkEvent():
                    print(f"  chunk[{event.chunk_index}]: {event.text!r}")
                case QuickCheckEvent(passed=False):
                    print(f"  FAIL at chunk {event.chunk_index}: {event.results[0].reason}")
                case StreamingDoneEvent():
                    print(f"  stream done — {len(event.full_text)} chars")
                case FullValidationEvent():
                    print(f"  final validation: {'pass' if event.passed else 'fail'}")
                case CompletedEvent():
                    print(f"  completed — success={event.success}")
                case _:
                    pass


asyncio.run(main())
```

```text Sample output
  chunk[0]: 'Water evaporates from oceans and lakes, rises into the atmosphere, and condenses into clouds.'
  chunk[1]: 'Precipitation then falls back to Earth as rain or snow, replenishing rivers, lakes, and groundwater.'
  stream done — 195 chars
  final validation: pass
  completed — success=True
```

> **Note:** LLM output is non-deterministic. Your result will vary in wording.

The event vocabulary:

- `ChunkEvent` — a validated chunk was delivered to the consumer.
- `QuickCheckEvent` — the result of validating one chunk; `passed=False` marks
  the requirement failure that ends the stream.
- `StreamingDoneEvent` — the token stream finished (natural completion only).
- `FullValidationEvent` — the final `validate()` pass over the whole output.
- `CompletedEvent` — the stream exited; always the last event, on every path.
- `ErrorEvent` — an exception occurred mid-stream.

Because the hook is global, a single plugin can observe many concurrent streams
at once — use `payload.streaming_id` to tell their events apart. See
[`docs/examples/streaming/multi_stream_events.py`](https://github.com/generative-computing/mellea/blob/main/docs/examples/streaming/multi_stream_events.py)
for a multi-stream consumer.

---

## Step 5: A custom chunking strategy

The built-in strategies cover the most common boundaries. For structured output
— numbered lists, code blocks, CSV rows — you can subclass `ChunkingStrategy`
and define your own split boundary.

Two methods to implement:

- **`split(text)`** — return all complete chunks in `text`, withholding any
  trailing fragment. Must be stateless and idempotent.
- **`flush(text)`** — called once at natural end of stream. Release the withheld
  trailing fragment, or return `[]` to discard it.

Here is a `LineChunking` strategy that splits on single newlines — natural for
numbered list output where each line is one item:

```python
# Requires: mellea
# Returns: None
import asyncio
import re

from mellea.core.backend import Backend
from mellea.core.base import Context
from mellea.core.requirement import PartialValidationResult, Requirement, ValidationResult
from mellea.stdlib.chunking import ChunkingStrategy
from mellea.stdlib.components import Instruction
from mellea.stdlib.streaming import stream

_NUMBERED_LINE = re.compile(r"^\s*\d+[\.\)]\s")


class LineChunking(ChunkingStrategy):
    """Emits one complete line per chunk, splitting on single newlines."""

    def split(self, text: str) -> list[str]:
        if "\n" not in text:
            return []
        last_nl = text.rfind("\n")
        return [line for line in text[:last_nl].split("\n") if line.strip()]

    def flush(self, text: str) -> list[str]:
        if not text:
            return []
        last_nl = text.rfind("\n")
        trailing = (text if last_nl == -1 else text[last_nl + 1 :]).strip()
        return [trailing] if trailing else []


class NumberedLineReq(Requirement):
    """Cancels the stream if any line does not begin with a number."""

    def format_for_llm(self) -> str:
        return "Every line must begin with a number followed by a period (e.g. '1. ')."

    async def stream_validate(
        self, chunk: str, *, backend: Backend, ctx: Context
    ) -> PartialValidationResult:
        if not _NUMBERED_LINE.match(chunk):
            return PartialValidationResult(
                "fail", reason=f"Line does not start with a number: {chunk.strip()!r}"
            )
        return PartialValidationResult("unknown")

    async def validate(
        self, backend: Backend, ctx: Context, *, format=None, model_options=None
    ) -> ValidationResult:
        # All format checking happens during streaming. Lines that reach validate()
        # are guaranteed to have passed stream_validate() already.
        return ValidationResult(result=True)


async def main() -> None:
    from mellea.stdlib.session import start_session

    m = start_session()

    async with await stream(
        Instruction(
            "List five world capitals, one per line, numbered 1 through 5. "
            "Use the format: '1. City'. Output only the numbered list, nothing else."
        ),
        m.backend,
        m.ctx,
        requirements=[NumberedLineReq()],
        chunking=LineChunking(),
    ) as streamer:
        async for line in streamer:
            print(f"  line: {line.strip()!r}")

    if streamer.streaming_failures:
        print(f"FAIL: {streamer.streaming_failures[0][1].reason}")
    else:
        print("Completed normally")


asyncio.run(main())
```

```text Sample output
  line: '1. London'
  line: '2. Paris'
  line: '3. Tokyo'
  line: '4. Ottawa'
  line: '5. Canberra'
Completed normally
```

> **Note:** LLM output is non-deterministic. Your result will vary in wording.

`validate()` on `NumberedLineReq` always returns `True` because all format
checking happens during streaming. If any line fails, the stream is cancelled
before reaching `validate()`. Lines that do reach it have already passed
`stream_validate()`. This pattern — enforce in `stream_validate`, pass in
`validate` — is common for requirements whose invariant is a property of
individual chunks rather than the full output.

Pass a `ChunkingStrategy` **instance** (not a string alias) to use a custom
chunker. The built-in strategies (`WordChunking`, `SentenceChunking`,
`ParagraphChunking`) are also available as instances if you need to pass one
explicitly or subclass to override `flush()`.

> **See also:** [`docs/examples/streaming/custom_chunking.py`](https://github.com/generative-computing/mellea/blob/main/docs/examples/streaming/custom_chunking.py)
> for an annotated version of this pattern with a more detailed `split()`/`flush()`
> contract walkthrough.

---

## What you built

| Concept | What it gives you |
| --- | --- |
| `stream()` + `requirements=` | Per-chunk validation with automatic early exit |
| `async for chunk in streamer` | Validated chunks as they arrive, inside `async with` for safe cleanup |
| `streamer.failed_early` / `streamer.streaming_failures` | Detect and inspect a mid-stream requirement failure |
| `stream(as_events=True)` | Typed event stream (an `EventStreamer`) — observe every chunk, validation result, and lifecycle signal |
| `"word"` / `"sentence"` / `"paragraph"` | Built-in chunking strategies trading reaction speed for context |
| `ChunkingStrategy` subclass | Custom split boundaries for structured output (lists, code, CSV) |

---

> **See also:**
> [How-to: Streaming with per-chunk validation](../how-to/use-async-and-streaming#streaming-with-per-chunk-validation) |
> [Concepts: The Requirements System — Streaming validation](../concepts/requirements-system#streaming-validation) |
> [Examples: streaming/](https://github.com/generative-computing/mellea/tree/main/docs/examples/streaming)
