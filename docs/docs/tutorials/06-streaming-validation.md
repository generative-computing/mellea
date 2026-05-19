---
canonical: "https://docs.mellea.ai/tutorials/06-streaming-validation"
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

- `stream_with_chunking()` — the streaming validation entry point
- The typed event vocabulary (`ChunkEvent`, `QuickCheckEvent`, …) from `result.events()`
- Early-exit cancellation and reading `streaming_failures`
- Choosing between `"word"`, `"sentence"`, and `"paragraph"` chunking
- Subclassing `ChunkingStrategy` to define a custom split boundary
- `result.astream()` for consumers that only need the validated chunks

**Prerequisites:** [Tutorial 02](./02-streaming-and-async) (async and streaming),
[Tutorial 04](./04-making-agents-reliable) (requirements and validation),
`pip install mellea`, Ollama running locally with `granite4.1:3b` downloaded.

---

## Step 1: Your first streaming validation call

`stream_with_chunking()` returns a `StreamChunkingResult` immediately. The
orchestrator runs in the background, splitting accumulated text into chunks and
calling `stream_validate()` on each one. Consume events with `result.events()`,
then call `result.acomplete()` to wait for the orchestrator to finish and raise
any exception it stored.

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
    stream_with_chunking,
)

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

    result = await stream_with_chunking(
        Instruction("Write a two-sentence summary of how photosynthesis works."),
        m.backend,
        m.ctx,
        requirements=[MaxSentencesReq(limit=3)],
        chunking="sentence",
    )

    async for event in result.events():
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

    await result.acomplete()
    print(f"\nFull text: {result.full_text!r}")


asyncio.run(main())
```

```text Sample output
  chunk[0]: 'Photosynthesis is the process by which plants use sunlight, water, and carbon dioxide to produce glucose and oxygen.'
  chunk[1]: 'This reaction takes place in the chloroplasts and is essential to nearly all life on Earth.'
  stream done — 222 chars
  final validation: pass
  completed — success=True

Full text: 'Photosynthesis is the process by which plants use sunlight, water, and carbon dioxide to produce glucose and oxygen. This reaction takes place in the chloroplasts and is essential to nearly all life on Earth.'
```

> **Note:** LLM output is non-deterministic. Your result will vary in wording.

Three things to notice:

- `stream_with_chunking()` is called with `await` but returns immediately — the
  orchestrator runs as a background task.
- `result.events()` is an async iterator that yields one event per semantic
  unit. The loop ends when the `CompletedEvent` is delivered.
- `result.acomplete()` must be called after the event loop drains to propagate
  any orchestrator exception and to ensure the background task has fully settled.

---

## Step 2: Early exit on failure

When `stream_validate()` returns `"fail"`, the orchestrator cancels the backend
immediately and stops the stream. No further chunks are delivered, and the
failure is recorded in `result.streaming_failures`.

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
from mellea.stdlib.streaming import ChunkEvent, CompletedEvent, QuickCheckEvent, stream_with_chunking

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
    result = await stream_with_chunking(
        Instruction("Write five sentences about the history of the internet."),
        m.backend,
        m.ctx,
        requirements=[MaxSentencesReq(limit=2)],
        chunking="sentence",
    )

    async for event in result.events():
        match event:
            case ChunkEvent():
                print(f"  chunk[{event.chunk_index}]: {event.text[:60]!r}...")
            case QuickCheckEvent(passed=False):
                print(f"  CANCELLED at chunk {event.chunk_index}")
            case CompletedEvent():
                print(f"  completed — success={event.success}")
            case _:
                pass

    await result.acomplete()

    if result.streaming_failures:
        req, pvr = result.streaming_failures[0]
        print(f"\nStreaming failure: {pvr.reason}")
        print(f"Text at cancellation:\n{result.full_text!r}")
    else:
        print(f"\nFull text: {result.full_text!r}")


asyncio.run(main())
```

```text Sample output
  chunk[0]: 'The internet began as ARPANET, a U.S. Defense Department pr'...
  chunk[1]: 'In the 1980s, the network expanded beyond government use and'...
  chunk[2]: 'Tim Berners-Lee invented the World Wide Web in 1989, transfo'...
  CANCELLED at chunk 2
  completed — success=False

Streaming failure: Exceeded 2-sentence limit
Text at cancellation:
'The internet began as ARPANET, a U.S. Defense Department project in the late 1960s. In the 1980s, the network expanded beyond government use and began connecting universities and research centres. Tim Berners-Lee invented the World Wide Web in 1989...'
```

> **Note:** Whether the stream is cancelled depends on whether the model
> exceeds the limit. If the model happens to comply, `streaming_failures` will
> be empty and `result.completed` will be `True`.

`result.full_text` always contains the text accumulated up to the point where
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

The trade-off is **latency vs context**. Word chunking fires after every token —
maximum reaction speed, but each chunk carries only a single word. Paragraph
chunking waits for blank lines — full paragraph context for the validator, but
detection is later and may happen after the model has produced a large amount
of invalid content.

Switch to word chunking for a forbidden-word check:

```python
# Requires: mellea
# Returns: None
import asyncio

from mellea.core.backend import Backend
from mellea.core.base import Context
from mellea.core.requirement import PartialValidationResult, Requirement, ValidationResult
from mellea.stdlib.components import Instruction
from mellea.stdlib.streaming import ChunkEvent, CompletedEvent, QuickCheckEvent, stream_with_chunking

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

    result = await stream_with_chunking(
        Instruction(
            "Describe three advantages of cloud-native development in two sentences."
        ),
        m.backend,
        m.ctx,
        requirements=[ForbiddenWordReq()],
        chunking="word",
    )

    word_count = 0
    async for event in result.events():
        match event:
            case ChunkEvent():
                word_count += 1
            case QuickCheckEvent(passed=False):
                print(f"  CANCELLED after word {event.chunk_index}: {event.results[0].reason}")
            case CompletedEvent():
                status = "cancelled" if not event.success else "ok"
                print(f"  {status} after {word_count} words")
            case _:
                pass

    await result.acomplete()

    if result.streaming_failures:
        print(f"Failure: {result.streaming_failures[0][1].reason}")
    else:
        print(f"OK — full text: {result.full_text!r}")


asyncio.run(main())
```

```text Sample output
  ok after 38 words
OK — full text: 'Cloud-native development enables scalable, resilient applications that can be deployed ...'
```

> **Note:** Your output may not trigger the forbidden words at all — the model
> tends to comply when they appear in `format_for_llm()`. The example shows what
> the output looks like on a clean run. Replace `_FORBIDDEN` with words you
> expect the model to produce to see early-exit behaviour.

---

## Step 4: A custom chunking strategy

The built-in strategies cover the most common boundaries. For structured output
— numbered lists, code blocks, CSV rows — you can subclass `ChunkingStrategy`
and define your own split boundary.

Two methods to implement:

- **`split(accumulated_text)`** — called on every new token delta. Return all
  complete chunks found so far; withhold any trailing fragment. Must be
  stateless: it receives the full accumulated text, not a delta.
- **`flush(accumulated_text)`** — called once at end of stream. Release the
  withheld trailing fragment, or return `[]` to discard it.

Here is a `LineChunker` that splits on single newlines — natural for numbered
list output where each line is one item:

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
from mellea.stdlib.streaming import ChunkEvent, CompletedEvent, QuickCheckEvent, stream_with_chunking

_NUMBERED_LINE = re.compile(r"^\s*\d+[\.\)]\s")


class LineChunker(ChunkingStrategy):
    """Emits one complete line per chunk, splitting on single newlines."""

    def split(self, accumulated_text: str) -> list[str]:
        if "\n" not in accumulated_text:
            return []
        last_nl = accumulated_text.rfind("\n")
        return [line for line in accumulated_text[:last_nl].split("\n") if line.strip()]

    def flush(self, accumulated_text: str) -> list[str]:
        if not accumulated_text:
            return []
        last_nl = accumulated_text.rfind("\n")
        trailing = (
            accumulated_text if last_nl == -1 else accumulated_text[last_nl + 1 :]
        ).strip()
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
        return ValidationResult(result=True)


async def main() -> None:
    from mellea.stdlib.session import start_session

    m = start_session()

    result = await stream_with_chunking(
        Instruction(
            "List five world capitals, one per line, numbered 1 through 5. "
            "Use the format: '1. City'. Output only the numbered list, nothing else."
        ),
        m.backend,
        m.ctx,
        requirements=[NumberedLineReq()],
        chunking=LineChunker(),
    )

    async for event in result.events():
        match event:
            case ChunkEvent():
                print(f"  line[{event.chunk_index}]: {event.text.strip()!r}")
            case QuickCheckEvent(passed=False):
                print(f"  FAIL: {event.results[0].reason}")
            case CompletedEvent():
                print(f"  completed — success={event.success}")
            case _:
                pass

    await result.acomplete()


asyncio.run(main())
```

```text Sample output
  line[0]: '1. London'
  line[1]: '2. Paris'
  line[2]: '3. Tokyo'
  line[3]: '4. Ottawa'
  line[4]: '5. Canberra'
  completed — success=True
```

Pass a `ChunkingStrategy` **instance** (not a string alias) to use a custom
chunker. Built-in chunkers are available as instances too — `LineChunker()`,
`WordChunker()`, etc. — if you want to override the default `flush()` behaviour.

---

## Step 5: Raw chunk access with `astream()`

If you only need the validated chunks and do not want event metadata, use
`result.astream()` instead of `result.events()`. It yields the text of each
validated chunk as a plain string:

```python
# Requires: mellea
# Returns: None
import asyncio
import re

from mellea.core.backend import Backend
from mellea.core.base import Context
from mellea.core.requirement import PartialValidationResult, Requirement, ValidationResult
from mellea.stdlib.components import Instruction
from mellea.stdlib.streaming import stream_with_chunking

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
            return PartialValidationResult("fail", reason=f"Exceeded {self._limit}-sentence limit")
        return PartialValidationResult("unknown")

    async def validate(
        self, backend: Backend, ctx: Context, *, format=None, model_options=None
    ) -> ValidationResult:
        return ValidationResult(result=self._count <= self._limit)


async def main() -> None:
    from mellea.stdlib.session import start_session

    m = start_session()

    result = await stream_with_chunking(
        Instruction("Write a two-sentence summary of the water cycle."),
        m.backend,
        m.ctx,
        requirements=[MaxSentencesReq(limit=3)],
        chunking="sentence",
    )

    # astream() yields only validated chunk text — no event wrapper.
    async for chunk in result.astream():
        print(chunk, end=" ", flush=True)
    print()

    await result.acomplete()
    print(f"completed={result.completed}")


asyncio.run(main())
```

```text Sample output
Water evaporates from oceans and lakes, rises into the atmosphere, and
condenses into clouds.  Precipitation falls back to Earth as rain or
snow, replenishing rivers, lakes, and groundwater.
completed=True
```

`astream()` and `events()` are independent — both are available on the same
result object and can even be consumed concurrently with `asyncio.gather`. Each
is **single-consumer**: calling either iterator a second time raises
`RuntimeError`.

---

## What you built

| Concept | What it gives you |
| --- | --- |
| `stream_with_chunking()` + `requirements=` | Per-chunk validation with automatic early exit |
| `result.events()` | Typed event stream — observe every chunk, validation result, and lifecycle signal |
| `QuickCheckEvent(passed=False)` | Detect the moment a requirement fails, mid-stream |
| `result.streaming_failures` | List of `(requirement, PartialValidationResult)` pairs for failed checks |
| `"word"` / `"sentence"` / `"paragraph"` | Built-in chunking strategies trading reaction speed for context |
| `ChunkingStrategy` subclass | Custom split boundaries for structured output (lists, code, CSV) |
| `result.astream()` | Raw validated chunks without event metadata |

---

> **See also:**
> [How-to: Streaming with per-chunk validation](../how-to/use-async-and-streaming#streaming-with-per-chunk-validation) |
> [Concepts: The Requirements System — Streaming validation](../concepts/requirements-system#streaming-validation) |
> [Examples: streaming/](../examples/index)
