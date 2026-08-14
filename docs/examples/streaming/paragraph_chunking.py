# pytest: ollama, e2e

"""Streaming generation with per-paragraph validation using ParagraphChunking.

Demonstrates:
- Using the `"paragraph"` chunking alias for coarse-grained, structure-aware
  validation
- A paragraph-length gate that cancels generation if any paragraph is too long
- How ParagraphChunking withholds text until a blank line (`\\n\\n`) is seen,
  then emits the entire paragraph as a single chunk
- The latency trade-off vs. SentenceChunking: fewer, larger chunks mean lower
  validation overhead but later detection

ParagraphChunking splits on two or more consecutive newlines.  Unlike
SentenceChunking, it waits for the model to produce a blank line before
emitting anything — so if the model writes everything as one long paragraph
the stream completes before any chunk is emitted.  Use ParagraphChunking when
the validation logic requires full paragraph context: topic coherence,
heading structure, citation presence, or overall paragraph quality.
"""

import asyncio
from typing import Any

from mellea.core.backend import Backend
from mellea.core.base import Context
from mellea.core.requirement import (
    PartialValidationResult,
    Requirement,
    ValidationResult,
)
from mellea.plugins import hook, register
from mellea.stdlib.components import Instruction
from mellea.stdlib.streaming import (
    ChunkEvent,
    CompletedEvent,
    FullValidationEvent,
    QuickCheckEvent,
    StreamingDoneEvent,
    stream,
)

_MAX_PARAGRAPH_WORDS = 60


class ParagraphLengthReq(Requirement):
    """Fails the stream if any paragraph exceeds a word-count limit.

    Each `stream_validate` call receives one complete paragraph (from
    `ParagraphChunking`).  The validator counts
    words and immediately fails the stream if the paragraph is too long.  This
    lets you enforce a maximum paragraph length at generation time rather than
    post-processing.
    """

    def __init__(self, max_words: int) -> None:
        super().__init__()
        self._max_words = max_words
        self._para_index = 0

    def format_for_llm(self) -> str:
        return f"Each paragraph must contain at most {self._max_words} words."

    async def stream_validate(
        self, chunk: str, *, backend: Backend, ctx: Context
    ) -> PartialValidationResult:
        self._para_index += 1
        word_count = len(chunk.split())
        if word_count > self._max_words:
            return PartialValidationResult(
                "fail",
                reason=(
                    f"Paragraph {self._para_index} has {word_count} words "
                    f"(limit: {self._max_words})"
                ),
            )
        return PartialValidationResult("unknown")

    async def validate(
        self,
        backend: Backend,
        ctx: Context,
        *,
        format: type | None = None,
        model_options: dict | None = None,
    ) -> ValidationResult:
        return ValidationResult(result=True)


async def main() -> None:
    from mellea.stdlib.session import start_session

    m = start_session()
    backend = m.backend
    ctx = m.ctx

    action = Instruction(
        "Write a two-paragraph explanation of how the internet works. "
        "Separate the two paragraphs with a blank line. "
        f"Keep each paragraph to at most {_MAX_PARAGRAPH_WORDS} words."
    )
    req = ParagraphLengthReq(max_words=_MAX_PARAGRAPH_WORDS)

    @hook("streaming_event")
    async def print_events(payload: Any, ctx: Any) -> None:
        event = payload.event
        match event:
            case ChunkEvent():
                word_count = len(event.text.split())
                preview = event.text[:80].replace("\n", "↵")
                print(
                    f"  PARAGRAPH[{event.chunk_index}]: {word_count} words — "
                    f"{preview!r}..."
                )
            case QuickCheckEvent(passed=False):
                print(
                    f"  QUICK_CHECK[para {event.chunk_index}]: FAIL — "
                    f"{event.results[0].reason if event.results else 'unknown'}"
                )
            case QuickCheckEvent():
                print(f"  QUICK_CHECK[para {event.chunk_index}]: pass")
            case StreamingDoneEvent():
                print(f"  STREAMING_DONE: {len(event.full_text)} chars accumulated")
            case FullValidationEvent():
                print(f"  FULL_VALIDATION: {'PASS' if event.passed else 'FAIL'}")
            case CompletedEvent():
                print(f"  COMPLETED: success={event.success}")
            case _:
                pass

    register(print_events)

    print("Stream events as they arrive (one per paragraph):")
    async with await stream(
        action, backend, ctx, requirements=[req], chunking="paragraph"
    ) as streamer:
        # Draining the stream fires the events; the hook does the printing.
        async for _paragraph in streamer:
            pass

    print(f"\nCompleted normally: {streamer.completed_normally}")
    if streamer.streaming_failures:
        for _req, pvr in streamer.streaming_failures:
            print(f"Streaming failure: {pvr.reason}")
    else:
        print(f"Full text:\n{streamer.full_text}")


asyncio.run(main())
