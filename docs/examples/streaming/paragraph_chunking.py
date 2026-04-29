# pytest: ollama, e2e

"""Streaming generation with per-paragraph validation using ParagraphChunker.

Demonstrates:
- Using the ``"paragraph"`` chunking alias for coarse-grained, structure-aware
  validation
- A paragraph-length gate that cancels generation if any paragraph is too long
- How ParagraphChunker withholds text until a blank line (``\\n\\n``) is seen,
  then emits the entire paragraph as a single chunk
- The latency trade-off vs. SentenceChunker: fewer, larger chunks mean lower
  validation overhead but later detection

ParagraphChunker splits on two or more consecutive newlines.  Unlike
SentenceChunker, it waits for the model to produce a blank line before
emitting anything — so if the model writes everything as one long paragraph
the stream completes before any chunk is emitted.  Use ParagraphChunker when
the validation logic requires full paragraph context: topic coherence,
heading structure, citation presence, or overall paragraph quality.
"""

import asyncio

from mellea.core.backend import Backend
from mellea.core.base import Context
from mellea.core.requirement import (
    PartialValidationResult,
    Requirement,
    ValidationResult,
)
from mellea.stdlib.components import Instruction
from mellea.stdlib.streaming import (
    ChunkEvent,
    CompletedEvent,
    FullValidationEvent,
    QuickCheckEvent,
    StreamingDoneEvent,
    stream_with_chunking,
)

_MAX_PARAGRAPH_WORDS = 60


class ParagraphLengthReq(Requirement):
    """Fails the stream if any paragraph exceeds a word-count limit.

    Each ``stream_validate`` call receives one complete paragraph (from
    :class:`~mellea.stdlib.chunking.ParagraphChunker`).  The validator counts
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

    result = await stream_with_chunking(
        action, backend, ctx, quick_check_requirements=[req], chunking="paragraph"
    )

    print("Streaming events as they arrive (one ChunkEvent per paragraph):")
    async for event in result.events():
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

    await result.acomplete()

    print(f"\nCompleted normally: {result.completed}")
    if result.streaming_failures:
        for _req, pvr in result.streaming_failures:
            print(f"Streaming failure: {pvr.reason}")
    else:
        print(f"Full text:\n{result.full_text}")


asyncio.run(main())
