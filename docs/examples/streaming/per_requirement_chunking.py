# pytest: ollama, e2e

"""Per-requirement chunking: each Requirement validates at its own granularity.

Demonstrates:
- Giving each `Requirement` its own `chunking=` strategy, independent of the stream's
- A sentence-level requirement and a word-level requirement validating one stream at
  different granularities at the same time

The stream uses its default chunking=None, so each `ChunkEvent.text` is a raw model delta.
Each requirement re-chunks those deltas with its own chunker, so its `_stream_validate`
receives one complete sentence or word at a time, not the raw deltas.
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
    QuickCheckEvent,
    StreamingDoneEvent,
    stream,
)


class MaxWordsPerSentence(Requirement):
    """Fails if any sentence exceeds `limit` words.

    Chunks at the sentence level: each `_stream_validate` call receives one complete
    sentence, so word counting is per-sentence.
    """

    def __init__(self, limit: int = 12) -> None:
        super().__init__(description="keep sentences short", chunking="sentence")
        self._limit = limit

    def format_for_llm(self) -> str:
        return f"Every sentence must be at most {self._limit} words long."

    async def _stream_validate(
        self, chunk: str, *, backend: Backend, ctx: Context
    ) -> PartialValidationResult:
        _ = backend, ctx
        words = len(chunk.split())
        if words > self._limit:
            return PartialValidationResult(
                "fail",
                reason=f"sentence has {words} words (> {self._limit}): {chunk!r}",
            )
        return PartialValidationResult("unknown")


class NoBannedWord(Requirement):
    """Fails if any word matches a banned term.

    Chunks at the word level: each `_stream_validate` call receives one word, so the check
    fires as soon as a banned word completes, without waiting for a sentence boundary.
    """

    def __init__(self, banned: set[str]) -> None:
        super().__init__(description="avoid banned words", chunking="word")
        self._banned = {w.lower() for w in banned}

    def format_for_llm(self) -> str:
        return f"Do not use any of these words: {', '.join(sorted(self._banned))}."

    async def _stream_validate(
        self, chunk: str, *, backend: Backend, ctx: Context
    ) -> PartialValidationResult:
        _ = backend, ctx
        word = chunk.strip(".,!?;:'\"").lower()
        if word in self._banned:
            return PartialValidationResult("fail", reason=f"banned word: {word!r}")
        return PartialValidationResult("unknown")


async def main() -> None:
    from mellea.stdlib.session import start_session

    m = start_session()
    backend = m.backend
    ctx = m.ctx

    action = Instruction(
        "Write three short sentences about the ocean. Keep each sentence under a dozen words."
    )
    # Two requirements, each with its own chunking granularity, on the same stream.
    sentence_req = MaxWordsPerSentence(limit=12)
    word_req = NoBannedWord(banned={"cyberspace", "blockchain"})

    print(
        "Stream events as they arrive; sentence- and word-level requirements validate in parallel:"
    )
    async with await stream(
        action, backend, ctx, requirements=[sentence_req, word_req], as_events=True
    ) as streamer:
        async for event in streamer:
            match event:
                case ChunkEvent():
                    # chunking=None: each ChunkEvent.text is a raw model delta
                    print(f"  DELTA[{event.chunk_index}]: {event.text!r}")
                case QuickCheckEvent(passed=False):
                    reasons = [r.reason for r in event.results if r.reason]
                    print(f"  QUICK_CHECK: FAIL — {reasons}")
                case StreamingDoneEvent():
                    print(f"  STREAMING_DONE: {len(event.full_text)} chars")
                case CompletedEvent():
                    print(f"  COMPLETED: success={event.success}")
                case _:
                    pass

    print(f"\nCompleted normally: {streamer.completed_normally}")
    if streamer.streaming_failures:
        for _req, pvr in streamer.streaming_failures:
            print(f"Streaming failure: {pvr.reason}")
    else:
        print(f"Full text:\n{streamer.full_text}")


asyncio.run(main())
