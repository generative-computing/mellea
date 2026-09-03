# pytest: ollama, e2e

"""Streaming generation with per-word validation using WordChunking.

Demonstrates:
- Using the `"word"` chunking alias for the finest-grained validation
- Detecting a forbidden word the moment it appears in the stream
- Early-exit cancelling generation before the consumer sees the bad word
- How WordChunking compares to SentenceChunking in reaction time

WordChunking splits on whitespace, so each `stream_validate` call receives
exactly one word.  This is the highest-sensitivity strategy: validation fires
before the model has finished even the current clause, letting you catch
prohibited content with minimal output produced.

The trade-off vs. SentenceChunking: validators that need sentence-level context
(grammar, coherence) cannot operate correctly at word granularity because each
chunk carries only a single token.  Use WordChunking when the check is
token-local — forbidden words, length budgets, numeric thresholds.
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
    stream,
)

# Words that must not appear in the model's response.
_FORBIDDEN = {"competitor", "CompetitorX", "legacy", "inferior", "obsolete"}


class ForbiddenWordReq(Requirement):
    """Fails the stream immediately if a forbidden word appears.

    Each `stream_validate` call receives a single word (from `WordChunking`).
    The check is O(1) per word — set membership test — so it adds negligible
    latency.
    """

    def __init__(self, forbidden: set[str]) -> None:
        super().__init__()
        self._forbidden_display = sorted(forbidden)
        self._forbidden = {w.lower() for w in forbidden}

    def format_for_llm(self) -> str:
        return f"Do not use any of the following words: {', '.join(self._forbidden_display)}."

    async def stream_validate(
        self, chunk: str, *, backend: Backend, ctx: Context
    ) -> PartialValidationResult:
        word = chunk.strip().lower().strip(".,!?;:\"'")
        if word in self._forbidden:
            return PartialValidationResult(
                "fail", reason=f"Forbidden word detected: {chunk.strip()!r}"
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
        "Describe three key advantages of cloud-native software development "
        "in two or three sentences."
    )
    req = ForbiddenWordReq(forbidden=_FORBIDDEN)

    word_count = 0

    print("Stream events as they arrive (one ChunkEvent per word):")
    async with await stream(
        action, backend, ctx, requirements=[req], chunking="word", as_events=True
    ) as streamer:
        async for event in streamer:
            match event:
                case ChunkEvent():
                    word_count += 1
                    # Only print every 5th word to keep output readable
                    if word_count % 5 == 1:
                        print(f"  ...word {word_count}: {event.text!r}")
                case QuickCheckEvent(passed=False):
                    print(
                        f"  QUICK_CHECK[word {event.chunk_index}]: FAIL — "
                        f"{event.results[0].reason if event.results else 'unknown'}"
                    )
                case StreamingDoneEvent():
                    print(
                        f"  STREAMING_DONE: {word_count} words, {len(event.full_text)} chars"
                    )
                case FullValidationEvent():
                    print(f"  FULL_VALIDATION: {'PASS' if event.passed else 'FAIL'}")
                case CompletedEvent():
                    print(f"  COMPLETED: success={event.success}")
                case _:
                    pass

    print(f"\nCompleted normally: {streamer.completed_normally}")
    if streamer.streaming_failures:
        for _req, pvr in streamer.streaming_failures:
            print(f"Streaming failure: {pvr.reason}")
        print(f"Text at cancellation: {streamer.full_text!r}")
    else:
        print(f"Full text: {streamer.full_text!r}")


asyncio.run(main())
