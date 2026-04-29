# pytest: ollama, e2e

"""Streaming generation with per-chunk validation using stream_with_chunking().

Demonstrates:
- Subclassing Requirement to override stream_validate() for early-exit checks
- Calling stream_with_chunking() with sentence-level chunking
- Observing the full event vocabulary via events() as they arrive
- Awaiting full completion with acomplete() to access final_validations and full_text
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


class MaxSentencesReq(Requirement):
    """Fails if the model generates more than *limit* sentences mid-stream.

    Each ``stream_validate`` call receives one complete sentence from the
    :class:`~mellea.stdlib.chunking.SentenceChunker`.  The running count is
    maintained on ``self`` — this is the standard pattern for requirements
    that need context beyond a single chunk.
    """

    def __init__(self, limit: int) -> None:
        super().__init__()
        self._limit = limit
        self._count = 0

    def format_for_llm(self) -> str:
        return f"The response must be at most {self._limit} sentences long."

    async def stream_validate(
        self, chunk: str, *, backend: Backend, ctx: Context
    ) -> PartialValidationResult:
        self._count += 1
        if self._count > self._limit:
            return PartialValidationResult(
                "fail",
                reason=f"Response exceeded {self._limit} sentence limit mid-stream",
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
        "Write a short paragraph about the water cycle in exactly two sentences."
    )
    req = MaxSentencesReq(limit=3)

    result = await stream_with_chunking(
        action, backend, ctx, quick_check_requirements=[req], chunking="sentence"
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
                pass  # RetryEvent and any future event types

    await result.acomplete()

    print(f"\nCompleted normally: {result.completed}")
    print(f"Full text: {result.full_text!r}")

    if result.streaming_failures:
        for _req, pvr in result.streaming_failures:
            print(f"Streaming failure: {pvr.reason}")

    if result.final_validations:
        for vr in result.final_validations:
            print(f"Final validation: {'PASS' if vr.as_bool() else 'FAIL'}")


asyncio.run(main())
