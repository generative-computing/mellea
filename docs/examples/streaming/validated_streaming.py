# pytest: ollama, e2e

"""Streaming generation with per-chunk validation using stream().

Demonstrates:
- Subclassing Requirement to override stream_validate() for early-exit checks
- Calling stream() with sentence-level chunking
- Observing the typed StreamEvents via the STREAMING_EVENT hook as they arrive
- Driving the stream with `async with` + `async for` for safe cleanup
- Reading terminal state (failed_early, full_text, final_validations) after the loop
"""

import asyncio
import re
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

# Crude sentence-terminator detector. A run of ``.``/``!``/``?`` counts once
# (so "..." and "!!!" are a single terminator). Good enough for an example;
# production code might use spaCy/NLTK for proper sentence segmentation.
_SENTENCE_END = re.compile(r"[.!?]+")


class MaxSentencesReq(Requirement):
    """Fails if the model generates more than *limit* sentences mid-stream.

    Counts sentence terminators in the chunk *text* rather than counting
    `stream_validate` calls.  This makes the requirement **chunker-agnostic**:
    the same instance behaves correctly with sentence, word, or paragraph
    chunking, because the semantics depend on content, not on the chunker's
    structural decisions.

    When writing your own streaming requirements, prefer this content-driven
    pattern over coupling the requirement to a specific chunker.  Reach for
    chunker-coupled logic only when the requirement is genuinely a property
    of chunk boundaries (e.g. "no chunk longer than N tokens").
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
        self._count += len(_SENTENCE_END.findall(chunk))
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
        return ValidationResult(result=self._count <= self._limit)


async def main() -> None:
    from mellea.stdlib.session import start_session

    m = start_session()
    backend = m.backend
    ctx = m.ctx

    action = Instruction(
        "Write a short paragraph about the water cycle in exactly two sentences."
    )
    req = MaxSentencesReq(limit=3)

    @hook("streaming_event")
    async def print_events(payload: Any, ctx: Any) -> None:
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
                pass  # RetryEvent and any future event types

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

    if streamer.streaming_failures:
        for _req, pvr in streamer.streaming_failures:
            print(f"Streaming failure: {pvr.reason}")

    if streamer.final_validations:
        for vr in streamer.final_validations:
            print(f"Final validation: {'PASS' if vr.as_bool() else 'FAIL'}")


asyncio.run(main())
