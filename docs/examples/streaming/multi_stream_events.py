# pytest: ollama, e2e

"""Observing streaming events across multiple concurrent streams via a plugin.

`stream()` yields validated chunks through `async for`, but its typed lifecycle
events (`ChunkEvent`, `QuickCheckEvent`, `StreamingDoneEvent`,
`FullValidationEvent`, `CompletedEvent`, `ErrorEvent`) are surfaced through the
`STREAMING_EVENT` plugin hook rather than the iterator.

Demonstrates:
- Registering a `@hook("streaming_event")` function to receive every stream's events
- Running two streams concurrently with `asyncio.gather`, so their events arrive
  interleaved
- Using `payload.streaming_id` to tell interleaved events apart — the same key a
  real consumer (dashboard, metrics sink, websocket) would demultiplex on
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
from mellea.stdlib.context import SimpleContext
from mellea.stdlib.streaming import (
    ChunkEvent,
    CompletedEvent,
    ErrorEvent,
    FullValidationEvent,
    QuickCheckEvent,
    StreamingDoneEvent,
    stream,
)


@hook("streaming_event")
async def print_streaming_events(payload: Any, ctx: Any) -> None:
    """Print each streaming event live, tagged with its stream's id."""
    event = payload.event
    match event:
        case ChunkEvent():
            summary = f"CHUNK[{event.chunk_index}]: {event.text!r}"
        case QuickCheckEvent():
            summary = f"QUICK_CHECK[{event.chunk_index}]: {'pass' if event.passed else 'FAIL'}"
        case StreamingDoneEvent():
            summary = f"STREAMING_DONE: {len(event.full_text)} chars"
        case FullValidationEvent():
            summary = f"FULL_VALIDATION: {'PASS' if event.passed else 'FAIL'}"
        case CompletedEvent():
            summary = f"COMPLETED: success={event.success}"
        case ErrorEvent():
            summary = f"ERROR: {event.exception_type}: {event.detail}"
        case _:
            summary = type(event).__name__
    print(f"  [{payload.streaming_id[:8]}] {summary}")


class MaxSentencesReq(Requirement):
    """Fails mid-stream once the response exceeds a sentence budget."""

    def __init__(self, limit: int) -> None:
        super().__init__()
        self._limit = limit
        self._count = 0

    def format_for_llm(self) -> str:
        return f"The response must be at most {self._limit} sentences long."

    async def stream_validate(
        self, chunk: str, *, backend: Backend, ctx: Context
    ) -> PartialValidationResult:
        self._count += sum(chunk.count(p) for p in ".!?")
        if self._count > self._limit:
            return PartialValidationResult(
                "fail", reason=f"Exceeded {self._limit} sentence limit mid-stream"
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

    # One backend serving two independent requests, each with a fresh context.
    backend = start_session().backend

    register(print_streaming_events)

    async def run(prompt: str) -> None:
        """Drive one validated stream to completion; the hook prints its events."""
        async with await stream(
            Instruction(prompt),
            backend,
            SimpleContext(),
            requirements=[MaxSentencesReq(limit=3)],
            chunking="sentence",
        ) as streamer:
            async for _chunk in streamer:
                pass

    # gather runs both at once, so events interleave — the id prefix says which stream.
    await asyncio.gather(
        run("Describe the water cycle in two sentences."),
        run("Describe photosynthesis in two sentences."),
    )


asyncio.run(main())
