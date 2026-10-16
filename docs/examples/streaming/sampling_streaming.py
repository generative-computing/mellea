# pytest: ollama, e2e

"""Streaming generation with mid-stream retry/repair via a SamplingStrategy.

Demonstrates:
- Passing `strategy=` to `stream()` so an attempt that fails mid-stream is aborted and repaired
- Observing `RetryEvent` and per-attempt `ChunkEvent`s via `stream(as_events=True)`
- Reading the per-attempt history from `streamer.attempts` after the run
"""

import asyncio
import re

from mellea.core.backend import Backend
from mellea.core.base import Context
from mellea.core.requirement import (
    PartialValidationResult,
    Requirement,
    ValidationResult,
)
from mellea.stdlib.components import Instruction
from mellea.stdlib.sampling import RepairTemplateStrategy
from mellea.stdlib.streaming import (
    ChunkEvent,
    CompletedEvent,
    FullValidationEvent,
    RetryEvent,
    stream,
)

_SENTENCE_END = re.compile(r"[.!?]+")


class MaxSentencesReq(Requirement):
    """Fails mid-stream once the response exceeds *limit* sentences.

    `stream_validate` returns `"fail"` the moment the cap is passed, so the
    attempt is aborted partway and repaired via `stream_repair` — no tokens spent
    finishing an over-long output.
    """

    def __init__(self, limit: int) -> None:
        super().__init__()
        self._limit = limit
        self._count = 0

    def format_for_llm(self) -> str:
        return f"The response must be at most {self._limit} sentences long."

    async def _stream_validate(
        self, chunk: str, *, backend: Backend, ctx: Context
    ) -> PartialValidationResult:
        self._count += len(_SENTENCE_END.findall(chunk))
        if self._count > self._limit:
            return PartialValidationResult(
                "fail", reason=f"exceeded {self._limit} sentences"
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

    action = Instruction("Write about the water cycle.")
    print("Stream events (attempt numbers increment across retries):")
    async with await stream(
        action,
        m.backend,
        m.ctx,
        requirements=[MaxSentencesReq(limit=2)],
        strategy=RepairTemplateStrategy(loop_budget=3),
        as_events=True,
    ) as streamer:
        async for event in streamer:
            match event:
                case ChunkEvent():
                    print(f"  CHUNK[attempt {event.attempt}]: {event.text!r}")
                case FullValidationEvent():
                    verdict = "PASS" if event.passed else "FAIL"
                    print(f"  FULL_VALIDATION[attempt {event.attempt}]: {verdict}")
                case RetryEvent():
                    print(f"  RETRY -> attempt {event.attempt}: {event.reason}")
                case CompletedEvent():
                    print(
                        f"  COMPLETED: success={event.success} "
                        f"attempts_used={event.attempts_used}"
                    )
                case _:
                    pass

    print(f"\nWinning text: {streamer.full_text!r}")
    print(f"Attempts run: {len(streamer.attempts)}")
    for a in streamer.attempts:
        print(
            f"  attempt {a.attempt}: failed_early={a.failed_early} "
            f"success={a.success} selected={a.selected}"
        )


asyncio.run(main())
