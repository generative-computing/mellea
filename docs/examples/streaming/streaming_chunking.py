# pytest: ollama, integration

"""Streaming generation with per-chunk validation using stream_with_chunking().

Demonstrates:
- Subclassing Requirement to override stream_validate() for early-exit checks
- Calling stream_with_chunking() with sentence-level chunking
- Consuming validated chunks via astream() as they arrive
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
from mellea.stdlib.streaming import stream_with_chunking


class MaxSentencesReq(Requirement):
    """Fails if the model generates more than *limit* sentences mid-stream."""

    def __init__(self, limit: int) -> None:
        self._limit = limit
        self._count = 0

    def format_for_llm(self) -> str:
        return f"The response must be at most {self._limit} sentences long."

    async def stream_validate(
        self, chunk: str, *, backend: Backend, ctx: Context
    ) -> PartialValidationResult:
        sentence_count = chunk.count(".") + chunk.count("!") + chunk.count("?")
        if sentence_count > self._limit:
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

    print("Streaming chunks as they arrive:")
    async for chunk in result.astream():
        print(f"  CHUNK: {chunk!r}")

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
