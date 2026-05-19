# pytest: ollama, e2e

"""Streaming generation with per-chunk validation using stream_with_chunking().

Demonstrates:
- Subclassing Requirement to override stream_validate() for early-exit checks
- Calling stream_with_chunking() with sentence-level chunking
- Consuming validated chunks via astream() as they arrive
- Awaiting full completion with acomplete() to access final_validations and full_text
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
from mellea.stdlib.streaming import stream_with_chunking

# Crude sentence-terminator detector. A run of ``.``/``!``/``?`` counts once
# (so "..." and "!!!" are a single terminator). Good enough for an example;
# production code might use spaCy/NLTK for proper sentence segmentation.
_SENTENCE_END = re.compile(r"[.!?]+")


class MaxSentencesReq(Requirement):
    """Fails if the model generates more than *limit* sentences mid-stream.

    Counts sentence terminators in the chunk *text* rather than counting
    ``stream_validate`` calls.  This makes the requirement **chunker-agnostic**:
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

    result = await stream_with_chunking(
        action, backend, ctx, requirements=[req], chunking="sentence"
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
