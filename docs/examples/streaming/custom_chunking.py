# pytest: ollama, e2e

"""Streaming generation with a custom ChunkingStrategy subclass.

Demonstrates:
- Subclassing `ChunkingStrategy` to define a new splitting boundary
- Implementing `split()` (stateless, idempotent) and `flush()` (end-of-stream
  release of any withheld trailing fragment)
- Using the custom chunker with `stream()` in place of a string alias
- Validating line-by-line output from a numbered-list prompt

`LineChunking` splits on single newlines (`\\n`), emitting one line per
`stream_validate` call.  It sits between `WordChunking` (one word) and
`SentenceChunking` (one sentence) in granularity, and is a natural fit for
list-formatted model output.

Extension pattern:
  1. Subclass `ChunkingStrategy`.
  2. Implement `split(text)` — return all complete chunks in `text`;
     withhold any trailing fragment.  It must be stateless and idempotent.
  3. Override `flush(text)` to release the withheld trailing fragment
     when the stream ends naturally.  The default base implementation returns `[]`
     (fragment discarded); override it when the trailing fragment is semantically
     significant.
"""

import asyncio
import re

from mellea.core.backend import Backend
from mellea.core.base import Context
from mellea.core.chunking import ChunkingStrategy
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

# Matches a leading list marker: "1.", "1)", "1 .", or a bare number followed
# by a space — covers common model output formats.
_NUMBERED_LINE = re.compile(r"^\s*\d+[\.\)]\s")


class LineChunking(ChunkingStrategy):
    """Splits text on single newlines, emitting one line per chunk.

    The line after the last `\\n` is withheld as a trailing fragment until
    the stream ends and `flush()` is called.  Blank lines are skipped —
    they carry no content for a line-level validator.

    This chunker is a good fit for numbered-list output, code listings, and
    any structured response where the model uses line breaks as separators
    rather than sentence-ending punctuation or double newlines.
    """

    def split(self, text: str) -> list[str]:
        """Return all complete lines (up to the last newline).

        Args:
            text: The text to split.

        Returns:
            Non-empty lines found before the last newline character.
            The text after the last newline is withheld as a trailing fragment.
        """
        if "\n" not in text:
            return []
        last_nl = text.rfind("\n")
        complete_section = text[:last_nl]
        return [line for line in complete_section.split("\n") if line.strip()]

    def flush(self, text: str) -> list[str]:
        """Release the trailing line fragment at end of stream.

        Args:
            text: The text whose trailing fragment to release.

        Returns:
            The text after the last newline as a single-element list (stripped),
            or an empty list if the text ends with a newline or is empty.
        """
        if not text:
            return []
        last_nl = text.rfind("\n")
        trailing = (text if last_nl == -1 else text[last_nl + 1 :]).strip()
        return [trailing] if trailing else []


class NumberedLineReq(Requirement):
    """Fails the stream if any line does not start with a list number.

    Each `stream_validate` call receives one complete line (from
    `LineChunking`).  This requirement enforces that every line follows
    the `N. item` format, catching unstructured paragraphs or stray headers
    that sneak into what should be a clean numbered list.
    """

    def format_for_llm(self) -> str:
        return "Every line must begin with a number followed by a period (e.g. '1. ')."

    async def _stream_validate(
        self, chunk: str, *, backend: Backend, ctx: Context
    ) -> PartialValidationResult:
        if not _NUMBERED_LINE.match(chunk):
            return PartialValidationResult(
                "fail", reason=f"Line does not start with a number: {chunk.strip()!r}"
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
        "List five world capitals, one per line, numbered 1 through 5. "
        "Use the format: '1. City'. Output only the numbered list, nothing else."
    )
    chunker = LineChunking()
    req = NumberedLineReq()

    print("Stream events as they arrive (one ChunkEvent per line):")
    async with await stream(
        action, backend, ctx, requirements=[req], chunking=chunker, as_events=True
    ) as streamer:
        async for event in streamer:
            match event:
                case ChunkEvent():
                    print(f"  LINE[{event.chunk_index}]: {event.text!r}")
                case QuickCheckEvent(passed=False):
                    print(
                        f"  QUICK_CHECK[line {event.chunk_index}]: FAIL — "
                        f"{event.results[0].reason if event.results else 'unknown'}"
                    )
                case QuickCheckEvent():
                    print(f"  QUICK_CHECK[line {event.chunk_index}]: pass")
                case StreamingDoneEvent():
                    print(f"  STREAMING_DONE: {len(event.full_text)} chars accumulated")
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
    else:
        print(f"Full text:\n{streamer.full_text}")


asyncio.run(main())
