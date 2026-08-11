# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""ChunkingStrategy, its built-in implementations, and the Chunker driver.

A `ChunkingStrategy` is the stateless "how to split": given text, it returns the
complete chunks and withholds any trailing fragment.

A `Chunker` is the stateful "how far through this stream we are": it wraps a
strategy and drives it incrementally, feeding one delta at a time and holding the
trailing fragment between calls.
"""

import re
from abc import ABC, abstractmethod

__all__ = [
    "Chunker",
    "ChunkingStrategy",
    "ParagraphChunking",
    "SentenceChunking",
    "WordChunking",
    "resolve_chunking_strategy",
]


class ChunkingStrategy(ABC):
    """Abstract base class for text chunking strategies used in streaming validation.

    A chunking strategy receives text and returns a list of complete chunks
    ready for downstream validation. Any trailing fragment that has not yet reached
    a chunk boundary is withheld — it is not included in the returned list. Each
    call is stateless and idempotent given the same input.

    End-of-stream contract: `split()` always withholds the trailing fragment;
    `flush()` releases it once no more text is coming.

    Note: this ABC operates on text streams only. Multi-modal output (audio
    segments, image regions) is not supported — the `text: str`
    signatures on `split` and `flush` preclude it.
    """

    @abstractmethod
    def split(self, text: str) -> list[str]:
        """Return complete chunks from text, excluding any trailing fragment.

        Args:
            text: The text to split.

        Returns:
            A list of complete chunks. If no chunk boundary has been reached yet,
            returns an empty list. Never includes the trailing incomplete fragment.
        """
        ...

    def flush(self, text: str) -> list[str]:
        """Return any trailing fragment that `split` withheld.

        Called once after the stream has ended naturally (not on early-exit
        cancellation).  Gives the strategy a chance to release the final fragment
        that did not reach a terminator.

        The default implementation returns an empty list — the trailing
        fragment is discarded.  Built-in chunkers override this to return
        the withheld fragment as a single-element list when non-empty.

        Args:
            text: The text whose trailing fragment to release.

        Returns:
            The trailing fragment as `[fragment]` if it should be treated
            as a final chunk, or an empty list to discard it.
        """
        _ = text
        return []


# Sentence boundary: sentence-ending punctuation, optionally followed by a closing
# quote or paren, then whitespace.
# Character class covers: straight double/single quotes, right double/single curly
# quotes (U+201D, U+2019), and closing paren.
_SENTENCE_BOUNDARY = re.compile("[.!?][\"'\u201d\u2019)]?\\s")

# Whitespace run separator used by WordChunking.
_WHITESPACE = re.compile(r"\s+")

# Paragraph boundary patterns used by ParagraphChunking.
_PARA_BOUNDARY = re.compile(r"\n{2,}")
_PARA_BOUNDARY_END = re.compile(r"\n{2,}$")


class SentenceChunking(ChunkingStrategy):
    """Splits text on sentence boundaries.

    Sentence boundaries are detected by `.`, `!`, or `?`, optionally
    followed by a closing quote (straight or curly) or parenthesis, then
    whitespace. The final sentence is only returned once it is followed by
    whitespace or another sentence — a trailing fragment with no following
    whitespace is withheld. Abbreviations are a known edge case: they will
    be split on (simple regex, not NLP). Leading and inter-sentence whitespace
    (including double-space or tab) is discarded — no chunk, including the first,
    begins with whitespace.
    """

    def split(self, text: str) -> list[str]:
        """Return complete sentences from text.

        Args:
            text: The text to split.

        Returns:
            Complete sentences detected so far. The trailing fragment (if any)
            is withheld.
        """
        if not text:
            return []

        chunks: list[str] = []
        # lstrip the leading edge so the first chunk, like the rest, never starts
        # with separator whitespace carried in from a prior boundary.
        remaining = text.lstrip()

        while True:
            match = _SENTENCE_BOUNDARY.search(remaining)
            if match is None:
                break
            # Include up to and including the punctuation (and optional quote/paren),
            # but not the trailing whitespace character.
            end = match.start() + len(match.group().rstrip())
            chunks.append(remaining[:end])
            # Advance past the entire whitespace separator; lstrip() handles
            # multi-character gaps (double-space, tab, etc.) so they don't
            # leak into the next chunk as leading whitespace.
            remaining = remaining[match.end() :].lstrip()

        return chunks

    def flush(self, text: str) -> list[str]:
        """Return the trailing sentence fragment (if any) as a final chunk.

        Leading and trailing whitespace on the fragment is non-semantic for
        sentence boundaries and is stripped, consistent with how `split` returns
        sentences with no surrounding whitespace.

        Args:
            text: The text whose trailing fragment to release.

        Returns:
            A single-element list containing the trailing sentence fragment
            with leading and trailing whitespace stripped, or an empty list
            when there is no fragment (all content ended in a sentence
            boundary or the input is empty/whitespace-only).
        """
        if not text:
            return []
        remaining = text.lstrip()
        while True:
            match = _SENTENCE_BOUNDARY.search(remaining)
            if match is None:
                break
            remaining = remaining[match.end() :].lstrip()
        trailing = remaining.rstrip()
        return [trailing] if trailing else []


class WordChunking(ChunkingStrategy):
    """Splits text on whitespace boundaries.

    Each word is a chunk. Trailing text not yet followed by whitespace is
    withheld.
    """

    def split(self, text: str) -> list[str]:
        """Return complete words from text.

        Args:
            text: The text to split.

        Returns:
            All whitespace-delimited words except the trailing fragment (if any).
            An empty list is returned when no whitespace boundary has been seen.
        """
        if not text:
            return []

        # Split on runs of whitespace; the last token is a trailing fragment
        # unless text ends with whitespace.
        parts = _WHITESPACE.split(text)

        # re.split on leading whitespace produces an empty first element; strip it.
        if parts and parts[0] == "":
            parts = parts[1:]
        if parts and parts[-1] == "":
            parts = parts[:-1]

        if not parts:
            return []

        # If the text does not end with whitespace, the last part is a fragment.
        if not text[-1].isspace():
            return parts[:-1]

        return parts

    def flush(self, text: str) -> list[str]:
        """Return the trailing word fragment (if any) as a final chunk.

        The trailing fragment is the text after the last whitespace run when
        the accumulated text does not end with whitespace.  When it does end
        with whitespace, every word is already complete and no fragment is
        released.

        Args:
            text: The text whose trailing fragment to release.

        Returns:
            A single-element list containing the trailing word fragment, or
            an empty list when the input ends with whitespace (every word
            already complete) or is empty.
        """
        if not text:
            return []
        if text[-1].isspace():
            return []
        parts = _WHITESPACE.split(text)
        for part in reversed(parts):
            if part:
                return [part]
        return []


class ParagraphChunking(ChunkingStrategy):
    r"""Splits text on double-newline paragraph boundaries.

    Two or more consecutive newline characters are treated as a paragraph
    separator. The trailing paragraph fragment (text not yet followed by `\n\n`)
    is withheld.

    Note: only Unix-style `\n\n` separators are recognised. CRLF
    (`\r\n\r\n`) paragraph separators are not supported.
    """

    def split(self, text: str) -> list[str]:
        """Return complete paragraphs from text.

        Args:
            text: The text to split.

        Returns:
            Complete paragraphs (separated by two or more newlines). The
            trailing incomplete paragraph is withheld. Returns an empty list
            if no paragraph boundary has been reached.
        """
        if not text:
            return []

        parts = _PARA_BOUNDARY.split(text)

        # If the text does not end with \n\n, the last part is a trailing fragment.
        if not _PARA_BOUNDARY_END.search(text):
            parts = parts[:-1]

        # _PARA_BOUNDARY.split on leading \n\n produces an empty first element.
        return [p for p in parts if p]

    def flush(self, text: str) -> list[str]:
        r"""Return the trailing paragraph fragment (if any) as a final chunk.

        Unlike `SentenceChunking.flush`, the fragment is returned
        byte-for-byte without stripping.  Internal whitespace — including
        a trailing single `\n` — can be semantically meaningful inside
        a paragraph (e.g. a list item or a deliberate line break), and a
        consumer validating paragraph content should see the fragment as
        it was withheld.

        Args:
            text: The text whose trailing fragment to release.

        Returns:
            A single-element list containing the trailing paragraph fragment
            byte-for-byte, or an empty list when the input ends with a
            paragraph boundary (`\n\n` or more) or is empty.
        """
        if not text:
            return []
        if _PARA_BOUNDARY_END.search(text):
            return []
        parts = _PARA_BOUNDARY.split(text)
        trailing = parts[-1] if parts else ""
        return [trailing] if trailing else []


_ALIASES: dict[str, type[ChunkingStrategy]] = {
    "sentence": SentenceChunking,
    "word": WordChunking,
    "paragraph": ParagraphChunking,
}


def resolve_chunking_strategy(
    chunking: str | ChunkingStrategy | None,
) -> ChunkingStrategy | None:
    """Resolve a chunking argument to a `ChunkingStrategy` instance, or `None`.

    Args:
        chunking: A `ChunkingStrategy` (returned as-is), a recognized alias string
            (instantiated to its strategy), or `None` (passed through, meaning no
            chunking).

    Returns:
        The resolved strategy, or `None`.

    Raises:
        ValueError: If `chunking` is a string that is not a recognized alias. The
            message lists the recognized aliases.
    """
    if isinstance(chunking, str):
        cls = _ALIASES.get(chunking)
        if cls is None:
            raise ValueError(
                f"Unknown chunking alias {chunking!r}. Choose from: {list(_ALIASES)}"
            )
        return cls()
    return chunking


class Chunker:
    """Drives a `ChunkingStrategy` incrementally over a stream of deltas.

    The stateful counterpart to a `ChunkingStrategy`: the strategy is the
    stateless "how to split," the `Chunker` holds "how far through this stream we
    are." Feed it one delta at a time with `feed()`; it returns any newly complete
    chunks and holds the trailing fragment until the next call. Call `flush()`
    once at stream end to release the final fragment.

    The `Chunker` holds only the pending fragment (text since the last boundary)
    — not the full accumulated text. A caller that needs the full raw stream keeps
    its own copy.

    Delta-invariant: feeding text in any delta slicing yields the same chunks as a
    single `split()` over the whole text.

    Args:
        strategy: The stateless chunking strategy that decides boundaries.
    """

    def __init__(self, strategy: ChunkingStrategy) -> None:
        """Wrap `strategy` for incremental driving."""
        self._strategy = strategy
        self._pending = ""

    def feed(self, delta: str) -> list[str]:
        """Add one stream delta and return any newly complete chunks.

        Args:
            delta: The new text received since the previous delta.

        Returns:
            The chunks completed by this delta, in order. Empty when the delta
            did not complete a boundary. The trailing fragment is withheld until
            a later `feed()` completes it or `flush()` releases it.
        """
        self._pending += delta
        chunks = self._strategy.split(self._pending)
        if not chunks:
            return []

        # Carry forward whatever follows the last emitted chunk. split() may drop
        # inter-chunk whitespace, so locate each chunk by position rather than
        # string-subtracting, then keep the raw suffix as the new pending fragment.
        cursor = 0
        for c in chunks:
            pos = self._pending.find(c, cursor)
            if pos >= 0:
                cursor = pos + len(c)
        self._pending = self._pending[cursor:]
        return chunks

    def flush(self) -> list[str]:
        """Release the trailing fragment withheld after the last boundary.

        Call once when the stream ends naturally. Returns whatever the strategy's
        `flush()` makes of the pending fragment (a single-element list, or empty
        when nothing remains).

        Returns:
            The final chunk as a one-element list, or an empty list when the
            pending fragment is empty or the strategy discards it.
        """
        return self._strategy.flush(self._pending)
