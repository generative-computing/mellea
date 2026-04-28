"""ChunkingStrategy ABC and built-in implementations for streaming validation."""

import re
from abc import ABC, abstractmethod

__all__ = ["ChunkingStrategy", "ParagraphChunker", "SentenceChunker", "WordChunker"]


class ChunkingStrategy(ABC):
    """Abstract base class for text chunking strategies used in streaming validation.

    A chunking strategy receives the full accumulated text so far and returns a
    list of complete chunks ready for downstream validation. Any trailing fragment
    that has not yet reached a chunk boundary is withheld — it is not included in
    the returned list. Each call is stateless and idempotent given the same input.

    End-of-stream contract: ``split()`` always withholds the trailing fragment.
    When the stream terminates, callers are responsible for processing any remainder:
    take the full accumulated text, identify everything after the last returned
    chunk boundary, and handle it appropriately (e.g. pass to a final validator
    or discard).
    """

    @abstractmethod
    def split(self, accumulated_text: str) -> list[str]:
        """Return complete chunks from accumulated_text, excluding any trailing fragment.

        Args:
            accumulated_text: The full text accumulated so far, including all
                previously seen tokens and the latest delta.

        Returns:
            A list of complete chunks. If no chunk boundary has been reached yet,
            returns an empty list. Never includes the trailing incomplete fragment.
        """
        ...

    def flush(self, accumulated_text: str) -> list[str]:
        """Return any trailing fragment that ``split`` withheld.

        Called once by the orchestrator after the stream has ended naturally
        (not on early-exit cancellation).  Gives the chunker a chance to
        release the final fragment that did not reach a terminator.

        The default implementation returns an empty list — the trailing
        fragment is discarded.  Built-in chunkers override this to return
        the withheld fragment as a single-element list when non-empty.

        Args:
            accumulated_text: The full accumulated text at stream end.

        Returns:
            The trailing fragment as ``[fragment]`` if it should be treated
            as a final chunk, or an empty list to discard it.
        """
        _ = accumulated_text
        return []


# Sentence boundary: sentence-ending punctuation, optionally followed by a closing
# quote or paren, then whitespace.
# Character class covers: straight double/single quotes, right double/single curly
# quotes (U+201D, U+2019), and closing paren.
_SENTENCE_BOUNDARY = re.compile("[.!?][\"'\u201d\u2019)]?\\s")

# Whitespace run separator used by WordChunker.
_WHITESPACE = re.compile(r"\s+")

# Paragraph boundary patterns used by ParagraphChunker.
_PARA_BOUNDARY = re.compile(r"\n{2,}")
_PARA_BOUNDARY_END = re.compile(r"\n{2,}$")


class SentenceChunker(ChunkingStrategy):
    """Splits accumulated text on sentence boundaries.

    Sentence boundaries are detected by ``.``, ``!``, or ``?``, optionally
    followed by a closing quote (straight or curly) or parenthesis, then
    whitespace. The final sentence is only returned once it is followed by
    whitespace or another sentence — a trailing fragment with no following
    whitespace is withheld. Abbreviations are a known edge case: they will
    be split on (simple regex, not NLP). Inter-sentence whitespace (including
    double-space or tab) is discarded and does not appear as leading whitespace
    in subsequent chunks.
    """

    def split(self, accumulated_text: str) -> list[str]:
        """Return complete sentences from accumulated_text.

        Args:
            accumulated_text: The full text accumulated so far.

        Returns:
            Complete sentences detected so far. The trailing fragment (if any)
            is withheld.
        """
        if not accumulated_text:
            return []

        chunks: list[str] = []
        remaining = accumulated_text

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

    def flush(self, accumulated_text: str) -> list[str]:
        """Return the trailing sentence fragment (if any) as a final chunk."""
        if not accumulated_text:
            return []
        remaining = accumulated_text
        while True:
            match = _SENTENCE_BOUNDARY.search(remaining)
            if match is None:
                break
            remaining = remaining[match.end() :].lstrip()
        trailing = remaining.strip()
        return [trailing] if trailing else []


class WordChunker(ChunkingStrategy):
    """Splits accumulated text on whitespace boundaries.

    Each word is a chunk. Trailing text not yet followed by whitespace is
    withheld.
    """

    def split(self, accumulated_text: str) -> list[str]:
        """Return complete words from accumulated_text.

        Args:
            accumulated_text: The full text accumulated so far.

        Returns:
            All whitespace-delimited words except the trailing fragment (if any).
            An empty list is returned when no whitespace boundary has been seen.
        """
        if not accumulated_text:
            return []

        # Split on runs of whitespace; the last token is a trailing fragment
        # unless accumulated_text ends with whitespace.
        parts = _WHITESPACE.split(accumulated_text)

        # re.split on leading whitespace produces an empty first element; strip it.
        if parts and parts[0] == "":
            parts = parts[1:]
        if parts and parts[-1] == "":
            parts = parts[:-1]

        if not parts:
            return []

        # If the text does not end with whitespace, the last part is a fragment.
        if not accumulated_text[-1].isspace():
            return parts[:-1]

        return parts

    def flush(self, accumulated_text: str) -> list[str]:
        """Return the trailing word fragment (if any) as a final chunk."""
        if not accumulated_text:
            return []
        if accumulated_text[-1].isspace():
            return []
        parts = _WHITESPACE.split(accumulated_text)
        for part in reversed(parts):
            if part:
                return [part]
        return []


class ParagraphChunker(ChunkingStrategy):
    r"""Splits accumulated text on double-newline paragraph boundaries.

    Two or more consecutive newline characters are treated as a paragraph
    separator. The trailing paragraph fragment (text not yet followed by ``\n\n``)
    is withheld.

    Note: only Unix-style ``\n\n`` separators are recognised. CRLF
    (``\r\n\r\n``) paragraph separators are not supported.
    """

    def split(self, accumulated_text: str) -> list[str]:
        """Return complete paragraphs from accumulated_text.

        Args:
            accumulated_text: The full text accumulated so far.

        Returns:
            Complete paragraphs (separated by two or more newlines). The
            trailing incomplete paragraph is withheld. Returns an empty list
            if no paragraph boundary has been reached.
        """
        if not accumulated_text:
            return []

        parts = _PARA_BOUNDARY.split(accumulated_text)

        # If the text does not end with \n\n, the last part is a trailing fragment.
        if not _PARA_BOUNDARY_END.search(accumulated_text):
            parts = parts[:-1]

        # _PARA_BOUNDARY.split on leading \n\n produces an empty first element.
        return [p for p in parts if p]

    def flush(self, accumulated_text: str) -> list[str]:
        """Return the trailing paragraph fragment (if any) as a final chunk."""
        if not accumulated_text:
            return []
        if _PARA_BOUNDARY_END.search(accumulated_text):
            return []
        parts = _PARA_BOUNDARY.split(accumulated_text)
        trailing = parts[-1] if parts else ""
        return [trailing] if trailing else []
