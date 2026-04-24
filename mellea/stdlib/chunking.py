"""ChunkingStrategy ABC and built-in implementations for streaming validation."""

import re
from abc import ABC, abstractmethod


class ChunkingStrategy(ABC):
    """Abstract base class for text chunking strategies used in streaming validation.

    A chunking strategy receives the full accumulated text so far and returns a
    list of complete chunks ready for downstream validation. Any trailing fragment
    that has not yet reached a chunk boundary is withheld — it is not included in
    the returned list. Each call is stateless and idempotent given the same input.
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


# Sentence boundary: sentence-ending punctuation optionally followed by closing
# quotes/parens, then either whitespace or end-of-string.
_SENTENCE_BOUNDARY = re.compile(r'[.!?]["\')]?\s')


class SentenceChunker(ChunkingStrategy):
    """Splits accumulated text on sentence boundaries.

    Sentence boundaries are detected by ``.``, ``!``, or ``?``, optionally
    followed by a closing quote or parenthesis, then whitespace. The final
    sentence is only returned once it is followed by whitespace or another
    sentence — a trailing fragment with no following whitespace is withheld.
    Abbreviations are a known edge case: they will be split on (simple regex,
    not NLP).
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
            # Advance past the whitespace separator
            remaining = remaining[match.end() :]

        return chunks


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
        parts = re.split(r"\s+", accumulated_text)

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


class ParagraphChunker(ChunkingStrategy):
    r"""Splits accumulated text on double-newline paragraph boundaries.

    Two or more consecutive newline characters are treated as a paragraph
    separator. The trailing paragraph fragment (text not yet followed by ``\n\n``)
    is withheld.
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

        parts = re.split(r"\n{2,}", accumulated_text)

        # If the text does not end with \n\n, the last part is a trailing fragment.
        if not re.search(r"\n{2,}$", accumulated_text):
            parts = parts[:-1]

        return [p for p in parts if p]
