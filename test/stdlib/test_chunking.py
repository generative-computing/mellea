"""Tests for ChunkingStrategy ABC and built-in chunker implementations."""

import pytest

from mellea.stdlib.chunking import (
    ChunkingStrategy,
    ParagraphChunker,
    SentenceChunker,
    WordChunker,
)


def test_chunking_strategy_is_abstract():
    with pytest.raises(TypeError):
        ChunkingStrategy()  # type: ignore[abstract]


# ---------------------------------------------------------------------------
# SentenceChunker
# ---------------------------------------------------------------------------


def test_sentence_chunker_empty():
    c = SentenceChunker()
    assert c.split("") == []


def test_sentence_chunker_no_boundary():
    c = SentenceChunker()
    assert c.split("The quick brown") == []


def test_sentence_chunker_one_sentence_no_trailing():
    # A sentence with no following whitespace is a trailing fragment — withheld.
    c = SentenceChunker()
    assert c.split("The quick brown fox.") == []


def test_sentence_chunker_one_sentence_with_space():
    # Sentence followed by a space signals completion.
    c = SentenceChunker()
    assert c.split("The quick brown fox. ") == ["The quick brown fox."]


def test_sentence_chunker_with_trailing():
    c = SentenceChunker()
    result = c.split("The quick brown fox. He")
    assert result == ["The quick brown fox."]


def test_sentence_chunker_multiple():
    c = SentenceChunker()
    result = c.split("Hello world. Goodbye world. ")
    assert result == ["Hello world.", "Goodbye world."]


def test_sentence_chunker_exclamation():
    c = SentenceChunker()
    result = c.split("Stop! Go. ")
    assert result == ["Stop!", "Go."]


def test_sentence_chunker_question():
    c = SentenceChunker()
    result = c.split("Are you sure? Yes. ")
    assert result == ["Are you sure?", "Yes."]


def test_sentence_chunker_closing_quote():
    c = SentenceChunker()
    result = c.split('He said "hello." She left. ')
    assert result == ['He said "hello."', "She left."]


def test_sentence_chunker_curly_quotes():
    # Verifies U+201D (right double curly quote) and U+2019 (right single curly quote)
    # are recognised as closing marks after sentence-ending punctuation.
    c = SentenceChunker()
    result = c.split("She said \u201cdone.\u201d Next sentence. ")
    assert result == ["She said \u201cdone.\u201d", "Next sentence."]


def test_sentence_chunker_unicode():
    c = SentenceChunker()
    result = c.split("Ça va bien. C'est délicieux. ")
    assert result == ["Ça va bien.", "C'est délicieux."]


def test_sentence_chunker_closing_paren():
    c = SentenceChunker()
    result = c.split("(See note.) Continue here. ")
    assert result == ["(See note.)", "Continue here."]


def test_sentence_chunker_double_space_separator():
    # Regression: double-space between sentences must not leak into next chunk.
    c = SentenceChunker()
    result = c.split("First.  Second. ")
    assert result == ["First.", "Second."]


def test_sentence_chunker_tab_separator():
    c = SentenceChunker()
    result = c.split("First.\tSecond. ")
    assert result == ["First.", "Second."]


def test_sentence_chunker_abbreviation_known_bad():
    # Known edge case: abbreviations cause a spurious split (simple regex, not NLP).
    c = SentenceChunker()
    result = c.split("Dr. Smith went home. He was tired. ")
    assert result == ["Dr.", "Smith went home.", "He was tired."]


def test_sentence_chunker_incremental_simulation():
    # Simulate accumulating text token by token.
    c = SentenceChunker()
    assert c.split("The") == []
    assert c.split("The quick") == []
    assert c.split("The quick brown fox.") == []
    assert c.split("The quick brown fox. He") == ["The quick brown fox."]
    assert c.split("The quick brown fox. He ran.") == ["The quick brown fox."]
    assert c.split("The quick brown fox. He ran. ") == [
        "The quick brown fox.",
        "He ran.",
    ]


# ---------------------------------------------------------------------------
# WordChunker
# ---------------------------------------------------------------------------


def test_word_chunker_empty():
    c = WordChunker()
    assert c.split("") == []


def test_word_chunker_no_boundary():
    c = WordChunker()
    assert c.split("hello") == []


def test_word_chunker_one_word_with_space():
    c = WordChunker()
    assert c.split("hello ") == ["hello"]


def test_word_chunker_trailing_fragment():
    c = WordChunker()
    result = c.split("hello world")
    assert result == ["hello"]


def test_word_chunker_multiple_words():
    c = WordChunker()
    result = c.split("one two three ")
    assert result == ["one", "two", "three"]


def test_word_chunker_multiple_spaces():
    c = WordChunker()
    result = c.split("one  two  three ")
    assert result == ["one", "two", "three"]


def test_word_chunker_unicode():
    c = WordChunker()
    result = c.split("naïve résumé ")
    assert result == ["naïve", "résumé"]


def test_word_chunker_incremental_simulation():
    c = WordChunker()
    assert c.split("foo") == []
    assert c.split("foobar") == []
    assert c.split("foobar ") == ["foobar"]
    assert c.split("foobar ba") == ["foobar"]
    assert c.split("foobar baz ") == ["foobar", "baz"]


def test_word_chunker_leading_whitespace():
    # re.split on " hello world" produces ['', 'hello', 'world'] — empty first
    # element must be stripped.
    c = WordChunker()
    result = c.split(" hello world ")
    assert result == ["hello", "world"]


# ---------------------------------------------------------------------------
# ParagraphChunker
# ---------------------------------------------------------------------------


def test_paragraph_chunker_empty():
    c = ParagraphChunker()
    assert c.split("") == []


def test_paragraph_chunker_no_boundary():
    c = ParagraphChunker()
    assert c.split("Just one paragraph with no double newline") == []


def test_paragraph_chunker_one_complete_paragraph():
    c = ParagraphChunker()
    result = c.split("First paragraph.\n\n")
    assert result == ["First paragraph."]


def test_paragraph_chunker_with_trailing():
    c = ParagraphChunker()
    result = c.split("First paragraph.\n\nSecond paragraph in progress")
    assert result == ["First paragraph."]


def test_paragraph_chunker_multiple():
    c = ParagraphChunker()
    result = c.split("Para one.\n\nPara two.\n\n")
    assert result == ["Para one.", "Para two."]


def test_paragraph_chunker_triple_newline():
    c = ParagraphChunker()
    result = c.split("Para one.\n\n\nPara two.\n\n")
    assert result == ["Para one.", "Para two."]


def test_paragraph_chunker_unicode():
    c = ParagraphChunker()
    result = c.split("Première partie.\n\nDeuxième partie.\n\n")
    assert result == ["Première partie.", "Deuxième partie."]


def test_paragraph_chunker_incremental_simulation():
    c = ParagraphChunker()
    assert c.split("First") == []
    assert c.split("First paragraph.") == []
    assert c.split("First paragraph.\n\n") == ["First paragraph."]
    assert c.split("First paragraph.\n\nSecond") == ["First paragraph."]
    assert c.split("First paragraph.\n\nSecond paragraph.\n\n") == [
        "First paragraph.",
        "Second paragraph.",
    ]


# ---------------------------------------------------------------------------
# flush() — trailing-fragment release at end of stream
# ---------------------------------------------------------------------------


def test_default_flush_returns_empty_list():
    """The ABC default discards the trailing fragment."""

    class Minimal(ChunkingStrategy):
        def split(self, accumulated_text: str) -> list[str]:
            _ = accumulated_text
            return []

    assert Minimal().flush("anything at all") == []
    assert Minimal().flush("") == []


def test_sentence_chunker_flush_empty():
    assert SentenceChunker().flush("") == []


def test_sentence_chunker_flush_only_complete():
    """All text ends in a complete sentence with trailing whitespace → no fragment."""
    assert SentenceChunker().flush("One. Two. ") == []


def test_sentence_chunker_flush_trailing_fragment():
    """Final sentence without trailing whitespace is released by flush."""
    assert SentenceChunker().flush("One. Two without period") == ["Two without period"]


def test_sentence_chunker_flush_terminated_no_trailing_space():
    """Final sentence with terminator but no trailing whitespace is a fragment
    under split() semantics and gets released by flush()."""
    assert SentenceChunker().flush("One. Two.") == ["Two."]


def test_sentence_chunker_flush_single_sentence_no_terminator():
    assert SentenceChunker().flush("Incomplete sentence") == ["Incomplete sentence"]


def test_word_chunker_flush_empty():
    assert WordChunker().flush("") == []


def test_word_chunker_flush_trailing_whitespace():
    """Trailing whitespace means all words are complete → no fragment."""
    assert WordChunker().flush("one two three ") == []


def test_word_chunker_flush_trailing_fragment():
    assert WordChunker().flush("one two three") == ["three"]


def test_word_chunker_flush_single_word():
    assert WordChunker().flush("solo") == ["solo"]


def test_paragraph_chunker_flush_empty():
    assert ParagraphChunker().flush("") == []


def test_paragraph_chunker_flush_only_complete():
    assert ParagraphChunker().flush("Para one.\n\nPara two.\n\n") == []


def test_paragraph_chunker_flush_trailing_fragment():
    assert ParagraphChunker().flush("Para one.\n\nPara two (no sep)") == [
        "Para two (no sep)"
    ]


def test_paragraph_chunker_flush_single_paragraph_no_separator():
    assert ParagraphChunker().flush("Only paragraph") == ["Only paragraph"]
