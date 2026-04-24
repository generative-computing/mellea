"""Tests for ChunkingStrategy ABC and built-in chunker implementations."""

import pytest

from mellea.stdlib.chunking import ParagraphChunker, SentenceChunker, WordChunker

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


def test_sentence_chunker_unicode():
    c = SentenceChunker()
    result = c.split("Ça va bien. C'est délicieux. ")
    assert result == ["Ça va bien.", "C'est délicieux."]


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
