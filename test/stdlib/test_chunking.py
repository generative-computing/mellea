# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for ChunkingStrategy ABC, built-in strategies, and the Chunker driver."""

import random
from collections.abc import Callable

import pytest

from mellea.stdlib.chunking import (
    Chunker,
    ChunkingStrategy,
    ParagraphChunking,
    SentenceChunking,
    WordChunking,
    resolve_chunking_strategy,
)


def test_chunking_strategy_is_abstract():
    with pytest.raises(TypeError):
        ChunkingStrategy()  # type: ignore[abstract]


# ---------------------------------------------------------------------------
# SentenceChunking
# ---------------------------------------------------------------------------


def test_sentence_chunker_empty():
    c = SentenceChunking()
    assert c.split("") == []


def test_sentence_chunker_no_boundary():
    c = SentenceChunking()
    assert c.split("The quick brown") == []


def test_sentence_chunker_one_sentence_no_trailing():
    # A sentence with no following whitespace is a trailing fragment — withheld.
    c = SentenceChunking()
    assert c.split("The quick brown fox.") == []


def test_sentence_chunker_one_sentence_with_space():
    # Sentence followed by a space signals completion.
    c = SentenceChunking()
    assert c.split("The quick brown fox. ") == ["The quick brown fox."]


def test_sentence_chunker_with_trailing():
    c = SentenceChunking()
    result = c.split("The quick brown fox. He")
    assert result == ["The quick brown fox."]


def test_sentence_chunker_multiple():
    c = SentenceChunking()
    result = c.split("Hello world. Goodbye world. ")
    assert result == ["Hello world.", "Goodbye world."]


def test_sentence_chunker_exclamation():
    c = SentenceChunking()
    result = c.split("Stop! Go. ")
    assert result == ["Stop!", "Go."]


def test_sentence_chunker_question():
    c = SentenceChunking()
    result = c.split("Are you sure? Yes. ")
    assert result == ["Are you sure?", "Yes."]


def test_sentence_chunker_closing_quote():
    c = SentenceChunking()
    result = c.split('He said "hello." She left. ')
    assert result == ['He said "hello."', "She left."]


def test_sentence_chunker_curly_quotes():
    # Verifies U+201D (right double curly quote) and U+2019 (right single curly quote)
    # are recognised as closing marks after sentence-ending punctuation.
    c = SentenceChunking()
    result = c.split("She said \u201cdone.\u201d Next sentence. ")
    assert result == ["She said \u201cdone.\u201d", "Next sentence."]


def test_sentence_chunker_unicode():
    c = SentenceChunking()
    result = c.split("Ça va bien. C'est délicieux. ")
    assert result == ["Ça va bien.", "C'est délicieux."]


def test_sentence_chunker_closing_paren():
    c = SentenceChunking()
    result = c.split("(See note.) Continue here. ")
    assert result == ["(See note.)", "Continue here."]


def test_sentence_chunker_double_space_separator():
    # Regression: double-space between sentences must not leak into next chunk.
    c = SentenceChunking()
    result = c.split("First.  Second. ")
    assert result == ["First.", "Second."]


def test_sentence_chunker_tab_separator():
    c = SentenceChunking()
    result = c.split("First.\tSecond. ")
    assert result == ["First.", "Second."]


def test_sentence_chunker_abbreviation_known_bad():
    # Known edge case: abbreviations cause a spurious split (simple regex, not NLP).
    c = SentenceChunking()
    result = c.split("Dr. Smith went home. He was tired. ")
    assert result == ["Dr.", "Smith went home.", "He was tired."]


def test_sentence_chunker_incremental_simulation():
    # Simulate accumulating text token by token.
    c = SentenceChunking()
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
# WordChunking
# ---------------------------------------------------------------------------


def test_word_chunker_empty():
    c = WordChunking()
    assert c.split("") == []


def test_word_chunker_no_boundary():
    c = WordChunking()
    assert c.split("hello") == []


def test_word_chunker_one_word_with_space():
    c = WordChunking()
    assert c.split("hello ") == ["hello"]


def test_word_chunker_trailing_fragment():
    c = WordChunking()
    result = c.split("hello world")
    assert result == ["hello"]


def test_word_chunker_multiple_words():
    c = WordChunking()
    result = c.split("one two three ")
    assert result == ["one", "two", "three"]


def test_word_chunker_multiple_spaces():
    c = WordChunking()
    result = c.split("one  two  three ")
    assert result == ["one", "two", "three"]


def test_word_chunker_unicode():
    c = WordChunking()
    result = c.split("naïve résumé ")
    assert result == ["naïve", "résumé"]


def test_word_chunker_incremental_simulation():
    c = WordChunking()
    assert c.split("foo") == []
    assert c.split("foobar") == []
    assert c.split("foobar ") == ["foobar"]
    assert c.split("foobar ba") == ["foobar"]
    assert c.split("foobar baz ") == ["foobar", "baz"]


def test_word_chunker_leading_whitespace():
    # re.split on " hello world" produces ['', 'hello', 'world'] — empty first
    # element must be stripped.
    c = WordChunking()
    result = c.split(" hello world ")
    assert result == ["hello", "world"]


# ---------------------------------------------------------------------------
# ParagraphChunking
# ---------------------------------------------------------------------------


def test_paragraph_chunker_empty():
    c = ParagraphChunking()
    assert c.split("") == []


def test_paragraph_chunker_no_boundary():
    c = ParagraphChunking()
    assert c.split("Just one paragraph with no double newline") == []


def test_paragraph_chunker_one_complete_paragraph():
    c = ParagraphChunking()
    result = c.split("First paragraph.\n\n")
    assert result == ["First paragraph."]


def test_paragraph_chunker_with_trailing():
    c = ParagraphChunking()
    result = c.split("First paragraph.\n\nSecond paragraph in progress")
    assert result == ["First paragraph."]


def test_paragraph_chunker_multiple():
    c = ParagraphChunking()
    result = c.split("Para one.\n\nPara two.\n\n")
    assert result == ["Para one.", "Para two."]


def test_paragraph_chunker_triple_newline():
    c = ParagraphChunking()
    result = c.split("Para one.\n\n\nPara two.\n\n")
    assert result == ["Para one.", "Para two."]


def test_paragraph_chunker_unicode():
    c = ParagraphChunking()
    result = c.split("Première partie.\n\nDeuxième partie.\n\n")
    assert result == ["Première partie.", "Deuxième partie."]


def test_paragraph_chunker_incremental_simulation():
    c = ParagraphChunking()
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
        def split(self, text: str) -> list[str]:
            _ = text
            return []

    assert Minimal().flush("anything at all") == []
    assert Minimal().flush("") == []


def test_sentence_chunker_flush_empty():
    assert SentenceChunking().flush("") == []


def test_sentence_chunker_flush_only_complete():
    """All text ends in a complete sentence with trailing whitespace → no fragment."""
    assert SentenceChunking().flush("One. Two. ") == []


def test_sentence_chunker_flush_trailing_fragment():
    """Final sentence without trailing whitespace is released by flush."""
    assert SentenceChunking().flush("One. Two without period") == ["Two without period"]


def test_sentence_chunker_flush_terminated_no_trailing_space():
    """Final sentence with terminator but no trailing whitespace is a fragment
    under split() semantics and gets released by flush()."""
    assert SentenceChunking().flush("One. Two.") == ["Two."]


def test_sentence_chunker_flush_single_sentence_no_terminator():
    assert SentenceChunking().flush("Incomplete sentence") == ["Incomplete sentence"]


def test_word_chunker_flush_empty():
    assert WordChunking().flush("") == []


def test_word_chunker_flush_trailing_whitespace():
    """Trailing whitespace means all words are complete → no fragment."""
    assert WordChunking().flush("one two three ") == []


def test_word_chunker_flush_trailing_fragment():
    assert WordChunking().flush("one two three") == ["three"]


def test_word_chunker_flush_single_word():
    assert WordChunking().flush("solo") == ["solo"]


def test_paragraph_chunker_flush_empty():
    assert ParagraphChunking().flush("") == []


def test_paragraph_chunker_flush_only_complete():
    assert ParagraphChunking().flush("Para one.\n\nPara two.\n\n") == []


def test_paragraph_chunker_flush_trailing_fragment():
    assert ParagraphChunking().flush("Para one.\n\nPara two (no sep)") == [
        "Para two (no sep)"
    ]


def test_paragraph_chunker_flush_single_paragraph_no_separator():
    assert ParagraphChunking().flush("Only paragraph") == ["Only paragraph"]


# ---------------------------------------------------------------------------
# resolve_chunking_strategy
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("alias", "expected"),
    [
        ("sentence", SentenceChunking),
        ("word", WordChunking),
        ("paragraph", ParagraphChunking),
    ],
)
def test_resolve_chunking_strategy_alias(alias: str, expected: type) -> None:
    assert isinstance(resolve_chunking_strategy(alias), expected)


def test_resolve_chunking_strategy_none_passes_through() -> None:
    assert resolve_chunking_strategy(None) is None


def test_resolve_chunking_strategy_instance_passes_through() -> None:
    strategy = SentenceChunking()
    assert resolve_chunking_strategy(strategy) is strategy


def test_resolve_chunking_strategy_unknown_alias_raises() -> None:
    with pytest.raises(ValueError, match="Unknown chunking alias"):
        resolve_chunking_strategy("nonsense")


# ---------------------------------------------------------------------------
# Chunker — stateful incremental driver
# ---------------------------------------------------------------------------


def _run_chunker(strategy: ChunkingStrategy, deltas: list[str]) -> list[str]:
    """Feed `deltas` through a Chunker and return all chunks including flush."""
    chunker = Chunker(strategy)
    out: list[str] = []
    for delta in deltas:
        out.extend(chunker.feed(delta))
    out.extend(chunker.flush())
    return out


def _reference(strategy: ChunkingStrategy, full: str) -> list[str]:
    """The one-shot result: split the whole text, then flush the remainder."""
    return strategy.split(full) + strategy.flush(full)


# Texts covering the delta-seam risks called out in #1440: sentence `.`+space,
# word partial-word, and the greedy paragraph `\n{2,}` boundary.
_INVARIANCE_TEXTS = {
    "sentence": [
        "Hello world. Goodbye now. Bye.",
        "First. Second. Third",
        "One. Two.  Three.   Four",
        '  Leading. She said "hi." Done.',
        "No boundary here",
        "Ends with terminator.",
        "",
    ],
    "word": [
        "foo bar baz qux",
        "one  two   three",
        "  spaced  out  ",
        "trailing fragment",
        "single",
        "",
    ],
    "paragraph": [
        "A\n\n\nB",
        "Para one\n\nPara two\n\nPara three",
        "x\n\ny\n\n\nz\n\n",
        "line\nstill same\n\nnext",
        "one\n\n\n\n\ntwo",
        "no break at all",
        "",
    ],
}

_STRATEGIES: dict[str, Callable[[], ChunkingStrategy]] = {
    "sentence": SentenceChunking,
    "word": WordChunking,
    "paragraph": ParagraphChunking,
}


def _all_texts() -> list[tuple[str, str]]:
    return [(name, text) for name, texts in _INVARIANCE_TEXTS.items() for text in texts]


@pytest.mark.parametrize(("name", "text"), _all_texts())
def test_chunker_delta_invariance_single_delta(name: str, text: str) -> None:
    """Feeding the whole text as one delta matches one-shot split+flush."""
    strategy_cls = _STRATEGIES[name]
    assert _run_chunker(strategy_cls(), [text]) == _reference(strategy_cls(), text)


@pytest.mark.parametrize(("name", "text"), _all_texts())
def test_chunker_delta_invariance_per_character(name: str, text: str) -> None:
    """Feeding one character per delta matches one-shot split+flush."""
    strategy_cls = _STRATEGIES[name]
    assert _run_chunker(strategy_cls(), list(text)) == _reference(strategy_cls(), text)


@pytest.mark.parametrize(("name", "text"), _all_texts())
def test_chunker_delta_invariance_random_splits(name: str, text: str) -> None:
    """Any random slicing into deltas matches one-shot split+flush.

    This is the core delta-invariance property: the boundary a chunk sits on may
    land anywhere relative to the delta seams, so the result must not depend on
    how the stream was chopped up.
    """
    strategy_cls = _STRATEGIES[name]
    reference = _reference(strategy_cls(), text)
    rng = random.Random(f"{name}:{text}")  # deterministic per case
    for _ in range(50):
        if len(text) <= 1:
            break
        k = rng.randint(1, min(6, len(text) - 1))
        cuts = sorted(set(rng.sample(range(1, len(text)), k)))
        deltas, prev = [], 0
        for cut in cuts:
            deltas.append(text[prev:cut])
            prev = cut
        deltas.append(text[prev:])
        assert _run_chunker(strategy_cls(), deltas) == reference, (
            f"delta-invariance broke for {name} on {text!r} sliced as {deltas!r}"
        )


def test_chunker_sentence_seam_period_then_space() -> None:
    """The sentence terminator and its following space arriving in separate deltas."""
    assert _run_chunker(SentenceChunking(), ["One two.", " Three four. "]) == [
        "One two.",
        "Three four.",
    ]


def test_chunker_word_seam_partial_word() -> None:
    """A word split across deltas is held until its trailing space arrives."""
    chunker = Chunker(WordChunking())
    assert chunker.feed("hel") == []  # codespell:ignore
    assert chunker.feed("lo wor") == ["hello"]
    assert chunker.feed("ld ") == ["world"]
    assert chunker.flush() == []


def test_chunker_paragraph_seam_split_newlines() -> None:
    """A `\\n\\n` boundary whose newlines arrive in separate deltas is not split early."""
    chunker = Chunker(ParagraphChunking())
    assert chunker.feed("Para one\n") == []
    # A lone second newline completes the boundary; the paragraph is emitted.
    assert chunker.feed("\nPara two") == ["Para one"]
    assert chunker.flush() == ["Para two"]


def test_chunker_flush_without_feed() -> None:
    """flush() on an unfed Chunker returns nothing."""
    assert Chunker(SentenceChunking()).flush() == []


def test_chunker_holds_only_pending_fragment() -> None:
    """The Chunker's state is the pending fragment, not the whole stream."""
    chunker = Chunker(WordChunking())
    chunker.feed("alpha beta gamma ")
    assert "alpha" not in chunker._pending
    chunker.feed("delta")
    # Only the un-terminated trailing word is held, never already-emitted content.
    assert "delta" in chunker._pending
    assert "alpha" not in chunker._pending
    assert len(chunker._pending) < len("alpha beta gamma delta")


def test_chunker_rejects_mutating_strategy() -> None:
    """feed() raises if split() returns a chunk that is not a verbatim substring."""

    class MutatingChunking(ChunkingStrategy):
        def split(self, text: str) -> list[str]:
            # Normalizes whitespace inside the chunk, so the returned text is no
            # longer a substring of the input.
            return [" ".join(text.split())] if text.strip() else []

    with pytest.raises(ValueError, match="verbatim substring"):
        Chunker(MutatingChunking()).feed("one   two")
