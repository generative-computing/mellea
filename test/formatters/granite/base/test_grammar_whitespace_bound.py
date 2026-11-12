# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Enforcement tests for the bounded llguidance whitespace_pattern.

The mock-based regression tests in `test/backends/test_huggingface_unit.py`
assert that `_LLGUIDANCE_GRAMMAR_DEFAULTS` is *passed* to
`grammar_from_json_schema`; these tests assert that llguidance actually
*enforces* the upper bound. On llguidance < 1.8.0 the JSON grammar's skip
lexeme could re-match at a single position, so a bounded pattern like
`[\\x20\\x0A\\x0D\\x09]{0,N}` silently permitted
unlimited whitespace.

These tests run against real llguidance (no model, no Ollama) using a
byte-level tokenizer: one token per byte plus EOS.
"""

import re

import pytest

pytest.importorskip(
    "llguidance", reason="llguidance not installed — install mellea[hf]"
)

# Third Party
import llguidance
from llguidance import LLMatcher, LLTokenizer, TokenizerWrapper

# First Party
from mellea.formatters.granite.base.util import _LLGUIDANCE_GRAMMAR_DEFAULTS

_WS_MATCH = re.search(
    r"\{0,(\d+)\}$", _LLGUIDANCE_GRAMMAR_DEFAULTS.get("whitespace_pattern") or ""
)
assert _WS_MATCH, "whitespace_pattern must carry a {0,N} upper bound"
_WS_BOUND = int(_WS_MATCH.group(1))
# Probe past the bound so an unenforced (unlimited) pattern is detected.
_PROBE_LIMIT = _WS_BOUND * 10
_ARRAY_SCHEMA = {"type": "array", "items": {"type": "string"}}
_SPACE = 0x20
_NEWLINE = 0x0A


class _ByteTokenizer:
    """Minimal byte-level tokenizer: one token per byte, plus EOS at 256."""

    eos_token_id = 256
    bos_token_id = None
    tokens = [bytes([i]) for i in range(256)] + [b"<eos>"]
    special_token_ids = [256]

    def __call__(self, b: bytes) -> str:
        return b.decode("utf-8", errors="replace")


@pytest.fixture(scope="module")
def ll_tokenizer() -> LLTokenizer:
    return LLTokenizer(TokenizerWrapper(_ByteTokenizer()))


def _token_allowed(matcher: LLMatcher, token_id: int) -> bool:
    bitmask = matcher.compute_bitmask()
    return bool(bitmask[token_id >> 3] & (1 << (token_id & 7)))


def _max_consecutive_whitespace(ll_tokenizer: LLTokenizer, ws_byte: int) -> int:
    """Count whitespace tokens accepted after '[' under the repo grammar defaults."""
    grammar = LLMatcher.grammar_from_json_schema(
        _ARRAY_SCHEMA, overrides=_LLGUIDANCE_GRAMMAR_DEFAULTS
    )
    matcher = LLMatcher(ll_tokenizer, grammar)
    assert matcher.try_consume_tokens([ord("[")]), matcher.get_error()
    count = 0
    while count < _PROBE_LIMIT:
        if not _token_allowed(matcher, ws_byte):
            break
        if not matcher.try_consume_tokens([ws_byte]):
            break
        count += 1
    return count


def _validate_text(ll_tokenizer: LLTokenizer, text: str) -> bool:
    """Check the full text (plus EOS) is accepted under the repo grammar defaults.

    `try_consume_tokens` returns the count of tokens consumed (not a bool), so
    full acceptance is `is_stopped()` after EOS — partial consumption of an
    invalid text leaves the matcher un-stopped.
    """
    grammar = LLMatcher.grammar_from_json_schema(
        _ARRAY_SCHEMA, overrides=_LLGUIDANCE_GRAMMAR_DEFAULTS
    )
    matcher = LLMatcher(ll_tokenizer, grammar)
    matcher.try_consume_tokens([*text.encode("utf-8"), _ByteTokenizer.eos_token_id])
    return matcher.is_stopped() and not matcher.is_error()


class TestWhitespaceBoundEnforced:
    """The upper bound in _LLGUIDANCE_GRAMMAR_DEFAULTS must hold in the mask."""

    def test_space_run_capped(self, ll_tokenizer: LLTokenizer):
        # A run of spaces past the bound must fall out of the acceptance mask,
        # otherwise a model can burn its entire token budget on whitespace.
        assert _max_consecutive_whitespace(ll_tokenizer, _SPACE) == _WS_BOUND

    def test_newline_run_capped(self, ll_tokenizer: LLTokenizer):
        # The runaway incident involved long newline/indent runs, not just spaces.
        assert _max_consecutive_whitespace(ll_tokenizer, _NEWLINE) == _WS_BOUND

    def test_spaced_json_still_accepted(self, ll_tokenizer: LLTokenizer):
        # The bound must not break ordinary spaced JSON (4-space indent, one level).
        assert _validate_text(ll_tokenizer, '["a",\n    "b"]')

    def test_overlong_whitespace_rejected(self, ll_tokenizer: LLTokenizer):
        # One more than the bound between '[' and the first value is rejected.
        assert not _validate_text(ll_tokenizer, '[{}"a"]'.format(" " * (_WS_BOUND + 1)))
