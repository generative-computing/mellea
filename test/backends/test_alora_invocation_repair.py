# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the aLoRA io.yaml invocation-sequence repair (issue #1679).

The repair under test (`_alora_invocation_repair`) handles published adapters
whose io.yaml instruction text does not tokenise to the invocation sequence
declared in the adapter's own config — specifically the `requirement-check`
aLoRA, where the instruction starts `<requirements>:` but the declared
sequence is `<requirements>` (no colon) and the Granite tokeniser merges `>:`
into a single token, so the sequence can never appear.

The fake tokenizer below mimics exactly that BPE quirk at word granularity:
`>:` is one token, a standalone `>` is a different one.
"""

# Standard
import re
from collections.abc import Sequence

# Third Party
import pytest

pytest.importorskip(
    "transformers", reason="transformers not installed — install mellea[hf]"
)

# First Party
from mellea.backends.huggingface import (
    _alora_invocation_repair,
    _token_sequence_present,
)

_TOKEN_RE = re.compile(r">:|\S+|\s+")


class _MergingTokenizer:
    """Word-level tokenizer where `>:` merges into one token, like the Granite BPE quirk."""

    def __init__(self, seed_text: str):
        self._ids: dict[str, int] = {}
        self._by_id: dict[int, str] = {}
        for m in _TOKEN_RE.finditer(seed_text):
            self._register(m.group())

    def _register(self, token: str) -> None:
        if token not in self._ids:
            self._ids[token] = len(self._ids) + 1
            self._by_id[self._ids[token]] = token

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        out = []
        for m in _TOKEN_RE.finditer(text):
            self._register(m.group())
            out.append(self._ids[m.group()])
        return out

    def decode(self, ids: Sequence[int], skip_special_tokens: bool = True) -> str:
        return "".join(self._by_id[i] for i in ids)


def _tokenizer() -> _MergingTokenizer:
    # Seed so both "<requirements>" and "<requirements>:" are encodable.
    return _MergingTokenizer("<requirements> <requirements>: x y")


def _invocation_no_colon() -> list[int]:
    return _tokenizer().encode("<requirements>")


def _invocation_with_colon() -> list[int]:
    return _tokenizer().encode("<requirements>:")


class TestTokenSequencePresent:
    def test_present(self):
        tok = _tokenizer()
        assert _token_sequence_present(
            tok, "a <requirements> b", tok.encode("<requirements>")
        )

    def test_absent_when_colon_merges(self):
        tok = _tokenizer()
        assert not _token_sequence_present(
            tok, "a <requirements>: b", tok.encode("<requirements>")
        )

    def test_empty_ids(self):
        assert not _token_sequence_present(_tokenizer(), "anything", [])


class TestAloraInvocationRepair:
    def test_healthy_instruction_unchanged(self):
        tok = _tokenizer()
        instruction = "<requirements> {requirement}\nEvaluate."
        assert (
            _alora_invocation_repair(tok, instruction, _invocation_no_colon()) is None
        )

    def test_repairs_colon_mismatch(self):
        tok = _tokenizer()
        instruction = "<requirements>: {requirement}\nEvaluate."
        repaired = _alora_invocation_repair(tok, instruction, _invocation_no_colon())
        assert repaired == "<requirements> {requirement}\nEvaluate."
        assert _token_sequence_present(tok, repaired, _invocation_no_colon())

    def test_self_terminates_when_publisher_fixes_tokens_instead(self):
        """If the publisher re-declares the invocation tokens to match the
        existing `<requirements>:` text (the other valid fix direction), the
        file is healthy and must be left untouched."""
        tok = _tokenizer()
        instruction = "<requirements>: {requirement}\nEvaluate."
        assert (
            _alora_invocation_repair(tok, instruction, _invocation_with_colon()) is None
        )

    def test_invocation_not_from_instruction_is_untouched(self):
        """Adapters whose invocation sequence is supplied by the chat template
        (e.g. role markers) never match instruction text; no repair applies."""
        tok = _tokenizer()
        instruction = "<requirements>: {requirement}\nEvaluate."
        assert (
            _alora_invocation_repair(tok, instruction, tok.encode("<|start_of_role|>"))
            is None
        )

    def test_unrepairable_mismatch_returns_none(self):
        """Declared sequence is `<requirements>:` (with colon) but the text
        only ever carries `<requirements>` with the colon elsewhere: the
        `broken`-form (`<requirements>:` + `:`) appears nowhere, so no repair
        applies and the text is left untouched for an upstream fix."""
        tok = _tokenizer()
        instruction = "x <requirements> y: {requirement}\nEvaluate."
        assert (
            _alora_invocation_repair(tok, instruction, _invocation_with_colon()) is None
        )
