# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the aLoRA io.yaml invocation-sequence repair (issue #1679).

The repair under test (`_alora_invocation_repair`) handles published adapters
whose io.yaml instruction text does not tokenise to the invocation sequence
declared in the adapter's own config — specifically the `requirement-check`
aLoRA, where the instruction starts `<requirements>:` but the declared
sequence is `<requirements>` (no colon) and the Granite tokeniser merges `>:`
into a single token, so the sequence can never appear.

The fake tokenizer below reproduces the real token shape: `<`, `requirements`
and `>` are separate tokens (the declared `[27, 71226, 29]` run), and `>`
merges with a following `:` (or `=`) into one token, so the broken text
shares the first two tokens of the run and differs only at the last.
"""

# Standard
import json
import logging
import re
from collections.abc import Sequence
from unittest.mock import mock_open, patch

# Third Party
import pytest

pytest.importorskip(
    "transformers", reason="transformers not installed — install mellea[hf]"
)
torch = pytest.importorskip("torch", reason="torch not installed — install mellea[hf]")

# First Party
from mellea.backends.adapters import AdapterType, IntrinsicAdapter
from mellea.backends.adapters._alora_repair import (
    _alora_invocation_repair,
    _token_sequence_present,
)
from mellea.backends.adapters._core import Adapter, Identity, LocalFileBinding
from mellea.backends.adapters.io_contracts import get_io_contract
from mellea.backends.huggingface import LocalHFBackend
from test.backends.test_huggingface_unit import _make_backend

_TOKEN_RE = re.compile(r"<|>:|>=|>|[^\s<>]+|\s+")


class _MergingTokenizer:
    """Tokenizer that splits `<`, `>` and words, merging `>:` and `>=` like Granite's BPE."""

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

    def test_colon_present_but_removal_does_not_restore_returns_none(self):
        """The colon form occurs, but dropping it still does not yield the
        declared run (here `<requirements>:=` -> `<requirements>=`, where `>=`
        merges into one token): the third check refuses rather than rewrite
        blindly."""
        tok = _tokenizer()
        instruction = "<requirements>:= {requirement}\nEvaluate."
        assert (
            _alora_invocation_repair(tok, instruction, _invocation_no_colon()) is None
        )


_BROKEN_CONFIG = {"instruction": "<requirements>: {requirement}\nEvaluate."}
_REPAIRED_INSTRUCTION = "<requirements> {requirement}\nEvaluate."


def _stub_backend(tokenizer: _MergingTokenizer) -> LocalHFBackend:
    """`_make_backend` (mock weights, no download) with the fake merging
    tokenizer swapped in for the repair to use."""
    backend = _make_backend()
    backend._tokenizer = tokenizer  # type: ignore[assignment]
    return backend


class TestRepairWiring:
    """`add_adapter` must apply the repair on both registration paths: the
    deprecated `IntrinsicAdapter` shim (weights load per generate call, after
    the rewriter has read the config) and the composed `Adapter`."""

    def test_intrinsic_adapter_shim_path_repairs_config(self, tmp_path, caplog):
        tok = _tokenizer()
        backend = _stub_backend(tok)
        (tmp_path / "adapter_config.json").write_text(
            json.dumps({"alora_invocation_tokens": _invocation_no_colon()})
        )
        caller_config = dict(_BROKEN_CONFIG)
        with pytest.warns(DeprecationWarning):
            adapter = IntrinsicAdapter(
                "requirement-check",
                adapter_type=AdapterType.ALORA,
                config_dict=caller_config,
                base_model_name=backend.base_model_name,
            )
        adapter.get_local_hf_path = lambda base_model_name: str(tmp_path)  # type: ignore[method-assign]

        with caplog.at_level(logging.WARNING, logger="mellea"):
            backend.add_adapter(adapter)

        assert adapter.config["instruction"] == _REPAIRED_INSTRUCTION
        assert caller_config == _BROKEN_CONFIG, "caller's dict must not be mutated"
        assert any(
            "'requirement-check_alora'" in r.message and "#1679" in r.message
            for r in caplog.records
        )
        _, config = backend._intrinsic_adapter_name_and_config(adapter)
        assert config["instruction"] == _REPAIRED_INSTRUCTION

    def test_intrinsic_adapter_shim_lora_left_untouched(self, tmp_path):
        backend = _stub_backend(_tokenizer())
        (tmp_path / "adapter_config.json").write_text(json.dumps({"r": 8}))
        with pytest.warns(DeprecationWarning):
            adapter = IntrinsicAdapter(
                "requirement-check",
                adapter_type=AdapterType.LORA,
                config_dict=dict(_BROKEN_CONFIG),
                base_model_name=backend.base_model_name,
            )
        adapter.get_local_hf_path = lambda base_model_name: str(tmp_path)  # type: ignore[method-assign]

        backend.add_adapter(adapter)

        assert adapter.config == _BROKEN_CONFIG

    def test_composed_adapter_path_repairs_config(self, tmp_path):
        tok = _tokenizer()
        backend = _stub_backend(tok)
        key = "requirement-check_alora"
        (tmp_path / "adapter_config.json").write_text(
            json.dumps({"alora_invocation_tokens": _invocation_no_colon()})
        )
        binding = LocalFileBinding(
            name="requirement-check",
            adapter_type=AdapterType.ALORA,
            repo_id="fake/repo",
        )
        binding.get_local_hf_path = lambda base_model_name: str(tmp_path)  # type: ignore[method-assign]
        composed = Adapter(
            identity=Identity(
                name="requirement-check",
                adapter_type="alora",
                capability="requirement_check",
            ),
            io_contract=get_io_contract("requirement-check"),
            weights=binding,
        )
        with (
            patch(
                "mellea.formatters.granite.intrinsics.obtain_io_yaml",
                return_value="/fake/adapter.yaml",
            ),
            patch("builtins.open", mock_open(read_data="key: value")),
            patch("yaml.safe_load", return_value=dict(_BROKEN_CONFIG)),
        ):
            backend.add_adapter(composed)

        assert (
            backend._composed_adapter_configs[key]["instruction"]
            == _REPAIRED_INSTRUCTION
        )
