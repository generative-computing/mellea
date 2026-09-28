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
import json
import logging
import re
from collections.abc import Sequence
from types import SimpleNamespace
from unittest.mock import MagicMock, mock_open, patch

# Third Party
import pytest

pytest.importorskip(
    "transformers", reason="transformers not installed — install mellea[hf]"
)
torch = pytest.importorskip("torch", reason="torch not installed — install mellea[hf]")

# First Party
from mellea.backends.adapters import AdapterType, IntrinsicAdapter
from mellea.backends.adapters._core import Adapter, Identity, LocalFileBinding
from mellea.backends.adapters.io_contracts import get_io_contract
from mellea.backends.huggingface import (
    LocalHFBackend,
    _alora_invocation_repair,
    _read_alora_invocation_tokens,
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

    def test_colon_present_but_removal_does_not_restore_returns_none(self):
        """The colon form occurs, but dropping it still does not yield the
        declared run (here `<requirements>:x` -> `<requirements>x`, a different
        token): the third check refuses rather than rewrite blindly."""
        tok = _tokenizer()
        instruction = "<requirements>:x {requirement}\nEvaluate."
        assert (
            _alora_invocation_repair(tok, instruction, _invocation_no_colon()) is None
        )


class TestReadAloraInvocationTokens:
    def test_reads_declared_tokens(self, tmp_path):
        (tmp_path / "adapter_config.json").write_text(
            json.dumps({"alora_invocation_tokens": [27, 71226, 29]})
        )
        assert _read_alora_invocation_tokens(str(tmp_path)) == [27, 71226, 29]

    def test_plain_lora_config_returns_none(self, tmp_path):
        (tmp_path / "adapter_config.json").write_text(json.dumps({"r": 8}))
        assert _read_alora_invocation_tokens(str(tmp_path)) is None

    def test_missing_config_file_returns_none(self, tmp_path):
        assert _read_alora_invocation_tokens(str(tmp_path)) is None


_BROKEN_CONFIG = {"instruction": "<requirements>: {requirement}\nEvaluate."}
_REPAIRED_INSTRUCTION = "<requirements> {requirement}\nEvaluate."


def _stub_backend(tokenizer: _MergingTokenizer) -> LocalHFBackend:
    """A real `LocalHFBackend` over mock model/tokenizer (no download), with
    the fake merging tokenizer swapped in for the repair to use."""
    mock_tok = MagicMock(eos_token_id=0, vocab_size=32000)
    mock_tok._tokenizer = MagicMock()
    mock_tok._tokenizer.get_vocab_size.return_value = 32000
    mock_tok.__len__ = MagicMock(return_value=32000)
    with (
        patch("mellea.backends.huggingface.llguidance") as mock_llg,
        patch("mellea.backends.huggingface.set_seed"),
    ):
        mock_llg.hf.from_tokenizer.return_value = MagicMock(vocab_size=32000)
        backend = LocalHFBackend(
            model_id="ibm-granite/granite-4.1-3b",
            custom_config=(mock_tok, MagicMock(vocab_size=32000), torch.device("cpu")),
        )
    backend._tokenizer = tokenizer  # type: ignore[assignment]
    return backend


class TestRepairAloraInstruction:
    def test_repair_returns_repaired_copy_and_warns(self, caplog):
        backend = _stub_backend(_tokenizer())
        config = dict(_BROKEN_CONFIG)
        with caplog.at_level(logging.WARNING, logger="mellea"):
            result = backend._repair_alora_instruction(
                config, _invocation_no_colon(), "requirement-check_alora"
            )
        assert result["instruction"] == _REPAIRED_INSTRUCTION
        assert result is not config
        assert config == _BROKEN_CONFIG, "caller's dict must not be mutated"
        assert any(
            "'requirement-check_alora'" in r.message and "#1679" in r.message
            for r in caplog.records
        )

    def test_healthy_config_returned_unchanged(self, caplog):
        backend = _stub_backend(_tokenizer())
        config = {"instruction": _REPAIRED_INSTRUCTION}
        with caplog.at_level(logging.WARNING, logger="mellea"):
            result = backend._repair_alora_instruction(
                config, _invocation_no_colon(), "requirement-check_alora"
            )
        assert result is config
        assert not caplog.records

    @pytest.mark.parametrize("tokens", [None, []])
    def test_no_declared_invocation_returns_config_unchanged(self, tokens):
        backend = _stub_backend(_tokenizer())
        config = dict(_BROKEN_CONFIG)
        assert backend._repair_alora_instruction(config, tokens, "x_lora") is config

    def test_no_instruction_returns_config_unchanged(self):
        backend = _stub_backend(_tokenizer())
        config = {"parameters": {}}
        assert (
            backend._repair_alora_instruction(config, _invocation_no_colon(), "x")
            is config
        )


class TestRepairWiring:
    """`add_adapter` must apply the repair on both registration paths: the
    deprecated `IntrinsicAdapter` shim (weights load per generate call, after
    the rewriter has read the config) and the composed `Adapter`."""

    def test_intrinsic_adapter_shim_path_repairs_config(self, tmp_path):
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

        backend.add_adapter(adapter)

        assert adapter.config["instruction"] == _REPAIRED_INSTRUCTION
        assert caller_config == _BROKEN_CONFIG, "caller's dict must not be mutated"
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

    def test_composed_adapter_path_repairs_config(self):
        tok = _tokenizer()
        backend = _stub_backend(tok)
        key = "requirement-check_alora"
        backend._model.peft_config = {
            key: SimpleNamespace(alora_invocation_tokens=_invocation_no_colon())
        }
        binding = LocalFileBinding(
            name="requirement-check",
            adapter_type=AdapterType.ALORA,
            repo_id="fake/repo",
        )
        binding.get_local_hf_path = lambda base_model_name: "/fake/path"  # type: ignore[method-assign]
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
