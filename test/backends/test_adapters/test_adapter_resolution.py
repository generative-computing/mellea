# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the availability-aware fallback chain walk.

Exercises `_register_available_composed_adapter` in isolation: the
`obtain_io_yaml` / `resolve_prompt_io_yaml` boundary is patched and `add_adapter`
is mocked, so each test asserts exactly what the walk registers for a given rung
of the chain, with no network, weights lifecycle, or model load.
"""

import logging
from unittest.mock import MagicMock, patch

import pytest
import yaml

from mellea.backends.adapters._core import LocalFileBinding, PromptBinding
from mellea.backends.adapters.adapter import _register_available_composed_adapter
from mellea.backends.adapters.catalog import AdapterType, fetch_intrinsic_metadata
from mellea.formatters.granite.intrinsics import AdapterNotFoundError

_ANSWERABILITY_META = fetch_intrinsic_metadata("answerability")
_BASE_MODEL = "granite-4.2-8b"
_INTRINSICS_MOD = "mellea.formatters.granite.intrinsics"


def _write_io_yaml(tmp_path):
    p = tmp_path / "io.yaml"
    p.write_text(yaml.safe_dump({"instruction": "Answer the question."}))
    return p


def _obtain_io_yaml_stub(*, alora_found: bool, lora_found: bool):
    """Build a stub `obtain_io_yaml`; `alora_found`/`lora_found` pick which types resolve.

    A found type returns a path; the rest raise `ValueError` — the not-found
    signal the walk advances on.
    """

    def stub(*args, **kwargs):
        adapter_type = kwargs.get("adapter_type", "lora")
        if adapter_type == "alora" and alora_found:
            return "/cache/alora/io.yaml"
        if adapter_type == "lora" and lora_found:
            return "/cache/lora/io.yaml"
        raise AdapterNotFoundError("not found")

    return stub


def test_alora_found_registers_local_file_binding():
    with (
        patch(
            f"{_INTRINSICS_MOD}.obtain_io_yaml",
            side_effect=_obtain_io_yaml_stub(alora_found=True, lora_found=True),
        ),
        patch(f"{_INTRINSICS_MOD}.resolve_prompt_io_yaml") as mock_prompt,
    ):
        self_mock = MagicMock()
        _register_available_composed_adapter(
            self_mock, "answerability", _BASE_MODEL, _ANSWERABILITY_META
        )
    self_mock.add_adapter.assert_called_once()
    adapter = self_mock.add_adapter.call_args.args[0]
    assert isinstance(adapter.weights, LocalFileBinding)
    assert adapter.weights.adapter_type is AdapterType.ALORA
    assert adapter.identity.adapter_type == "alora"
    # aLoRA won, so the prompt rungs are never probed.
    mock_prompt.assert_not_called()


def test_lora_found_when_no_alora():
    with (
        patch(
            f"{_INTRINSICS_MOD}.obtain_io_yaml",
            side_effect=_obtain_io_yaml_stub(alora_found=False, lora_found=True),
        ),
        patch(f"{_INTRINSICS_MOD}.resolve_prompt_io_yaml") as mock_prompt,
    ):
        self_mock = MagicMock()
        _register_available_composed_adapter(
            self_mock, "answerability", _BASE_MODEL, _ANSWERABILITY_META
        )
    self_mock.add_adapter.assert_called_once()
    adapter = self_mock.add_adapter.call_args.args[0]
    assert isinstance(adapter.weights, LocalFileBinding)
    assert adapter.weights.adapter_type is AdapterType.LORA
    mock_prompt.assert_not_called()


def test_model_specific_prompt_when_no_trained_adapter(tmp_path, caplog):
    io_yaml = _write_io_yaml(tmp_path)
    with (
        patch(
            f"{_INTRINSICS_MOD}.obtain_io_yaml",
            side_effect=_obtain_io_yaml_stub(alora_found=False, lora_found=False),
        ),
        patch(
            f"{_INTRINSICS_MOD}.resolve_prompt_io_yaml", return_value=(io_yaml, False)
        ),
    ):
        with caplog.at_level(logging.WARNING, logger="mellea"):
            self_mock = MagicMock()
            _register_available_composed_adapter(
                self_mock, "answerability", _BASE_MODEL, _ANSWERABILITY_META
            )
    self_mock.add_adapter.assert_called_once()
    adapter = self_mock.add_adapter.call_args.args[0]
    assert isinstance(adapter.weights, PromptBinding)
    assert adapter.weights.target_model_name == _BASE_MODEL
    assert adapter.identity.adapter_type == "prompt"
    # The parsed io.yaml is handed to add_adapter as config=.
    assert self_mock.add_adapter.call_args.kwargs["config"] == {
        "instruction": "Answer the question."
    }
    # No "any" warning for a model-specific prompt.
    assert not any("model-agnostic 'any'" in r.message for r in caplog.records)


def test_any_prompt_warns_and_registers_any_binding(tmp_path, caplog):
    io_yaml = _write_io_yaml(tmp_path)
    with (
        patch(
            f"{_INTRINSICS_MOD}.obtain_io_yaml",
            side_effect=_obtain_io_yaml_stub(alora_found=False, lora_found=False),
        ),
        patch(
            f"{_INTRINSICS_MOD}.resolve_prompt_io_yaml", return_value=(io_yaml, True)
        ),
    ):
        with caplog.at_level(logging.WARNING, logger="mellea"):
            self_mock = MagicMock()
            _register_available_composed_adapter(
                self_mock, "answerability", _BASE_MODEL, _ANSWERABILITY_META
            )
    self_mock.add_adapter.assert_called_once()
    adapter = self_mock.add_adapter.call_args.args[0]
    assert isinstance(adapter.weights, PromptBinding)
    assert adapter.weights.target_model_name == "any"
    assert any("model-agnostic 'any'" in r.message for r in caplog.records)


def test_hard_error_aborts_without_fallthrough():
    # A non-ValueError on the first (aLoRA) probe must propagate, not advance to
    # LoRA / prompt rungs.
    def fail(*args, **kwargs):
        raise RuntimeError("network down")

    with (
        patch(f"{_INTRINSICS_MOD}.obtain_io_yaml", side_effect=fail),
        patch(f"{_INTRINSICS_MOD}.resolve_prompt_io_yaml") as mock_prompt,
    ):
        self_mock = MagicMock()
        with pytest.raises(RuntimeError):
            _register_available_composed_adapter(
                self_mock, "answerability", _BASE_MODEL, _ANSWERABILITY_META
            )
    mock_prompt.assert_not_called()
    self_mock.add_adapter.assert_not_called()


def test_nothing_found_raises_value_error():
    with (
        patch(
            f"{_INTRINSICS_MOD}.obtain_io_yaml",
            side_effect=_obtain_io_yaml_stub(alora_found=False, lora_found=False),
        ),
        patch(
            f"{_INTRINSICS_MOD}.resolve_prompt_io_yaml",
            side_effect=AdapterNotFoundError("not found"),
        ),
    ):
        self_mock = MagicMock()
        with pytest.raises(ValueError, match=r"No adapter .* or prompt fallback"):
            _register_available_composed_adapter(
                self_mock, "answerability", _BASE_MODEL, _ANSWERABILITY_META
            )
    self_mock.add_adapter.assert_not_called()
