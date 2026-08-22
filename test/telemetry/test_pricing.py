# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the litellm-backed pricing module."""

import builtins
import json
import logging
import sys
from unittest.mock import MagicMock, patch

import pytest

from mellea.telemetry import pricing
from test.telemetry.conftest import reset_pricing_state


@pytest.fixture()
def restore_pricing(monkeypatch):
    """Restore pricing module state after each test."""
    yield
    monkeypatch.delenv("MELLEA_PRICING_ENABLED", raising=False)
    monkeypatch.delenv("MELLEA_PRICING_FILE", raising=False)
    reset_pricing_state()


@pytest.fixture()
def mock_litellm_pricing():
    """Patch pricing into an enabled state with a mock litellm."""
    mock = MagicMock()
    mock.cost_per_token.return_value = (0.001, 0.002)
    with (
        patch("mellea.telemetry.pricing._PRICING_ENABLED", True),
        patch.dict(sys.modules, {"litellm": mock}),
        patch("mellea.telemetry.pricing._warned_models", set()),
    ):
        yield mock


@pytest.fixture()
def unimportable_litellm(monkeypatch):
    """Pricing enabled, but `import litellm` fails.

    A `None` entry in `sys.modules` is CPython's "blocked module" sentinel:
    `import litellm` raises ImportError while the distribution stays on disk,
    which is exactly the skewed-install case `find_spec` cannot detect.
    """
    monkeypatch.setattr(pricing, "_LITELLM_AVAILABLE", True)
    monkeypatch.setattr(pricing, "_PRICING_ENABLED", True)
    monkeypatch.setattr(pricing, "_warned_models", set())
    monkeypatch.setitem(sys.modules, "litellm", None)
    yield
    # `_import_litellm` clears the module globals. Undo the patches first, then
    # re-run setup, so teardown ordering against `restore_pricing` (which also
    # depends on monkeypatch, and so is torn down last) is deterministic.
    monkeypatch.undo()
    reset_pricing_state()


# ---------------------------------------------------------------------------
# Tri-state flag — module init logic
# ---------------------------------------------------------------------------


def test_tri_state_false_disables_explicitly(monkeypatch, restore_pricing):
    """MELLEA_PRICING_ENABLED=false disables pricing regardless of litellm."""
    monkeypatch.setenv("MELLEA_PRICING_ENABLED", "false")
    monkeypatch.setattr(pricing, "_LITELLM_AVAILABLE", True)
    reset_pricing_state()
    assert pricing._PRICING_ENABLED is False


def test_tri_state_true_with_litellm_enables(monkeypatch, restore_pricing):
    """MELLEA_PRICING_ENABLED=true with litellm present enables pricing."""
    monkeypatch.setenv("MELLEA_PRICING_ENABLED", "true")
    monkeypatch.setattr(pricing, "_LITELLM_AVAILABLE", True)
    reset_pricing_state()
    assert pricing._PRICING_ENABLED is True


def test_tri_state_true_without_litellm_warns_and_disables(
    monkeypatch, restore_pricing
):
    """MELLEA_PRICING_ENABLED=true without litellm emits a warning and disables."""
    monkeypatch.setenv("MELLEA_PRICING_ENABLED", "true")
    monkeypatch.setattr(pricing, "_LITELLM_AVAILABLE", False)
    with pytest.warns(UserWarning, match="litellm is not installed"):
        reset_pricing_state()
    assert pricing._PRICING_ENABLED is False


def test_tri_state_unset_with_litellm_auto_enables(monkeypatch, restore_pricing):
    """Unset MELLEA_PRICING_ENABLED with litellm present auto-enables pricing."""
    monkeypatch.delenv("MELLEA_PRICING_ENABLED", raising=False)
    monkeypatch.setattr(pricing, "_LITELLM_AVAILABLE", True)
    reset_pricing_state()
    assert pricing._PRICING_ENABLED is True


def test_tri_state_unset_without_litellm_silent_disable(
    monkeypatch, restore_pricing, recwarn
):
    """Unset MELLEA_PRICING_ENABLED without litellm silently disables (no warning)."""
    monkeypatch.delenv("MELLEA_PRICING_ENABLED", raising=False)
    monkeypatch.setattr(pricing, "_LITELLM_AVAILABLE", False)
    reset_pricing_state()
    assert pricing._PRICING_ENABLED is False
    assert not any("litellm is not installed" in str(w.message) for w in recwarn.list)


# ---------------------------------------------------------------------------
# MELLEA_PRICING_FILE — _register_custom_pricing
# ---------------------------------------------------------------------------


def test_register_custom_pricing_calls_register_model(tmp_path):
    """_register_custom_pricing passes loaded JSON to litellm.register_model."""
    data = {"my-model": {"input_cost_per_token": 1e-6, "output_cost_per_token": 2e-6}}
    pricing_file = tmp_path / "pricing.json"
    pricing_file.write_text(json.dumps(data))

    mock_litellm = MagicMock()
    with patch.dict(sys.modules, {"litellm": mock_litellm}):
        pricing._register_custom_pricing(str(pricing_file))

    mock_litellm.register_model.assert_called_once_with(data)


def test_register_custom_pricing_file_not_found(caplog):
    """Missing file path logs a warning without raising."""
    import logging

    with caplog.at_level(logging.WARNING, logger="mellea.telemetry.pricing"):
        pricing._register_custom_pricing("/nonexistent/pricing.json")

    assert any("nonexistent" in r.message for r in caplog.records)


def test_register_custom_pricing_invalid_json(tmp_path, caplog):
    """Invalid JSON logs a warning without raising."""
    import logging

    bad = tmp_path / "bad.json"
    bad.write_text("not json {{{")

    with caplog.at_level(logging.WARNING, logger="mellea.telemetry.pricing"):
        pricing._register_custom_pricing(str(bad))

    assert any("Invalid JSON" in r.message for r in caplog.records)


def test_register_custom_pricing_not_a_dict(tmp_path, caplog):
    """Non-dict JSON logs a warning without raising."""
    import logging

    list_file = tmp_path / "list.json"
    list_file.write_text(json.dumps([1, 2, 3]))

    with caplog.at_level(logging.WARNING, logger="mellea.telemetry.pricing"):
        pricing._register_custom_pricing(str(list_file))

    assert any("JSON object" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# compute_cost — delegation to litellm
# ---------------------------------------------------------------------------


def test_compute_cost_delegates_to_litellm(mock_litellm_pricing):
    """compute_cost passes correct args to litellm.cost_per_token."""
    mock_litellm_pricing.cost_per_token.return_value = (0.0025, 0.0075)
    cost = pricing.compute_cost("gpt-5.4", "openai", 1000, 500)

    assert cost == pytest.approx(0.010)
    mock_litellm_pricing.cost_per_token.assert_called_once_with(
        model="gpt-5.4",
        custom_llm_provider="openai",
        prompt_tokens=1000,
        completion_tokens=500,
        cache_read_input_tokens=0,
        cache_creation_input_tokens=0,
    )


def test_compute_cost_passes_full_prompt_tokens_to_litellm(mock_litellm_pricing):
    """prompt_tokens is forwarded as-is; litellm handles cached-token deduction internally."""
    pricing.compute_cost(
        "gpt-5.4",
        "openai",
        prompt_tokens=100,
        completion_tokens=20,
        cached_tokens=50,
        cache_creation_tokens=10,
    )

    _, kwargs = mock_litellm_pricing.cost_per_token.call_args
    assert kwargs["prompt_tokens"] == 100
    assert kwargs["cache_read_input_tokens"] == 50
    assert kwargs["cache_creation_input_tokens"] == 10


def test_compute_cost_none_tokens_treated_as_zero(mock_litellm_pricing):
    """None token and provider values are treated as zero/None."""
    pricing.compute_cost("gpt-5.4", None, None, None)

    _, kwargs = mock_litellm_pricing.cost_per_token.call_args
    assert kwargs["prompt_tokens"] == 0
    assert kwargs["completion_tokens"] == 0
    assert kwargs["custom_llm_provider"] is None


def test_compute_cost_disabled_returns_none():
    """compute_cost returns None immediately when pricing is disabled."""
    mock_litellm = MagicMock()
    with (
        patch("mellea.telemetry.pricing._PRICING_ENABLED", False),
        patch.dict(sys.modules, {"litellm": mock_litellm}),
    ):
        cost = pricing.compute_cost("gpt-5.4", None, 100, 50)

    assert cost is None
    mock_litellm.cost_per_token.assert_not_called()


def test_compute_cost_unknown_model_warns_once(mock_litellm_pricing, caplog):
    """Exception from litellm → None + log warning on first failure only."""
    mock_litellm_pricing.cost_per_token.side_effect = ValueError("No model data")
    with caplog.at_level(logging.WARNING, logger="mellea.telemetry.pricing"):
        cost = pricing.compute_cost("unknown-model-xyz", None, 100, 50)
        assert cost is None
        assert any("unknown-model-xyz" in r.message for r in caplog.records)
        caplog.clear()
        pricing.compute_cost("unknown-model-xyz", None, 100, 50)

    assert not any("unknown-model-xyz" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# litellm discoverable via find_spec but not importable
# ---------------------------------------------------------------------------


def test_compute_cost_unimportable_litellm_returns_none(unimportable_litellm):
    """A litellm that fails to import yields None instead of propagating."""
    assert pricing.compute_cost("gpt-5.4", "openai", 100, 50) is None


def test_compute_cost_unimportable_litellm_disables_pricing(unimportable_litellm):
    """A failed litellm import latches pricing off for the rest of the process."""
    pricing.compute_cost("gpt-5.4", "openai", 100, 50)

    assert pricing.is_pricing_enabled() is False


def test_compute_cost_unimportable_litellm_warns_once(unimportable_litellm, caplog):
    """The import failure is logged on the first call only."""
    with caplog.at_level(logging.WARNING, logger="mellea.telemetry.pricing"):
        pricing.compute_cost("gpt-5.4", "openai", 100, 50)
        assert any("failed to import" in r.message for r in caplog.records)
        caplog.clear()
        pricing.compute_cost("gpt-5.4", "openai", 100, 50)

    assert not any("failed to import" in r.message for r in caplog.records)


def test_compute_cost_non_import_error_from_litellm_returns_none(monkeypatch):
    """A skewed install that raises AttributeError on import is absorbed too."""
    real_import = builtins.__import__

    def failing_import(name, *args, **kwargs):
        if name == "litellm":
            raise AttributeError("partially-installed dependency")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(pricing, "_LITELLM_AVAILABLE", True)
    monkeypatch.setattr(pricing, "_PRICING_ENABLED", True)
    monkeypatch.setattr(builtins, "__import__", failing_import)

    assert pricing.compute_cost("gpt-5.4", "openai", 100, 50) is None


def test_register_custom_pricing_unimportable_litellm_does_not_raise(
    tmp_path, unimportable_litellm
):
    """A valid pricing file plus an unimportable litellm is a no-op, not an error."""
    pricing_file = tmp_path / "pricing.json"
    pricing_file.write_text(json.dumps({"m": {"input_cost_per_token": 1e-6}}))

    pricing._register_custom_pricing(str(pricing_file))  # must not raise

    assert pricing.is_pricing_enabled() is False


def test_setup_pricing_unimportable_litellm_does_not_raise(
    tmp_path, monkeypatch, unimportable_litellm
):
    """`import mellea` survives MELLEA_PRICING_FILE plus an unimportable litellm."""
    pricing_file = tmp_path / "pricing.json"
    pricing_file.write_text(json.dumps({"m": {"input_cost_per_token": 1e-6}}))
    monkeypatch.setenv("MELLEA_PRICING_ENABLED", "true")
    monkeypatch.setenv("MELLEA_PRICING_FILE", str(pricing_file))

    pricing._setup_pricing()  # must not raise

    assert pricing.is_pricing_enabled() is False
