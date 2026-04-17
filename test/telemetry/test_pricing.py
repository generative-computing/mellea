"""Unit tests for the pricing registry."""

import importlib
import json

import pytest


@pytest.fixture
def custom_pricing(tmp_path, monkeypatch):
    """Load a custom pricing dict into the module singleton; restore on teardown."""
    import mellea.telemetry.pricing

    def _load(pricing_dict: dict) -> None:
        pricing_file = tmp_path / "pricing.json"
        pricing_file.write_text(json.dumps(pricing_dict))
        monkeypatch.setenv("MELLEA_PRICING_FILE", str(pricing_file))
        importlib.reload(mellea.telemetry.pricing)

    yield _load

    monkeypatch.delenv("MELLEA_PRICING_FILE", raising=False)
    importlib.reload(mellea.telemetry.pricing)


@pytest.fixture
def fresh_registry(monkeypatch):
    """Return a fresh PricingRegistry with no MELLEA_PRICING_FILE set."""
    monkeypatch.delenv("MELLEA_PRICING_FILE", raising=False)
    from mellea.telemetry.pricing import PricingRegistry

    return PricingRegistry()


def test_compute_cost_known_model(fresh_registry):
    """Known model returns a non-None float cost."""
    cost = fresh_registry.compute_cost("gpt-5.4", input_tokens=1000, output_tokens=500)
    assert cost is not None
    # gpt-5.4: 1000 * 0.0025/1k + 500 * 0.015/1k = 0.0025 + 0.0075 = 0.010
    assert abs(cost - 0.010) < 1e-9


def test_compute_cost_unknown_model(fresh_registry, caplog):
    """Unknown model returns None and logs a warning."""
    import logging

    with caplog.at_level(logging.WARNING, logger="mellea.telemetry.pricing"):
        cost = fresh_registry.compute_cost("nonexistent-model-xyz", 100, 50)

    assert cost is None
    assert any("nonexistent-model-xyz" in record.message for record in caplog.records)


def test_compute_cost_none_tokens(fresh_registry):
    """None tokens are treated as zero without raising."""
    cost = fresh_registry.compute_cost("gpt-5.4", input_tokens=None, output_tokens=None)
    assert cost == 0.0


def test_compute_cost_zero_tokens(fresh_registry):
    """Zero tokens produce zero cost."""
    cost = fresh_registry.compute_cost(
        "claude-sonnet-4-6", input_tokens=0, output_tokens=0
    )
    assert cost == 0.0


def test_custom_pricing_override(custom_pricing):
    """MELLEA_PRICING_FILE overrides built-in prices."""
    custom_pricing({"my-custom-model": {"input_per_1k": 0.001, "output_per_1k": 0.002}})

    from mellea.telemetry.pricing import compute_cost

    cost = compute_cost("my-custom-model", 1000, 1000)
    assert cost is not None
    assert abs(cost - 0.003) < 1e-9


def test_custom_pricing_overrides_builtin(custom_pricing):
    """Custom file can override built-in prices for existing models."""
    custom_pricing({"gpt-5.4": {"input_per_1k": 0.999, "output_per_1k": 0.999}})

    from mellea.telemetry.pricing import compute_cost

    cost = compute_cost("gpt-5.4", 1000, 0)
    assert cost is not None
    assert abs(cost - 0.999) < 1e-9


def test_custom_pricing_file_not_found(monkeypatch, caplog):
    """Missing MELLEA_PRICING_FILE logs warning and falls back to built-ins."""
    import logging

    monkeypatch.setenv("MELLEA_PRICING_FILE", "/nonexistent/path/pricing.json")

    from mellea.telemetry.pricing import PricingRegistry

    with caplog.at_level(logging.WARNING, logger="mellea.telemetry.pricing"):
        registry = PricingRegistry()

    assert any("nonexistent" in record.message for record in caplog.records)
    # Built-in pricing still works
    assert registry.compute_cost("gpt-5.4", 1000, 0) is not None


def test_custom_pricing_invalid_json(tmp_path, monkeypatch, caplog):
    """Invalid JSON in MELLEA_PRICING_FILE logs warning and falls back to built-ins."""
    import logging

    bad_file = tmp_path / "bad.json"
    bad_file.write_text("this is not json {{{")

    monkeypatch.setenv("MELLEA_PRICING_FILE", str(bad_file))

    from mellea.telemetry.pricing import PricingRegistry

    with caplog.at_level(logging.WARNING, logger="mellea.telemetry.pricing"):
        registry = PricingRegistry()

    assert any("Invalid JSON" in record.message for record in caplog.records)
    # Built-in pricing still works
    assert registry.compute_cost("claude-sonnet-4-6", 1000, 0) is not None
