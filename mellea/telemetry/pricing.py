"""LLM pricing registry for cost estimation.

Provides per-request cost estimates based on token usage and model pricing data.
Built-in prices are loaded from ``builtin_pricing.json`` in this package directory.
Custom overrides are loaded from a path set via ``MELLEA_PRICING_FILE``.

Pricing data sources (last verified 2026-04-17):
  - Anthropic (2026-04-17): https://platform.claude.com/docs/en/about-claude/pricing
  - OpenAI (2026-04-17): https://platform.openai.com/docs/pricing

Prices change over time. To override or supplement built-in prices, create a JSON
file in the same format as ``builtin_pricing.json`` and point ``MELLEA_PRICING_FILE``
to it. Custom entries take precedence over built-ins.

Environment variables:
  - MELLEA_PRICING_FILE: Path to a JSON file with custom model pricing overrides.

Custom pricing file format::

    {
      "my-model": {"input_per_1m": 1.0, "output_per_1m": 2.0}
    }
"""

import json
import logging
import os
from importlib.resources import files
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _load_pricing_json(path: str | Path) -> dict[str, Any]:
    """Load a pricing JSON file, returning an empty dict on any error."""
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except OSError as exc:
        logger.warning("Failed to load pricing file %r: %s", str(path), exc)
        return {}
    except json.JSONDecodeError as exc:
        logger.warning("Invalid JSON in pricing file %r: %s", str(path), exc)
        return {}


class PricingRegistry:
    """Registry of LLM model pricing for per-request cost estimation.

    Merges built-in pricing (from ``builtin_pricing.json``) with optional custom
    overrides loaded from ``MELLEA_PRICING_FILE``. Custom entries take precedence.

    Args:
        pricing_file: Path to a custom pricing JSON file. Defaults to the value of
            the ``MELLEA_PRICING_FILE`` environment variable.

    """

    def __init__(self, pricing_file: str | None = None) -> None:
        """Load built-in pricing and merge any custom overrides."""
        builtin_text = (
            files("mellea.telemetry")
            .joinpath("builtin_pricing.json")
            .read_text(encoding="utf-8")
        )
        builtin = json.loads(builtin_text)
        custom_path = pricing_file or os.getenv("MELLEA_PRICING_FILE")
        custom = _load_pricing_json(custom_path) if custom_path else {}
        self._pricing: dict[str, dict[str, float]] = {**builtin, **custom}
        self._warned_models: set[str] = set()

    def compute_cost(
        self, model: str, input_tokens: int | None, output_tokens: int | None
    ) -> float | None:
        """Estimate request cost in USD.

        Args:
            model: Model identifier (e.g. ``"gpt-5.4"``, ``"claude-sonnet-4-6"``).
            input_tokens: Number of input/prompt tokens, or ``None``.
            output_tokens: Number of output/completion tokens, or ``None``.

        Returns:
            Estimated cost in USD, or ``None`` if no pricing data exists for the model.
        """
        entry = self._pricing.get(model)
        if entry is None:
            if model not in self._warned_models:
                self._warned_models.add(model)
                logger.warning(
                    "No pricing data for model %r — cost metric will not be recorded. "
                    "Set MELLEA_PRICING_FILE to add custom pricing.",
                    model,
                )
            return None
        input_cost = ((input_tokens or 0) / 1_000_000.0) * entry.get(
            "input_per_1m", 0.0
        )
        output_cost = ((output_tokens or 0) / 1_000_000.0) * entry.get(
            "output_per_1m", 0.0
        )
        return input_cost + output_cost


_registry: PricingRegistry | None = None


def _get_registry() -> PricingRegistry:
    global _registry
    if _registry is None:
        _registry = PricingRegistry()
    return _registry


def compute_cost(
    model: str, input_tokens: int | None, output_tokens: int | None
) -> float | None:
    """Estimate request cost in USD using the default pricing registry.

    Args:
        model: Model identifier (e.g. ``"gpt-5.4"``, ``"claude-sonnet-4-6"``).
        input_tokens: Number of input/prompt tokens, or ``None``.
        output_tokens: Number of output/completion tokens, or ``None``.

    Returns:
        Estimated cost in USD, or ``None`` if no pricing data exists for the model.
    """
    return _get_registry().compute_cost(model, input_tokens, output_tokens)
