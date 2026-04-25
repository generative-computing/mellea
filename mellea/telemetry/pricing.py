"""LLM pricing registry for cost estimation.

Provides per-request cost estimates based on token usage and model pricing data.
Built-in prices are loaded from ``builtin_pricing.json`` in this package directory.
Custom overrides are loaded from a path set via ``MELLEA_PRICING_FILE``.

Pricing data sources:
  - Anthropic (2026-04-24): https://platform.claude.com/docs/en/about-claude/pricing
    ``cache_write_per_1m`` = 5-minute write rate (1.25x base input). 1-hour writes cost
    2x base, but the API does not distinguish write duration in
    ``cache_creation_input_tokens``, so cost will be underestimated for 1-hour writes.
  - OpenAI (2026-04-17): https://platform.openai.com/docs/pricing
    ``cache_read_per_1m`` = 50% of base input. OpenAI has no separate write cost.

Prices change over time. To override or supplement built-in prices, create a JSON
file in the same format as ``builtin_pricing.json`` and point ``MELLEA_PRICING_FILE``
to it. Custom entries take precedence over built-ins.

Environment variables:
  - MELLEA_PRICING_FILE: Path to a JSON file with custom model pricing overrides.

Custom pricing file format::

    {
      "my-model": {
        "input_per_1m": 1.0,
        "output_per_1m": 2.0,
        "cache_write_per_1m": 1.25,
        "cache_read_per_1m": 0.10
      }
    }

``cache_write_per_1m`` and ``cache_read_per_1m`` are optional. Models without
these fields report $0 for cache token costs.
"""

import json
import logging
import math
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


_PRICING_KEYS = {"input_per_1m", "output_per_1m"}


def _validate_pricing_entry(model: str, entry: Any) -> bool:
    """Return True if entry is a valid pricing dict; log and return False otherwise."""
    if not isinstance(entry, dict):
        logger.warning("Pricing entry for %r is not a dict — skipping.", model)
        return False
    for key, val in entry.items():
        if not isinstance(val, (int, float)) or val < 0 or not math.isfinite(val):
            logger.warning(
                "Pricing entry for %r has invalid %r: %r — skipping.", model, key, val
            )
            return False
    if not _PRICING_KEYS & entry.keys():
        logger.warning(
            "Pricing entry for %r has no recognised keys (%s) — skipping.",
            model,
            ", ".join(sorted(_PRICING_KEYS)),
        )
        return False
    return True


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
        self._pricing: dict[str, dict[str, float]] = {
            model: entry
            for model, entry in {**builtin, **custom}.items()
            if _validate_pricing_entry(model, entry)
        }
        self._warned_models: set[str] = set()

    def compute_cost(
        self,
        model: str,
        input_tokens: int | None,
        output_tokens: int | None,
        cached_tokens: int | None = None,
        cache_creation_tokens: int | None = None,
    ) -> float | None:
        """Estimate request cost in USD.

        Args:
            model: Model identifier (e.g. ``"gpt-5.4"``, ``"claude-sonnet-4-6"``).
            input_tokens: Number of input/prompt tokens, or ``None``.
            output_tokens: Number of output/completion tokens, or ``None``.
            cached_tokens: Tokens served from prompt cache, or ``None``.
            cache_creation_tokens: Tokens written to prompt cache, or ``None``.

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
        cache_read_cost = ((cached_tokens or 0) / 1_000_000.0) * entry.get(
            "cache_read_per_1m", 0.0
        )
        cache_creation_cost = ((cache_creation_tokens or 0) / 1_000_000.0) * entry.get(
            "cache_write_per_1m", 0.0
        )
        return input_cost + output_cost + cache_read_cost + cache_creation_cost


_registry: PricingRegistry | None = None


def _get_registry() -> PricingRegistry:
    global _registry
    if _registry is None:
        _registry = PricingRegistry()
    return _registry


def compute_cost(
    model: str,
    input_tokens: int | None,
    output_tokens: int | None,
    cached_tokens: int | None = None,
    cache_creation_tokens: int | None = None,
) -> float | None:
    """Estimate request cost in USD using the default pricing registry.

    Args:
        model: Model identifier (e.g. ``"gpt-5.4"``, ``"claude-sonnet-4-6"``).
        input_tokens: Number of input/prompt tokens, or ``None``.
        output_tokens: Number of output/completion tokens, or ``None``.
        cached_tokens: Tokens served from prompt cache, or ``None``.
        cache_creation_tokens: Tokens written to prompt cache, or ``None``.

    Returns:
        Estimated cost in USD, or ``None`` if no pricing data exists for the model.
    """
    return _get_registry().compute_cost(
        model, input_tokens, output_tokens, cached_tokens, cache_creation_tokens
    )
