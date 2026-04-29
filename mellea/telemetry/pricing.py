"""LLM pricing via litellm's pricing API.

Pricing metrics require the litellm package (``mellea[litellm]``). Pricing is
auto-enabled when litellm is installed and can be explicitly controlled via the
``MELLEA_PRICING_ENABLED`` environment variable.

``MELLEA_PRICING_ENABLED`` tri-state:
  - ``"true"``  + litellm installed  → enabled
  - ``"true"``  + litellm absent     → warning, disabled
  - ``"false"`` (any)                → disabled (silent)
  - unset       + litellm installed  → enabled (auto)
  - unset       + litellm absent     → disabled (silent)

Pricing is only active when ``MELLEA_METRICS_ENABLED`` is also set.

Custom pricing:
  Set ``MELLEA_PRICING_FILE`` to a JSON file using litellm's native per-token
  schema. Minimal entries with only cost fields are supported::

      {
        "my-model": {
          "input_cost_per_token": 0.000003,
          "output_cost_per_token": 0.000015
        }
      }

  Optional cache fields: ``cache_read_input_token_cost``,
  ``cache_creation_input_token_cost``.

Environment variables:
  - MELLEA_PRICING_ENABLED: Tri-state pricing flag (true/false/unset).
  - MELLEA_PRICING_FILE: Path to a JSON file with custom model pricing.
"""

import json
import logging
import os
import warnings
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    import litellm  # type: ignore[import-not-found]

    _LITELLM_AVAILABLE = True
except ImportError:
    litellm = None  # type: ignore
    _LITELLM_AVAILABLE = False


def _resolve_pricing_enabled() -> bool:
    env = os.getenv("MELLEA_PRICING_ENABLED")
    if env is not None and env.lower() in ("false", "0", "no"):
        return False
    if env is not None and env.lower() in ("true", "1", "yes"):
        if _LITELLM_AVAILABLE:
            return True
        warnings.warn(
            "MELLEA_PRICING_ENABLED=true but litellm is not installed — "
            "pricing metrics disabled. Install with: pip install 'mellea[litellm]'",
            stacklevel=2,
        )
        return False
    return _LITELLM_AVAILABLE


_PRICING_ENABLED = _resolve_pricing_enabled()

_warned_models: set[str] = set()


def _register_custom_pricing(path: str | Path) -> None:
    """Load MELLEA_PRICING_FILE and register entries with litellm."""
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except OSError as exc:
        logger.warning("Failed to load custom pricing file %r: %s", str(path), exc)
        return
    except json.JSONDecodeError as exc:
        logger.warning("Invalid JSON in custom pricing file %r: %s", str(path), exc)
        return
    if not isinstance(data, dict):
        logger.warning(
            "Custom pricing file %r must be a JSON object — skipping.", str(path)
        )
        return
    try:
        litellm.register_model(data)
    except Exception as exc:
        logger.warning("Failed to register custom pricing from %r: %s", str(path), exc)


_custom_path = os.getenv("MELLEA_PRICING_FILE")
if _PRICING_ENABLED and _custom_path:
    _register_custom_pricing(_custom_path)


def compute_cost(
    model: str,
    provider: str | None,
    prompt_tokens: int | None,
    completion_tokens: int | None,
    cached_tokens: int | None = None,
    cache_creation_tokens: int | None = None,
) -> float | None:
    """Estimate request cost in USD using litellm's pricing data.

    Args:
        model: Model identifier (e.g. ``"gpt-5.4"``, ``"claude-sonnet-4-6"``).
        provider: Provider name from the backend (e.g. ``"openai"``, ``"watsonx"``).
            Passed to litellm as ``custom_llm_provider`` to aid model resolution —
            e.g. ``"watsonx"`` causes litellm to try ``watsonx/ibm/granite-4-h-small``.
        prompt_tokens: Total prompt tokens including any cached tokens, or ``None``.
        completion_tokens: Number of completion tokens, or ``None``.
        cached_tokens: Tokens served from prompt cache, or ``None``.
        cache_creation_tokens: Tokens written to prompt cache, or ``None``.

    Returns:
        Estimated cost in USD, or ``None`` if pricing is disabled or no pricing
        data exists for the model.
    """
    if not _PRICING_ENABLED:
        return None
    try:
        prompt_cost, completion_cost = litellm.cost_per_token(
            model=model,
            custom_llm_provider=provider or None,
            prompt_tokens=prompt_tokens or 0,
            completion_tokens=completion_tokens or 0,
            cache_read_input_tokens=cached_tokens or 0,
            cache_creation_input_tokens=cache_creation_tokens or 0,
        )
        return prompt_cost + completion_cost
    except Exception:
        if model not in _warned_models:
            _warned_models.add(model)
            logger.warning(
                "No pricing data for model %r — cost metric will not be recorded.",
                model,
            )
        return None


def is_pricing_enabled() -> bool:
    """Return True if pricing metrics are enabled.

    Returns:
        True if litellm is available and pricing is not explicitly disabled.
    """
    return _PRICING_ENABLED
