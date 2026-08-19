# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""LLM pricing via litellm's pricing API.

Pricing metrics require the litellm package (`mellea[litellm]`). Pricing is
auto-enabled when litellm is installed and can be explicitly controlled via the
`MELLEA_PRICING_ENABLED` environment variable.

`MELLEA_PRICING_ENABLED` tri-state:
  - `"true"`  + litellm importable → enabled
  - `"true"`  + litellm absent     → warning, disabled
  - `"false"` (any)                → disabled (silent)
  - unset     + litellm importable → enabled (auto)
  - unset     + litellm absent     → disabled (silent)

If litellm is discoverable but fails to import (a skewed install), pricing is
disabled on first use with a logged warning; it never raises into caller code.

Pricing is only active when `MELLEA_METRICS_ENABLED` is also set.

Custom pricing:
  Set `MELLEA_PRICING_FILE` to a JSON file using litellm's native per-token
  schema. Minimal entries with only cost fields are supported:

      {
        "my-model": {
          "input_cost_per_token": 0.000003,
          "output_cost_per_token": 0.000015
        }
      }
  Optional cache fields: `cache_read_input_token_cost`,
  `cache_creation_input_token_cost`.

Environment variables:
  - MELLEA_PRICING_ENABLED: Tri-state pricing flag (true/false/unset).
  - MELLEA_PRICING_FILE: Path to a JSON file with custom model pricing.
"""

import importlib.util
import json
import logging
import os
import warnings
from pathlib import Path
from types import ModuleType

logger = logging.getLogger(__name__)

# Availability is probed without importing: `import litellm` costs ~1s and
# transitively pulls in openai and pandas, which every `import mellea` would pay
# even when pricing is disabled. `find_spec` only proves a loader exists, so the
# real import happens at the point of use via `_import_litellm`, which downgrades
# to "pricing disabled" if the module fails to execute. Either way the cost lands
# on the first priced request instead of on `import mellea`.
_LITELLM_AVAILABLE = importlib.util.find_spec("litellm") is not None


def _resolve_pricing_enabled() -> bool:
    env = os.getenv("MELLEA_PRICING_ENABLED")
    if env is not None and env.lower() in ("false", "0", "no"):
        return False
    if env is not None and env.lower() in ("true", "1", "yes"):
        if _LITELLM_AVAILABLE:
            return True
        warnings.warn(
            "MELLEA_PRICING_ENABLED=true but litellm is not installed or could "
            "not be imported — pricing metrics disabled. "
            "Install with: pip install 'mellea[litellm]'",
            stacklevel=2,
        )
        return False
    return _LITELLM_AVAILABLE


_PRICING_ENABLED: bool = False

_warned_models: set[str] = set()


def _import_litellm() -> ModuleType | None:
    """Import litellm, disabling pricing if the import fails.

    `_LITELLM_AVAILABLE` is probed with `find_spec`, which proves only that a
    loader exists — not that the module executes. A skewed install (mismatched
    pydantic, half-installed openai, wrong native ABI) can raise almost anything
    from litellm's module body, so every failure is treated as "no pricing"
    rather than propagated: this is a best-effort telemetry path and
    `compute_cost` is called from a fire-and-forget metrics hook.

    Both `_LITELLM_AVAILABLE` and `_PRICING_ENABLED` are cleared on failure, so
    the warning is emitted at most once per process.

    Returns:
        The imported `litellm` module, or `None` if it could not be imported.
    """
    global _LITELLM_AVAILABLE, _PRICING_ENABLED
    try:
        import litellm  # type: ignore[import-not-found]

        return litellm
    except Exception as exc:
        logger.warning(
            "litellm was discoverable but failed to import (%s: %s) — "
            "pricing metrics disabled.",
            type(exc).__name__,
            exc,
        )
        _LITELLM_AVAILABLE = _PRICING_ENABLED = False
        return None


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
    litellm = _import_litellm()
    if litellm is None:
        return

    try:
        litellm.register_model(data)
    except Exception as exc:
        logger.warning("Failed to register custom pricing from %r: %s", str(path), exc)


def _setup_pricing() -> None:
    """Read env vars and register custom pricing if configured.

    Reads `MELLEA_PRICING_ENABLED` and `MELLEA_PRICING_FILE` at call time so
    that environment changes made after module import are respected without
    requiring a module reload.
    """
    global _PRICING_ENABLED
    _PRICING_ENABLED = _resolve_pricing_enabled()
    if not _PRICING_ENABLED:
        return
    custom_path = os.getenv("MELLEA_PRICING_FILE")
    if custom_path:
        _register_custom_pricing(custom_path)


_setup_pricing()


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
        model: Model identifier (e.g. `"gpt-5.4"`, `"claude-sonnet-4-6"`).
        provider: Provider name from the backend (e.g. `"openai"`, `"watsonx"`).
            Passed to litellm as `custom_llm_provider` to aid model resolution —
            e.g. `"watsonx"` causes litellm to try `watsonx/ibm/granite-4-h-small`.
        prompt_tokens: Total prompt tokens including any cached tokens, or `None`.
        completion_tokens: Number of completion tokens, or `None`.
        cached_tokens: Tokens served from prompt cache, or `None`.
        cache_creation_tokens: Tokens written to prompt cache, or `None`.

    Returns:
        Estimated cost in USD, or `None` if pricing is disabled or no pricing
        data exists for the model.
    """
    if not _PRICING_ENABLED:
        return None

    litellm = _import_litellm()
    if litellm is None:
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
