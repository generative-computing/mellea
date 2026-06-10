"""Built-in debug plugin for validation pipeline.

Provides tracing for requirement validation including pre-check setup, post-check
results, and detailed requirement-by-requirement analysis.

Examples:
    Enable validation tracing:

        from mellea.plugins.builtin_debug.validation import (
            log_validation_pre_check,
            log_validation_post_check,
        )
        from mellea.plugins import register

        register([
            log_validation_pre_check,
            log_validation_post_check,
        ])

        with start_session() as m:
            result = m.instruct("...", requirements=[...])
"""

from __future__ import annotations

import logging
from typing import Any

from mellea.plugins import HookType, hook
from mellea.plugins.hooks.validation import (
    ValidationPostCheckPayload,
    ValidationPreCheckPayload,
)

logger = logging.getLogger(__name__)


@hook(HookType.VALIDATION_PRE_CHECK)
async def log_validation_pre_check(
    payload: ValidationPreCheckPayload, ctx: dict[str, Any]
) -> None:
    """Log validation setup before requirements are checked.

    Args:
        payload: ValidationPreCheckPayload containing requirements and target.
        ctx: Plugin context for hook execution.
    """
    num_reqs = len(payload.requirements)
    target_type = type(payload.target).__name__ if payload.target else "None"

    logger.info(
        f"[🔍 VALIDATION-PRE-CHECK] requirements={num_reqs} | target={target_type}"
    )

    if payload.requirements:
        logger.debug("   Requirements to validate:")
        for i, req in enumerate(payload.requirements, 1):
            req_desc = getattr(req, "description", str(req))
            req_type = type(req).__name__
            logger.debug(f"     {i}. [{req_type}] {req_desc}")


@hook(HookType.VALIDATION_POST_CHECK)
async def log_validation_post_check(
    payload: ValidationPostCheckPayload, ctx: dict[str, Any]
) -> None:
    """Log validation results after requirements are checked.

    Args:
        payload: ValidationPostCheckPayload with passed_count, failed_count, results.
        ctx: Plugin context for hook execution.
    """
    passed = payload.passed_count
    failed = payload.failed_count
    total = len(payload.requirements)
    all_passed = payload.all_validations_passed

    if all_passed:
        logger.info(
            f"[✅ VALIDATION-POST-CHECK] ALL PASSED: {passed}/{total} requirements"
        )
    else:
        logger.info(
            f"[❌ VALIDATION-POST-CHECK] MIXED RESULTS: {passed}/{total} passed, "
            f"{failed}/{total} failed"
        )

    # Log detailed results per requirement
    if payload.results:
        for i, (req, result) in enumerate(
            zip(payload.requirements, payload.results), 1
        ):
            req_desc = getattr(req, "description", str(req))
            is_passed = result.as_bool()
            status = "✓" if is_passed else "❌"

            if is_passed:
                logger.debug(f"    {status} {req_desc}")
            else:
                logger.info(f"    {status} {req_desc}")

                # Show reason if available and informative
                reason = getattr(result, "reason", None)
                if reason and reason not in ("yes", "no", ""):
                    logger.info(f"       └─ {reason}")

                # Show score if available
                score = getattr(result, "score", None)
                if score is not None:
                    logger.debug(f"       └─ score: {score:.2f}")
