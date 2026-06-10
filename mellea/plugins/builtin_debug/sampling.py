"""Built-in debug plugin for sampling pipeline.

Provides tracing for sampling strategies including iteration tracking, validation
results, repair events, and success/failure analysis.

Examples:
    Enable sampling tracing:

        from mellea.plugins.builtin_debug.sampling import (
            log_sampling_loop_start,
            log_sampling_iteration,
            log_sampling_repair,
            log_sampling_loop_end,
        )
        from mellea.plugins import register

        register([
            log_sampling_loop_start,
            log_sampling_iteration,
            log_sampling_repair,
            log_sampling_loop_end,
        ])

        with start_session() as m:
            result = m.instruct("...", strategy=SamplingStrategy(...))
"""

from __future__ import annotations

import logging
from typing import Any

from mellea.plugins import HookType, hook
from mellea.plugins.hooks.sampling import (
    SamplingIterationPayload,
    SamplingLoopEndPayload,
    SamplingLoopStartPayload,
    SamplingRepairPayload,
)

logger = logging.getLogger(__name__)


@hook(HookType.SAMPLING_LOOP_START)
async def log_sampling_loop_start(
    payload: SamplingLoopStartPayload, ctx: dict[str, Any]
) -> None:
    """Log sampling strategy initialization with budget and requirement count.

    Args:
        payload: SamplingLoopStartPayload with strategy_name, loop_budget, requirements.
        ctx: Plugin context for hook execution.
    """
    strategy = payload.strategy_name
    budget = payload.loop_budget
    num_reqs = len(payload.requirements)

    logger.info(
        f"[🎯 SAMPLING-START] strategy={strategy} | loop_budget={budget} | "
        f"requirements={num_reqs}"
    )

    if payload.requirements:
        for i, req in enumerate(payload.requirements, 1):
            req_desc = getattr(req, "description", str(req))
            logger.debug(f"  {i}. {req_desc}")


@hook(HookType.SAMPLING_ITERATION)
async def log_sampling_iteration(
    payload: SamplingIterationPayload, ctx: dict[str, Any]
) -> None:
    """Log validation results for each sampling attempt.

    Args:
        payload: SamplingIterationPayload with iteration, valid_count, validation_results.
        ctx: Plugin context for hook execution.
    """
    iteration = payload.iteration
    passed = payload.valid_count
    total = payload.total_count

    if payload.all_validations_passed:
        logger.info(
            f"[✅ SAMPLING-ITER {iteration}] SUCCESS: {passed}/{total} validations passed"
        )
    else:
        logger.info(
            f"[❌ SAMPLING-ITER {iteration}] FAILED: {passed}/{total} validations passed"
        )

        if payload.validation_results:
            for req_obj, result in payload.validation_results:
                req_desc = getattr(req_obj, "description", str(req_obj))
                status = "✓" if result.as_bool() else "❌"

                if result.as_bool():
                    logger.debug(f"    {status} {req_desc}")
                else:
                    logger.info(f"    {status} {req_desc}")
                    # Show detailed reason only if informative
                    reason = getattr(result, "reason", None)
                    if reason and reason not in ("yes", "no"):
                        logger.info(f"       └─ {reason}")


@hook(HookType.SAMPLING_REPAIR)
async def log_sampling_repair(
    payload: SamplingRepairPayload, ctx: dict[str, Any]
) -> None:
    """Log when repair is triggered during sampling iterations.

    Args:
        payload: SamplingRepairPayload with repair_iteration, repair_type, failed_validations.
        ctx: Plugin context for hook execution.
    """
    iteration = payload.repair_iteration
    repair_type = payload.repair_type

    logger.info(f"\n[🔧 REPAIR-TRIGGERED] at iteration {iteration}")
    logger.info(f"   repair_type={repair_type}")
    logger.info("   failed_validations:")

    for req_obj, result in payload.failed_validations:
        if not result.as_bool():
            req_desc = getattr(req_obj, "description", str(req_obj))
            logger.info(f"     • {req_desc}")


@hook(HookType.SAMPLING_LOOP_END)
async def log_sampling_loop_end(
    payload: SamplingLoopEndPayload, ctx: dict[str, Any]
) -> None:
    """Log sampling completion with success status and attempt statistics.

    Args:
        payload: SamplingLoopEndPayload with success, iterations_used, all_results, all_validations.
        ctx: Plugin context for hook execution.
    """
    strategy = payload.strategy_name
    iterations = payload.iterations_used
    success = payload.success
    failure_reason = payload.failure_reason

    if success:
        logger.info(
            f"\n[🎉 SAMPLING-END] SUCCESS in {iterations} iteration(s) using {strategy}"
        )
    else:
        logger.info(
            f"\n[💥 SAMPLING-END] FAILED after {iterations} iteration(s): "
            f"{failure_reason}"
        )

    # Summary statistics
    total_results = len(payload.all_results)
    logger.info(f"   total_attempts={total_results}")

    # Show best attempt statistics
    if payload.all_validations:
        best_valid_count = 0
        for validation_list in payload.all_validations:
            valid_count = sum(1 for _, result in validation_list if result.as_bool())
            best_valid_count = max(best_valid_count, valid_count)

        total_reqs = len(payload.all_validations[0]) if payload.all_validations else 0
        logger.info(f"   best_validation_score={best_valid_count}/{total_reqs}")
