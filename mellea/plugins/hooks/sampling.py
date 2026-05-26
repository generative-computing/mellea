"""Sampling pipeline hook payloads."""

from __future__ import annotations

from typing import Any

from mellea.plugins.base import MelleaBasePayload


class SamplingLoopStartPayload(MelleaBasePayload):
    """Payload for ``sampling_loop_start`` — when sampling strategy begins.

    Attributes:
        strategy_name: Class name of the sampling strategy (e.g. ``"RejectionSamplingStrategy"``).
        action: The ``Component`` being sampled.

        context: The ``Context`` at the start of sampling.

        requirements: List of ``Requirement`` instances to validate against.
        loop_budget: Maximum number of sampling iterations allowed (writable).
    """

    strategy_name: str = ""
    action: Any = None
    context: Any = None
    requirements: list[Any] = []
    loop_budget: int = 0


class SamplingIterationPayload(MelleaBasePayload):
    """Payload for ``sampling_iteration`` — after each sampling attempt.

    Attributes:
        strategy_name: Class name of the sampling strategy (e.g. ``"RejectionSamplingStrategy"``).
        iteration: 1-based iteration number within the sampling loop.
        action: The ``Component`` used for this attempt.

        result: The ``ModelOutputThunk`` produced by this attempt.
        validation_results: List of ``(Requirement, ValidationResult)`` tuples.
        all_validations_passed: ``True`` when **every** requirement in ``validation_results``
            passed for this iteration (i.e., the sampling attempt succeeded).
        valid_count: Number of requirements that passed.
        total_count: Total number of requirements evaluated.
    """

    strategy_name: str = ""
    iteration: int = 0
    action: Any = None
    result: Any = None
    validation_results: list[tuple[Any, Any]] = []
    all_validations_passed: bool = False
    valid_count: int = 0
    total_count: int = 0


class SamplingRepairPayload(MelleaBasePayload):
    """Payload for ``sampling_repair`` — when repair is invoked after validation failure.

    Attributes:
        repair_type: Kind of repair (strategy-dependent, e.g. ``"rejection"``, ``"template"``).
        failed_action: The ``Component`` that failed validation.

        failed_result: The ``ModelOutputThunk`` that failed validation.
        failed_validations: List of ``(Requirement, ValidationResult)`` tuples that failed.
        repair_action: The repaired ``Component`` to use for the next attempt.
        repair_context: The ``Context`` to use for the next attempt.
        repair_iteration: 1-based iteration at which the repair was triggered.
    """

    repair_type: str = ""
    failed_action: Any = None
    failed_result: Any = None
    failed_validations: list[tuple[Any, Any]] = []
    repair_action: Any = None
    repair_context: Any = None
    repair_iteration: int = 0


class SamplingLoopEndPayload(MelleaBasePayload):
    """Payload for ``sampling_loop_end`` — when sampling completes.

    Attributes:
        strategy_name: Class name of the sampling strategy (e.g. ``"RejectionSamplingStrategy"``).
        success: ``True`` if at least one attempt passed all requirements.
        iterations_used: Total number of iterations the loop executed.
        final_result: The selected ``ModelOutputThunk`` (best success or best failure).
        final_action: The ``Component`` that produced ``final_result``.

        final_context: The ``Context`` associated with ``final_result``.

        failure_reason: Human-readable reason when ``success`` is ``False``.
        all_results: List of ``ModelOutputThunk`` from every iteration.
        all_validations: Nested list — ``all_validations[i]`` is the list of
            ``(Requirement, ValidationResult)`` tuples for iteration *i*.
    """

    strategy_name: str = ""
    success: bool = False
    iterations_used: int = 0
    final_result: Any = None
    final_action: Any = None
    final_context: Any = None
    failure_reason: str | None = None
    all_results: list[Any] = []
    all_validations: list[list[tuple[Any, Any]]] = []
