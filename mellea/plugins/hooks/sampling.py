"""Sampling pipeline hook payloads."""

from __future__ import annotations

from typing import Any

from mellea.plugins.base import MelleaBasePayload


class SamplingLoopStartPayload(MelleaBasePayload):
    """Payload for ``sampling_loop_start`` — when sampling strategy begins."""

    strategy_name: str = ""
    action: Any = None  # Component
    context: Any = None  # Context
    requirements: list[Any] = []  # list[Requirement]
    loop_budget: int = 0


class SamplingIterationPayload(MelleaBasePayload):
    """Payload for ``sampling_iteration`` — after each sampling attempt."""

    iteration: int = 0
    action: Any = None  # Component
    result: Any = None  # ModelOutputThunk
    validation_results: list[
        tuple[Any, Any]
    ] = []  # list[tuple[Requirement, ValidationResult]]
    all_valid: bool = False
    valid_count: int = 0
    total_count: int = 0


class SamplingRepairPayload(MelleaBasePayload):
    """Payload for ``sampling_repair`` — when repair is invoked after validation failure."""

    repair_type: str = ""
    failed_action: Any = None  # Component
    failed_result: Any = None  # ModelOutputThunk
    failed_validations: list[tuple[Any, Any]] = []
    repair_action: Any = None  # Component
    repair_context: Any = None  # Context
    repair_iteration: int = 0


class SamplingLoopEndPayload(MelleaBasePayload):
    """Payload for ``sampling_loop_end`` — when sampling completes."""

    success: bool = False
    iterations_used: int = 0
    final_result: Any = None  # ModelOutputThunk | None
    final_action: Any = None  # Component | None
    final_context: Any = None  # Context | None
    failure_reason: str | None = None
    all_results: list[Any] = []  # list[ModelOutputThunk]
    all_validations: list[list[tuple[Any, Any]]] = []
