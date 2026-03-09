"""Validation hook payloads."""

from __future__ import annotations

from typing import Any

from mellea.plugins.base import MelleaBasePayload, WeakProxy


class ValidationPreCheckPayload(MelleaBasePayload):
    """Payload for ``validation_pre_check`` — before requirement validation.

    Attributes:
        requirements: List of ``Requirement`` instances to validate (writable).
        target: The ``CBlock`` being validated, or ``None`` when validating the full context.
            Held as a weak reference — do not cache this payload.
        context: The ``Context`` used for validation.
            Held as a weak reference — do not cache this payload.
        model_options: Dict of model options for backend-based validators (writable).
    """

    requirements: list[Any] = []
    target: WeakProxy = None
    context: WeakProxy = None
    model_options: dict[str, Any] = {}


class ValidationPostCheckPayload(MelleaBasePayload):
    """Payload for ``validation_post_check`` — after validation completes.

    Attributes:
        requirements: List of ``Requirement`` instances that were evaluated.
        results: List of ``ValidationResult`` instances (writable).
        all_validations_passed: ``True`` when every requirement passed (writable).
        passed_count: Number of requirements that passed.
        failed_count: Number of requirements that failed.
    """

    requirements: list[Any] = []
    results: list[Any] = []
    all_validations_passed: bool = False
    passed_count: int = 0
    failed_count: int = 0
