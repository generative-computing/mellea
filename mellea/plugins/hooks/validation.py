# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validation hook payloads."""

from __future__ import annotations

from typing import Any

from mellea.plugins.base import MelleaBasePayload


class ValidationPreCheckPayload(MelleaBasePayload):
    """Payload for `validation_pre_check` — before requirement validation.

    Attributes:
        validation_id: UUID correlating the pre/post hooks for a single
            requirement-validation batch.
        requirements: List of `Requirement` instances to validate (writable).
        target: The `CBlock` being validated, or `None` when validating the full context.

        context: The `Context` used for validation.

        model_options: Dict of model options for backend-based validators (writable).
    """

    validation_id: str = ""
    requirements: list[Any] = []
    target: Any = None
    context: Any = None
    model_options: dict[str, Any] = {}


class ValidationPostCheckPayload(MelleaBasePayload):
    """Payload for `validation_post_check` — after validation completes.

    Fires on every completing path: a normal check (any pass/fail mix) and an
    unhandled exception. On the exception path `results` and the counts are empty.

    Attributes:
        validation_id: UUID correlating with the matching `validation_pre_check`.
        requirements: List of `Requirement` instances that were evaluated.
        results: List of `ValidationResult` instances (writable).
        all_validations_passed: `True` when every requirement passed (writable).
        passed_count: Number of requirements that passed.
        failed_count: Number of requirements that failed.
        exception: The exception raised during validation, or `None` when it completed.
    """

    validation_id: str = ""
    requirements: list[Any] = []
    results: list[Any] = []
    all_validations_passed: bool = False
    passed_count: int = 0
    failed_count: int = 0
    exception: BaseException | None = None
