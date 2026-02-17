"""Validation hook payloads."""

from __future__ import annotations

from typing import Any

from mellea.plugins.base import MelleaBasePayload


class ValidationPreCheckPayload(MelleaBasePayload):
    """Payload for ``validation_pre_check`` — before requirement validation."""

    requirements: list[Any] = []  # list[Requirement]
    target: Any = None  # CBlock | None
    context: Any = None  # Context
    model_options: dict[str, Any] = {}


class ValidationPostCheckPayload(MelleaBasePayload):
    """Payload for ``validation_post_check`` — after validation completes."""

    requirements: list[Any] = []  # list[Requirement]
    results: list[Any] = []  # list[ValidationResult]
    all_passed: bool = False
    passed_count: int = 0
    failed_count: int = 0
