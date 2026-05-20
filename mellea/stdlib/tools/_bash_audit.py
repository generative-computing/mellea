"""Audit trail for bash guardrails violations.

Records all command rejections with pattern, severity, category, and execution
context for compliance audits, security monitoring, and observability integration.
"""

import threading
import time
from dataclasses import dataclass
from typing import Any

from mellea.core.utils import MelleaLogger
from mellea.telemetry.context import get_current_context
from mellea.telemetry.metrics import create_counter


@dataclass
class BashViolation:
    """Record of a guardrail violation.

    Attributes:
        timestamp: Unix timestamp when violation occurred.
        command: Original command string.
        argv: Tokenized command arguments.
        pattern: Pattern name that detected the violation (e.g., "DangerousCommandPattern").
        category: Violation category (e.g., "PRIVILEGE_ESCALATION", "DESTRUCTIVE").
        severity: Severity level ("CRITICAL", "HIGH", "MEDIUM", "LOW").
        reason: Human-readable explanation of why it was rejected.
        working_dir: Working directory for execution context.
        allowed_paths: Path restrictions that were in effect.
        session_id: Session ID from context if available.
        request_id: Request ID from context if available.
    """

    timestamp: float
    command: str
    argv: list[str]
    pattern: str
    category: str
    severity: str
    reason: str
    working_dir: str | None = None
    allowed_paths: list[str] | None = None
    session_id: str | None = None
    request_id: str | None = None


class BashAuditTrail:
    """Singleton audit trail for bash guardrails violations.

    Records, queries, and exports metrics for all command rejections.
    Thread-safe with in-memory storage suitable for typical workflows
    where violations are rare.
    """

    _instance: "BashAuditTrail | None" = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        self._violations: list[BashViolation] = []
        self._violations_by_session: dict[str, list[BashViolation]] = {}
        self._violations_by_pattern: dict[str, list[BashViolation]] = {}
        self._storage_lock = threading.Lock()

    @classmethod
    def get_instance(cls) -> "BashAuditTrail":
        """Get singleton audit trail instance (thread-safe)."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def record_violation(self, violation: BashViolation) -> None:
        """Record a guardrail violation.

        Args:
            violation: The violation to record.
        """
        with self._storage_lock:
            self._violations.append(violation)

            if violation.session_id:
                if violation.session_id not in self._violations_by_session:
                    self._violations_by_session[violation.session_id] = []
                self._violations_by_session[violation.session_id].append(violation)

            if violation.pattern not in self._violations_by_pattern:
                self._violations_by_pattern[violation.pattern] = []
            self._violations_by_pattern[violation.pattern].append(violation)

        _log_violation(violation)
        _record_violation_metrics(violation.category, violation.severity)

    def get_violations(
        self,
        session_id: str | None = None,
        pattern: str | None = None,
        category: str | None = None,
        severity: str | None = None,
        limit: int | None = None,
    ) -> list[BashViolation]:
        """Query recorded violations with optional filters.

        Args:
            session_id: Filter by session ID.
            pattern: Filter by pattern name.
            category: Filter by category.
            severity: Filter by severity level.
            limit: Maximum number of results to return.

        Returns:
            List of matching violations.
        """
        with self._storage_lock:
            results = self._violations[:]

        for violation in results:
            if session_id and violation.session_id != session_id:
                results.remove(violation)
            elif pattern and violation.pattern != pattern:
                results.remove(violation)
            elif category and violation.category != category:
                results.remove(violation)
            elif severity and violation.severity != severity:
                results.remove(violation)

        if limit:
            results = results[:limit]

        return results

    def export_metrics(self) -> dict[str, Any]:
        """Export violation metrics.

        Returns:
            Dictionary with counts by severity, category, and pattern.
        """
        with self._storage_lock:
            violations = self._violations[:]

        metrics: dict[str, Any] = {"total": len(violations)}

        severity_counts: dict[str, int] = {}
        category_counts: dict[str, int] = {}
        pattern_counts: dict[str, int] = {}

        for v in violations:
            severity_counts[v.severity] = severity_counts.get(v.severity, 0) + 1
            category_counts[v.category] = category_counts.get(v.category, 0) + 1
            pattern_counts[v.pattern] = pattern_counts.get(v.pattern, 0) + 1

        for severity, count in severity_counts.items():
            metrics[f"severity_{severity}"] = count

        for category, count in category_counts.items():
            metrics[f"category_{category}"] = count

        for pattern, count in pattern_counts.items():
            metrics[f"pattern_{pattern}"] = count

        return metrics

    def clear(self) -> None:
        """Clear all recorded violations (primarily for testing)."""
        with self._storage_lock:
            self._violations.clear()
            self._violations_by_session.clear()
            self._violations_by_pattern.clear()


def record_bash_violation(
    command: str,
    argv: list[str],
    pattern_name: str,
    category: str,
    severity: str,
    reason: str,
    working_dir: str | None = None,
    allowed_paths: list[str] | None = None,
) -> None:
    """Record a bash guardrail violation (public entry point).

    Args:
        command: Original command string.
        argv: Tokenized command arguments.
        pattern_name: Pattern that detected the violation.
        category: Violation category.
        severity: Severity level.
        reason: Why it was rejected.
        working_dir: Working directory context.
        allowed_paths: Path restrictions context.
    """
    context = get_current_context()
    session_id = context.get("session_id") if context else None
    request_id = context.get("request_id") if context else None

    violation = BashViolation(
        timestamp=time.time(),
        command=command[:200],
        argv=argv,
        pattern=pattern_name,
        category=category,
        severity=severity,
        reason=reason,
        working_dir=working_dir,
        allowed_paths=allowed_paths,
        session_id=session_id,
        request_id=request_id,
    )

    trail = BashAuditTrail.get_instance()
    trail.record_violation(violation)


def get_bash_violations(
    session_id: str | None = None,
    pattern: str | None = None,
    category: str | None = None,
    severity: str | None = None,
    limit: int | None = None,
) -> list[BashViolation]:
    """Query recorded bash violations (public entry point).

    Args:
        session_id: Filter by session ID.
        pattern: Filter by pattern name.
        category: Filter by category.
        severity: Filter by severity level.
        limit: Maximum number of results.

    Returns:
        List of matching violations.
    """
    trail = BashAuditTrail.get_instance()
    return trail.get_violations(
        session_id=session_id,
        pattern=pattern,
        category=category,
        severity=severity,
        limit=limit,
    )


def _log_violation(violation: BashViolation) -> None:
    """Log violation using structured logging."""
    logger = MelleaLogger.get_logger()
    logger.warning(
        "Bash guardrail violation",
        extra={
            "bash_violation": True,
            "pattern": violation.pattern,
            "category": violation.category,
            "severity": violation.severity,
            "reason": violation.reason,
            "session_id": violation.session_id,
            "request_id": violation.request_id,
        },
    )


def _record_violation_metrics(category: str, severity: str) -> None:
    """Record violation metrics."""
    counter = create_counter(
        "bash.guardrail.violations",
        description="Count of bash guardrail violations",
        unit="1",
    )
    counter.add(1, {"category": category, "severity": severity})
