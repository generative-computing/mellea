# pytest: ollama, e2e
"""Validation tracing plugin example.

This example demonstrates the validation debug plugins from mellea.plugins.builtin_debug,
which trace requirement validation with pre-check setup and detailed per-requirement
results including pass/fail status, reasons, and scores.

The plugin logs:
- Pre-check: requirements about to be validated, target being checked
- Per-requirement: pass/fail status with reasons for failures
- Post-check: aggregate pass/fail counts and summary

Run:
    uv run python docs/examples/plugins/builtin_validation_tracing.py

Watch the logs to see:
    [🔍 VALIDATION-PRE-CHECK] requirements=... | target=...
    [❌ VALIDATION-POST-CHECK] MIXED RESULTS: ... passed, ... failed
    ✓ Passed requirement
    ❌ Failed requirement
       └─ reason why it failed
"""

import logging

import mellea
from mellea.core import Requirement
from mellea.plugins import register
from mellea.plugins.builtin_debug.validation import (
    log_validation_post_check,
    log_validation_pre_check,
)
from mellea.stdlib.requirements import check, req, simple_validate

logging.basicConfig(
    level=logging.DEBUG,  # DEBUG to see passed requirements too
    format="%(levelname)s: %(message)s",
)
log = logging.getLogger(__name__)

# Enable validation tracing
register([log_validation_pre_check, log_validation_post_check])


def is_lowercase_only(text: str) -> bool:
    """Validation: text must be all lowercase."""
    return text == text.lower()


def has_required_phrase(text: str) -> bool:
    """Validation: text must contain 'thank you'."""
    return "thank you" in text.lower()


def has_proper_length(text: str) -> bool:
    """Validation: text should be between 10 and 500 characters."""
    return 10 <= len(text) <= 500


requirements: list[Requirement | str] = [
    req(
        "Use only lowercase letters (no capitals)",
        validation_fn=simple_validate(is_lowercase_only),
    ),
    req(
        "Include the phrase 'thank you'",
        validation_fn=simple_validate(has_required_phrase),
    ),
    req(
        "Text length between 10 and 500 characters",
        validation_fn=simple_validate(has_proper_length),
    ),
    check("Response should be helpful and polite"),
]


def main():
    """Example: Use validation tracing to debug requirement checking."""
    log.info("=" * 70)
    log.info("Validation Tracing Plugin Example")
    log.info("=" * 70)
    log.info("")
    log.info("Watch the logs for validation lifecycle events:")
    log.info("  [🔍 VALIDATION-PRE-CHECK]  - setup phase before validation")
    log.info("  [❌ VALIDATION-POST-CHECK]  - results after validation")
    log.info("")
    log.info("Requirements being validated:")
    for i, req_obj in enumerate(requirements, 1):
        req_desc = getattr(req_obj, "description", str(req_obj))
        log.info(f"  {i}. {req_desc}")
    log.info("")

    with mellea.start_session() as m:
        log.info("Test 1: Good response (should pass most requirements)")
        log.info("-" * 70)

        result1 = m.instruct(
            "Say thank you",
            requirements=requirements[:3],  # Use first 3 requirements only
        )
        log.info("")
        log.info(f"Result:\n{result1}")
        log.info("")

        log.info("Test 2: Uppercase response (should fail lowercase requirement)")
        log.info("-" * 70)

        result2 = m.instruct("Say THANK YOU in all caps", requirements=requirements[:3])
        log.info("")
        log.info(f"Result:\n{result2}")
        log.info("")

        log.info("Test 3: Short response (should fail length requirement)")
        log.info("-" * 70)

        result3 = m.instruct("Say hi", requirements=requirements[:3])
        log.info("")
        log.info(f"Result:\n{result3}")
        log.info("")

        log.info("=" * 70)
        log.info("Validation tracing complete!")
        log.info("=" * 70)


if __name__ == "__main__":
    main()
