# pytest: ollama, e2e
"""Strict validation tracing example with intentional failures.

This example demonstrates validation tracing with STRICT requirements designed
to trigger failures so you can see the full validation output including:
- Failed requirement details
- Failure reasons
- Pass/fail counts

Run:
    uv run python docs/examples/plugins/builtin_validation_strict.py

Watch the logs to see validation failures:
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
from mellea.stdlib.requirements import req, simple_validate

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# Enable validation tracing
register([log_validation_pre_check, log_validation_post_check])


def is_single_word(text: str) -> bool:
    """Validation: response must be exactly one word."""
    return len(text.strip().split()) == 1


def is_all_caps(text: str) -> bool:
    """Validation: all letters must be uppercase."""
    return text == text.upper() and text.isalpha()


def is_very_short(text: str) -> bool:
    """Validation: must be 5 characters or less."""
    return len(text) <= 5


requirements: list[Requirement | str] = [
    req(
        "Response must be exactly one word",
        validation_fn=simple_validate(is_single_word),
    ),
    req("All letters must be UPPERCASE", validation_fn=simple_validate(is_all_caps)),
    req(
        "Response must be 5 characters or less",
        validation_fn=simple_validate(is_very_short),
    ),
]


def main():
    """Example with strict requirements that will fail."""
    log.info("=" * 70)
    log.info("Strict Validation Tracing Example")
    log.info("=" * 70)
    log.info("")
    log.info("Intentionally strict requirements to demonstrate validation failures:")
    for i, req_obj in enumerate(requirements, 1):
        req_desc = getattr(req_obj, "description", str(req_obj))
        log.info(f"  {i}. {req_desc}")
    log.info("")

    with mellea.start_session() as m:
        log.info(
            "Generating multi-word response (will fail 'single word' requirement):"
        )
        log.info("-" * 70)

        result = m.instruct("Say hello world", requirements=requirements)
        log.info(f"Result: {result}")
        log.info("")

        log.info("=" * 70)
        log.info("Validation tracing complete!")
        log.info("=" * 70)


if __name__ == "__main__":
    main()
