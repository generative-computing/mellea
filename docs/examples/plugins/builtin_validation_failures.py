# pytest: ollama, e2e
"""Validation tracing example showing real failures.

This example uses instruct with immediate validation (no repair strategy),
so we'll see real validation failures in the logs.

Run:
    uv run python docs/examples/plugins/builtin_validation_failures.py

Watch the logs to see:
    [🔍 VALIDATION-PRE-CHECK] requirements setup
    [❌ VALIDATION-POST-CHECK] with MIXED RESULTS showing:
    ✓ Passed requirements (debug level)
    ❌ Failed requirements (info level) with reasons
"""

import logging

import mellea
from mellea.core import Requirement
from mellea.plugins import plugin_scope
from mellea.plugins.builtin_debug.validation import (
    log_validation_post_check,
    log_validation_pre_check,
)
from mellea.stdlib.requirements import req, simple_validate

logging.basicConfig(level=logging.DEBUG, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


def requires_hello(text: str) -> bool:
    """Validation: must contain the word 'hello'."""
    return "hello" in text.lower()


def no_exclamation(text: str) -> bool:
    """Validation: no exclamation marks allowed."""
    return "!" not in text


def is_lowercase_only(text: str) -> bool:
    """Validation: all lowercase."""
    return text == text.lower()


def is_short(text: str) -> bool:
    """Validation: must be 20 characters or less."""
    return len(text) <= 20


requirements: list[Requirement | str] = [
    req("Must contain the word 'hello'", validation_fn=simple_validate(requires_hello)),
    req("No exclamation marks allowed", validation_fn=simple_validate(no_exclamation)),
    req("All lowercase", validation_fn=simple_validate(is_lowercase_only)),
    req("Must be 20 characters or less", validation_fn=simple_validate(is_short)),
]


def main():
    """Example that will show validation failures."""
    log.info("=" * 70)
    log.info("Validation Failures Example")
    log.info("=" * 70)
    log.info("")
    log.info("Requirements (with potential for failures):")
    for i, req_obj in enumerate(requirements, 1):
        req_desc = getattr(req_obj, "description", str(req_obj))
        log.info(f"  {i}. {req_desc}")
    log.info("")

    with plugin_scope([log_validation_pre_check, log_validation_post_check]):
        with mellea.start_session() as m:
            log.info(
                "Test: Generate casual greeting (likely to fail some requirements)"
            )
            log.info("-" * 70)
            log.info("")

            # Use immediate validation (no repair) to see failures
            result = m.instruct(
                "Say a casual greeting with punctuation", requirements=requirements
            )
            log.info("")
            log.info(f"Generated text: {result}")
            log.info("")
            log.info("=" * 70)


if __name__ == "__main__":
    main()
