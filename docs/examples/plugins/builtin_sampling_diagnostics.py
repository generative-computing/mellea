# pytest: ollama, e2e
"""Built-in sampling diagnostics plugin example.

This example demonstrates the sampling hooks from mellea.plugins.builtin_debug,
which trace the sampling strategy pipeline with iteration tracking, validation
results, repair events, and success/failure analysis.

The plugin logs:
- Strategy initialization with loop budget and requirements count
- Each iteration with validation pass/fail status
- Detailed validation results per requirement
- Repair events when triggered
- Final sampling result with success/failure reason

Run:
    uv run python docs/examples/plugins/builtin_sampling_diagnostics.py

Watch the logs to see:
    [🎯 SAMPLING-START] strategy=... | loop_budget=... | requirements=...
    [✅ SAMPLING-ITER 1] SUCCESS: ... validations passed
    [❌ SAMPLING-ITER 2] FAILED: ... validations passed
    [🔧 REPAIR-TRIGGERED] at iteration ...
    [🎉 SAMPLING-END] SUCCESS in ... iterations
"""

import logging

import mellea
from mellea.core import Requirement
from mellea.plugins import plugin_scope
from mellea.plugins.builtin_debug.sampling import (
    log_sampling_iteration,
    log_sampling_loop_end,
    log_sampling_loop_start,
    log_sampling_repair,
)
from mellea.stdlib.requirements import req, simple_validate
from mellea.stdlib.sampling import RepairTemplateStrategy

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


def is_lowercase_only(text: str) -> bool:
    """Validation: text must be all lowercase."""
    return text == text.lower()


def has_thank_you(text: str) -> bool:
    """Validation: text must contain 'thank you'."""
    return "thank you" in text.lower()


requirements: list[Requirement | str] = [
    req("Start with a greeting"),
    req(
        "Use only lowercase letters (no capitals)",
        validation_fn=simple_validate(is_lowercase_only),
    ),
    req("Include the phrase 'thank you'", validation_fn=simple_validate(has_thank_you)),
]


def main():
    """Example: Use sampling diagnostics to debug repair strategies."""
    log.info("=" * 70)
    log.info("Sampling Diagnostics Plugin Example")
    log.info("=" * 70)
    log.info("")
    log.info("Watch the logs for sampling lifecycle events:")
    log.info("  [🎯 SAMPLING-START] - strategy begins")
    log.info("  [✅/❌ SAMPLING-ITER] - per-iteration validation results")
    log.info("  [🔧 REPAIR-TRIGGERED] - repair invoked (RepairTemplateStrategy only)")
    log.info("  [🎉/💥 SAMPLING-END] - final result")
    log.info("")

    with plugin_scope(
        [
            log_sampling_loop_start,
            log_sampling_iteration,
            log_sampling_repair,
            log_sampling_loop_end,
        ]
    ):
        with mellea.start_session() as m:
            log.info("Generating text with strict requirements...")
            log.info("")

            result = m.instruct(
                "Write a thank you note",
                requirements=requirements,
                strategy=RepairTemplateStrategy(loop_budget=3),
            )

            log.info("")
            log.info("=" * 70)
            log.info("Final result:")
            log.info(str(result)[:300] + ("..." if len(str(result)) > 300 else ""))
            log.info("=" * 70)


if __name__ == "__main__":
    main()
