# pytest: ollama, e2e
"""Complete diagnostics example with all debug plugins.

This example combines generation, sampling, AND validation tracing for maximum
visibility into the entire sampling + validation pipeline.

You'll see:
1. Validation pre-check: requirements about to be checked
2. Generation pre/post-call: model called for main task + validation
3. Validation post-check: results per requirement
4. Sampling iteration: aggregate pass/fail count
5. Repair events: when and why repairs triggered
6. Final result: success/failure with statistics

This provides complete end-to-end traceability for:
- Model behavior (generation)
- Validation logic (what passes/fails and why)
- Repair strategy (how feedback improves results)
- Overall sampling loop (iterations and budget)

Run:
    uv run python docs/examples/plugins/builtin_complete_diagnostics.py

Watch the complete flow with all lifecycle events visible.
"""

import logging

import mellea
from mellea.core import Requirement
from mellea.plugins import register
from mellea.plugins.builtin_debug.generation import (
    log_generation_post_call,
    log_generation_pre_call,
)
from mellea.plugins.builtin_debug.sampling import (
    log_sampling_iteration,
    log_sampling_loop_end,
    log_sampling_loop_start,
    log_sampling_repair,
)
from mellea.plugins.builtin_debug.validation import (
    log_validation_post_check,
    log_validation_pre_check,
)
from mellea.stdlib.requirements import req, simple_validate
from mellea.stdlib.sampling import RepairTemplateStrategy

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# Enable ALL debug plugins
register(
    [
        # Generation pipeline
        log_generation_pre_call,
        log_generation_post_call,
        # Validation pipeline
        log_validation_pre_check,
        log_validation_post_check,
        # Sampling pipeline
        log_sampling_loop_start,
        log_sampling_iteration,
        log_sampling_repair,
        log_sampling_loop_end,
    ]
)


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
    """Complete diagnostics example."""
    log.info("=" * 70)
    log.info("Complete Diagnostics Example (All Debug Plugins)")
    log.info("=" * 70)
    log.info("")
    log.info("All debug plugins are enabled:")
    log.info("  [📤/📥 GEN-*]        - model calls and responses")
    log.info("  [🔍 VALIDATION-*]    - requirement validation")
    log.info("  [❌ SAMPLING-ITER]   - iteration results")
    log.info("  [🔧 REPAIR]          - repair events")
    log.info("")

    with mellea.start_session() as m:
        log.info("Generating text with strict requirements and repair strategy...")
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
