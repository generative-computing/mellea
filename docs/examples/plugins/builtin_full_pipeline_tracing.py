# pytest: ollama, e2e
"""Full pipeline tracing example combining generation and sampling diagnostics.

This example demonstrates both the generation and sampling debug plugins working
together to provide end-to-end visibility into the entire sampling loop.

You'll see:
1. Generation pre-call: what's being sent to the LLM
2. Generation post-call: model response, latency, and tokens
3. Sampling iteration: validation results per attempt
4. Repair events: when and why repairs are triggered
5. Final result: success/failure with statistics

This combined view helps debug complex interactions between:
- Model behavior (generation tracing)
- Validation logic (sampling diagnostics)
- Repair strategies (how feedback improves results)

Run:
    uv run python docs/examples/plugins/builtin_full_pipeline_tracing.py

Watch the complete flow:
    [📤 GEN-PRE-CALL] → [📥 GEN-POST-CALL] (model called)
    [❌ SAMPLING-ITER] → [🔧 REPAIR-TRIGGERED] (validation failed)
    [📤 GEN-PRE-CALL] → [📥 GEN-POST-CALL] (repair attempt)
    [❌ SAMPLING-ITER] → [🔧 REPAIR-TRIGGERED] (still failing)
    ...
    [💥 SAMPLING-END] (final result)
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
from mellea.stdlib.requirements import req, simple_validate
from mellea.stdlib.sampling import RepairTemplateStrategy

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# Enable both generation and sampling tracing
register(
    [
        # Generation pipeline
        log_generation_pre_call,
        log_generation_post_call,
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
    """Full pipeline tracing example."""
    log.info("=" * 70)
    log.info("Full Pipeline Tracing Example")
    log.info("=" * 70)
    log.info("")
    log.info("Watch for both generation AND sampling events:")
    log.info("  [📤 GEN-PRE-CALL]     - what's sent to the LLM")
    log.info("  [📥 GEN-POST-CALL]    - model response + latency")
    log.info("  [❌ SAMPLING-ITER]    - validation results")
    log.info("  [🔧 REPAIR-TRIGGERED] - repair strategy kicks in")
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
