# pytest: huggingface, e2e
"""Guardian safety checks as requirements inside a repair loop.

Wraps `guardian_check()` in a `Requirement` so a failed safety check feeds back
into `RepairTemplateStrategy` and the model gets another attempt. The adapter
function returns a risk score and no reasoning text, so the repair feedback
names the criteria and the threshold rather than explaining what was wrong.

Runs the model locally, so it needs a GPU or Apple Silicon Mac; the model and
adapter are downloaded from Hugging Face on first run.

Prerequisites:
    uv sync --extra hf

Usage:
    uv run python docs/examples/safety/repair_with_guardian.py
"""

from mellea import MelleaSession, model_ids
from mellea.backends.huggingface import LocalHFBackend
from mellea.core import Context, Requirement, ValidationResult
from mellea.stdlib.components.intrinsic import guardian
from mellea.stdlib.context import ChatContext
from mellea.stdlib.sampling import RepairTemplateStrategy

# One backend serves both generation and the Guardian adapter. A separate
# LocalHFBackend works too, at the cost of loading another model.
backend = LocalHFBackend(model_id=model_ids.IBM_GRANITE_4_1_3B)

# A Guardian check evaluates a conversation, so the session needs a ChatContext.
# The default SimpleContext keeps no history, leaving the adapter nothing to judge.
session = MelleaSession(backend=backend, ctx=ChatContext())

scores: list[float] = []


def harm_check(threshold: float) -> Requirement:
    """Build a requirement that fails when Guardian reports harm at or above `threshold`."""

    def validate(ctx: Context) -> ValidationResult:
        if not isinstance(ctx, ChatContext):
            raise TypeError(
                "Guardian checks need a ChatContext; create the session with "
                "ctx=ChatContext()"
            )
        score = guardian.guardian_check(ctx, backend, criteria="harm")
        scores.append(score)
        # Keep the reason low-cardinality: it is both repair feedback and a
        # metric attribute (mellea.requirement.failures). Log the score separately.
        return ValidationResult(
            score < threshold,
            reason=f"Guardian flagged harm at or above threshold {threshold}",
        )

    return Requirement(
        "The response must not be harmful", validation_fn=validate, check_only=True
    )


def run(threshold: float) -> None:
    """Generate under a Guardian requirement and report what the loop did.

    The printed response is the one `RepairTemplateStrategy` selects, which on a
    failed run is the first attempt (`select_from_failure` always returns 0).
    """
    # Each stage starts from an empty conversation.
    session.reset()
    before = len(scores)
    result = session.instruct(
        "Write a short, firm reply to a customer who has asked for a refund "
        "three times and been refused twice.",
        requirements=[harm_check(threshold)],
        strategy=RepairTemplateStrategy(loop_budget=3),
        return_sampling_results=True,
    )
    assert len(scores) > before, "validation_fn never ran"
    print(f"  Guardian scored {scores[before:]}")
    print(
        f"  Passed: {result.success} after {len(result.sample_generations)} attempt(s)"
    )
    print(f"  Selected response:\n{result.result}\n")


# 1. A realistic threshold: a firm but civil reply is expected to clear it on the
#    first attempt, so the loop short-circuits and repair never runs.
print("--- Threshold 0.05: low risk ---")
run(threshold=0.05)

# 2. A zero-tolerance threshold. `score < 0.0` is false for any probability, so
#    every attempt fails by construction: the loop uses its full budget and the
#    Guardian reason is fed back between attempts. This stage shows the repair
#    mechanics, not a recovery. With greedy decoding and a constant reason, the
#    retry after the first repair repeats itself, so the last two scores match.
print("--- Threshold 0.0: zero tolerance, repair path ---")
run(threshold=0.0)
