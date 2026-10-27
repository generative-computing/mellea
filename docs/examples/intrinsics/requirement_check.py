# pytest: huggingface, e2e
"""Validating a response against a requirement with the requirement-check adapter.

Demonstrates `core.requirement_check()`, which scores (0.0 to 1.0) whether an
assistant response satisfies a natural-language requirement. This is the same
adapter function that `ALoraRequirement` uses internally.

The script uses `LocalHFBackend`. The same adapter function also runs on
`OpenAIBackend` with a Granite Switch model; see `docs/examples/granite-switch/`.
Runs the model locally, so it needs a GPU or Apple Silicon Mac; the model and
adapter are downloaded from Hugging Face on first run.

Prerequisites:
    uv sync --extra hf

Usage:
    uv run python docs/examples/intrinsics/requirement_check.py
"""

from mellea import model_ids, start_backend
from mellea.stdlib.components import Message
from mellea.stdlib.components.intrinsic import core

user_text = "Invite for an IBM office party."
response_text = """
Dear Team,

To celebrate our recent successes and take a well-deserved moment to recharge,
you are cordially invited to a team social. Please join us for an evening of
live music, appetizers, and drinks as we recognize our collective wins.

Event Details
* **Date:** Saturday, April 25, 2026
* **Time:** 6:00 PM
* **Location:** Ryan’s Bar, Chelsea, NY
* **Highlights:** Live entertainment and refreshments

RSVP
To ensure we have an accurate headcount for catering, please confirm your
attendance by **Friday, April 10, 2026**.

We look forward to seeing everyone there and celebrating our hard work together.

**Best regards,**
[Your Name/Management Team]
"""
requirement = "Use a professional tone."

ctx, backend = start_backend(
    "hf", model_id=model_ids.IBM_GRANITE_4_1_3B, context_type="chat"
)
# NOTE: This example can also be run with the OpenAIBackend using a GraniteSwitch model. See docs/examples/granite-switch/.

ctx = ctx.add(Message("user", user_text)).add(Message("assistant", response_text))

result = core.requirement_check(ctx, backend, requirement)
print(f"Requirements Satisfied: {result}")  # float between 0.0 and 1.0
