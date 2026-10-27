# pytest: huggingface, e2e
"""Checking scenario compliance with a written policy using the policy-guardrails adapter.

Demonstrates `guardian.policy_guardrails()`, which uses the Granite Guardian 4.0
Micro LoRA adapter (policy_guardrails) through Mellea's intrinsic pipeline to
label whether a scenario complies with a policy text.

The script uses `LocalHFBackend`. The same adapter function also runs on
`OpenAIBackend` with a Granite Switch model; see `docs/examples/granite-switch/`.
Runs the model locally, so it needs a GPU or Apple Silicon Mac; the model and
adapter are downloaded from Hugging Face on first run.

Prerequisites:
    uv sync --extra hf

Usage:
    uv run python docs/examples/intrinsics/policy_guardrails.py
"""

from mellea import model_ids, start_backend
from mellea.stdlib.components import Message
from mellea.stdlib.components.intrinsic import guardian

# Create the backend.
ctx, backend = start_backend(
    "hf", model_id=model_ids.IBM_GRANITE_4_1_3B, context_type="chat"
)
# NOTE: This example can also be run with the OpenAIBackend using a GraniteSwitch model. See docs/examples/granite-switch/.

policy_text = "hiring managers should steer away from any questions that directly seek information about protected classes—such as “how old are you,” “where are you from,” “what year did you graduate” or “what are your plans for having kids.”"
user_text = "During a job interview, Alex focuses on asking candidates about their work experience, specific skills, and past accomplishments, without delving into their age, personal background, or family status."

print("--- Checking scenario compliance with policy ---")
ctx = ctx.add(Message("user", user_text))

label = guardian.policy_guardrails(ctx, backend, policy_text=policy_text)
print(f"Label: {label}")
print()
