# pytest: llamacpp, e2e, skip

"""Example: one-token decisions against a llama.cpp server.

Decisions answer a fixed-answer question about some state with a typed
answer (probability, choice, or score) read from the model's first
generated token. This example shows how to serve a Granite Switch
checkpoint with llama.cpp and use the decision adapter functions
against it.

Prerequisites:
- llama.cpp built from main (Granite Switch support merged 2026-08-10)
- A Granite Switch GGUF file. `scripts/start_llamacpp_switch.sh` converts
  a checkpoint with the pinned converter and serves it; it comes with the
  llama-server support PR (#1736). If you have a GPU, the vLLM path (see
  `decisions.py`) is simpler and works with the Hugging Face checkpoint
  directly.

Serve the model, then point this example at it:

```bash
llama-server -m granite-switch-4.1-8b-f16.gguf --port 8080 -ngl 99
SWITCH_DECISIONS_BASE_URL=http://localhost:8080/v1 \\
SWITCH_DECISIONS_MODEL_ID=ibm-granite/granite-switch-4.1-8b \\
uv run python docs/examples/intrinsics/decisions_llamacpp.py
```

To run this script from the root of the Mellea source tree, use the command:
```
uv run python docs/examples/intrinsics/decisions_llamacpp.py
```
"""

import os
import sys

from mellea.backends.openai import OpenAIBackend
from mellea.formatters import TemplateFormatter
from mellea.stdlib.components.intrinsic import decisions

base_url = os.environ.get("SWITCH_DECISIONS_BASE_URL")
model_id = os.environ.get(
    "SWITCH_DECISIONS_MODEL_ID", "ibm-granite/granite-switch-4.1-8b"
)
if base_url is None:
    print(
        "SKIPPED: no decision endpoint configured. Serve a Granite Switch "
        "GGUF with llama-server (llama.cpp from main, which has Granite "
        "Switch support), then set SWITCH_DECISIONS_BASE_URL to run this "
        "example."
    )
    sys.exit(0)

# The model_id is used for client-side adapter discovery (io.yaml
# download); the llama-server endpoint serves the GGUF regardless of
# the model name in the request.
backend = OpenAIBackend(
    model_id=model_id,
    formatter=TemplateFormatter(model_id=model_id),
    base_url=base_url,
    api_key=os.environ.get("SWITCH_DECISIONS_API_KEY", "unused"),
    load_embedded_adapters=True,
)

ticket = (
    "My payouts have failed three times. The bank says everything is fine. "
    "Can someone please fix this?"
)

# All three decision types work identically against llama.cpp — the
# readout is transport-agnostic (it reads the first token's logprobs
# from the OpenAI-compatible response).

answer = decisions.decide(ticket, backend, "Does this require urgent attention?")
print(f"decide: P(urgent) = {answer.probability:.3f}")
if answer >= 0.7:
    print("  -> escalate")

route = decisions.choose(
    ticket,
    backend,
    "Which team handles this?",
    {"payments": "Payout failures and refunds", "account": "Login and account access"},
)
print(
    f"choose: {route.choice} ({ {k: round(v, 3) for k, v in route.probabilities.items()} })"
)

severity = decisions.score(
    ticket, backend, "How urgent is this?", ["can wait a week", "this week", "today"]
)
print(
    f"score: {severity.score:.2f} ({ {k: round(v, 3) for k, v in severity.probabilities.items()} })"
)
