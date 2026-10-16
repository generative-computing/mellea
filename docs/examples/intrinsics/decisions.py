# pytest: e2e, vllm, skip

"""Example usage of the decision adapter functions: decide, choose, and score.

Decisions answer a fixed-answer question about some state with a typed
answer — a probability, a choice key, or a score — read from the model's
first generated token. One forward pass per decision.

Requires a Granite Switch checkpoint that embeds the decision adapter
function (e.g. `ibm-granite/granite-switch-4.1-8b`), served via vLLM with
`--max-logprobs 20`. Point the example at it with:

```
SWITCH_DECISIONS_BASE_URL=http://localhost:8000/v1 \\
SWITCH_DECISIONS_MODEL_ID=ibm-granite/granite-switch-4.1-8b \\
uv run python docs/examples/intrinsics/decisions.py
```

To run this script from the root of the Mellea source tree, use the command:
```
uv run python docs/examples/intrinsics/decisions.py
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
        "checkpoint with the decision adapter function via vLLM "
        "(--max-logprobs 20), then set SWITCH_DECISIONS_BASE_URL "
        "(e.g. http://localhost:8000/v1) to run this example."
    )
    sys.exit(0)

backend = OpenAIBackend(
    model_id=model_id,
    formatter=TemplateFormatter(model_id=model_id),
    base_url=base_url,
    api_key=os.environ.get("SWITCH_DECISIONS_API_KEY", "EMPTY"),
    load_embedded_adapters=True,
)

ticket = (
    "My payouts have failed three times. The bank says everything is fine. "
    "Can someone please fix this?"
)

# Predicate: a probability you can threshold directly.
answer = decisions.decide(ticket, backend, "Does this require urgent attention?")
print(f"decide: P(urgent) = {answer.probability:.3f}")
if answer >= 0.7:
    print("  -> escalate")

# Choice: the best option key plus the full distribution.
route = decisions.choose(
    ticket,
    backend,
    "Which team handles this?",
    {"payments": "Payout failures and refunds", "account": "Login and account access"},
)
print(
    f"choose: {route.choice} ({ {k: round(v, 3) for k, v in route.probabilities.items()} })"
)

# Score: expected level (may land between levels) plus per-level probabilities.
severity = decisions.score(
    ticket, backend, "How urgent is this?", ["can wait a week", "this week", "today"]
)
print(
    f"score: {severity.score:.2f} ({ {k: round(v, 3) for k, v in severity.probabilities.items()} })"
)
