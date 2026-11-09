# pytest: e2e, llamacpp_switch

"""Example: one-token decisions against a llama.cpp server.

Experimental: provided purely as an example. Discussion of the appropriate
decision APIs is ongoing, and this API may change or be removed at any time
without a deprecation period.

Decisions answer a fixed-answer question about some state with a typed
answer (probability, choice, or score) read from the model's first
generated token. This example runs the decision adapter functions against
a Granite Switch checkpoint served by llama-server (llama.cpp v0.1.0 or
newer). If you have a GPU, the vLLM path (`decisions.py`) is simpler.

The `decision` adapter is embedded in the released `granite-switch-4.1-8b`
and `-30b`, not the 3b preview the other llama.cpp example uses. Serve IBM's
published 8b GGUF (Q4_K_M, ~6.4 GB, downloaded and cached on first run):

```bash
llama-server -hf ibm-granite/granite-switch-4.1-8b-GGUF:Q4_K_M \\
  --alias ibm-granite/granite-switch-4.1-8b --port 8081 --jinja
```

`--alias` makes the server report the name this example checks for. The
server listens on `http://127.0.0.1:8081`; set
`LLAMACPP_SWITCH_BASE_URL` to point elsewhere. This example skips, naming
what it found, when no server is reachable or the server is not serving the
8b.

To run this script from the root of the Mellea source tree, use the command:
```
uv run python docs/examples/intrinsics/decisions_llamacpp.py
```
"""

import os
import sys

import requests

BASE_URL = os.environ.get("LLAMACPP_SWITCH_BASE_URL", "http://127.0.0.1:8081/v1")

try:
    response = requests.get(f"{BASE_URL}/models", timeout=5)
    response.raise_for_status()
    served = {model["id"] for model in response.json()["data"]}
except (requests.RequestException, ValueError, KeyError, TypeError):
    # Detected by docs/examples/conftest.py subprocess runner as a skip.
    print(
        f"Skipped: no llama-server ready at {BASE_URL} "
        "(start llama-server -hf ibm-granite/granite-switch-4.1-8b-GGUF:Q4_K_M "
        "--alias ibm-granite/granite-switch-4.1-8b --port 8081 --jinja, "
        "or set LLAMACPP_SWITCH_BASE_URL)",
        file=sys.stderr,
    )
    raise SystemExit(1)

from mellea.backends.model_ids import IBM_GRANITE_SWITCH_4_1_8B
from mellea.backends.openai import OpenAIBackend
from mellea.formatters import TemplateFormatter
from mellea.stdlib.components.intrinsic import decisions

model_id = IBM_GRANITE_SWITCH_4_1_8B.hf_model_name
assert model_id is not None

if model_id not in served:
    print(
        f"Skipped: llama-server at {BASE_URL} is not serving {model_id} "
        f"(found: {', '.join(sorted(served)) or 'nothing'}); start it with "
        f"llama-server -hf {model_id}-GGUF:Q4_K_M --alias {model_id} "
        "--port 8081 --jinja",
        file=sys.stderr,
    )
    raise SystemExit(1)

# The model_id is also used for client-side adapter discovery (io.yaml
# download), so it must name the checkpoint the server serves.
backend = OpenAIBackend(
    model_id=model_id,
    formatter=TemplateFormatter(model_id=model_id),
    base_url=BASE_URL,
    # llama-server ignores the api_key; the OpenAI client requires one.
    api_key="llamacpp",
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
