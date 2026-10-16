# pytest: e2e, llamacpp_switch, qualitative

"""Example: Granite Switch adapter functions against a local llama.cpp server.

llama-server speaks the OpenAI chat-completions API, so Mellea reaches it
through the `openai` backend with a `base_url`, the same path as the vLLM
example in this directory, including per-request adapter activation through
`chat_template_kwargs.adapter_name` (llama.cpp v0.1.0 or newer carries the
granite-switch architecture; this example is tested against v0.6.0).

Marked `qualitative` alongside `e2e` because it asserts on the model's
answerability labels rather than only on structure, per the marker rules in
`test/README.md`.

Prerequisites (nothing in this example downloads a model; the server must
already be running):

    ./scripts/start_llamacpp_switch.sh

    The first run of that script downloads the checkpoint from the Hugging Face
    Hub (~7 GB for the 3b), converts it to GGUF (~9 GB more, both cached under
    ~/.cache/mellea/llamacpp-switch-build), and serves it on
    http://127.0.0.1:8081. Later runs reuse the cache and start in seconds.
    The port is deliberately not the base-Granite server's 8080, so both
    servers can run side by side.

When this example runs, as a plain script or collected by pytest:

- A llama-server is reachable at LLAMACPP_SWITCH_BASE_URL (default
  http://127.0.0.1:8081/v1) AND serves the Granite Switch alias → it runs.
- No server there, or the server is still loading → it skips, with the start
  command in the reason.
- A server is there but serves a different model → it skips, naming what it
  found, so the adapter calls are never handed the wrong weights.

The example pins the 3b preview alias (`IBM_GRANITE_SWITCH_4_1_3B_PREVIEW`);
serving a different Switch size makes it skip, naming what the server
presented, so swap the identifier in the example if you serve the 8b or 30b.

The example never downloads model weights. Its first run does fetch the small
adapter configuration files (`adapter_index.json` and the `io_configs/`) for
`load_embedded_adapters=True` from the Hub, which are cached.

To run this script from the root of the Mellea source tree:
```
uv run python docs/examples/granite-switch/answerability_llamacpp.py
```

This example carries the `llamacpp_switch` marker, deliberately not `llamacpp`:
that marker's conftest gate probes `LLAMACPP_BASE_URL` (default 8080) for the
base-Granite server, while `llamacpp_switch` probes `LLAMACPP_SWITCH_BASE_URL`
(default 8081), the server `scripts/start_llamacpp_switch.sh` starts.
"""

import os
import sys

import requests

BASE_URL = os.environ.get("LLAMACPP_SWITCH_BASE_URL", "http://127.0.0.1:8081/v1")

try:
    response = requests.get(f"{BASE_URL}/models", timeout=5)
    response.raise_for_status()
except requests.RequestException:
    # Covers both "nothing listening" and "still loading weights": llama-server
    # answers 503 from every endpoint until the model is in. Detected by
    # docs/examples/conftest.py subprocess runner as a skip.
    print(
        f"Skipped: no llama-server ready at {BASE_URL} "
        "(start scripts/start_llamacpp_switch.sh, or set LLAMACPP_SWITCH_BASE_URL)",
        file=sys.stderr,
    )
    raise SystemExit(1)

from mellea.backends.model_ids import IBM_GRANITE_SWITCH_4_1_3B_PREVIEW
from mellea.backends.openai import OpenAIBackend
from mellea.formatters import TemplateFormatter
from mellea.stdlib.components import Document, Message
from mellea.stdlib.components.intrinsic import rag
from mellea.stdlib.context import ChatContext

SWITCH_MODEL_ID = IBM_GRANITE_SWITCH_4_1_3B_PREVIEW.hf_model_name
assert SWITCH_MODEL_ID is not None

served = {model["id"] for model in response.json()["data"]}
if SWITCH_MODEL_ID not in served:
    # Detected by docs/examples/conftest.py subprocess runner as a skip.
    print(
        f"Skipped: llama-server at {BASE_URL} is not serving {SWITCH_MODEL_ID} "
        f"(found: {', '.join(sorted(served)) or 'nothing'}); "
        "start scripts/start_llamacpp_switch.sh",
        file=sys.stderr,
    )
    raise SystemExit(1)

backend = OpenAIBackend(
    model_id=SWITCH_MODEL_ID,
    formatter=TemplateFormatter(model_id=SWITCH_MODEL_ID),
    base_url=BASE_URL,
    # llama-server ignores the api_key; the OpenAI client requires one.
    api_key="llamacpp",
    load_embedded_adapters=True,
)

context = ChatContext().add(Message("assistant", "Hello there, how can I help you?"))
question = "What is the square root of 4?"

# Both calls must reach the documents: before issue #1735's fix, llama-server
# dropped them and every question came back "unanswerable".
result_answerable = rag.check_answerability(
    question, [Document("The square root of 4 is 2.")], context, backend
)
print(f"Answerability (answer in docs): {result_answerable}")
assert result_answerable == "answerable", (
    f"expected 'answerable' when the documents contain the answer, "
    f"got {result_answerable!r}"
)

result_unanswerable = rag.check_answerability(
    question, [Document("The square root of 8 is not 2.")], context, backend
)
print(f"Answerability (answer NOT in docs): {result_unanswerable}")
assert result_unanswerable == "unanswerable", (
    f"expected 'unanswerable' when the documents lack the answer, "
    f"got {result_unanswerable!r}"
)
