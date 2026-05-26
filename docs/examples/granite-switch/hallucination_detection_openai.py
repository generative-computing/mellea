# pytest: e2e, vllm, skip

"""Example: running hallucination detection via OpenAI backend with Granite Switch.

Requires a vLLM server hosting a Granite Switch model.

To start the server:
    python -m vllm.entrypoints.openai.api_server \
        --model <granite-switch-model-id> \
        --dtype bfloat16 --enable-prefix-caching

To run this script from the root of the Mellea source tree:
    uv run python docs/examples/granite-switch/hallucination_detection_openai.py
"""

import json
import os
import sys

import requests

VLLM_BASE_URL = os.environ.get("VLLM_SWITCH_TEST_BASE_URL", "http://localhost:8000")
try:
    requests.get(f"{VLLM_BASE_URL}/v1/models", timeout=2)
except requests.ConnectionError:
    print(f"Skipped: vLLM server not reachable at {VLLM_BASE_URL}", file=sys.stderr)
    raise SystemExit(1)

from mellea.backends.model_ids import IBM_GRANITE_SWITCH_4_1_3B_PREVIEW
from mellea.backends.openai import OpenAIBackend
from mellea.formatters import TemplateFormatter
from mellea.stdlib.components import Document, Message
from mellea.stdlib.components.intrinsic import rag
from mellea.stdlib.context import ChatContext

SWITCH_MODEL_ID = IBM_GRANITE_SWITCH_4_1_3B_PREVIEW.hf_model_name
assert SWITCH_MODEL_ID is not None

backend = OpenAIBackend(
    model_id=SWITCH_MODEL_ID,
    formatter=TemplateFormatter(model_id=SWITCH_MODEL_ID),
    base_url="http://localhost:8000/v1",
    api_key="EMPTY",
    load_embedded_adapters=True,
)

context = (
    ChatContext()
    .add(Message("assistant", "Hello there, how can I help you?"))
    .add(Message("user", "Tell me about some yellow fish."))
)

assistant_response = "Purple bumble fish are yellow. Green bumble fish are also yellow."

documents = [
    Document(
        doc_id="1",
        text="The only type of fish that is yellow is the purple bumble fish.",
    )
]

result = rag.flag_hallucinated_content(assistant_response, documents, context, backend)
print(f"Result of hallucination check: {json.dumps(result, indent=2)}")
