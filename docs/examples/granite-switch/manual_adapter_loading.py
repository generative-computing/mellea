# pytest: e2e, vllm, skip

"""Example: manually loading embedded adapters for the OpenAI backend.

Instead of using ``load_embedded_adapters=True`` (which loads all adapters from
the model repo at init), this example shows how to create an OpenAIBackend with
``load_embedded_adapters=False`` and then manually load individual adapters using
``EmbeddedIntrinsicAdapter.from_hub()`` or ``from_model_directory()``.

This is useful when:
- You only need a subset of the model's embedded adapters.
- You want to load adapters from a local directory rather than HuggingFace Hub.
- You want more control over adapter registration.

Requires a vLLM server hosting a Granite Switch model.

To start the server:
    python -m vllm.entrypoints.openai.api_server \
        --model <granite-switch-model-id> \
        --dtype bfloat16 --enable-prefix-caching

To run this script from the root of the Mellea source tree:
    uv run python docs/examples/granite-switch/manual_adapter_loading.py
"""

import os
import sys

import requests

VLLM_BASE_URL = os.environ.get("VLLM_SWITCH_TEST_BASE_URL", "http://localhost:8000")
try:
    requests.get(f"{VLLM_BASE_URL}/v1/models", timeout=2)
except requests.ConnectionError:
    print(f"Skipped: vLLM server not reachable at {VLLM_BASE_URL}", file=sys.stderr)
    raise SystemExit(1)

from mellea.backends.adapters.adapter import EmbeddedIntrinsicAdapter
from mellea.backends.model_ids import IBM_GRANITE_SWITCH_4_1_8B
from mellea.backends.openai import OpenAIBackend
from mellea.formatters import TemplateFormatter
from mellea.stdlib.components import Document, Message
from mellea.stdlib.components.intrinsic import rag
from mellea.stdlib.context import ChatContext

# Create the backend WITHOUT auto-loading adapters.
backend = OpenAIBackend(
    model_id=IBM_GRANITE_SWITCH_4_1_8B.hf_model_name,
    formatter=TemplateFormatter(model_id=IBM_GRANITE_SWITCH_4_1_8B.hf_model_name),
    base_url="http://localhost:8000/v1",
    api_key="EMPTY",
    load_embedded_adapters=False,
)

# --- Option A: Load a single adapter from HuggingFace Hub ---
adapters = EmbeddedIntrinsicAdapter.from_hub(
    IBM_GRANITE_SWITCH_4_1_8B.hf_model_name, intrinsic_name="answerability"
)
for adapter in adapters:
    backend.add_adapter(adapter)

# --- Option B (alternative): Load from a local model directory ---
# adapters = EmbeddedIntrinsicAdapter.from_model_directory(
#     "/path/to/local/granite-switch-model",
#     intrinsic_name="answerability",
# )
# for adapter in adapters:
#     backend.add_adapter(adapter)

print(f"Registered adapters: {backend.list_adapters()}")

# Now use the intrinsic as usual.
context = ChatContext().add(Message("assistant", "Hello there, how can I help you?"))
question = "What is the square root of 4?"
documents = [Document("The square root of 4 is 2.")]

result = rag.check_answerability(question, documents, context, backend)
print(f"Answerability result: {result}")
