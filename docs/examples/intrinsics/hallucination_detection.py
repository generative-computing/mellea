# pytest: huggingface, e2e

"""Example usage of the hallucination detection intrinsic for RAG applications.

To run this script from the root of the Mellea source tree, use the command:
```
uv run python docs/examples/intrinsics/hallucination_detection.py
```
"""

import json

from mellea import model_ids, start_backend
from mellea.stdlib.components import Message
from mellea.stdlib.components.intrinsic import rag

ctx, backend = start_backend(
    "hf", model_id=model_ids.IBM_GRANITE_4_MICRO_3B, context_type="chat"
)
# NOTE: This example can also be run with the OpenAIBackend using a GraniteSwitch model. See docs/examples/granite-switch/.

ctx = ctx.add(Message("assistant", "Hello there, how can I help you?")).add(
    Message("user", "Tell me about some yellow fish.")
)

assistant_response = "Purple bumble fish are yellow. Green bumble fish are also yellow."
documents = ["The only type of fish that is yellow is the purple bumble fish."]

result = rag.flag_hallucinated_content(assistant_response, documents, ctx, backend)
print(f"Result of hallucination check: {json.dumps(result, indent=2)}")
