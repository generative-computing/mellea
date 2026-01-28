"""Example usage of the hallucination detection intrinsic for RAG applications.

To run this script from the root of the Mellea source tree, use the command:
```
uv run python docs/examples/intrinsics/hallucination_detection.py
```
"""

import json

from mellea.backends.huggingface import LocalHFBackend
from mellea.stdlib.components import Document, Message
from mellea.stdlib.components.intrinsic import rag
from mellea.stdlib.context import ChatContext

backend = LocalHFBackend(model_id="ibm-granite/granite-4.0-micro")
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
