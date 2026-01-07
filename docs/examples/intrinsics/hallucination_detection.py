"""
Example usage of the hallucination detection intrinsic for RAG applications.

To run this script from the root of the Mellea source tree, use the command:
```
uv run python docs/examples/intrinsics/hallucination_detection.py
```
"""

from mellea.backends.huggingface import LocalHFBackend
from mellea.stdlib.base import ChatContext, Document
from mellea.stdlib.chat import Message
from mellea.stdlib.intrinsics import rag
import json


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
