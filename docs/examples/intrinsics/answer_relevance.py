"""
Example usage of the answer relevance intrinsic for RAG applications.

To run this script from the root of the Mellea source tree, use the command:
```
uv run python docs/examples/intrinsics/answer_relevance.py
```
"""

from mellea.backends.huggingface import LocalHFBackend
from mellea.stdlib.base import ChatContext, Document
from mellea.stdlib.chat import Message
from mellea.stdlib.intrinsic import rag


backend = LocalHFBackend(model_id="ibm-granite/granite-4.0-micro")
context = ChatContext().add(Message("user", "Who attended the meeting?"))
documents = [
    Document("Meeting attendees: Alice, Bob, Carol."),
    Document("Meeting time: 9:00 am to 11:00 am."),
]
original_result = "Many people attended the meeting."

print(f"Result before relevance intrinsic: {original_result}")
result = rag.rewrite_answer_for_relevance(original_result, documents, context, backend)
print(f"Result after relevance intrinsic: {result}")
