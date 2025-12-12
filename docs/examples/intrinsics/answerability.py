"""
Example usage of the answerability intrinsic for RAG applications.

To run this script from the root of the Mellea source tree, use the command:
```
uv run python docs/examples/intrinsics/answerability.py
```
"""

from mellea.backends.huggingface import LocalHFBackend
from mellea.stdlib.base import ChatContext, Document
from mellea.stdlib.chat import Message
from mellea.stdlib.intrinsics import rag


backend = LocalHFBackend(model_id="ibm-granite/granite-3.3-2b-instruct")
context = ChatContext().add(Message("assistant", "Hello there, how can I help you?"))
next_user_turn = "What is the square root of 4?"
documents_answerable = [Document("The square root of 4 is 2.")]
documents_unanswerable = [Document("The square root of 8 is not 2.")]

result = rag.check_answerability(next_user_turn, documents_answerable, context, backend)
print(f"Result of answerability check when answer is in documents: {result}")

result = rag.check_answerability(
    next_user_turn, documents_unanswerable, context, backend
)
print(f"Result of answerability check when answer is not in documents: {result}")
