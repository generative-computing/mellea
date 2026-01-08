"""
Example usage of the context relevance intrinsic for RAG applications.

To run this script from the root of the Mellea source tree, use the command:
```
uv run python docs/examples/intrinsics/context_relevance.py
```
"""

from mellea.backends.huggingface import LocalHFBackend
from mellea.stdlib.base import ChatContext, Document
from mellea.stdlib.chat import Message
from mellea.stdlib.intrinsics import rag


backend = LocalHFBackend(model_id="ibm-granite/granite-4.0-micro")
context = ChatContext()
question = "Who is the CEO of Microsoft?"
document = Document(
    # Document text does not say who is the CEO.
    "Microsoft Corporation is an American multinational corporation and technology "
    "conglomerate headquartered in Redmond, Washington.[2] Founded in 1975, the "
    "company became influential in the rise of personal computers through software "
    "like Windows, and the company has since expanded to Internet services, cloud "
    "computing, video gaming and other fields. Microsoft is the largest software "
    "maker, one of the most valuable public U.S. companies,[a] and one of the most "
    "valuable brands globally."
)

result = rag.check_context_relevance(question, document, context, backend)
print(f"Result of context relevance check with irrelevant document: {result}")
