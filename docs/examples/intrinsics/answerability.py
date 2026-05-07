# pytest: huggingface, e2e

"""Example usage of the answerability intrinsic for RAG applications.

To run this script from the root of the Mellea source tree, use the command:
```
uv run python docs/examples/intrinsics/answerability.py
```
"""

from mellea import model_ids, start_backend
from mellea.stdlib.components.intrinsic import rag

ctx, backend = start_backend(
    "hf", model_id=model_ids.IBM_GRANITE_4_1_3B, context_type="chat"
)
# NOTE: This example can also be run with the OpenAIBackend using a GraniteSwitch model. See docs/examples/granite-switch/.

result = rag.check_answerability(
    "What is the square root of 4?",
    documents=["The square root of 4 is 2."],
    context=ctx,
    backend=backend,
)
print(f"Result of answerability check when answer is in documents: {result}")

result = rag.check_answerability(
    "What is the square root of 4?",
    documents=["The square root of 8 is not 2."],
    context=ctx,
    backend=backend,
)
print(f"Result of answerability check when answer is not in documents: {result}")
