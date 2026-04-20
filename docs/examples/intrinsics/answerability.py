# pytest: huggingface, e2e

"""Example usage of the answerability intrinsic for RAG applications.

To run this script from the root of the Mellea source tree, use the command:
```
uv run python docs/examples/intrinsics/answerability.py
```
"""

from mellea.backends.huggingface import LocalHFBackend
from mellea.stdlib.components import Document, Message
from mellea.stdlib.components.intrinsic import rag
from mellea.stdlib.context import ChatContext

backend = LocalHFBackend(model_id="ibm-granite/granite-4.0-micro")
# --- Alternative: OpenAI backend with Granite Switch (requires vLLM server) ---
# Requires the adapter for this intrinsic to be embedded in the Granite Switch
# model. See docs/examples/granite-switch/ for a full runnable example.
# from mellea.backends.openai import OpenAIBackend
# from mellea.backends.model_ids import IBM_GRANITE_SWITCH_4_1_8B
# from mellea.formatters import TemplateFormatter
#
# backend = OpenAIBackend(
#     model_id=IBM_GRANITE_SWITCH_4_1_8B.hf_model_name,
#     formatter=TemplateFormatter(model_id=IBM_GRANITE_SWITCH_4_1_8B.hf_model_name),
#     base_url="http://localhost:8000/v1",  # vLLM server URL
#     api_key="EMPTY",
#     load_embedded_adapters=True,
# )
# --- End alternative ---
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
