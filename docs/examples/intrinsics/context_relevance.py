# pytest: huggingface, e2e

"""Example usage of the context relevance intrinsic for RAG applications.

To run this script from the root of the Mellea source tree, use the command:
```
uv run python docs/examples/intrinsics/context_relevance.py
```
"""

from mellea.backends.huggingface import LocalHFBackend
from mellea.stdlib.components import Document
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
