# pytest: huggingface, e2e

"""Example usage of the query rewrite intrinsic for RAG applications.

To run this script from the root of the Mellea source tree, use the command:
```
uv run python docs/examples/intrinsics/query_rewrite.py
```
"""

from mellea.backends.huggingface import LocalHFBackend
from mellea.stdlib.components import Message
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
context = (
    ChatContext()
    .add(Message("assistant", "Welcome to pet questions!"))
    .add(Message("user", "I have two pets, a dog named Rex and a cat named Lucy."))
    .add(
        Message(
            "assistant",
            "Rex spends a lot of time in the backyard and outdoors, "
            "and Luna is always inside.",
        )
    )
    .add(
        Message(
            "user",
            "Sounds good! Rex must love exploring outside, while Lucy "
            "probably enjoys her cozy indoor life.",
        )
    )
)
next_user_turn = "But is he more likely to get fleas because of that?"

print(f"Original user question: {next_user_turn}")

result = rag.rewrite_question(next_user_turn, context, backend)
print(f"Rewritten user question: {result}")
