# pytest: huggingface, e2e

"""Example usage of the uncertainty/certainty intrinsic.

Evaluates how certain the model is about its response to a user question.
The context should contain a user question followed by an assistant answer.

To run this script from the root of the Mellea source tree, use the command:
```
uv run python docs/examples/intrinsics/uncertainty.py
```
"""

from mellea.backends.huggingface import LocalHFBackend
from mellea.stdlib.components import Message
from mellea.stdlib.components.intrinsic import core
from mellea.stdlib.context import ChatContext

backend = LocalHFBackend(model_id="ibm-granite/granite-4.0-micro")
# --- Alternative: OpenAI backend with Granite Switch (requires vLLM server) ---
# Requires the adapter for this intrinsic to be embedded in the Granite Switch
# model. See docs/examples/granite-switch/ for a full runnable example.
# from mellea.backends.openai import OpenAIBackend
# from mellea.backends.model_ids import IBM_GRANITE_SWITCH_4_1_3B
# from mellea.formatters import TemplateFormatter
#
# backend = OpenAIBackend(
#     model_id=IBM_GRANITE_SWITCH_4_1_3B.hf_model_name,
#     formatter=TemplateFormatter(model_id=IBM_GRANITE_SWITCH_4_1_3B.hf_model_name),
#     base_url="http://localhost:8000/v1",  # vLLM server URL
#     api_key="EMPTY",
#     load_embedded_adapters=True,
# )
# --- End alternative ---
context = (
    ChatContext()
    .add(Message("user", "What is the square root of 4?"))
    .add(Message("assistant", "The square root of 4 is 2."))
)

result = core.check_certainty(context, backend)
print(f"Certainty score: {result}")
