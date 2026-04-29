# pytest: huggingface, e2e

"""Example usage of the factuality detection intrinsic.

To run this script from the root of the Mellea source tree, use the command:
```
uv run python docs/examples/intrinsics/factuality_detection.py
```
"""

from mellea.backends.huggingface import LocalHFBackend
from mellea.stdlib.components import Document, Message
from mellea.stdlib.components.intrinsic import guardian
from mellea.stdlib.context import ChatContext

user_text = "Is Ozzy Osbourne still alive?"
response_text = "Yes, Ozzy Osbourne is alive in 2025 and preparing for another world tour, continuing to amaze fans with his energy and resilience."

document = Document(
    # Context says Ozzy Osbourne is dead, but the response says he is alive.
    "Ozzy Osbourne passed away on July 22, 2025, at the age of 76 from a heart attack. "
    "He died at his home in Buckinghamshire, England, with contributing conditions "
    "including coronary artery disease and Parkinson's disease. His final "
    "performance took place earlier that month in Birmingham."
)

# Create the backend.
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
    .add(document)
    .add(Message("user", user_text))
    .add(Message("assistant", response_text))
)

result = guardian.factuality_detection(context, backend)
print(f"Result of factuality detection: {result}")  # string "yes" or "no"
