# pytest: huggingface, e2e

import mellea.stdlib.functional as mfuncs
from mellea.backends import model_ids
from mellea.backends.adapters.adapter import AdapterType, IntrinsicAdapter
from mellea.backends.huggingface import LocalHFBackend
from mellea.stdlib.components import Intrinsic, Message
from mellea.stdlib.context import ChatContext

# This is an example for how you would directly use intrinsics. See `mellea/stdlib/intrinsics/rag.py`
# for helper functions.

backend = LocalHFBackend(model_id=model_ids.IBM_GRANITE_4_1_3B)
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

# Create the Adapter. IntrinsicAdapter's default to ALORAs.
req_adapter = IntrinsicAdapter(
    "requirement_check", base_model_name=backend.base_model_name
)

# Add the adapter to the backend.
backend.add_adapter(req_adapter)

ctx = ChatContext()
ctx = ctx.add(Message("user", "Hi, can you help me?"))
ctx = ctx.add(Message("assistant", "Hello; yes! What can I help with?"))

# Generate from an intrinsic with the same name as the adapter. By default, it will look for
# ALORA and then LORA adapters.
out, new_ctx = mfuncs.act(
    Intrinsic(
        "requirement_check",
        intrinsic_kwargs={"requirement": "The assistant is helpful."},
    ),
    ctx,
    backend,
)

# Print the output. The requirement_check adapter has a specific output format:
print(out)  # {"requirement_likelihood": 1.0}

# The AloraRequirement uses this adapter. It automatically parses that output
# when validating the output.
