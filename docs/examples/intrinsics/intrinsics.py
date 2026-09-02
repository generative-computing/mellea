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
# --- Alternative: local Granite Switch checkpoint ---
# Requires: uv sync --extra hf
# See docs/examples/granite-switch/answerability_local_hf.py for a runnable example.
# from mellea.backends.huggingface import LocalHFBackend
# from mellea.backends.model_ids import IBM_GRANITE_SWITCH_4_1_3B_PREVIEW
#
# backend = LocalHFBackend(
#     model_id=IBM_GRANITE_SWITCH_4_1_3B_PREVIEW,
#     load_embedded_adapters=True,
# )
# --- End alternative ---

# Create the Adapter. IntrinsicAdapter's default to ALORAs.
req_adapter = IntrinsicAdapter(
    "requirement-check", base_model_name=backend.base_model_name
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
        "requirement-check",
        intrinsic_kwargs={"requirement": "The assistant is helpful."},
    ),
    ctx,
    backend,
)

# Print the output. The requirement-check adapter has a specific output format:
print(out)  # {"requirement_check": {"score": 0.41272119992000356}}

# The AloraRequirement uses this adapter. It automatically parses that output
# when validating the output.
