import mellea.stdlib.functional as mfuncs
from mellea.backends.adapters.adapter import AdapterType, GraniteCommonAdapter
from mellea.backends.huggingface import LocalHFBackend
from mellea.backends.openai import OpenAIBackend, _ServerType
from mellea.stdlib.components import Intrinsic, Message
from mellea.stdlib.context import ChatContext

# This is an example for how you would directly use intrinsics. See `mellea/stdlib/intrinsics/rag.py`
# for helper functions.

# Create the backend. Example for a VLLM Server. Commented out in favor of the hugging face code for now.
# # Assumes a locally running VLLM server.
# backend = OpenAIBackend(
#     model_id="ibm-granite/granite-3.3-8b-instruct",
#     base_url="http://0.0.0.0:8000/v1",
#     api_key="EMPTY",
# )

# # If using a remote VLLM server, utilize the `test/backends/test_openai_vllm/serve.sh`
# # script with `export VLLM_DOWNLOAD_RAG_INTRINSICS=True`. This will download the granite_common
# # adapters on the server.
# backend._server_type = _ServerType.REMOTE_VLLM

backend = LocalHFBackend(model_id="ibm-granite/granite-3.3-8b-instruct")

# Create the Adapter. GraniteCommonAdapter's default to ALORAs.
req_adapter = GraniteCommonAdapter(
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
