# pytest: skip
# SKIP REASON: Requires an AWS bearer token for Bedrock.
#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "mellea[litellm]",
#   "boto3" # including so that this example works before the next release.
# ]
# ///
import os

import mellea
from mellea.backends.bedrock import create_bedrock_litellm_backend
from mellea.backends.model_ids import MISTRALAI_DEVSTRAL_2_123B
from mellea.stdlib.context import SimpleContext

try:
    import boto3
except Exception:
    raise Exception(
        "Using Bedrock requires separately installing boto3. "
        "Run `uv pip install mellea[litellm]`"
    )

MODEL_ID = MISTRALAI_DEVSTRAL_2_123B

backend = create_bedrock_litellm_backend(MODEL_ID)
ctx = SimpleContext()
m = mellea.MelleaSession(backend, ctx)

result = m.chat("What model am I talking to rn?")

print(result.content)
