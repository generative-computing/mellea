# pytest: skip
# SKIP REASON: Requires an AWS bearer token for Bedrock.
#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "mellea[openai]"
# ]
# ///
import os

from mellea import MelleaSession
from mellea.backends.openai import OpenAIBackend
from mellea.stdlib.context import ChatContext

assert "AWS_BEARER_TOKEN_BEDROCK" in os.environ.keys(), (
    "Using AWS Bedrock requires setting a AWS_BEARER_TOKEN_BEDROCK environment variable.\n\nTo proceed:\n"
    "\n\t1. Generate a key from the AWS console at: https://us-east-1.console.aws.amazon.com/bedrock/home?region=us-east-1#/api-keys?tab=long-term "
    "\n\t2. Run `export AWS_BEARER_TOKEN_BEDROCK=<insert your key here>"
)

MODEL_ID = "openai.gpt-oss-120b-1:0"

m = MelleaSession(
    backend=OpenAIBackend(
        base_url="https://bedrock-runtime.us-west-2.amazonaws.com/openai/v1",
        model_id=MODEL_ID,
        api_key=os.environ["AWS_BEARER_TOKEN_BEDROCK"],
    ),
    ctx=ChatContext(),
)

result = m.chat("Give me three facts about Amazon.")

print(result.content)
