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
from mellea.backends import model_ids
from mellea.backends.bedrock import create_bedrock_mantle_backend
from mellea.backends.openai import OpenAIBackend
from mellea.stdlib.context import ChatContext

assert "AWS_BEARER_TOKEN_BEDROCK" in os.environ.keys(), (
    "Using AWS Bedrock requires setting a AWS_BEARER_TOKEN_BEDROCK environment variable.\n\nTo proceed:\n"
    "\n\t1. Generate a key from the AWS console at: https://us-east-1.console.aws.amazon.com/bedrock/home?region=us-east-1#/api-keys?tab=long-term "
    "\n\t2. Run `export AWS_BEARER_TOKEN_BEDROCK=<insert your key here>"
)

m = MelleaSession(
    backend=create_bedrock_mantle_backend(model_id=model_ids.OPENAI_GPT_OSS_120B),
    ctx=ChatContext(),
)

result = m.chat("Give me three facts about Amazon.")

print(result.content)
