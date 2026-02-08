# pytest: skip
# SKIP REASON: Requires an AWS bearer token for Bedrock.
#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "mellea[litellm]",
#   "boto3" # including so that this example works before the next release.
# ]
# ///
import os

import mellea

try:
    import boto3
except Exception:
    raise Exception(
        "Using Bedrock requires separately installing boto3. "
        "Run `uv pip install mellea[litellm]`"
    )

assert "AWS_BEARER_TOKEN_BEDROCK" in os.environ.keys(), (
    "Using AWS Bedrock requires setting a AWS_BEARER_TOKEN_BEDROCK environment variable. "
    "Generate a key from the AWS console at: https://us-east-1.console.aws.amazon.com/bedrock/home?region=us-east-1#/api-keys?tab=long-term "
    "Then run `export AWS_BEARER_TOKEN_BEDROCK=<insert your key here>"
)

MODEL_ID = "bedrock/converse/us.amazon.nova-pro-v1:0"

m = mellea.start_session(backend_name="litellm", model_id=MODEL_ID)

result = m.chat("Give me three facts about Amazon.")

print(result.content)
