import os

import mellea

try:
    import boto3
except Exception:
    raise Exception(
        "Using Bedrock requires separately installing boto3."
        "Run `uv pip install mellea[aws]`"
    )

assert "AWS_BEARER_TOKEN_BEDROCK" in os.environ.keys(), (
    "Using AWS Bedrock requires setting a AWS_BEARER_TOKEN_BEDROCK environment variable. "
    "Generate a key from the AWS console at: https://us-east-1.console.aws.amazon.com/bedrock/home?region=us-east-1#/api-keys?tab=long-term "
    "Then run `export AWS_BEARER_TOKEN_BEDROCK=<insert your key here>"
)

MODEL_ID = "bedrock/converse/openai.gpt-oss-120b-1:0"

m = mellea.start_session(backend_name="litellm", model_id=MODEL_ID)

result = m.chat("Give me three facts about Amazon.")

print(result.content)
