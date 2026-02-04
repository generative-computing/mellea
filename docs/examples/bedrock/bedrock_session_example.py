try:
    import boto3
except:
    raise Exception(
        "Using Bedrock requires separately installing boto3."
        "Run `uv pip install mellea[aws]`"
    )

import mellea

MODEL_ID = "bedrock/converse/openai.gpt-oss-120b-1:0"

m = mellea.start_session(backend_name="litellm", model_id=MODEL_ID)

result = m.chat("Give me three facts about Amazon.")

print(result.content)
