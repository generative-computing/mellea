# Using Mellea with Bedrock

Mellea can be used with Bedrock models via Mellea's LiteLLM or OpenAI backend support.
For setup, model IDs, regions, and troubleshooting, see
[AWS Bedrock](../../docs/integrations/bedrock.md).

You can use your own model IDs as strings, as long as they're accessible using the [mantle endpoints](https://docs.aws.amazon.com/bedrock/latest/userguide/bedrock-mantle.html).

You can get a list of all models that are available at the mantle endpoint for a region by running this utility script:

```python
from mellea.backends.bedrock import stringify_mantle_model_ids
REGION = "us-east-1" # change this to see other region availability.
print(f"Available Models in {REGION}:\n{stringify_mantle_model_ids(region=REGION)}")
```

## Examples

| File | Description |
| :--- | :--- |
| [`bedrock_litellm_example.py`](./bedrock_litellm_example.py) | Using LiteLLM with Bedrock (based on [litellm's docs](https://docs.litellm.ai/docs/providers/bedrock)). |
| [`bedrock_openai_example.py`](./bedrock_openai_example.py) | Stand-alone email using the mantle OpenAI backends. |
