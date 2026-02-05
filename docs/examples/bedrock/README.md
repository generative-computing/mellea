# Using Mellea with Bedrock

Mellea can be used with Bedrock models via Mellea's LiteLLM or OpenAI backend support.

## Pre-requisites

To get started:

1. Set the `AWS_BEARER_TOKEN_BEDROCK` environment variable.

```bash
export AWS_BEARER_TOKEN_BEDROCK=<your API key goes here>
```

2. If you want to use litellm, you need to install the optional `litellm` dependencies:

```python
uv pip install mellea[litellm]
```

3. You can now use Bedrock. We've included some built-in model ids for convienance:


```python
from mellea import MelleaSession
from mellea.backends.bedrock import create_bedrock_mantle_backend
from mellea.backends.model_ids import OPENAI_GPT_OSS_120B
from mellea.stdlib.context import ChatContext

bedrock_oai_backend = create_bedrock_mantle_backend(model_id=OPENAI_GPT_OSS_120B, region="us-east-1")

m = MelleaSession(backend=bedrock_oai_backend, ctx=ChatContext())

print(m.chat("Tell me 3 facts about Amazon.").content)
```

You can also use your own model IDs as strings, as long as they're accessible using the [mantle endpoints](https://docs.aws.amazon.com/bedrock/latest/userguide/bedrock-mantle.html):

```python
from mellea import MelleaSession
from mellea.backends.bedrock import create_bedrock_mantle_backend
from mellea.stdlib.context import ChatContext

bedrock_oai_backend = create_bedrock_mantle_backend(
    model_id="qwen.qwen3-coder-480b-a35b-instruct", 
    region="us-east-1"
)

m = MelleaSession(backend=bedrock_oai_backend, ctx=ChatContext())

print(m.chat("Tell me 3 facts about Amazon.").content)
```

You can get a list of all models that are available at the mantle endpoint for a region by running this utility script:

```python
from mellea.backends.bedrock import stringify_mantle_model_ids
REGION = "us-east-1" # change this to see other region availability.
print(f"Available Models in {REGION}:\n{stringify_mantle_model_ids(region=REGION)}")
```

## Other Examples

Using LiteLLM with Bedrock (based on [litellm's docs](https://docs.litellm.ai/docs/providers/bedrock)):

```
uv run bedrock_litellm_example.py
```

Stand-alone email using the mantle OpenAI backends:

```python
uv run bedrock_openai_example.py
```