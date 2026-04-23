---
title: "AWS Bedrock"
description: "Run Mellea with AWS Bedrock models using the Bedrock Mantle backend or LiteLLM."
# diataxis: how-to
---

Mellea accesses AWS Bedrock via the **Bedrock Mantle** endpoint, which exposes an
OpenAI-compatible API authenticated with an AWS Bearer Token.

**Prerequisites:** `pip install mellea` (no extra needed — uses the OpenAI client
already included), a valid `AWS_BEARER_TOKEN_BEDROCK` value.

## Getting a Bedrock API key

Generate a long-term API key from the AWS console:
[us-east-1 Bedrock API keys](https://us-east-1.console.aws.amazon.com/bedrock/home?region=us-east-1#/api-keys?tab=long-term)

Export it before running Mellea:

```bash
export AWS_BEARER_TOKEN_BEDROCK=your-bedrock-key
```

## Connecting with `create_bedrock_mantle_backend`

```python
# Requires: mellea[openai]
# Returns: ModelOutputThunk
from mellea import MelleaSession
from mellea.backends import model_ids
from mellea.backends.bedrock import create_bedrock_mantle_backend
from mellea.stdlib.context import ChatContext

m = MelleaSession(
    backend=create_bedrock_mantle_backend(model_id=model_ids.OPENAI_GPT_OSS_120B),
    ctx=ChatContext(),
)

result = m.chat("Give me three facts about the Amazon rainforest.")
print(str(result))
# Output will vary — LLM responses depend on model and temperature.
```

`create_bedrock_mantle_backend` returns an [`OpenAIBackend`](../reference/glossary#backend) pointed at the Bedrock
Mantle endpoint. Pass it to [`MelleaSession`](../reference/glossary#melleasession) as shown above. It reads `AWS_BEARER_TOKEN_BEDROCK` from the environment and checks
that the requested model is available in the target region before returning.

## Specifying a region

The default region is `us-east-1`. Pass `region` to target a different region:

```python
# Requires: mellea[openai]
# Returns: MelleaSession
from mellea import MelleaSession
from mellea.backends.bedrock import create_bedrock_mantle_backend

m = MelleaSession(
    backend=create_bedrock_mantle_backend(
        model_id="amazon.nova-pro-v1:0",
        region="eu-west-1",
    )
)
```

## Using a model string directly

If the `ModelIdentifier` for a Bedrock model is not in `model_ids`, pass the Bedrock
model ID string directly:

```python
# Requires: mellea[openai]
# Returns: MelleaSession
from mellea import MelleaSession
from mellea.backends.bedrock import create_bedrock_mantle_backend

m = MelleaSession(
    backend=create_bedrock_mantle_backend(
        model_id="anthropic.claude-3-haiku-20240307-v1:0"
    )
)
```

Listing available models in your region:

```python
# Returns: str
from mellea.backends.bedrock import stringify_mantle_model_ids

print(stringify_mantle_model_ids())
```

## Bedrock via LiteLLM

An alternative path to Bedrock is the [`LiteLLMBackend`](../reference/glossary#litellm--litellmbackend),
which uses the standard AWS credentials chain (IAM roles, `~/.aws/credentials`,
environment variables):

```bash
pip install 'mellea[litellm]'
export AWS_BEARER_TOKEN_BEDROCK=your-bedrock-key
```

```python
# Requires: mellea[litellm]
# Returns: str
import mellea

m = mellea.start_session(
    backend_name="litellm",
    model_id="bedrock/converse/us.amazon.nova-pro-v1:0",
)
result = m.chat("Give me three facts about the Amazon rainforest.")
print(str(result))
# Output will vary — LLM responses depend on model and temperature.
```

The LiteLLM model ID format for Bedrock is `bedrock/converse/<bedrock-model-id>`.
See the [LiteLLM documentation](https://docs.litellm.ai/docs/providers/bedrock) for
available model IDs and credential setup.

> **Full example:** [`docs/examples/bedrock/bedrock_openai_example.py`](https://github.com/generative-computing/mellea/blob/main/docs/examples/bedrock/bedrock_openai_example.py)

## Troubleshooting

**`AWS_BEARER_TOKEN_BEDROCK` not set:**

```text
AssertionError: Using AWS Bedrock requires setting a AWS_BEARER_TOKEN_BEDROCK environment variable.
```

Export the environment variable before running your script:

```bash
export AWS_BEARER_TOKEN_BEDROCK=your-key
```

**Model not available in region:**

```text
Model X is not supported in region us-east-1.
```

Either enable model access for the requested model in your AWS account at
[Bedrock Model Access](https://us-east-1.console.aws.amazon.com/bedrock/home#/model-access),
or pass a different `region` to `create_bedrock_mantle_backend`.

## Vision support

Bedrock models accessed via the Mantle endpoint use the `OpenAIBackend` under the hood,
so vision-capable models (e.g., `amazon.nova-pro-v1:0`) support image input via
`images=[...]`. Pass a PIL image or an [`ImageBlock`](../reference/glossary#imageblock) to
`instruct()` or `chat()`. See [Use Images and Vision Models](../how-to/use-images-and-vision).

---

**See also:** [Backends and Configuration](../how-to/backends-and-configuration)
