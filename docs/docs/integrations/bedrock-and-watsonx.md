---
title: "AWS Bedrock and IBM WatsonX"
description: "Run Mellea with AWS Bedrock models and IBM WatsonX using the Bedrock Mantle and WatsonX backends."
# diataxis: how-to
---

# AWS Bedrock and IBM WatsonX

Mellea provides backends for AWS Bedrock and IBM WatsonX for enterprise deployments.
Both require cloud credentials and optional extra packages.

## AWS Bedrock

Mellea accesses AWS Bedrock via the **Bedrock Mantle** endpoint, which exposes an
OpenAI-compatible API. Authentication uses an AWS Bearer Token.

**Prerequisites:** `pip install mellea` (no extra needed — uses the OpenAI client
already included), a valid `AWS_BEARER_TOKEN_BEDROCK` value.

### Getting a Bedrock API key

Generate a long-term API key from the AWS console:
[us-east-1 Bedrock API keys](https://us-east-1.console.aws.amazon.com/bedrock/home?region=us-east-1#/api-keys?tab=long-term)

Export it before running Mellea:

```bash
export AWS_BEARER_TOKEN_BEDROCK=your-bedrock-key
```

### Connecting with `create_bedrock_mantle_backend`

```python
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

`create_bedrock_mantle_backend` returns an `OpenAIBackend` pointed at the Bedrock
Mantle endpoint. It reads `AWS_BEARER_TOKEN_BEDROCK` from the environment and checks
that the requested model is available in the target region before returning.

### Specifying a region

The default region is `us-east-1`. Pass `region` to target a different region:

```python
from mellea import MelleaSession
from mellea.backends.bedrock import create_bedrock_mantle_backend

m = MelleaSession(
    backend=create_bedrock_mantle_backend(
        model_id="amazon.nova-pro-v1:0",
        region="eu-west-1",
    )
)
```

### Using a model string directly

If the `ModelIdentifier` for a Bedrock model is not in `model_ids`, pass the Bedrock
model ID string directly:

```python
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
from mellea.backends.bedrock import stringify_mantle_model_ids

print(stringify_mantle_model_ids())
```

### Bedrock via LiteLLM

An alternative path to Bedrock is the LiteLLM backend, which uses the standard AWS
credentials chain (IAM roles, `~/.aws/credentials`, environment variables):

```bash
pip install 'mellea[litellm]'
export AWS_BEARER_TOKEN_BEDROCK=your-bedrock-key
```

```python
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

---

## IBM WatsonX

The WatsonX backend connects to IBM's managed AI platform. It requires an API key,
project ID, and service URL.

**Prerequisites:** `pip install 'mellea[watsonx]'` and IBM Cloud credentials.

### Credentials

```bash
export WATSONX_URL=https://us-south.ml.cloud.ibm.com
export WATSONX_API_KEY=your-watsonx-api-key
export WATSONX_PROJECT_ID=your-project-id
```

Obtain these from the IBM Cloud console:

- **API key:** [IBM Cloud IAM](https://cloud.ibm.com/iam/apikeys)
- **Project ID:** Your Watson Studio project settings
- **URL:** Region-specific endpoint (e.g., `https://us-south.ml.cloud.ibm.com`)

### Connecting

```python
from mellea import start_session

m = start_session(
    backend_name="watsonx",
    model_id="ibm/granite-4-h-small",
)
result = m.instruct("Summarise this document in three bullet points.")
print(str(result))
# Output will vary — LLM responses depend on model and temperature.
```

Or construct the backend directly for full control:

```python
from mellea import MelleaSession
from mellea.backends.watsonx import WatsonxAIBackend
from mellea.backends import model_ids

m = MelleaSession(
    WatsonxAIBackend(model_id=model_ids.IBM_GRANITE_4_HYBRID_SMALL)
)
```

Credentials are read from the environment variables by default. Pass them explicitly
if needed:

```python
from mellea import MelleaSession
from mellea.backends.watsonx import WatsonxAIBackend

m = MelleaSession(
    WatsonxAIBackend(
        model_id="ibm/granite-3-3-8b-instruct",
        base_url="https://us-south.ml.cloud.ibm.com",
        api_key="your-api-key",
        project_id="your-project-id",
    )
)
```

### Available WatsonX models

| `model_ids` constant | WatsonX model name | Notes |
| -------------------- | ------------------ | ----- |
| `IBM_GRANITE_4_HYBRID_SMALL` | `ibm/granite-4-h-small` | Default WatsonX model |
| `IBM_GRANITE_3_3_8B` | `ibm/granite-3-3-8b-instruct` | |
| `IBM_GRANITE_3_2_8B` | `ibm/granite-3-2b-instruct` | |

Pass the WatsonX model name string directly for any model not listed in `model_ids`.

---

## Troubleshooting

### Bedrock: `AWS_BEARER_TOKEN_BEDROCK` not set

```text
AssertionError: Using AWS Bedrock requires setting a AWS_BEARER_TOKEN_BEDROCK environment variable.
```

Export the environment variable before running your script:

```bash
export AWS_BEARER_TOKEN_BEDROCK=your-key
```

### Bedrock: model not available in region

```text
Model X is not supported in region us-east-1.
```

Either enable model access for the requested model in your AWS account
[Bedrock Model Access](https://us-east-1.console.aws.amazon.com/bedrock/home#/model-access),
or pass a different `region` to `create_bedrock_mantle_backend`.

### WatsonX: missing credentials

```text
KeyError: WATSONX_URL / WATSONX_API_KEY / WATSONX_PROJECT_ID
```

All three environment variables must be set. Check your IBM Cloud project settings
for the correct values.

### WatsonX: `pip install mellea[watsonx]` required

The WatsonX backend requires the `ibm-watson-machine-learning` package, which is not
installed by default:

```bash
pip install 'mellea[watsonx]'
```

---

**Previous:** [OpenAI and OpenAI-Compatible APIs](./openai.md) |
**Next:** [HuggingFace and vLLM](./huggingface-and-vllm.md)

**See also:** [Backends and Configuration](../guide/backends-and-configuration.md)
