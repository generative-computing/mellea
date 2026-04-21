---
title: "IBM WatsonX"
description: "Run Mellea with IBM WatsonX AI using the WatsonxAIBackend."
# diataxis: how-to
---

> **Deprecated:** The native WatsonX backend is deprecated since v0.4. Use the
> [LiteLLM](../how-to/backends-and-configuration#litellm-backend) or
> [OpenAI](../how-to/backends-and-configuration#openai-backend) backend with a
> WatsonX-compatible endpoint instead.

The WatsonX backend connects to IBM's managed AI platform. It requires an API key,
project ID, and service URL.

**Prerequisites:** `pip install 'mellea[watsonx]'` and IBM Cloud credentials.

## Credentials

```bash
export WATSONX_URL=https://us-south.ml.cloud.ibm.com
export WATSONX_API_KEY=your-watsonx-api-key
export WATSONX_PROJECT_ID=your-project-id
```

Obtain these from the IBM Cloud console:

- **API key:** [IBM Cloud IAM](https://cloud.ibm.com/iam/apikeys)
- **Project ID:** Your Watson Studio project settings
- **URL:** Region-specific endpoint (e.g., `https://us-south.ml.cloud.ibm.com`)

## Connecting

The quickest path is [`start_session()`](../guide/glossary#melleasession) with `backend_name="watsonx"`:

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

Or construct the [`Backend`](../guide/glossary#backend) directly for full control:

```python
from mellea import MelleaSession
from mellea.backends import model_ids
from mellea.backends.watsonx import WatsonxAIBackend

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

## Available models

| `model_ids` constant | WatsonX model name | Notes |
| -------------------- | ------------------ | ----- |
| `IBM_GRANITE_4_HYBRID_SMALL` | `ibm/granite-4-h-small` | Default WatsonX model |
| `IBM_GRANITE_3_3_8B` | `ibm/granite-3-3-8b-instruct` | |
| `IBM_GRANITE_3_2_8B` | `ibm/granite-3-2b-instruct` | |

Pass the WatsonX model name string directly for any model not listed in `model_ids`.

## Troubleshooting

**Missing credentials:**

```text
KeyError: WATSONX_URL / WATSONX_API_KEY / WATSONX_PROJECT_ID
```

All three environment variables must be set. Check your IBM Cloud project settings
for the correct values.

**`pip install "mellea[watsonx]"` required:**

The WatsonX backend requires the `ibm-watson-machine-learning` package, which is not
installed by default:

```bash
pip install 'mellea[watsonx]'
```

## Vision support

> **Note:** `WatsonxAIBackend` does not currently support image input. Passing
> `images=[...]` to `instruct()` or `chat()` will raise an error. Use the
> [OpenAI backend](./openai) or [Ollama](./ollama) for vision tasks.

---

**See also:** [Backends and Configuration](../how-to/backends-and-configuration)
