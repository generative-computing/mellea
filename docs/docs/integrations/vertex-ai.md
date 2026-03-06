---
title: "Vertex AI"
description: "Connect Mellea to Google Vertex AI models via LiteLLM."
# diataxis: how-to
---

Mellea reaches Google Vertex AI through the `LiteLLMBackend`. There is no
separate native Vertex backend — LiteLLM handles authentication and request
translation.

**Prerequisites:**

```bash
pip install 'mellea[litellm]'
pip install google-cloud-aiplatform
```

You also need a Google Cloud project with the Vertex AI API enabled.

## Authentication

LiteLLM supports two authentication methods for Vertex AI.

### Application Default Credentials (recommended for local development)

Run the following command once to authenticate with your Google account:

```bash
gcloud auth application-default login
```

This stores credentials that LiteLLM picks up automatically. No environment
variable pointing to a file is required.

### Service account key file

For production deployments, create a service account in the Google Cloud
console, grant it the `Vertex AI User` role, download the JSON key, and export
its path:

```bash
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account-key.json
```

> **Note:** Never commit service account key files to source control. Store
> them in a secrets manager or inject them as environment variables at deploy
> time.

## Required environment variables

LiteLLM reads the project and region from these two variables:

```bash
export VERTEXAI_PROJECT=your-gcp-project-id
export VERTEXAI_LOCATION=us-central1
```

Set `VERTEXAI_LOCATION` to the region where your Vertex AI endpoints are
deployed. Common values are `us-central1`, `europe-west4`, and `asia-east1`.

## Connecting Mellea to Vertex AI

Use `LiteLLMBackend` with a `vertex_ai/` or `vertex_ai_beta/` model string:

```python
import os

from mellea import MelleaSession
from mellea.backends.litellm import LiteLLMBackend

backend = LiteLLMBackend(
    model_id="vertex_ai/gemini-1.5-pro",
    model_options={
        "vertex_project": os.environ["VERTEXAI_PROJECT"],
        "vertex_location": os.environ["VERTEXAI_LOCATION"],
    },
)
m = MelleaSession(backend=backend)

result = m.instruct("Summarise the key points of the Vertex AI documentation.")
print(str(result))
# Output will vary — LLM responses depend on model and temperature.
```

> **Note (review needed):** The `vertex_project` and `vertex_location` keys
> shown above follow the LiteLLM convention for per-call overrides. Hendrik,
> please confirm whether these keys are correct and whether they are required
> when `VERTEXAI_PROJECT` / `VERTEXAI_LOCATION` are already set in the
> environment, or whether they are only needed to override the environment
> values.

## Model string format

The LiteLLM model string for Vertex AI follows this pattern:

```text
vertex_ai/<model-name>
vertex_ai_beta/<model-name>
```

Use `vertex_ai_beta/` for models that are only available through the Vertex AI
Preview SDK endpoint. Common model strings:

| Model | LiteLLM string |
| ----- | -------------- |
| Gemini 1.5 Pro | `vertex_ai/gemini-1.5-pro` |
| Gemini 1.5 Flash | `vertex_ai/gemini-1.5-flash` |
| Gemini Pro | `vertex_ai/gemini-pro` |
| Gemini 2.0 Flash (preview) | `vertex_ai_beta/gemini-2.0-flash-exp` |

Check the [LiteLLM Vertex AI documentation](https://docs.litellm.ai/docs/providers/vertex)
for the full list of supported model strings.

## Using `chat()` and `instruct()`

Both `chat()` and `instruct()` work with `LiteLLMBackend` in the same way as
other backends:

```python
import os

from mellea import MelleaSession
from mellea.backends.litellm import LiteLLMBackend
from mellea.stdlib.context import ChatContext

backend = LiteLLMBackend(
    model_id="vertex_ai/gemini-1.5-flash",
    model_options={
        "vertex_project": os.environ["VERTEXAI_PROJECT"],
        "vertex_location": os.environ["VERTEXAI_LOCATION"],
    },
)
m = MelleaSession(backend=backend, ctx=ChatContext())

reply = m.chat("What is the capital of France?")
print(str(reply))
# Output will vary — LLM responses depend on model and temperature.
```

## Structured output

Use the `format` parameter with a Pydantic model to get typed responses:

```python
import os

from pydantic import BaseModel

from mellea import MelleaSession
from mellea.backends.litellm import LiteLLMBackend


class KeyPoints(BaseModel):
    points: list[str]
    source_quality: str


backend = LiteLLMBackend(
    model_id="vertex_ai/gemini-1.5-pro",
    model_options={
        "vertex_project": os.environ["VERTEXAI_PROJECT"],
        "vertex_location": os.environ["VERTEXAI_LOCATION"],
    },
)
m = MelleaSession(backend=backend)

result = m.instruct(
    "Extract the key points from this text: {{text}}",
    format=KeyPoints,
    user_variables={"text": "...your document..."},
)
parsed = KeyPoints.model_validate_json(str(result))
print(parsed.points)
```

## Model options

Pass generation parameters with `ModelOption`:

```python
import os

from mellea import MelleaSession
from mellea.backends import ModelOption
from mellea.backends.litellm import LiteLLMBackend

backend = LiteLLMBackend(
    model_id="vertex_ai/gemini-1.5-pro",
    model_options={
        "vertex_project": os.environ["VERTEXAI_PROJECT"],
        "vertex_location": os.environ["VERTEXAI_LOCATION"],
        ModelOption.TEMPERATURE: 0.2,
        ModelOption.MAX_NEW_TOKENS: 512,
    },
)
m = MelleaSession(backend=backend)
```

Options set at construction time apply to all calls on that session. Options
passed to `instruct()` or `chat()` apply to that call only and take precedence.

## Troubleshooting

### `VERTEXAI_PROJECT` or `VERTEXAI_LOCATION` not set

LiteLLM raises an error if the project or location cannot be determined. Export
the variables before running your script:

```bash
export VERTEXAI_PROJECT=your-gcp-project-id
export VERTEXAI_LOCATION=us-central1
```

### Authentication error

If you see a `google.auth.exceptions.DefaultCredentialsError`, run:

```bash
gcloud auth application-default login
```

or confirm that `GOOGLE_APPLICATION_CREDENTIALS` points to a valid service
account key file.

### Model not available in region

Not all Gemini models are available in every Vertex AI region. Check model
availability in the
[Vertex AI model garden](https://cloud.google.com/vertex-ai/generative-ai/docs/learn/models)
and update `VERTEXAI_LOCATION` accordingly.

### `google-cloud-aiplatform` not installed

```text
ModuleNotFoundError: No module named 'google.cloud.aiplatform'
```

Install the package:

```bash
pip install google-cloud-aiplatform
```

---

**See also:** [OpenAI and OpenAI-Compatible APIs](../integrations/openai) |
[Backends and Configuration](../guide/backends-and-configuration)
