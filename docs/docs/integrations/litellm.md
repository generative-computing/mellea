---
title: "LiteLLM"
description: "Reach 100+ LLM providers through Mellea's LiteLLMBackend: direct calls, a self-hosted proxy, and the recommended migration path off the deprecated WatsonX backend."
sidebar_label: "LiteLLM"
# diataxis: how-to
---

The [`LiteLLMBackend`](../reference/glossary.md#litellm--litellmbackend) gives Mellea a
single code path to 100+ model providers (Anthropic, AWS Bedrock, Azure OpenAI, IBM
WatsonX, Google Vertex AI, and more) by delegating provider auth and request
translation to [LiteLLM](https://docs.litellm.ai/). You switch providers by changing
the `model_id` prefix; your Mellea code stays the same.

**Prerequisites:** `pip install 'mellea[litellm]'` and provider credentials set as
environment variables (each provider is listed below).

## When to use LiteLLM

- **Multi-provider access.** One backend reaches most hosted providers, so you can
  compare models or fail over between vendors without rewriting session code.
- **WatsonX migration.** The native `WatsonxAIBackend` is deprecated since v0.4;
  LiteLLM (or the OpenAI backend) is the recommended replacement. See
  [Migrating from the WatsonX backend](#migrating-from-the-watsonx-backend).
- **Central governance.** Paired with a self-hosted proxy, LiteLLM centralizes API
  keys, routing, rate limits, and cost tracking across teams.

For providers that already have a dedicated Mellea page, prefer it for provider-specific
detail: [AWS Bedrock](./bedrock.md) and [Vertex AI](./vertex-ai.md) both use this
backend under the hood.

## Direct mode vs the LiteLLM Proxy

There are two ways to use LiteLLM. **Direct mode** calls the provider straight from
your process. The **LiteLLM Proxy** is a standalone server you (or your platform team)
run; Mellea talks to the proxy, and the proxy talks to the providers.

| | Direct mode | LiteLLM Proxy |
| --- | --- | --- |
| Setup | `pip install 'mellea[litellm]'` only | Run a separate proxy server |
| Credentials | Provider keys live in each app's environment | Keys live on the proxy; apps hold only a proxy key |
| Routing / fallback / rate limits | Per-app | Centralized on the proxy |
| Cost & usage tracking | Per-app | Centralized on the proxy |
| Best for | Single apps, local dev, quick provider comparison | Shared infrastructure, many apps, governed key management |

## Direct mode

The quickest path is [`start_session()`](../reference/glossary.md#melleasession) with
`backend_name="litellm"`. The `model_id` is a LiteLLM model string of the form
`<provider>/<model>`:

```python
# Requires: mellea[litellm]
# Returns: str
import mellea

m = mellea.start_session(
    backend_name="litellm",
    model_id="anthropic/claude-sonnet-4-20250514",
)
result = m.chat("Give me three facts about the Amazon rainforest.")
print(str(result))
# Output will vary — LLM responses depend on model and temperature.
```

> **Note:** For cloud providers, leave `base_url` unset (the default). LiteLLM infers
> the correct endpoint from the `model_id` prefix. Only set `base_url` when you target
> a proxy or a local server (see [Self-hosted LiteLLM Proxy](#self-hosted-litellm-proxy)).

For full control, construct the [`Backend`](../reference/glossary.md#backend) directly and
pass it to [`MelleaSession`](../reference/glossary.md#melleasession):

```python
# Requires: mellea[litellm]
# Returns: MelleaSession
from mellea import MelleaSession
from mellea.backends.litellm import LiteLLMBackend

m = MelleaSession(
    LiteLLMBackend(model_id="anthropic/claude-sonnet-4-20250514"),
)
```

## Provider configuration examples

Set the provider's credentials in the environment, then pass the matching `model_id`
prefix. The [environment variable reference](#environment-variable-reference) below
lists every provider in one table.

### Anthropic

```bash
export ANTHROPIC_API_KEY=your-api-key-here
```

```python
# Requires: mellea[litellm]
# Returns: MelleaSession
from mellea import MelleaSession
from mellea.backends.litellm import LiteLLMBackend

m = MelleaSession(
    LiteLLMBackend(model_id="anthropic/claude-sonnet-4-20250514"),
)
```

### Azure OpenAI

The `model_id` is `azure/<your-deployment-name>`, your Azure deployment name rather
than a base model name:

```bash
export AZURE_API_KEY=your-api-key-here
export AZURE_API_BASE=https://your-resource.openai.azure.com
export AZURE_API_VERSION=2024-02-15-preview
```

```python
# Requires: mellea[litellm]
# Returns: MelleaSession
from mellea import MelleaSession
from mellea.backends.litellm import LiteLLMBackend

m = MelleaSession(
    LiteLLMBackend(model_id="azure/my-gpt-4o-deployment"),
)
```

### IBM WatsonX

LiteLLM reaches WatsonX with the `watsonx/` prefix. Note the API-key variable is
`WATSONX_APIKEY` (no underscore before `KEY`), which differs from the native backend's
`WATSONX_API_KEY`:

```bash
export WATSONX_URL=https://us-south.ml.cloud.ibm.com
export WATSONX_APIKEY=your-api-key-here
export WATSONX_PROJECT_ID=your-project-id
```

```python
# Requires: mellea[litellm]
# Returns: MelleaSession
from mellea import MelleaSession
from mellea.backends.litellm import LiteLLMBackend

m = MelleaSession(
    LiteLLMBackend(model_id="watsonx/ibm/granite-3-3-8b-instruct"),
)
```

### AWS Bedrock and Google Vertex AI

Both use `LiteLLMBackend` and have dedicated pages with credential setup and model-string
tables:

- **Bedrock**: `bedrock/converse/<model-id>`. See [AWS Bedrock](./bedrock.md).
- **Vertex AI**: `vertex_ai/<model>`. See [Vertex AI](./vertex-ai.md).

## Self-hosted LiteLLM Proxy

The [LiteLLM Proxy](https://docs.litellm.ai/docs/simple_proxy) is a server that holds
your provider keys centrally and exposes them behind a single proxy key. Define your
models in a `config.yaml`:

```yaml
model_list:
  - model_name: my-model
    litellm_params:
      model: anthropic/claude-sonnet-4-20250514
      api_key: os.environ/ANTHROPIC_API_KEY
```

Start the proxy (defaults to port 4000):

```bash
pip install 'litellm[proxy]'
litellm --config config.yaml
```

Point Mellea at the proxy with the `litellm_proxy/` prefix and the proxy URL as
`base_url` (forwarded to LiteLLM as `api_base`). Authenticate with the proxy's key via
`LITELLM_PROXY_API_KEY`:

```bash
export LITELLM_PROXY_API_KEY=sk-...
```

```python
# Requires: mellea[litellm]
# Returns: MelleaSession
from mellea import MelleaSession
from mellea.backends.litellm import LiteLLMBackend

m = MelleaSession(
    LiteLLMBackend(
        model_id="litellm_proxy/my-model",
        base_url="http://localhost:4000",
    ),
)
```

`my-model` is the `model_name` you defined in the proxy's `config.yaml`; the app never
sees which provider or model backs it. To pass the proxy key explicitly instead of using
the environment variable, add it to `model_options`:

```python
# Requires: mellea[litellm]
# Returns: MelleaSession
from mellea import MelleaSession
from mellea.backends.litellm import LiteLLMBackend

m = MelleaSession(
    LiteLLMBackend(
        model_id="litellm_proxy/my-model",
        base_url="http://localhost:4000",
        model_options={"api_key": "sk-..."},
    ),
)
```

## Migrating from the WatsonX backend

The native `WatsonxAIBackend` is deprecated since v0.4. To move an existing WatsonX
session onto LiteLLM, swap the backend and prefix the model with `watsonx/`.

Before (deprecated):

```python
# Requires: mellea[watsonx]
# Returns: MelleaSession
from mellea import MelleaSession
from mellea.backends.watsonx import WatsonxAIBackend

m = MelleaSession(
    WatsonxAIBackend(model_id="ibm/granite-3-3-8b-instruct"),
)
```

After (LiteLLM):

```python
# Requires: mellea[litellm]
# Returns: MelleaSession
from mellea import MelleaSession
from mellea.backends.litellm import LiteLLMBackend

m = MelleaSession(
    LiteLLMBackend(model_id="watsonx/ibm/granite-3-3-8b-instruct"),
)
```

> **Warning:** LiteLLM reads the WatsonX API key from `WATSONX_APIKEY`, whereas the
> native backend read `WATSONX_API_KEY`. Rename the variable when you migrate, or the
> new backend will not find your credentials. `WATSONX_URL` and `WATSONX_PROJECT_ID`
> are unchanged.

## Environment variable reference

Set these before creating the session. LiteLLM reads them automatically based on the
`model_id` prefix.

| Provider | `model_id` prefix | Environment variables |
| --- | --- | --- |
| Anthropic | `anthropic/` | `ANTHROPIC_API_KEY` |
| Azure OpenAI | `azure/` | `AZURE_API_KEY`, `AZURE_API_BASE`, `AZURE_API_VERSION` |
| AWS Bedrock | `bedrock/converse/` | `AWS_BEARER_TOKEN_BEDROCK` (or standard AWS credentials) |
| IBM WatsonX | `watsonx/` | `WATSONX_URL`, `WATSONX_APIKEY`, `WATSONX_PROJECT_ID` |
| Google Vertex AI | `vertex_ai/` | `VERTEXAI_PROJECT`, `VERTEXAI_LOCATION` (see [Vertex AI](./vertex-ai.md)) |
| LiteLLM Proxy | `litellm_proxy/` | `LITELLM_PROXY_API_KEY`, `LITELLM_PROXY_API_BASE` (or pass `base_url`) |

See the [LiteLLM providers documentation](https://docs.litellm.ai/docs/providers) for
the full list and any provider-specific variables.

## Model options

Pass generation parameters with [`ModelOption`](../reference/glossary#modeloption), the
same as any other backend. Options set at construction apply to all calls; options
passed to `instruct()` or `chat()` apply to that call only and take precedence:

```python
# Requires: mellea[litellm]
# Returns: MelleaSession
from mellea import MelleaSession
from mellea.backends import ModelOption
from mellea.backends.litellm import LiteLLMBackend

m = MelleaSession(
    LiteLLMBackend(
        model_id="anthropic/claude-sonnet-4-20250514",
        model_options={ModelOption.TEMPERATURE: 0.2, ModelOption.MAX_NEW_TOKENS: 512},
    ),
)
```

See [Configure Model Options](../how-to/configure-model-options.md) for the full list of
`ModelOption` keys.

## Troubleshooting

**`ImportError: The LiteLLM backend requires extra dependencies`:**

```bash
pip install 'mellea[litellm]'
```

**"litellm allows for unknown / non-openai input params" or "litellm may drop the
following openai keys" warnings:** LiteLLM supports different parameters per provider.
Mellea logs which model options are unrecognized or may be dropped for the current
model, and passes the request through anyway. These warnings are informational: the
unsupported OpenAI parameters are dropped automatically (`drop_params=True`), and there
are occasional false positives. Remove the flagged options if a call misbehaves.

**"There is a known bug with litellm. This generation call may fail" warning:** This
appears for WatsonX-over-LiteLLM when Mellea detects generation calls running across
multiple asyncio event loops. Run only synchronous Mellea functions, or run your async
Mellea code from a single `asyncio.run()` call.

---

**See also:** [Backends and Configuration](../how-to/backends-and-configuration.md) |
[AWS Bedrock](./bedrock.md) | [Vertex AI](./vertex-ai.md) | [IBM WatsonX](./watsonx.md)
