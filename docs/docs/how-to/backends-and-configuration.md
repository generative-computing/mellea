---
title: "Backends and Configuration"
description: "Configure Mellea to use Ollama, OpenAI, LiteLLM, HuggingFace, or WatsonX backends."
# diataxis: how-to
---

**Prerequisites:** `pip install mellea`, [Ollama](https://ollama.ai) for local inference
or appropriate credentials for cloud backends.

A backend is the engine that runs the LLM. Mellea ships with backends for Ollama,
OpenAI-compatible APIs, LiteLLM, HuggingFace transformers, and IBM WatsonX. You
configure the backend when you create a session.

## Default backend

`start_session()` defaults to **Ollama** with **IBM Granite 4 Micro** (`granite4:micro`).
No API keys needed — just have Ollama running:

```python
# Requires: mellea
# Returns: MelleaSession
import mellea

m = mellea.start_session()
```

## Backend comparison

The following table shows all available backends, their class names, import paths, required extras, and `start_session()` configuration:

| Backend | Class | Import | Required extras | `start_session()` |
| ------- | ----- | ------ | --------------- | ----------------- |
| [Ollama](../integrations/ollama) | `OllamaModelBackend` | `mellea.backends.ollama` | base | `backend_name="ollama"` |
| [OpenAI](../integrations/openai) | `OpenAIBackend` | `mellea.backends.openai` | base | `backend_name="openai"` |
| [LiteLLM](../integrations/bedrock) | `LiteLLMBackend` | `mellea.backends.litellm` | `mellea[litellm]` | `backend_name="litellm"` |
| [HuggingFace](../integrations/huggingface) | `LocalHFBackend` | `mellea.backends.huggingface` | `mellea[hf]` | `backend_name="hf"` |
| [WatsonX](../integrations/watsonx) | `WatsonxAIBackend` | `mellea.backends.watsonx` | `mellea[watsonx]` | `backend_name="watsonx"` (deprecated) |

> **Note:** Vertex AI uses the LiteLLM backend with appropriate model IDs. See the [Vertex AI integration](../integrations/vertex-ai) for details. For detailed setup instructions, click the backend name in the table above.

## Switching the model

Pass any model string your backend supports:

```python
# Requires: mellea
# Returns: MelleaSession
import mellea

m = mellea.start_session(model_id="llama3.2:3b")
```

Use `model_ids` constants for known models:

```python
# Requires: mellea
# Returns: MelleaSession
from mellea import start_session
from mellea.backends import model_ids

m = start_session(model_id=model_ids.IBM_GRANITE_3_3_8B)
```

## OpenAI backend

> **Backend note:** This section requires `pip install mellea` (no extras needed — the
> OpenAI client is included). Needs a valid `api_key` for the OpenAI API; local
> endpoints such as LM Studio and Ollama's OpenAI endpoint do not require a real key.

Use any OpenAI-compatible API — OpenAI itself, LM Studio, vLLM, or Ollama's
OpenAI-compatible endpoint:

```python
# Requires: mellea
# Returns: MelleaSession
from mellea import MelleaSession
from mellea.backends.openai import OpenAIBackend
from mellea.stdlib.context import ChatContext

# OpenAI API
m = MelleaSession(
    OpenAIBackend(model_id="gpt-4o", api_key="sk-..."),
    ctx=ChatContext(),
)
```

```python
# Requires: mellea
# Returns: MelleaSession
from mellea import MelleaSession
from mellea.backends.openai import OpenAIBackend

# LM Studio (local, no real key needed)
m = MelleaSession(
    OpenAIBackend(model_id="qwen2.5vl:7b", base_url="http://127.0.0.1:1234/v1"),
)

# Ollama via OpenAI-compatible endpoint
m = MelleaSession(
    OpenAIBackend(
        model_id="qwen2.5vl:7b",
        base_url="http://localhost:11434/v1",
        api_key="ollama",
    ),
)
```

## LiteLLM backend

> **Backend note:** Requires `pip install "mellea[litellm]"`. Provider-specific
> environment variables must be set (e.g., `AWS_BEARER_TOKEN_BEDROCK` for Bedrock).
> See the [LiteLLM docs](https://docs.litellm.ai/) for your provider's setup.

LiteLLM provides unified access to 100+ providers — Anthropic, AWS Bedrock, Azure,
and more:

```python
# Requires: mellea, litellm
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

## HuggingFace backend

> **Backend note:** Requires `pip install "mellea[hf]"`. Models are downloaded from
> HuggingFace Hub on first use. GPU recommended for reasonable inference speed.
> Required for [Intrinsics](../advanced/intrinsics).

Run models locally using HuggingFace transformers:

```python
# Requires: mellea, transformers
# Returns: MelleaSession
from mellea import MelleaSession
from mellea.backends.huggingface import LocalHFBackend

backend = LocalHFBackend(model_id="ibm-granite/granite-4.0-micro")
m = MelleaSession(backend=backend)
```

## WatsonX backend

> **Deprecated:** The native WatsonX backend is deprecated. Use the **LiteLLM** or
> **OpenAI** backend with a WatsonX-compatible endpoint instead.
> See [IBM WatsonX integration](/integrations/watsonx) for the recommended setup.

## Model options

`ModelOption` provides backend-agnostic keys for common generation parameters.
Options set at session level apply to all calls; options passed to `instruct()` or
`chat()` apply to that call only and take precedence:

```python
# Requires: mellea
# Returns: str
from mellea import MelleaSession
from mellea.backends import ModelOption
from mellea.backends.ollama import OllamaModelBackend

# Set seed for all calls in this session
m = MelleaSession(
    backend=OllamaModelBackend(model_options={ModelOption.SEED: 42})
)

# Override temperature and token limit for a single call
answer = m.instruct(
    "What is 2 × 2?",
    model_options={
        ModelOption.TEMPERATURE: 0.5,
        ModelOption.MAX_NEW_TOKENS: 15,
    },
)
print(str(answer))
# Output will vary — LLM responses depend on model and temperature.
```

Available `ModelOption` constants:

| Constant | Description |
| -------- | ----------- |
| `ModelOption.TEMPERATURE` | Sampling temperature |
| `ModelOption.MAX_NEW_TOKENS` | Maximum tokens to generate |
| `ModelOption.SEED` | Random seed for reproducibility |
| `ModelOption.SYSTEM_PROMPT` | System prompt override |
| `ModelOption.THINKING` | Enable thinking / reasoning mode |
| `ModelOption.STREAM` | Enable streaming output |
| `ModelOption.TOOLS` | List of tools available to the model |
| `ModelOption.CONTEXT_WINDOW` | Context window size |

You can also pass raw backend-native keys alongside `ModelOption` constants. If
the same parameter is specified both ways, `ModelOption` takes precedence.

### System prompt

`ModelOption.SYSTEM_PROMPT` is the recommended way to set a system message. It is
translated correctly for all backends regardless of how each provider serializes the
system role:

```python
# Requires: mellea
# Returns: str
from mellea import start_session
from mellea.backends import ModelOption

m = start_session(model_options={ModelOption.SYSTEM_PROMPT: "You are a concise assistant."})
reply = m.chat("What is the capital of France?")
print(str(reply))
# Output will vary — LLM responses depend on model and temperature.
```

## Direct backend construction

For full control, construct the backend and pass it to `MelleaSession` directly:

```python
# Requires: mellea
# Returns: MelleaSession
import mellea
from mellea.backends.ollama import OllamaModelBackend
from mellea.stdlib.context import ChatContext

backend = OllamaModelBackend(model_id="phi4-mini:latest")
m = mellea.MelleaSession(backend=backend, ctx=ChatContext())
```

`start_session()` accepts the same arguments as keyword parameters:

```python
# Requires: mellea
# Returns: MelleaSession
import mellea
from mellea.backends import ModelOption
from mellea.stdlib.context import ChatContext

m = mellea.start_session(
    backend_name="ollama",
    model_id="phi4-mini:latest",
    ctx=ChatContext(),
    model_options={ModelOption.TEMPERATURE: 0.1},
)
```

Valid `backend_name` values: `"ollama"`, `"openai"`, `"hf"`, `"litellm"`, `"watsonx"`.

---

**See also:** [Configure Model Options](../how-to/configure-model-options) | [Integrations](../integrations/ollama)
