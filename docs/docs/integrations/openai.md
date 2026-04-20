---
title: "OpenAI and OpenAI-Compatible APIs"
description: "Use Mellea with OpenAI's API and any OpenAI-compatible endpoint — LM Studio, vLLM, Anthropic, and more."
# diataxis: how-to
---

`OpenAIBackend` connects Mellea to the OpenAI API and to any server that implements
the OpenAI HTTP API — including LM Studio, Ollama's OpenAI endpoint, vLLM, and
OpenAI-compatible providers.

**Prerequisites:** `pip install mellea`, a valid API key for the OpenAI API or a
local OpenAI-compatible server running.

## OpenAI API

Set your API key as an environment variable (recommended):

```bash
export OPENAI_API_KEY=sk-...
```

Then create a session:

```python
from mellea import MelleaSession
from mellea.backends.openai import OpenAIBackend
from mellea.stdlib.context import ChatContext

m = MelleaSession(
    OpenAIBackend(model_id="gpt-4o"),
    ctx=ChatContext(),
)
reply = m.chat("What is the capital of France?")
print(str(reply))
# Output will vary — LLM responses depend on model and temperature.
```

Pass the key directly if you prefer not to use an environment variable:

```python
from mellea import MelleaSession
from mellea.backends.openai import OpenAIBackend

m = MelleaSession(
    OpenAIBackend(model_id="gpt-4o", api_key="sk-..."),
)
```

> **Note:** Never commit API keys to source control. Use environment variables or
> a secrets manager in production.

## OpenAI-compatible local servers

`OpenAIBackend` works with any server that implements the OpenAI HTTP API. No real
API key is needed for local servers — pass any non-empty string:

### LM Studio

```python
from mellea import MelleaSession
from mellea.backends.openai import OpenAIBackend

m = MelleaSession(
    OpenAIBackend(
        model_id="qwen/qwen2.5-vl-7b",
        base_url="http://127.0.0.1:1234/v1",
    )
)
```

### Ollama's OpenAI endpoint

```python
from mellea import MelleaSession
from mellea.backends.openai import OpenAIBackend
from mellea.stdlib.context import ChatContext

m = MelleaSession(
    OpenAIBackend(
        model_id="qwen2.5vl:7b",
        base_url="http://localhost:11434/v1",
        api_key="ollama",              # Ollama ignores the key; any value works
    ),
    ctx=ChatContext(),
)
```

### vLLM

```python
from mellea import MelleaSession
from mellea.backends.openai import OpenAIBackend

m = MelleaSession(
    OpenAIBackend(
        model_id="ibm-granite/granite-3.3-8b-instruct",
        base_url="http://localhost:8000/v1",
        api_key="your-vllm-key",
    )
)
```

## Using `base_url` from the environment

Set `OPENAI_BASE_URL` to avoid repeating the base URL in your code:

```bash
export OPENAI_BASE_URL=http://localhost:11434/v1
export OPENAI_API_KEY=ollama
```

```python
from mellea import MelleaSession
from mellea.backends.openai import OpenAIBackend

# Reads OPENAI_BASE_URL and OPENAI_API_KEY from environment
m = MelleaSession(OpenAIBackend(model_id="qwen2.5vl:7b"))
```

`base_url` and `api_key` constructor parameters take precedence over environment
variables if both are set.

## Vision and multimodal input

`OpenAIBackend` supports image inputs for vision-capable models. Pass a PIL image
or a Mellea `ImageBlock`:

```python
from PIL import Image
from mellea import MelleaSession
from mellea.backends.openai import OpenAIBackend
from mellea.core import ImageBlock
from mellea.stdlib.context import ChatContext

m = MelleaSession(
    OpenAIBackend(
        model_id="gpt-4o",
        api_key="sk-...",
    ),
    ctx=ChatContext(),
)

pil_image = Image.open("screenshot.png")
img_block = ImageBlock.from_pil_image(pil_image)

response = m.instruct(
    "Describe the content of this image and identify any text visible.",
    images=[img_block],
)
print(str(response))
# Output will vary — LLM responses depend on model and temperature.
```

You can also pass PIL `Image` objects directly without wrapping them:

```python
chat_response = m.chat(
    "How many people are in this image?",
    images=[pil_image],
)
```

> **Backend note:** Vision requires a model that supports image inputs (e.g., `gpt-4o`,
> `qwen2.5vl:7b`). Text-only models will raise an error if images are passed.

## Structured output with `format`

Use the `format` parameter to constrain generation to a Pydantic schema:

```python
from pydantic import BaseModel
from mellea import MelleaSession
from mellea.backends.openai import OpenAIBackend

class Summary(BaseModel):
    title: str
    key_points: list[str]
    word_count: int

m = MelleaSession(OpenAIBackend(model_id="gpt-4o", api_key="sk-..."))
result = m.instruct(
    "Summarise this article: {{text}}",
    format=Summary,
    user_variables={"text": "...your article text..."},
)
parsed = Summary.model_validate_json(str(result))
print(parsed.title)
```

## Model options

Set generation parameters with `ModelOption`:

```python
from mellea import MelleaSession
from mellea.backends import ModelOption
from mellea.backends.openai import OpenAIBackend

m = MelleaSession(
    OpenAIBackend(
        model_id="gpt-4o",
        api_key="sk-...",
        model_options={
            ModelOption.TEMPERATURE: 0.3,
            ModelOption.MAX_NEW_TOKENS: 500,
            ModelOption.SYSTEM_PROMPT: "You are a concise technical writer.",
        },
    )
)
```

Options set at construction time apply to all calls. Options passed to `instruct()`
or `chat()` apply to that call only and take precedence.

## Anthropic via OpenAI-compatible endpoint

Anthropic's API is not OpenAI-compatible natively, but if you access it through a
proxy that exposes an OpenAI-compatible interface, you can use `OpenAIBackend`:

```python
from mellea import MelleaSession
from mellea.backends.openai import OpenAIBackend

# Example: accessing Claude via a proxy with OpenAI-compatible interface
m = MelleaSession(
    OpenAIBackend(
        model_id="claude-3-haiku-20240307",
        api_key="your-anthropic-key",
        base_url="https://api.anthropic.com/v1/",
    )
)
```

> **Note (review needed):** Direct Anthropic API compatibility via this path has not
> been verified against the current Mellea version. If you are using Anthropic,
> LiteLLM provides a verified integration — see
> [Backends and Configuration](../guide/backends-and-configuration).

## Intrinsics with Granite Switch

Granite Switch models embed LoRA/aLoRA adapters directly in the model weights.
When served via vLLM, these adapters enable intrinsic functions (RAG quality
checks, safety evaluation, requirement validation) through the OpenAI-compatible
API without loading adapter weights at runtime.

Start a vLLM server with the Granite Switch model:

```bash
python -m vllm.entrypoints.openai.api_server \
    --model <granite-switch-model-id> \
    --dtype bfloat16 \
    --enable-prefix-caching
```

Then create a backend with `load_embedded_adapters=True`:

```python
from mellea.backends.openai import OpenAIBackend
from mellea.backends.model_ids import IBM_GRANITE_SWITCH_4_1_8B
from mellea.formatters import TemplateFormatter

backend = OpenAIBackend(
    model_id=IBM_GRANITE_SWITCH_4_1_8B.hf_model_name,
    formatter=TemplateFormatter(model_id=IBM_GRANITE_SWITCH_4_1_8B.hf_model_name),
    base_url="http://localhost:8000/v1",
    api_key="EMPTY",
    load_embedded_adapters=True,
)
```

The high-level intrinsic wrappers (`rag.check_answerability`,
`core.check_certainty`, etc.) work identically with this backend. See
[Intrinsics](../advanced/intrinsics) for the full list of available intrinsics.

> **Note:** `load_embedded_adapters=True` downloads adapter I/O configurations
> from the model's HuggingFace repository on first use. No adapter weights are
> transferred — the adapters are already part of the model. Only intrinsics
> embedded in the model are available — check the model's `adapter_index.json`
> for the list.

For more control, load adapters manually with `load_embedded_adapters=False`:

```python
from mellea.backends.adapters.adapter import EmbeddedIntrinsicAdapter
from mellea.backends.openai import OpenAIBackend
from mellea.backends.model_ids import IBM_GRANITE_SWITCH_4_1_8B
from mellea.formatters import TemplateFormatter

backend = OpenAIBackend(
    model_id=IBM_GRANITE_SWITCH_4_1_8B.hf_model_name,
    formatter=TemplateFormatter(model_id=IBM_GRANITE_SWITCH_4_1_8B.hf_model_name),
    base_url="http://localhost:8000/v1",
    api_key="EMPTY",
    load_embedded_adapters=False,
)

# Load a single adapter from the model's HuggingFace repo
adapters = EmbeddedIntrinsicAdapter.from_hub(
    IBM_GRANITE_SWITCH_4_1_8B.hf_model_name,
    intrinsic_name="answerability",
)
for adapter in adapters:
    backend.add_adapter(adapter)
```

## Troubleshooting

### `OPENAI_API_KEY` not set error

Either export the environment variable or pass `api_key` directly to `OpenAIBackend`.
For local servers, pass any non-empty string (e.g., `api_key="local"`).

### Connection refused at custom `base_url`

Confirm the local server is running and listening on the expected port. For Ollama,
run `ollama serve`; for LM Studio, start the local server from the LM Studio UI.

### Model not found

The model string must exactly match the name your server recognises. For OpenAI,
refer to the [OpenAI models page](https://platform.openai.com/docs/models). For
local servers, list available models from the server's API or UI.

---

**See also:** [Backends and Configuration](../guide/backends-and-configuration) |
[Enforce Structured Output](../how-to/enforce-structured-output) |
[Intrinsics](../advanced/intrinsics)
