---
canonical: "https://docs.mellea.ai/integrations/ollama"
title: "Ollama"
description: "Run Mellea with local models via Ollama — the default backend."
# diataxis: how-to
---

[Ollama](https://ollama.ai) is the default backend for Mellea. It runs models locally
with no API key, making it the fastest way to get started.

**Prerequisites:** [Ollama](https://ollama.ai) installed and the Ollama server running,
`pip install mellea`.

## Install Ollama

Download the installer from [ollama.ai](https://ollama.ai) or:

```bash
# macOS
brew install ollama

# Linux (one-line installer)
curl -fsSL https://ollama.ai/install.sh | sh
```

Start the server before running any Mellea code:

```bash
ollama serve
```

On macOS, installing via Homebrew or the `.dmg` starts the server automatically as a
background service.

## Default setup

`start_session()` connects to Ollama on `localhost:11434` and uses
**IBM Granite 4 Micro** (`granite4:micro`) by default. On first run, Mellea
automatically pulls the model if it is not already downloaded:

```python
# Returns: ModelOutputThunk
import mellea

m = mellea.start_session()
email = m.instruct("Write an email inviting the team to a meeting.")
print(str(email))
# Output will vary — LLM responses depend on model and temperature.
```

> **Note:** The first run pulls `granite4:micro` (~2 GB). Subsequent runs start
> immediately from the local cache.

## Switching models

Pass any model name that Ollama supports:

```python
# Returns: MelleaSession
import mellea

m = mellea.start_session(model_id="llama3.2:3b")
```

Use `model_ids` constants for well-known models — they carry the correct Ollama
model name automatically:

```python
# Returns: MelleaSession
from mellea import start_session
from mellea.backends import model_ids

m = start_session(model_id=model_ids.IBM_GRANITE_3_3_8B)
```

Pull models before using them (or let Mellea pull on first use):

```bash
ollama pull granite4:micro
ollama pull llama3.2:3b
ollama pull mistral:7b
```

## Recommended models

| `model_ids` constant | Ollama name | Notes |
| -------------------- | ----------- | ----- |
| `IBM_GRANITE_4_MICRO_3B` | `granite4:micro` | Default. Fast, low memory (~2 GB). |
| `IBM_GRANITE_4_HYBRID_MICRO` | `granite4:micro-h` | Hybrid variant with extended thinking. |
| `IBM_GRANITE_3_3_8B` | `granite3.3:8b` | Higher quality, ~5 GB. |
| `IBM_GRANITE_3_3_VISION_2B` | `ibm/granite3.3-vision:2b` | Vision model for image inputs. |
| `META_LLAMA_3_2_3B` | `llama3.2:3b` | Compact Llama model. |
| `MISTRALAI_MISTRAL_0_3_7B` | `mistral:7b` | Mistral 7B. |
| `QWEN3_8B` | `qwen3:8b` | Qwen3 8B. |
| `DEEPSEEK_R1_8B` | `deepseek-r1:8b` | Reasoning-capable model. |

Run `ollama list` to see which models are already downloaded locally.

## Direct backend construction

For full control, construct `OllamaModelBackend` directly:

```python
# Returns: MelleaSession
from mellea import MelleaSession
from mellea.backends.ollama import OllamaModelBackend
from mellea.backends import model_ids
from mellea.stdlib.context import ChatContext

backend = OllamaModelBackend(
    model_id=model_ids.IBM_GRANITE_3_3_8B,
)
m = MelleaSession(backend=backend, ctx=ChatContext())
```

## Custom host

Mellea reads the `OLLAMA_HOST` environment variable or accepts a `base_url`
parameter. Use this to connect to Ollama running on a remote machine or a
non-standard port:

```bash
# Environment variable
export OLLAMA_HOST=http://my-gpu-server:11434
```

```python
# Requires: mellea
# Returns: MelleaSession
from mellea import MelleaSession
from mellea.backends.ollama import OllamaModelBackend

m = MelleaSession(
    OllamaModelBackend(
        model_id="granite4:micro",
        base_url="http://my-gpu-server:11434",
    )
)
```

`base_url` takes precedence over `OLLAMA_HOST` if both are set.

## Model options

Pass generation parameters via `ModelOption`:

```python
# Requires: mellea
# Returns: MelleaSession
from mellea import MelleaSession
from mellea.backends import ModelOption, model_ids
from mellea.backends.ollama import OllamaModelBackend

m = MelleaSession(
    OllamaModelBackend(
        model_id=model_ids.IBM_GRANITE_4_MICRO_3B,
        model_options={
            ModelOption.TEMPERATURE: 0.1,
            ModelOption.SEED: 42,
        },
    )
)
```

Options set at construction time apply to all calls. Options passed to `instruct()`
or `chat()` apply to that call only and take precedence.

## Vision models

Ollama hosts vision-capable models. Use `IBM_GRANITE_3_3_VISION_2B` or any Ollama
vision model via the OpenAI-compatible endpoint:

```python
# Requires: mellea, pillow
# Returns: str
from PIL import Image
from mellea import MelleaSession
from mellea.backends.ollama import OllamaModelBackend
from mellea.backends import model_ids
from mellea.core import ImageBlock

backend = OllamaModelBackend(model_id=model_ids.IBM_GRANITE_3_3_VISION_2B)
m = MelleaSession(backend=backend)

pil_image = Image.open("photo.jpg")
img_block = ImageBlock.from_pil_image(pil_image)

response = m.instruct(
    "Describe what you see in this image.",
    images=[img_block],
)
print(str(response))
# Output will vary — LLM responses depend on model and temperature.
```

> **Backend note:** Vision requires a model that supports image inputs. The default
> `granite4:micro` is text-only. Pull a vision model explicitly before using images:
> `ollama pull ibm/granite3.3-vision:2b`.

## Ollama's OpenAI-compatible endpoint

Ollama exposes an OpenAI-compatible API at `http://localhost:11434/v1`. Use this
with the `OpenAIBackend` to access any Ollama model with OpenAI-style tool calling
or vision support:

```python
# Requires: mellea[openai]
# Returns: MelleaSession
from mellea import MelleaSession
from mellea.backends.openai import OpenAIBackend

m = MelleaSession(
    OpenAIBackend(
        model_id="qwen2.5vl:7b",
        base_url="http://localhost:11434/v1",
        api_key="ollama",          # required by the client; value is ignored by Ollama
    )
)
```

See [Backends and Configuration](../how-to/backends-and-configuration) for the
full `OpenAIBackend` reference.

## Troubleshooting

### Connection refused on port 11434

The Ollama server is not running. Start it with `ollama serve`, or on macOS,
launch the Ollama app from Applications.

### Model not found

The model has not been pulled. Run `ollama pull <model-name>` before using it, or
let Mellea pull it automatically on first use.

### Slow first run

Ollama loads the model into memory on the first request. Subsequent requests in the
same session are much faster. On machines with less than 8 GB RAM, consider using
`granite4:micro` or `llama3.2:1b`.

### Intel Mac torch errors

Some dependencies require a Rosetta-compatible environment on Intel Macs. Create a
conda environment and install `torchvision` before `pip install mellea`:

```bash
conda create -n mellea python=3.12
conda activate mellea
conda install 'torchvision>=0.22.0'
pip install mellea
```

---

**See also:** [Backends and Configuration](../how-to/backends-and-configuration) |
[Getting Started](../getting-started/installation)
