# Audio-Text-to-Text Examples

Examples for audio-text-to-text workflows using Mellea with an OpenAI-compatible backend.

## Prerequisites

**llama-server** running with an audio-capable model (e.g. Gemma 4):

```bash
llama-server \
  --model gemma-4-12b-it-Q8_0.gguf \
  --mmproj mmproj-F16.gguf \
  --n-gpu-layers 99 \
  --ctx-size 32768 \
  --flash-attn on \
  --parallel 1 \
  --cache-type-k q8_0 \
  --cache-type-v q8_0 \
  --temp 0.1 --top-k 1 \
  --jinja \
  --host 0.0.0.0 --port 8088
```

## Quick Start

```bash
uv run python docs/examples/audio_text_models/audio_examples.py
```

## Usage Pattern

```python
import base64

from mellea import start_session
from mellea.backends import ModelOption
from mellea.core import AudioBlock

# Load audio
with open("speech.wav", "rb") as f:
    audio_b64 = base64.b64encode(f.read()).decode()

# IMPORTANT: pass modalities=['text'] — tells the server to return text, not audio
session = start_session(
    "openai",
    model_id="gemma-4-12b-it-Q8_0.gguf",
    base_url="http://localhost:8088/v1",
    api_key="default",
    model_options={
        ModelOption.MAX_NEW_TOKENS: 1000,
        "modalities": ["text"],
    },
)

audio = AudioBlock(f"data:audio/wav;base64,{audio_b64}")

# Use session.instruct() for task-style prompts
response = session.instruct("Summarize this audio", audio=[audio], strategy=None)
print(response.value)

# Use session.chat() for conversational use
response = session.chat("What do you hear?", audio=[audio])
print(response.content)
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `LLAMA_SERVER_URL` | `http://localhost:8088/v1` | llama-server endpoint |
| `LLAMA_SERVER_API_KEY` | `default` | API key |
| `LLAMA_SERVER_MODEL` | `gemma-4-12b-it-Q8_0.gguf` | Model identifier |

## Files

- **`audio_examples.py`** — Downloads sample audio from OpenAI CDN, demonstrates both `session.instruct()` and `session.chat()`
