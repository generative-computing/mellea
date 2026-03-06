---
title: "Use Images and Vision Models"
description: "Pass images to instruct() and chat() calls, and configure vision-capable backends."
# diataxis: how-to
---

Mellea supports multimodal input: pass images alongside your text prompt to any
`instruct()` or `chat()` call using the `images` parameter.

**Prerequisites:** `pip install mellea pillow`, a vision-capable model downloaded and
running.

> **Backend note:** The default Ollama model (`granite4:micro`) does not support image
> input. You must switch to a vision-capable model such as `granite3.2-vision` or
> `llava`. Not all backends support vision — see backend notes below.

---

## Basic usage with Ollama

Start a session with a vision-capable model, then pass a [Pillow](https://python-pillow.org/)
`Image` object in the `images` list:

```python
import pathlib
from PIL import Image
from mellea import start_session

m = start_session(model_id="granite3.2-vision")

img = Image.open("photo.jpg")
result = m.instruct("Is the subject in this image smiling?", images=[img])
print(str(result))
# Output will vary — LLM responses depend on model and temperature.
```

Other vision-capable Ollama models: `llava`, `llava-phi3`, `moondream`, `qwen2.5vl:7b`.

---

## Using ImageBlock for explicit control

For the OpenAI backend (and compatible endpoints), convert the PIL image to an
`ImageBlock` first:

```python
import pathlib
from PIL import Image
from mellea import MelleaSession
from mellea.backends.openai import OpenAIBackend
from mellea.core import ImageBlock
from mellea.stdlib.context import ChatContext

# Point the OpenAI backend at a local vision model (e.g., via Ollama's OpenAI layer)
m = MelleaSession(
    OpenAIBackend(
        model_id="qwen2.5vl:7b",
        base_url="http://localhost:11434/v1",
        api_key="ollama",
    ),
    ctx=ChatContext(),
)

img = Image.open("photo.jpg")
img_block = ImageBlock.from_pil_image(img)

result = m.instruct(
    "Is there a person in this image? Are they smiling?",
    images=[img_block],
)
print(str(result))
# Output will vary — LLM responses depend on model and temperature.
```

Both PIL images and `ImageBlock` objects are accepted in the `images` list. Use
`ImageBlock` when you need to work with an already-encoded representation or when
the PIL image is not directly available.

---

## Multi-turn vision with ChatContext

Images passed to `instruct()` or `chat()` are stored in the [`ChatContext`](../guide/glossary#context)
turn history. Subsequent calls in the same session can reference the image without
passing it again:

```python
from PIL import Image
from mellea import start_session
from mellea.stdlib.context import ChatContext

m = start_session(model_id="granite3.2-vision", ctx=ChatContext())

img = Image.open("photo.jpg")

# First turn — attach the image
r1 = m.instruct("Is the subject in the image smiling?", images=[img])
print(str(r1))

# Second turn — the image is still in context
r2 = m.instruct("How many eyes can you identify in the image? Explain.")
print(str(r2))
```

To remove images from context on the next turn, pass `images=[]` explicitly.

---

## Backend support

| Backend | Vision support | Notes |
| ------- | -------------- | ----- |
| `OllamaModelBackend` | ✓ | Requires a vision model (e.g., `granite3.2-vision`, `llava`) |
| `OpenAIBackend` | ✓ | Use with `gpt-4o`, or a local vision model via OpenAI-compatible endpoint |
| `LiteLLMBackend` | ✓ | Depends on the underlying provider |
| `LocalHFBackend` | Partial | Model-dependent; experimental |
| `LocalVLLMBackend` | Partial | Model-dependent |
| `WatsonxAIBackend` | ✗ | Not currently supported |

> **Full example (Ollama):** [`docs/examples/image_text_models/vision_ollama_chat.py`](../../examples/image_text_models/vision_ollama_chat.py)
> **Full example (OpenAI backend):** [`docs/examples/image_text_models/vision_openai_examples.py`](../../examples/image_text_models/vision_openai_examples.py)

---


**See also:** [Working with Data](../guide/working-with-data) |
[The Instruction Model](../concepts/instruct-validate-repair)
