# Vision/Multimodal Examples

This directory contains examples for working with vision-language models that can process both images and text.

## Files

### vision_ollama_chat.py
Demonstrates using vision models through Ollama backend with chat interface.

**Key Features:**
- Loading and processing images
- Using vision models for image understanding
- Chat-based interaction with images

### vision_openai_examples.py
Shows how to use OpenAI-compatible vision models (including local VLLM servers).

### vision_litellm_backend.py
Examples using LiteLLM backend for vision model access.

## Supporting Files

### pointing_up.jpg
Sample image used in the examples for testing vision capabilities.

## Concepts Demonstrated

- **Multimodal Input**: Combining text and images in prompts
- **Vision Understanding**: Asking questions about image content
- **Backend Flexibility**: Using different backends (Ollama, OpenAI, LiteLLM) for vision
- **Image Processing**: Loading and formatting images for LLM consumption

## Basic Usage

```python
from mellea import start_session
from mellea.stdlib.components import Message

# Load image
with open("pointing_up.jpg", "rb") as f:
    image_data = f.read()

# Create session with vision model
m = start_session(model_id="llava:7b")

# Ask about the image
response = m.chat(
    Message(
        role="user",
        content="What do you see in this image?",
        images=[image_data]
    )
)
```

## Supported Models

- **Ollama**: llava, bakllava, llava-phi3, moondream
- **OpenAI**: gpt-4-vision-preview, gpt-4o
- **LiteLLM**: Various vision models through unified interface

## Related Documentation

- See `test/backends/test_vision_*.py` for more examples
- See `mellea/stdlib/components/chat.py` for Message API