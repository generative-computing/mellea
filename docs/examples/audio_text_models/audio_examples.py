# pytest: openai, e2e, qualitative
"""Audio-text-to-text examples using session.instruct() and session.chat().

Prerequisites:
    - llama-server running with gemma-4-12b-it-Q8_0.gguf on port 8088

Usage:
    uv run python docs/examples/audio_text_models/audio_examples.py
"""

import base64
import os

import requests

from mellea import start_session
from mellea.backends import ModelOption
from mellea.core import AudioBlock


def _download_audio() -> AudioBlock:
    """Download the sample WAV from OpenAI CDN and return an AudioBlock."""
    print("Downloading sample audio from OpenAI CDN...")
    response = requests.get("https://cdn.openai.com/API/docs/audio/alloy.wav")
    response.raise_for_status()
    wav_data = response.content
    print(f"Audio downloaded: {len(wav_data)} bytes")
    audio_b64 = f"data:audio/wav;base64,{base64.b64encode(wav_data).decode()}"
    block = AudioBlock(audio_b64)
    print(f"AudioBlock created: format={block.format}")
    return block


def main():
    """Run audio-text-to-text examples with session.instruct() and session.chat()."""
    base_url = os.environ.get("LLAMA_SERVER_URL", "http://localhost:8088/v1")
    api_key = os.environ.get("LLAMA_SERVER_API_KEY", "default")
    model_id = os.environ.get("LLAMA_SERVER_MODEL", "gemma-4-12b-it-Q8_0.gguf")

    print(f"Connecting to llama-server at {base_url}")
    print(f"Model: {model_id}")

    audio_block = _download_audio()

    with start_session(
        "openai",
        model_id=model_id,
        base_url=base_url,
        api_key=api_key,
        model_options={
            ModelOption.MAX_NEW_TOKENS: 1000,
            "modalities": ["text"],  # text output only
        },
    ) as session:
        print("\n--- session.instruct() ---")
        result = session.instruct(
            "Explain what is in this recording using bullet points",
            audio=[audio_block],
            strategy=None,
        )
        print(f"Response: {result.value}")

        print("\n--- session.chat() ---")
        message = session.chat(
            "Summarise what you hear in one sentence", audio=[audio_block]
        )
        print(f"Response: {message.content}")


if __name__ == "__main__":
    main()
