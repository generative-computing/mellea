# pytest: ollama, e2e, qualitative

"""Example: M serve with audio input via Ollama — two-step transcribe + chat.

Ollama does not pass audio through its generic chat backends, but since
version 0.7 it exposes an OpenAI-compatible ``POST /v1/audio/transcriptions``
endpoint that accepts multipart WAV/MP3 uploads and returns a JSON transcript.
This example uses that endpoint to split the work into two steps:

1. **Transcribe** — POST the raw audio bytes to Ollama's
   ``/v1/audio/transcriptions`` endpoint using
   ``gabegoodhart/granite4.1-speech:2b``.
2. **Chat** — pass the transcript to a standard ``granite3.3`` session
   (via Mellea's Ollama backend) so the model can reason about it and
   produce a response to the caller's question.

This emulates audio-text-to-text on an all-Ollama/Granite stack without
requiring a separate llama-server or cloud API.

Prerequisites:
    - Ollama running locally with both models pulled:

    ```bash
    ollama pull gabegoodhart/granite4.1-speech:2b
    ollama pull granite3.3
    ```

Environment variables (all optional):
    OLLAMA_HOST            Ollama server base URL  (default: http://localhost:11434)
    GRANITE_SPEECH_MODEL   speech model name       (default: gabegoodhart/granite4.1-speech:2b)
    GRANITE_CHAT_MODEL     chat model name         (default: granite3.3)

Usage:
    m serve docs/examples/m_serve/m_serve_example_multimodal_audio_granite.py

Then test with:
    uv run python docs/examples/m_serve/client_multimodal_audio.py
"""

import base64
import io
import os
from typing import Any, cast

import httpx

from mellea import start_session
from mellea.core import AudioBlock, AudioUrlBlock, ModelOutputThunk
from mellea.serve import ChatMessage

_ollama_host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
_speech_model = os.environ.get(
    "GRANITE_SPEECH_MODEL", "gabegoodhart/granite4.1-speech:2b"
)
_chat_model = os.environ.get("GRANITE_CHAT_MODEL", "granite3.3")

# Standard Mellea session for the chat step.
chat_session = start_session(model_id=_chat_model)


def _audio_block_to_bytes(block: AudioBlock) -> tuple[bytes, str]:
    """Decode an AudioBlock's base64 value to raw bytes and return (bytes, format).

    AudioBlock values may be stored in two forms:
    - Data URI: ``data:audio/wav;base64,<b64>`` (e.g. from session.chat() audio)
    - Raw base64 string without a prefix (e.g. from ``get_audio_blocks()`` on a
      serve ChatMessage whose ``InputAudioData.data`` is plain base64)
    """
    value = block.value or ""
    fmt = block.format or "wav"
    if isinstance(value, str):
        raw_b64 = value.split("base64,")[-1] if "base64," in value else value
        return base64.b64decode(raw_b64), fmt
    return value, fmt  # type: ignore[return-value]


async def _transcribe(audio_blocks: list[AudioBlock | AudioUrlBlock]) -> str:
    """Transcribe audio using Ollama's OpenAI-compatible transcriptions endpoint.

    POSTs each AudioBlock as a multipart WAV upload to
    ``/v1/audio/transcriptions`` and concatenates the results.

    A fresh httpx.AsyncClient is created per call so that it always belongs
    to the running event loop (avoids loop-mismatch errors when the serve
    function is called from different async contexts).

    AudioUrlBlock is not supported (skipped); download it first if needed.
    """
    parts: list[str] = []
    async with httpx.AsyncClient(base_url=_ollama_host, timeout=120) as client:
        for block in audio_blocks:
            if not isinstance(block, AudioBlock):
                continue
            audio_bytes, fmt = _audio_block_to_bytes(block)
            response = await client.post(
                "/v1/audio/transcriptions",
                files={
                    "file": (f"audio.{fmt}", io.BytesIO(audio_bytes), f"audio/{fmt}")
                },
                data={"model": _speech_model},
            )
            response.raise_for_status()
            parts.append(response.json().get("text", ""))

    return " ".join(p for p in parts if p)


async def serve(
    input: list[ChatMessage],
    requirements: list[str] | None = None,
    model_options: dict[str, Any] | None = None,
) -> ModelOutputThunk:
    """Serve function that emulates audio-text-to-text via two Ollama calls.

    Step 1: granite-speech transcribes the audio.
    Step 2: granite3.3 answers the user's text question, informed by the transcript.
    """

    _ = requirements, model_options  # Not used in this example

    if not input:
        return ModelOutputThunk(value="No input provided")

    last_message = input[-1]
    user_text = last_message.get_text_content() or "What is in this audio?"
    audio_blocks: list[AudioBlock | AudioUrlBlock] = list(
        last_message.get_audio_blocks()
    )

    transcript = ""
    if audio_blocks:
        transcript = await _transcribe(audio_blocks)
        print(f"Transcript: {transcript[:120]}...")

    # Build the prompt that combines the user question with the transcript.
    if transcript:
        prompt = f"{user_text}\n\n[Audio transcript: {transcript}]"
    else:
        prompt = user_text

    result = chat_session.chat(content=prompt)

    print(f"Result content: {result.content[:100] if result.content else 'None'}...")
    return cast(ModelOutputThunk, chat_session.ctx.as_list()[-1])
