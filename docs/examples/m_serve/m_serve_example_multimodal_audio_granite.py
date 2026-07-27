# pytest: ollama, e2e, qualitative

"""Example: M serve with audio input via Ollama — two-step transcribe + chat.

Ollama does not pass audio through its generic chat backends, but since
version 0.7 it exposes an OpenAI-compatible `POST /v1/audio/transcriptions`
endpoint that accepts multipart WAV/MP3 uploads and returns a JSON transcript.
This example uses that endpoint to split the work into two steps:

1. **Transcribe** — POST the raw audio bytes to Ollama's
   `/v1/audio/transcriptions` endpoint using
   `hf.co/ibm-granite/granite-speech-4.1-2b-GGUF:Q4_K_M`.
2. **Chat** — pass the transcript to a standard `granite4.1:3b` session
   (via Mellea's Ollama backend) so the model can reason about it and
   produce a response to the caller's question.

This emulates audio-text-to-text on an all-Ollama/Granite stack without
requiring a separate llama-server or cloud API. Because the audio is
transcribed to plain text before reaching the chat model, only word-level
content survives the pipeline — tone of voice, emotion, background sound,
and speaker identity are not available to the chat model on this path.

The session uses `ChatContext` so conversation history accumulates across
turns: follow-up questions like "who sang it?" work without re-uploading
the audio.  The serve function also honours caller-supplied `requirements`
and `model_options`, and uses `session.ainstruct()` with a built-in
`Requirement` to keep answers grounded in the transcript.

Prerequisites:
    - Ollama running locally with both models pulled:

    ```bash
    ollama pull hf.co/ibm-granite/granite-speech-4.1-2b-GGUF:Q4_K_M
    ollama pull granite4.1:3b
    ```

Environment variables (all optional):
    OLLAMA_HOST            Ollama server base URL  (default: http://localhost:11434)
    GRANITE_SPEECH_MODEL   speech model name       (default: hf.co/ibm-granite/granite-speech-4.1-2b-GGUF:Q4_K_M)
    GRANITE_CHAT_MODEL     chat model name         (default: granite4.1:3b)

Usage:
    m serve docs/examples/m_serve/m_serve_example_multimodal_audio_granite.py

Then test with:
    uv run python docs/examples/m_serve/client_multimodal_audio.py
"""

import base64
import io
import os
from typing import Any

import httpx

from mellea import start_session
from mellea.core import AudioBlock, AudioUrlBlock, ModelOutputThunk, Requirement
from mellea.serve import ChatMessage
from mellea.stdlib.context import ChatContext
from mellea.stdlib.requirements import simple_validate
from mellea.stdlib.sampling import RejectionSamplingStrategy

_ollama_host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
_speech_model = os.environ.get(
    "GRANITE_SPEECH_MODEL", "hf.co/ibm-granite/granite-speech-4.1-2b-GGUF:Q4_K_M"
)
_chat_model = os.environ.get("GRANITE_CHAT_MODEL", "granite4.1:3b")

# ChatContext accumulates conversation history across turns so follow-up
# questions work without re-uploading the audio.
chat_session = start_session(model_id=_chat_model, ctx=ChatContext())


def _audio_block_to_bytes(block: AudioBlock) -> tuple[bytes, str]:
    """Decode an AudioBlock's base64 value to raw bytes and return (bytes, format).

    AudioBlock values may be stored in two forms:
    - Data URI: `data:audio/wav;base64,<b64>` (e.g. from session.chat() audio)
    - Raw base64 string without a prefix (e.g. from `get_audio_blocks()` on a
      serve ChatMessage whose `InputAudioData.data` is plain base64)
    """
    value = block.value or ""
    fmt = block.format or "wav"
    raw_b64 = value.split("base64,")[-1] if "base64," in value else value
    return base64.b64decode(raw_b64), fmt


async def _transcribe(audio_blocks: list[AudioBlock | AudioUrlBlock]) -> str:
    """Transcribe audio using Ollama's OpenAI-compatible transcriptions endpoint.

    POSTs each AudioBlock as a multipart WAV upload to
    `/v1/audio/transcriptions` and concatenates the results.

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

    Step 1: `hf.co/ibm-granite/granite-speech-4.1-2b-GGUF:Q4_K_M` transcribes the audio.
    Step 2: `granite4.1:3b` answers the user's text question, informed by the transcript.

    The session retains conversation history across calls (via `ChatContext`),
    so follow-up questions work without re-uploading audio.  Caller-supplied
    `requirements` and `model_options` are forwarded to `ainstruct`.
    """
    if not input:
        return ModelOutputThunk(value="No input provided")

    last_message = input[-1]
    user_text = last_message.get_text_content() or "What is in this audio?"
    audio_blocks: list[AudioBlock | AudioUrlBlock] = list(
        last_message.get_audio_blocks()
    )

    if not audio_blocks:
        raise ValueError(
            "No audio provided. Please include an audio clip in your message."
        )

    transcript = await _transcribe(audio_blocks)
    if not transcript:
        raise ValueError("Audio was provided but could not be transcribed.")
    print(f"Transcript: {transcript[:120]}...")

    prompt = f"{user_text}\n\n[Audio transcript: {transcript}]"

    # Python-checkable grounding requirement: verify the answer shares at least
    # one word with the transcript.  This avoids LLM-as-a-Judge overhead and
    # makes the RejectionSamplingStrategy retry loop deterministic.
    _words = set(transcript.lower().split())
    grounding_req: Requirement = Requirement(
        "Base your answer only on the provided audio transcript",
        validation_fn=simple_validate(
            lambda output: bool(_words & set(output.lower().split()))
        ),
    )

    result = await chat_session.ainstruct(
        description=prompt,
        requirements=[
            grounding_req,
            *[Requirement(r) for r in (requirements or []) if r],
        ],
        strategy=RejectionSamplingStrategy(loop_budget=2),
        model_options=model_options,
        await_result=True,
    )

    print(f"Result content: {result.value[:100] if result.value else 'None'}...")
    return result
