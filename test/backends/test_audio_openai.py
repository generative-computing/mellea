"""E2E tests for audio-text-to-text with OpenAI-compatible backends.

These tests require a local llama-server with an audio-capable model.

**Setup:**

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

**Environment variables** (all optional):

```
LLAMA_SERVER_URL   base URL of the server  (default: http://localhost:8088/v1)
LLAMA_SERVER_API_KEY  API key              (default: default)
LLAMA_SERVER_MODEL    model name           (default: gemma-4-12b-it-Q8_0.gguf)
```

**Run these tests explicitly:**

```bash
LLAMA_SERVER_URL=http://localhost:8088/v1 uv run pytest test/backends/test_audio_openai.py -v
```
"""

import base64
import os

import pytest
import requests
from pydantic import BaseModel

# Skipped by default — set LLAMA_SERVER_URL to opt in (see module docstring).
pytestmark = [
    pytest.mark.e2e,
    pytest.mark.skipif(
        not os.environ.get("LLAMA_SERVER_URL"),
        reason="requires local llama-server — set LLAMA_SERVER_URL to run (see module docstring)",
    ),
]

from mellea import MelleaSession, start_session
from mellea.backends import ModelOption
from mellea.core import AudioBlock, ModelOutputThunk
from mellea.stdlib.components import Message


def _make_session() -> MelleaSession:
    """Create a fresh session with the local llama-server."""
    base_url = os.environ.get("LLAMA_SERVER_URL", "http://localhost:8088/v1")
    api_key = os.environ.get("LLAMA_SERVER_API_KEY", "default")
    model_id = os.environ.get("LLAMA_SERVER_MODEL", "gemma-4-12b-it-Q8_0.gguf")
    return start_session(
        "openai",
        model_id=model_id,
        base_url=base_url,
        api_key=api_key,
        model_options={ModelOption.MAX_NEW_TOKENS: 512, "modalities": ["text"]},
    )


# The Gemma audio sample says "Roses are red, violets are blue."
_AUDIO_URL = "https://ai.google.dev/gemma/docs/audio/roses-are.wav"


class AudioContent(BaseModel):
    """Structured extraction from the audio clip."""

    colors: list[str]
    flowers: list[str]


@pytest.fixture(scope="module")
def sample_audio_wav() -> str:
    """Download the Gemma sample speech WAV ('Roses are red, violets are blue.')."""
    response = requests.get(_AUDIO_URL, timeout=30)
    response.raise_for_status()
    encoded = base64.b64encode(response.content).decode("utf-8")
    return f"data:audio/wav;base64,{encoded}"


def test_audio_block_construction(sample_audio_wav: str):
    """Test that AudioBlock can be constructed from base64 WAV data."""
    audio_block = AudioBlock(sample_audio_wav)
    assert isinstance(audio_block, AudioBlock)
    assert isinstance(audio_block.value, str)
    assert audio_block.format == "wav"


def test_audio_block_serialization(sample_audio_wav: str):
    """Test AudioBlock serialization with Message and session.act().

    Verifies that audio blocks are correctly serialized to OpenAI format
    in the prompt sent to the backend.
    """
    audio_block = AudioBlock(sample_audio_wav)

    msg = Message(
        role="user",
        content="Transcribe exactly what is said in this audio.",
        audio=[audio_block],
    )

    with _make_session() as session:
        result = session.act(msg)
        assert isinstance(result, ModelOutputThunk)

        # Verify the last action has audio attached
        turn = session.ctx.last_turn()
        assert turn is not None
        last_action = turn.model_input
        assert isinstance(last_action, Message)
        assert last_action.audio is not None
        assert len(last_action.audio) > 0

        # First audio should match the input block
        first_audio = last_action.audio[0]
        assert isinstance(first_audio, AudioBlock)
        assert first_audio.value == audio_block.value

        # Verify the prompt message structure
        lp = turn.output._generate_log.prompt  # type: ignore
        assert isinstance(lp, list)
        assert len(lp) == 1

        prompt_msg = lp[0]
        assert isinstance(prompt_msg, dict)

        content_list = prompt_msg.get("content", None)
        assert isinstance(content_list, list)
        assert len(content_list) == 2  # text + audio

        content_audio = content_list[1]
        assert isinstance(content_audio, dict)
        assert content_audio.get("type") in ("audio_url", "input_audio")

        if "audio_url" in content_audio:
            audio_url = content_audio["audio_url"]
            assert audio_url is not None
            assert "url" in audio_url
            url_value = audio_url["url"]
            assert isinstance(url_value, str)
            assert audio_block.value is not None
            assert audio_block.value[:100] in url_value
        elif "input_audio" in content_audio:
            input_audio = content_audio["input_audio"]
            assert input_audio is not None
            assert "data" in input_audio
            assert input_audio.get("format") == "wav"


@pytest.mark.qualitative
def test_session_instruct_with_audio(sample_audio_wav: str):
    """Test session.instruct() with audio blocks and structured output.

    Asks the model to extract colors and flowers from the audio clip
    ('Roses are red, violets are blue.') and validates the structured response.
    """
    from mellea.stdlib.components.instruction import Instruction

    audio_block = AudioBlock(sample_audio_wav)

    with _make_session() as session:
        result = session.instruct(
            "Listen to the audio and return the colors and flowers mentioned.",
            audio=[audio_block],
            strategy=None,
            format=AudioContent,
        )

        assert isinstance(result, ModelOutputThunk)
        assert result.value is not None
        parsed = AudioContent.model_validate_json(result.value)
        colors = [c.lower() for c in parsed.colors]
        flowers = [f.lower() for f in parsed.flowers]
        assert any("red" in c or c == "red" for c in colors), (
            f"Expected 'red' in colors, got: {parsed.colors!r}"
        )
        assert any("blue" in c or c == "blue" for c in colors), (
            f"Expected 'blue' in colors, got: {parsed.colors!r}"
        )
        assert any("rose" in f for f in flowers), (
            f"Expected 'rose(s)' in flowers, got: {parsed.flowers!r}"
        )
        assert any("violet" in f for f in flowers), (
            f"Expected 'violet(s)' in flowers, got: {parsed.flowers!r}"
        )

        # Verify audio was attached to the context
        turn = session.ctx.last_turn()
        assert turn is not None
        assert isinstance(turn.model_input, Instruction)
        assert turn.model_input._audio is not None
        assert len(turn.model_input._audio) > 0


@pytest.mark.qualitative
def test_session_chat_with_audio(sample_audio_wav: str):
    """Test session.chat() with audio blocks and structured output.

    Asks the model to extract colors and flowers from the audio clip
    ('Roses are red, violets are blue.') and validates the structured response.
    """
    audio_block = AudioBlock(sample_audio_wav)

    with _make_session() as session:
        result = session.chat(
            "Listen to the audio and return the colors and flowers mentioned.",
            audio=[audio_block],
            format=AudioContent,
        )

        assert isinstance(result, Message)
        assert result.content is not None
        parsed = AudioContent.model_validate_json(result.content)
        colors = [c.lower() for c in parsed.colors]
        flowers = [f.lower() for f in parsed.flowers]
        assert any("red" in c or c == "red" for c in colors), (
            f"Expected 'red' in colors, got: {parsed.colors!r}"
        )
        assert any("blue" in c or c == "blue" for c in colors), (
            f"Expected 'blue' in colors, got: {parsed.colors!r}"
        )
        assert any("rose" in f for f in flowers), (
            f"Expected 'rose(s)' in flowers, got: {parsed.flowers!r}"
        )
        assert any("violet" in f for f in flowers), (
            f"Expected 'violet(s)' in flowers, got: {parsed.flowers!r}"
        )

        # Verify audio was attached to the context
        turn = session.ctx.last_turn()
        assert turn is not None
        assert isinstance(turn.model_input, Message)
        assert turn.model_input.audio is not None
        assert len(turn.model_input.audio) > 0


if __name__ == "__main__":
    pytest.main([__file__])
