# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for audio payload shape on OpenAI-compatible backends (mocked).

Verifies that `AudioBlock` is correctly serialised into the ``input_audio``
content-part format that OpenAI Chat Completions expects, without requiring a
live server.  Mirrors the tier-2 structural tests in ``test_vision_ollama.py``.
"""

import base64
import struct
from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch

import pytest
from openai.types.chat import ChatCompletion
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from openai.types.completion_usage import CompletionUsage

from mellea import MelleaSession
from mellea.backends import ModelOption
from mellea.backends.openai import OpenAIBackend
from mellea.core import AudioBlock, AudioUrlBlock

# ---------------------------------------------------------------------------
# Minimal WAV helper — 44-byte header + 1 silent sample so AudioBlock accepts it
# ---------------------------------------------------------------------------

_SILENT_SAMPLE = struct.pack("<h", 0)
_WAV_HEADER = (
    b"RIFF"
    + struct.pack("<I", 36 + len(_SILENT_SAMPLE))
    + b"WAVEfmt "
    + struct.pack("<IHHIIHH", 16, 1, 1, 16000, 32000, 2, 16)
    + b"data"
    + struct.pack("<I", len(_SILENT_SAMPLE))
)
_MINIMAL_WAV = _WAV_HEADER + _SILENT_SAMPLE
_B64_WAV = base64.b64encode(_MINIMAL_WAV).decode()

# Canned non-streaming ChatCompletion that processing() can handle.
_CANNED_RESPONSE = ChatCompletion(
    id="chatcmpl-test",
    choices=[
        Choice(
            finish_reason="stop",
            index=0,
            message=ChatCompletionMessage(role="assistant", content="ok"),
        )
    ],
    created=0,
    model="test-model",
    object="chat.completion",
    usage=CompletionUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
)


@pytest.fixture
def audio_block() -> AudioBlock:
    return AudioBlock(_B64_WAV, format="wav")


# ---------------------------------------------------------------------------
# Shared mocked OpenAI session fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def mocked_openai_session():
    """OpenAIBackend with chat.completions.create replaced by an AsyncMock."""
    with (
        patch("mellea.backends.openai.openai.OpenAI"),
        patch("mellea.backends.openai._server_type", return_value=MagicMock()),
    ):
        backend = OpenAIBackend(
            model_id="test-model",
            base_url="http://localhost:9999/v1",
            api_key="test",
            model_options={ModelOption.MAX_NEW_TOKENS: 5},
        )

    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(return_value=_CANNED_RESPONSE)

    with patch.object(
        type(backend),
        "_async_client",
        new_callable=PropertyMock,
        return_value=mock_client,
    ):
        yield MelleaSession(backend), mock_client


# ---------------------------------------------------------------------------
# Structural payload tests
# ---------------------------------------------------------------------------


def test_audio_block_in_instruct_payload_shape(
    mocked_openai_session: tuple, audio_block: AudioBlock
):
    """AudioBlock is serialised as an ``input_audio`` content part in the outgoing payload."""
    session, _ = mocked_openai_session

    session.instruct("Transcribe this audio.", audio=[audio_block], strategy=None)

    turn = session.ctx.last_turn()
    assert turn is not None
    lp = turn.output._generate_log.prompt  # type: ignore[union-attr]
    assert isinstance(lp, list)
    # Find the user message (last non-system entry)
    user_msg = lp[-1]
    content = user_msg["content"]
    assert isinstance(content, list), "content should be a list when audio is present"

    audio_parts = [p for p in content if p.get("type") == "input_audio"]
    assert len(audio_parts) == 1, f"expected 1 input_audio part, got {audio_parts}"

    ia = audio_parts[0]["input_audio"]
    assert ia["format"] == "wav"
    # The data field must be raw base64 (no data: URI prefix).
    assert not ia["data"].startswith("data:"), (
        "data field must be raw base64, not a data URI"
    )


def test_audio_url_block_rejected_by_openai(mocked_openai_session: tuple):
    """AudioUrlBlock raises ValueError before the request is sent."""
    session, _ = mocked_openai_session
    url_block = AudioUrlBlock("https://example.com/audio.wav", format="wav")
    with pytest.raises(ValueError, match="AudioUrlBlock"):
        session.instruct("Transcribe this.", audio=[url_block], strategy=None)
