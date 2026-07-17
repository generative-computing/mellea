# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests verifying MelleaSession forwards audio params to mfuncs — no backend required."""

import base64
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mellea.core import AudioBlock, AudioUrlBlock
from mellea.stdlib.components import Message
from mellea.stdlib.context import SimpleContext
from mellea.stdlib.session import MelleaSession


def _make_audio_block() -> AudioBlock:
    """Return a minimal valid AudioBlock."""
    wav_bytes = (
        b"RIFF$\x00\x00\x00WAVEfmt "
        b"\x10\x00\x00\x00\x01\x00\x01\x00"
        b"@\x1f\x00\x00\x80>\x00\x00"
        b"\x02\x00\x10\x00data\x00\x00\x00\x00"
    )
    return AudioBlock(value=base64.b64encode(wav_bytes).decode(), format="wav")


def _make_session() -> MagicMock:
    """Return a mock MelleaSession with a live context."""
    session = MagicMock(spec=MelleaSession)
    session.ctx = SimpleContext()
    session.backend = MagicMock()
    return session


# --- session.instruct() ---


@patch("mellea.stdlib.session.mfuncs")
def test_session_instruct_forwards_audio(mock_mfuncs):
    """session.instruct() must pass audio through to mfuncs.instruct()."""
    mock_mfuncs.instruct.return_value = (MagicMock(), SimpleContext())
    audio = [_make_audio_block()]

    MelleaSession.instruct(
        _make_session(), "describe audio", audio=audio, strategy=None
    )

    _, kwargs = mock_mfuncs.instruct.call_args
    assert kwargs["audio"] == audio


# --- session.chat() ---


@patch("mellea.stdlib.session.mfuncs")
def test_session_chat_forwards_audio(mock_mfuncs):
    """session.chat() must pass audio through to mfuncs.chat()."""
    mock_mfuncs.chat.return_value = (
        Message(role="assistant", content="reply"),
        SimpleContext(),
    )
    audio = [_make_audio_block()]

    MelleaSession.chat(_make_session(), "hello", audio=audio)

    _, kwargs = mock_mfuncs.chat.call_args
    assert kwargs["audio"] == audio


# --- session.ainstruct() ---


@pytest.mark.asyncio
@patch("mellea.stdlib.session.mfuncs")
async def test_session_ainstruct_forwards_audio(mock_mfuncs):
    """session.ainstruct() must pass audio through to mfuncs.ainstruct()."""
    mock_mfuncs.ainstruct = AsyncMock(return_value=(MagicMock(), SimpleContext()))
    audio = [_make_audio_block()]

    await MelleaSession.ainstruct(
        _make_session(), "describe audio", audio=audio, strategy=None
    )

    _, kwargs = mock_mfuncs.ainstruct.call_args
    assert kwargs["audio"] == audio


# --- session.achat() ---


@pytest.mark.asyncio
@patch("mellea.stdlib.session.mfuncs")
async def test_session_achat_forwards_audio(mock_mfuncs):
    """session.achat() must pass audio through to mfuncs.achat()."""
    mock_mfuncs.achat = AsyncMock(
        return_value=(Message(role="assistant", content="reply"), SimpleContext())
    )
    audio = [AudioUrlBlock("https://example.com/clip.wav", format="wav")]

    await MelleaSession.achat(_make_session(), "hello", audio=audio)

    _, kwargs = mock_mfuncs.achat.call_args
    assert kwargs["audio"] == audio


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
