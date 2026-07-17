# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for functional.py pure helpers — no backend, no LLM required.

Covers image preprocessing plus chat()/instruct() forwarding of multimodal inputs.
"""

import base64
import io
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image as PILImage

from mellea.core import AudioBlock, ImageBlock
from mellea.stdlib.components import Document, Instruction, Message
from mellea.stdlib.context import SimpleContext
from mellea.stdlib.functional import (
    _parse_and_clean_image_args,
    achat,
    ainstruct,
    chat,
    instruct,
)


def _make_image_block() -> ImageBlock:
    """Return a valid ImageBlock backed by a 1x1 red PNG."""
    img = PILImage.new("RGB", (1, 1), color="red")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    return ImageBlock(value=b64)


def _make_audio_block() -> AudioBlock:
    """Return a valid AudioBlock backed by a short WAV payload."""
    wav_bytes = (
        b"RIFF$\x00\x00\x00WAVEfmt "
        b"\x10\x00\x00\x00\x01\x00\x01\x00"
        b"@\x1f\x00\x00\x80>\x00\x00"
        b"\x02\x00\x10\x00data\x00\x00\x00\x00"
    )
    b64 = base64.b64encode(wav_bytes).decode()
    return AudioBlock(value=b64, format="wav")


# --- _parse_and_clean_image_args ---


def test_none_returns_none():
    assert _parse_and_clean_image_args(None) is None


def test_empty_list_returns_none():
    assert _parse_and_clean_image_args([]) is None


def test_image_blocks_passed_through():
    ib = _make_image_block()
    result = _parse_and_clean_image_args([ib])
    assert result == [ib]


def test_multiple_image_blocks_preserved():
    ib1 = _make_image_block()
    ib2 = _make_image_block()
    result = _parse_and_clean_image_args([ib1, ib2])
    assert result is not None
    assert len(result) == 2
    assert result[0] is ib1
    assert result[1] is ib2


def test_pil_images_converted_to_image_blocks():
    pil_img = PILImage.new("RGB", (1, 1), color="blue")
    result = _parse_and_clean_image_args([pil_img])
    assert result is not None
    assert len(result) == 1
    assert isinstance(result[0], ImageBlock)


def test_non_list_raises():
    with pytest.raises(AssertionError, match="Images should be a list"):
        _parse_and_clean_image_args("not_a_list")  # type: ignore


# --- chat() document forwarding ---


@patch("mellea.stdlib.functional.act")
def test_chat_forwards_documents_to_message(mock_act):
    """Verify that chat() passes documents through to the Message it constructs."""
    # Set up mock to return a fake assistant message and context
    assistant_msg = Message(role="assistant", content="reply")
    mock_result = MagicMock()
    mock_result.parsed_repr = assistant_msg
    mock_ctx = SimpleContext()
    mock_act.return_value = (mock_result, mock_ctx)

    backend = MagicMock()
    ctx = SimpleContext()

    chat("hello", ctx, backend, documents=["grounding text", "more context"])

    # Inspect the Message that was passed to act()
    user_message = mock_act.call_args[0][0]
    assert isinstance(user_message, Message)
    assert user_message._docs is not None
    assert len(user_message._docs) == 2
    assert all(isinstance(d, Document) for d in user_message._docs)
    assert user_message._docs[0].text == "grounding text"
    assert user_message._docs[1].text == "more context"


@patch("mellea.stdlib.functional.act")
def test_chat_no_documents_by_default(mock_act):
    """Verify that chat() passes None documents when not specified."""
    assistant_msg = Message(role="assistant", content="reply")
    mock_result = MagicMock()
    mock_result.parsed_repr = assistant_msg
    mock_act.return_value = (mock_result, SimpleContext())

    chat("hello", SimpleContext(), MagicMock())

    user_message = mock_act.call_args[0][0]
    assert isinstance(user_message, Message)
    assert user_message._docs is None


@patch("mellea.stdlib.functional.act")
def test_chat_forwards_audio_and_images(mock_act):
    """Verify that chat() passes multimodal inputs through to the Message."""
    assistant_msg = Message(role="assistant", content="reply")
    mock_result = MagicMock()
    mock_result.parsed_repr = assistant_msg
    mock_act.return_value = (mock_result, SimpleContext())

    image = PILImage.new("RGB", (1, 1), color="green")
    audio = _make_audio_block()

    chat("hello", SimpleContext(), MagicMock(), images=[image], audio=[audio])

    user_message = mock_act.call_args[0][0]
    assert isinstance(user_message, Message)
    assert user_message.audio == [audio]
    assert user_message.images is not None
    assert len(user_message.images) == 1
    assert isinstance(user_message.images[0], ImageBlock)


@patch("mellea.stdlib.functional.act")
def test_instruct_forwards_audio_to_instruction(mock_act):
    """Verify that instruct() forwards audio blocks into the Instruction."""
    mock_act.return_value = (MagicMock(), SimpleContext())
    audio = _make_audio_block()

    instruct("describe this audio", SimpleContext(), MagicMock(), audio=[audio])

    instruction = mock_act.call_args[0][0]
    assert isinstance(instruction, Instruction)
    assert instruction._audio == [audio]


@patch("mellea.stdlib.functional.act")
def test_instruct_converts_pil_images_before_forwarding(mock_act):
    """Verify that instruct() converts PIL images before building the Instruction."""
    mock_act.return_value = (MagicMock(), SimpleContext())
    image = PILImage.new("RGB", (1, 1), color="yellow")

    instruct("describe this image", SimpleContext(), MagicMock(), images=[image])

    instruction = mock_act.call_args[0][0]
    assert isinstance(instruction, Instruction)
    assert instruction._images is not None
    assert len(instruction._images) == 1
    assert isinstance(instruction._images[0], ImageBlock)


# --- achat() and ainstruct() async forwarding ---


@pytest.mark.asyncio
@patch("mellea.stdlib.functional.aact")
async def test_achat_forwards_audio(mock_aact):
    """Verify that achat() passes audio through to the Message."""
    assistant_msg = Message(role="assistant", content="reply")
    mock_result = MagicMock()
    mock_result.parsed_repr = assistant_msg
    mock_aact.return_value = (mock_result, SimpleContext())

    audio = _make_audio_block()
    await achat("hello", SimpleContext(), MagicMock(), audio=[audio])

    user_message = mock_aact.call_args[0][0]
    assert isinstance(user_message, Message)
    assert user_message.audio == [audio]


@pytest.mark.asyncio
@patch("mellea.stdlib.functional.aact")
async def test_ainstruct_forwards_audio(mock_aact):
    """Verify that ainstruct() forwards audio blocks into the Instruction."""
    mock_aact.return_value = (MagicMock(), SimpleContext())
    audio = _make_audio_block()

    await ainstruct("describe this audio", SimpleContext(), MagicMock(), audio=[audio])

    instruction = mock_aact.call_args[0][0]
    assert isinstance(instruction, Instruction)
    assert instruction._audio == [audio]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
