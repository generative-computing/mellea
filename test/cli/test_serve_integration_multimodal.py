# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for multimodal content in m serve endpoint.

Tests that the serve endpoint correctly passes ChatMessage objects
with multimodal content to the user's serve function.
"""

import base64
from unittest.mock import Mock

import pytest

from cli.serve.app import make_chat_endpoint
from cli.serve.models import (
    ChatCompletionRequest,
    ChatMessage,
    ImageUrlContent,
    InputAudioContent,
    InputAudioData,
    TextContent,
)
from mellea.core import AudioBlock, ModelOutputThunk

# --- Helper functions ---


def _make_valid_png_base64() -> str:
    """Create a minimal valid PNG as base64."""
    png_data = (
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
        b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\xcf"
        b"\xc0\x00\x00\x00\x00\xff\xff\x03\x00\x05\x00\x02\xfe\xdc\xccY"
        b"\x00\x00\x00\x00IEND\xaeB`\x82"
    )
    return base64.b64encode(png_data).decode("utf-8")


def _make_valid_wav_base64() -> str:
    """Create a minimal valid WAV file (4 silent 16-bit PCM samples) as base64."""
    import struct

    num_samples = 4
    sample_rate = 16000
    data_size = num_samples * 2
    chunk_size = 36 + data_size
    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        chunk_size,
        b"WAVE",
        b"fmt ",
        16,
        1,
        1,
        sample_rate,
        sample_rate * 2,
        2,
        16,
        b"data",
        data_size,
    )
    samples = b"\x00\x00" * num_samples
    return base64.b64encode(header + samples).decode("utf-8")


# --- Test fixtures ---


@pytest.fixture
def mock_output():
    """Create a mock ModelOutputThunk."""
    output = Mock(spec=ModelOutputThunk)
    output.value = "Test response"
    output.mot = Mock()
    output.mot.generation = Mock()
    output.mot.generation.usage = {
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "total_tokens": 15,
    }
    output.mot.generation.finish_reason = "stop"
    output.mot.tool_calls = None
    return output


# --- Integration tests ---


@pytest.mark.integration
class TestServeEndpointWithImages:
    """Test serve endpoint with image content."""

    async def test_endpoint_passes_images_inside_input_messages(self, mock_output):
        """Endpoint passes ChatMessage objects with multimodal content to serve function."""
        received_input = None

        mock_module = Mock()
        mock_module.__name__ = "test_module"

        async def mock_serve(input, requirements=None, model_options=None):
            nonlocal received_input
            received_input = input
            return mock_output

        mock_module.serve = mock_serve
        endpoint = make_chat_endpoint(mock_module)

        png_b64 = _make_valid_png_base64()
        request = ChatCompletionRequest(
            model="test-model",
            messages=[
                ChatMessage(
                    role="user",
                    content=[
                        TextContent(type="text", text="What's in this image?"),
                        ImageUrlContent(
                            type="image_url",
                            image_url={"url": f"data:image/png;base64,{png_b64}"},
                        ),
                    ],
                )
            ],
        )

        await endpoint(request)

        assert received_input is not None
        assert len(received_input) == 1
        # Verify it's a ChatMessage with multimodal content
        assert isinstance(received_input[0], ChatMessage)
        assert received_input[0].get_text_content() == "What's in this image?"
        image_urls = received_input[0].get_image_urls()
        assert len(image_urls) == 1
        assert image_urls[0].startswith("data:image/png;base64,")

    async def test_endpoint_preserves_images_on_corresponding_input_messages(
        self, mock_output
    ):
        """Endpoint preserves per-message image association in ChatMessage objects."""
        received_input = None

        mock_module = Mock()
        mock_module.__name__ = "test_module"

        async def mock_serve(input, requirements=None, model_options=None):
            nonlocal received_input
            received_input = input
            return mock_output

        mock_module.serve = mock_serve
        endpoint = make_chat_endpoint(mock_module)

        png_b64 = _make_valid_png_base64()
        request = ChatCompletionRequest(
            model="test-model",
            messages=[
                ChatMessage(
                    role="user",
                    content=[
                        TextContent(type="text", text="first"),
                        ImageUrlContent(
                            type="image_url",
                            image_url={"url": f"data:image/png;base64,{png_b64}"},
                        ),
                    ],
                ),
                ChatMessage(
                    role="user",
                    content=[
                        TextContent(type="text", text="second"),
                        ImageUrlContent(
                            type="image_url",
                            image_url={"url": f"data:image/png;base64,{png_b64}"},
                        ),
                    ],
                ),
            ],
        )

        await endpoint(request)

        assert received_input is not None
        assert len(received_input) == 2
        # Verify both are ChatMessage objects with their own images
        assert isinstance(received_input[0], ChatMessage)
        assert isinstance(received_input[1], ChatMessage)
        assert received_input[0].get_text_content() == "first"
        assert received_input[1].get_text_content() == "second"
        assert len(received_input[0].get_image_urls()) == 1
        assert len(received_input[1].get_image_urls()) == 1

    async def test_endpoint_passes_text_only_messages(self, mock_output):
        """Text-only requests are passed as ChatMessage objects."""
        received_input = None

        mock_module = Mock()
        mock_module.__name__ = "test_module"

        async def mock_serve(input, requirements=None, model_options=None):
            nonlocal received_input
            received_input = input
            return mock_output

        mock_module.serve = mock_serve
        endpoint = make_chat_endpoint(mock_module)

        request = ChatCompletionRequest(
            model="test-model", messages=[ChatMessage(role="user", content="Hello")]
        )

        await endpoint(request)

        assert received_input is not None
        assert len(received_input) == 1
        assert isinstance(received_input[0], ChatMessage)
        assert received_input[0].content == "Hello"
        assert len(received_input[0].get_image_urls()) == 0


@pytest.mark.integration
class TestServeEndpointWithMixedMultimodal:
    """Test serve endpoint with mixed multimodal content."""

    async def test_endpoint_backward_compatible_text_only(self, mock_output):
        """Endpoint remains backward compatible with text-only messages."""
        mock_module = Mock()
        mock_module.__name__ = "test_module"

        # Old-style serve function without images/audio parameters
        async def mock_serve(input, requirements=None, model_options=None):
            return mock_output

        mock_module.serve = mock_serve

        endpoint = make_chat_endpoint(mock_module)

        request = ChatCompletionRequest(
            model="test-model", messages=[ChatMessage(role="user", content="Hello")]
        )

        response = await endpoint(request)
        assert response is not None


@pytest.mark.integration
class TestServeEndpointSyncFunction:
    """Test serve endpoint with synchronous serve functions."""

    async def test_sync_serve_with_images(self, mock_output):
        """Endpoint works with sync serve function receiving ChatMessage objects."""
        received_input = None

        mock_module = Mock()
        mock_module.__name__ = "test_module"

        def mock_serve(input, requirements=None, model_options=None):
            nonlocal received_input
            received_input = input
            return mock_output

        mock_module.serve = mock_serve

        endpoint = make_chat_endpoint(mock_module)

        png_b64 = _make_valid_png_base64()
        request = ChatCompletionRequest(
            model="test-model",
            messages=[
                ChatMessage(
                    role="user",
                    content=[
                        ImageUrlContent(
                            type="image_url",
                            image_url={"url": f"data:image/png;base64,{png_b64}"},
                        )
                    ],
                )
            ],
        )

        await endpoint(request)

        assert received_input is not None
        assert len(received_input) == 1
        assert isinstance(received_input[0], ChatMessage)
        image_urls = received_input[0].get_image_urls()
        assert len(image_urls) == 1
        assert image_urls[0].startswith("data:image/png;base64,")


# --- Audio unit tests ---


class TestGetAudioBlocks:
    """Unit tests for ChatMessage.get_audio_blocks()."""

    def test_returns_empty_list_for_string_content(self):
        """get_audio_blocks() returns [] when content is a plain string."""
        msg = ChatMessage(role="user", content="hello")
        assert msg.get_audio_blocks() == []

    def test_returns_empty_list_for_none_content(self):
        """get_audio_blocks() returns [] when content is None."""
        msg = ChatMessage(role="user", content=None)
        assert msg.get_audio_blocks() == []

    def test_returns_audio_block_for_valid_wav(self):
        """get_audio_blocks() returns an AudioBlock for a valid base64 WAV part."""
        wav_b64 = _make_valid_wav_base64()
        msg = ChatMessage(
            role="user",
            content=[
                InputAudioContent(
                    type="input_audio",
                    input_audio=InputAudioData(data=wav_b64, format="wav"),
                )
            ],
        )
        blocks = msg.get_audio_blocks()
        assert len(blocks) == 1
        assert isinstance(blocks[0], AudioBlock)
        assert blocks[0].format == "wav"

    def test_returns_empty_list_for_no_audio_parts(self):
        """get_audio_blocks() returns [] when content has only text parts."""
        msg = ChatMessage(role="user", content=[TextContent(type="text", text="hello")])
        assert msg.get_audio_blocks() == []

    def test_raises_value_error_for_invalid_base64(self):
        """get_audio_blocks() raises ValueError for an invalid base64 payload."""
        msg = ChatMessage(
            role="user",
            content=[
                InputAudioContent(
                    type="input_audio",
                    input_audio=InputAudioData(
                        data="not-valid-base64!!!", format="wav"
                    ),
                )
            ],
        )
        with pytest.raises(ValueError, match="Invalid audio data"):
            msg.get_audio_blocks()

    def test_get_text_content_ignores_audio_parts(self):
        """get_text_content() skips InputAudioContent items."""
        wav_b64 = _make_valid_wav_base64()
        msg = ChatMessage(
            role="user",
            content=[
                TextContent(type="text", text="describe this"),
                InputAudioContent(
                    type="input_audio",
                    input_audio=InputAudioData(data=wav_b64, format="wav"),
                ),
            ],
        )
        assert msg.get_text_content() == "describe this"

    def test_deserialises_mixed_text_and_audio(self):
        """ChatMessage deserialises a mixed list of TextContent and InputAudioContent."""
        wav_b64 = _make_valid_wav_base64()
        raw = {
            "role": "user",
            "content": [
                {"type": "text", "text": "what is this?"},
                {
                    "type": "input_audio",
                    "input_audio": {"data": wav_b64, "format": "wav"},
                },
            ],
        }
        msg = ChatMessage.model_validate(raw)
        assert len(msg.content) == 2  # type: ignore[arg-type]
        assert isinstance(msg.content[0], TextContent)  # type: ignore[index]
        assert isinstance(msg.content[1], InputAudioContent)  # type: ignore[index]
