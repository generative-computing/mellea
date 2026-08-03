# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""User-facing types for `m serve`."""

from typing import Any, Literal

from pydantic import BaseModel

from mellea.core import AudioBlock, ImageBlock, ImageUrlBlock


class TextContent(BaseModel):
    """Text content in a multimodal message."""

    type: Literal["text"]
    text: str


class ImageUrlContent(BaseModel):
    """Image URL content in a multimodal message.

    Supports both data URIs (base64-encoded images) and HTTP(S) URLs.
    """

    type: Literal["image_url"]
    image_url: dict[str, str]
    """Image URL object containing 'url' key and optional 'detail' key."""


class InputAudioData(BaseModel):
    """Audio data payload for an `input_audio` content part."""

    data: str
    """Base64-encoded audio data."""
    format: str
    """Audio format, e.g. `"wav"` or `"mp3"`."""


class InputAudioContent(BaseModel):
    """Audio content part in an OpenAI-compatible multimodal message.

    Matches the `{"type": "input_audio", "input_audio": {"data": "...", "format": "wav"}}`
    wire format used by the OpenAI Chat Completions API.
    """

    type: Literal["input_audio"]
    input_audio: InputAudioData


# Union type for all content types
MessageContent = TextContent | ImageUrlContent | InputAudioContent


class ChatMessage(BaseModel):
    """Chat message with support for text-only or multimodal content.

    The content field can be:
    - A string (text-only, backward compatible)
    - None (for messages without content)
    - A list of content objects (multimodal: text, images, audio)
    """

    role: Literal["system", "user", "assistant", "tool", "function"]
    content: str | list[MessageContent] | None = None
    name: str | None = None
    tool_call_id: str | None = None
    function_call: dict[str, Any] | None = None  # For function/tool messages

    def get_text_content(self) -> str:
        """Extract text content from message, handling both string and multimodal formats.

        Returns:
            Concatenated text from all TextContent items, or empty string if no text.
            Images and audio are ignored (handled separately via extraction utilities).
        """
        if isinstance(self.content, str):
            return self.content
        elif isinstance(self.content, list):
            return " ".join(
                item.text for item in self.content if isinstance(item, TextContent)
            )
        return ""

    def get_image_urls(self) -> list[str]:
        """Extract image URLs from message content.

        Returns:
            List of image URL strings from all ImageUrlContent items.
            Empty list if content is a string or contains no images.
        """
        if not isinstance(self.content, list):
            return []
        urls = []
        for item in self.content:
            if isinstance(item, ImageUrlContent):
                url = item.image_url.get("url")
                if url:
                    urls.append(url)
        return urls

    def get_image_blocks(self) -> list[ImageBlock | ImageUrlBlock]:
        """Extract image blocks from message content.

        Returns:
            List of `ImageBlock` (for base64/data-URI images) or
            `ImageUrlBlock` (for http/https URLs) from all ImageUrlContent
            items. Empty list if content is a string or contains no images.

        Raises:
            ValueError: If an image URL is invalid or cannot be processed.
        """
        image_urls = self.get_image_urls()
        image_blocks: list[ImageBlock | ImageUrlBlock] = []
        for url in image_urls:
            if url.startswith(("http://", "https://")):
                image_blocks.append(ImageUrlBlock(url))
            elif url.startswith("data:"):
                try:
                    image_blocks.append(ImageBlock(url))
                except ValueError as e:
                    # Raise ValueError for invalid data so the client gets a clear 400 error
                    # rather than silently processing a request without the expected images
                    raise ValueError(
                        f"Invalid image data: {url[:100]}{'...' if len(url) > 100 else ''}. "
                        f"Error: {e}"
                    ) from e
            else:
                raise ValueError(
                    f"Invalid image URL: {url[:100]}{'...' if len(url) > 100 else ''}. "
                    "Expected an http/https URL or a data: URI."
                )
        return image_blocks

    def get_audio_blocks(self) -> list[AudioBlock]:
        """Extract audio blocks from message content.

        Returns:
            List of `AudioBlock` instances from all `InputAudioContent` items.
            Empty list if content is a string or contains no audio parts.

        Raises:
            ValueError: If an audio data payload is invalid or cannot be processed.
        """
        if not isinstance(self.content, list):
            return []
        audio_blocks: list[AudioBlock] = []
        for item in self.content:
            if isinstance(item, InputAudioContent):
                try:
                    audio_blocks.append(
                        AudioBlock(item.input_audio.data, item.input_audio.format)
                    )
                except ValueError as e:
                    raise ValueError(
                        f"Invalid audio data: {item.input_audio.data[:100]}"
                        f"{'...' if len(item.input_audio.data) > 100 else ''}. "
                        f"Error: {e}"
                    ) from e
        return audio_blocks
