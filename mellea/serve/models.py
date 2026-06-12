"""User-facing types for `m serve`."""

from typing import Any, Literal

from pydantic import BaseModel

from mellea.core import ImageBlock, ImageUrlBlock


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


# Union type for all content types
MessageContent = TextContent | ImageUrlContent


class ChatMessage(BaseModel):
    """Chat message with support for text-only or multimodal content.

    The content field can be:
    - A string (text-only, backward compatible)
    - None (for messages without content)
    - A list of content objects (multimodal: text, images)
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
            Images are ignored (handled separately via extraction utilities).
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
            List of ``ImageBlock`` (for base64/data-URI images) or
            ``ImageUrlBlock`` (for http/https URLs) from all ImageUrlContent
            items. Empty list if content is a string or contains no images.

        Raises:
            ValueError: If an image URL is invalid or cannot be processed.
        """
        image_urls = self.get_image_urls()
        image_blocks: list[ImageBlock | ImageUrlBlock] = []
        for url in image_urls:
            if url.startswith(("http://", "https://")):
                image_blocks.append(ImageUrlBlock(url))
            else:
                try:
                    image_blocks.append(ImageBlock(url))
                except AssertionError as e:
                    # Raise ValueError for invalid data so the client gets a clear 400 error
                    # rather than silently processing a request without the expected images
                    raise ValueError(
                        f"Invalid image data: {url[:100]}{'...' if len(url) > 100 else ''}. "
                        f"Error: {e}"
                    ) from e
        return image_blocks
