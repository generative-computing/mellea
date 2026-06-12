"""Unit tests for multimodal content support in CLI serve models."""

import base64

import pytest
from pydantic import ValidationError

from mellea.core import ImageBlock
from mellea.serve.models import ChatMessage, ImageUrlContent, TextContent


class TestTextContent:
    """Test TextContent model."""

    def test_text_content_valid(self):
        """TextContent accepts valid text."""
        content = TextContent(type="text", text="Hello world")
        assert content.type == "text"
        assert content.text == "Hello world"

    def test_text_content_requires_text_field(self):
        """TextContent requires text field."""
        with pytest.raises(ValidationError):
            TextContent(type="text")

    def test_text_content_requires_type_text(self):
        """TextContent type must be 'text'."""
        with pytest.raises(ValidationError):
            TextContent(type="image", text="Hello")


class TestImageUrlContent:
    """Test ImageUrlContent model."""

    def test_image_url_content_valid(self):
        """ImageUrlContent accepts valid image URL."""
        content = ImageUrlContent(
            type="image_url", image_url={"url": "data:image/png;base64,iVBORw0KGgo..."}
        )
        assert content.type == "image_url"
        assert content.image_url["url"].startswith("data:image/png")

    def test_image_url_content_http_url(self):
        """ImageUrlContent accepts HTTP URLs."""
        content = ImageUrlContent(
            type="image_url", image_url={"url": "https://example.com/image.png"}
        )
        assert content.image_url["url"] == "https://example.com/image.png"

    def test_image_url_content_requires_image_url_field(self):
        """ImageUrlContent requires image_url field."""
        with pytest.raises(ValidationError):
            ImageUrlContent(type="image_url")

    def test_image_url_content_requires_url_in_dict(self):
        """ImageUrlContent requires 'url' key in image_url dict."""
        # Pydantic allows empty dict, but we can test with missing field
        content = ImageUrlContent(type="image_url", image_url={})
        assert content.image_url == {}

    def test_image_url_content_with_detail(self):
        """ImageUrlContent accepts optional detail parameter."""
        content = ImageUrlContent(
            type="image_url",
            image_url={"url": "data:image/png;base64,iVBOR...", "detail": "high"},
        )
        assert content.image_url["detail"] == "high"


class TestChatMessageTextOnly:
    """Test ChatMessage with text-only content (backward compatibility)."""

    def test_chat_message_string_content(self):
        """ChatMessage accepts simple string content."""
        msg = ChatMessage(role="user", content="Hello world")
        assert msg.role == "user"
        assert msg.content == "Hello world"
        assert isinstance(msg.content, str)

    def test_chat_message_none_content(self):
        """ChatMessage accepts None content."""
        msg = ChatMessage(role="assistant", content=None)
        assert msg.content is None

    def test_chat_message_empty_string(self):
        """ChatMessage accepts empty string content."""
        msg = ChatMessage(role="user", content="")
        assert msg.content == ""


class TestChatMessageMultimodal:
    """Test ChatMessage with multimodal content arrays."""

    def test_chat_message_text_and_image(self):
        """ChatMessage accepts text and image content."""
        msg = ChatMessage(
            role="user",
            content=[
                {"type": "text", "text": "What's in this image?"},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,iVBOR..."},
                },
            ],
        )
        assert isinstance(msg.content, list)
        assert len(msg.content) == 2
        assert msg.content[0].type == "text"
        assert msg.content[1].type == "image_url"

    def test_chat_message_multiple_images(self):
        """ChatMessage accepts multiple images."""
        msg = ChatMessage(
            role="user",
            content=[
                {"type": "text", "text": "Compare these images"},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,iVBOR..."},
                },
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,iVBOR..."},
                },
            ],
        )
        assert len(msg.content) == 3
        image_count = sum(1 for item in msg.content if item.type == "image_url")
        assert image_count == 2

    def test_chat_message_image_only(self):
        """ChatMessage accepts image-only content."""
        msg = ChatMessage(
            role="user",
            content=[
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,iVBOR..."},
                }
            ],
        )
        assert len(msg.content) == 1
        assert msg.content[0].type == "image_url"

    def test_chat_message_empty_content_array(self):
        """ChatMessage accepts empty content array."""
        msg = ChatMessage(role="user", content=[])
        assert msg.content == []


class TestChatMessageValidation:
    """Test ChatMessage validation rules."""

    def test_chat_message_requires_role(self):
        """ChatMessage requires role field."""
        with pytest.raises(ValidationError):
            ChatMessage(content="Hello")

    def test_chat_message_invalid_role(self):
        """ChatMessage rejects invalid role."""
        with pytest.raises(ValidationError):
            ChatMessage(role="invalid", content="Hello")

    def test_chat_message_valid_roles(self):
        """ChatMessage accepts all valid roles."""
        valid_roles = ["system", "user", "assistant", "tool", "function"]
        for role in valid_roles:
            msg = ChatMessage(role=role, content="Test")
            assert msg.role == role

    def test_chat_message_invalid_content_type(self):
        """ChatMessage rejects invalid content types."""
        with pytest.raises(ValidationError):
            ChatMessage(role="user", content=123)

    def test_chat_message_invalid_multimodal_type(self):
        """ChatMessage rejects invalid content type in array."""
        with pytest.raises(ValidationError):
            ChatMessage(role="user", content=[{"type": "invalid", "data": "..."}])


class TestChatMessageWithOtherFields:
    """Test ChatMessage with additional fields (tool_call_id, name, etc.)."""

    def test_chat_message_with_name(self):
        """ChatMessage accepts name field."""
        msg = ChatMessage(role="user", content="Hello", name="Alice")
        assert msg.name == "Alice"

    def test_chat_message_with_tool_call_id(self):
        """ChatMessage accepts tool_call_id field."""
        msg = ChatMessage(role="tool", content="Result", tool_call_id="call_123")
        assert msg.tool_call_id == "call_123"

    def test_chat_message_multimodal_with_name(self):
        """ChatMessage accepts multimodal content with name."""
        msg = ChatMessage(
            role="user",
            name="Alice",
            content=[
                {"type": "text", "text": "Hello"},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,iVBOR..."},
                },
            ],
        )
        assert msg.name == "Alice"
        assert len(msg.content) == 2


class TestBackwardCompatibility:
    """Test that existing text-only usage still works."""

    def test_existing_text_message_unchanged(self):
        """Existing text message creation still works."""
        msg = ChatMessage(role="user", content="Hello")
        assert msg.content == "Hello"

    def test_existing_none_content_unchanged(self):
        """Existing None content still works."""
        msg = ChatMessage(role="assistant", content=None)
        assert msg.content is None

    def test_existing_tool_message_unchanged(self):
        """Existing tool message still works."""
        msg = ChatMessage(role="tool", content="Result", tool_call_id="call_123")
        assert msg.content == "Result"


class TestChatMessageImageBlocks:
    """Test ChatMessage.get_image_blocks() method."""

    def test_get_image_blocks_valid_base64(self):
        """ChatMessage.get_image_blocks() returns ImageBlock objects for valid base64 images."""
        # Create a minimal valid PNG
        png_data = (
            b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
            b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\xcf"
            b"\xc0\x00\x00\x00\x00\xff\xff\x03\x00\x05\x00\x02\xfe\xdc\xccY"
            b"\x00\x00\x00\x00IEND\xaeB`\x82"
        )
        png_b64 = base64.b64encode(png_data).decode("utf-8")

        msg = ChatMessage(
            role="user",
            content=[
                ImageUrlContent(
                    type="image_url",
                    image_url={"url": f"data:image/png;base64,{png_b64}"},
                )
            ],
        )
        blocks = msg.get_image_blocks()
        assert len(blocks) == 1
        assert isinstance(blocks[0], ImageBlock)

    def test_get_image_blocks_multiple_images(self):
        """ChatMessage.get_image_blocks() returns multiple ImageBlock objects."""
        png_data = (
            b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
            b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\xcf"
            b"\xc0\x00\x00\x00\x00\xff\xff\x03\x00\x05\x00\x02\xfe\xdc\xccY"
            b"\x00\x00\x00\x00IEND\xaeB`\x82"
        )
        png_b64 = base64.b64encode(png_data).decode("utf-8")

        msg = ChatMessage(
            role="user",
            content=[
                ImageUrlContent(
                    type="image_url",
                    image_url={"url": f"data:image/png;base64,{png_b64}"},
                ),
                ImageUrlContent(
                    type="image_url",
                    image_url={"url": f"data:image/png;base64,{png_b64}"},
                ),
            ],
        )
        blocks = msg.get_image_blocks()
        assert len(blocks) == 2

    def test_get_image_blocks_invalid_url(self):
        """ChatMessage.get_image_blocks() raises ValueError for invalid URLs."""
        msg = ChatMessage(
            role="user",
            content=[
                ImageUrlContent(
                    type="image_url", image_url={"url": "invalid-url-format"}
                )
            ],
        )
        with pytest.raises(ValueError, match="Invalid image URL"):
            msg.get_image_blocks()

    def test_get_image_blocks_invalid_base64(self):
        """ChatMessage.get_image_blocks() raises ValueError for invalid base64 data."""
        msg = ChatMessage(
            role="user",
            content=[
                ImageUrlContent(
                    type="image_url",
                    image_url={"url": "data:image/png;base64,not-valid-base64!!!"},
                )
            ],
        )
        with pytest.raises(ValueError, match="Invalid image data"):
            msg.get_image_blocks()

    def test_get_image_blocks_empty_list(self):
        """ChatMessage.get_image_blocks() returns empty list for text-only content."""
        msg = ChatMessage(role="user", content="Hello world")
        blocks = msg.get_image_blocks()
        assert blocks == []

    def test_get_image_blocks_mixed_content(self):
        """ChatMessage.get_image_blocks() extracts only images from mixed content."""
        png_data = (
            b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
            b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\xcf"
            b"\xc0\x00\x00\x00\x00\xff\xff\x03\x00\x05\x00\x02\xfe\xdc\xccY"
            b"\x00\x00\x00\x00IEND\xaeB`\x82"
        )
        png_b64 = base64.b64encode(png_data).decode("utf-8")

        msg = ChatMessage(
            role="user",
            content=[
                TextContent(type="text", text="What's in this image?"),
                ImageUrlContent(
                    type="image_url",
                    image_url={"url": f"data:image/png;base64,{png_b64}"},
                ),
            ],
        )
        blocks = msg.get_image_blocks()
        assert len(blocks) == 1

    def test_get_image_blocks_truncates_long_url_in_error(self):
        """ChatMessage.get_image_blocks() truncates long URLs in error messages."""
        long_url = "data:image/png;base64," + "A" * 200
        msg = ChatMessage(
            role="user",
            content=[ImageUrlContent(type="image_url", image_url={"url": long_url})],
        )
        with pytest.raises(ValueError) as exc_info:
            msg.get_image_blocks()
        # Error message should contain truncated URL (first 100 chars + ...)
        assert "..." in str(exc_info.value)
        assert len(str(exc_info.value)) < len(long_url)
