import base64
import io
from typing import Any

import pytest
from PIL import Image as PILImage

from mellea.core import CBlock, Component, ImageBlock, ModelOutputThunk
from mellea.stdlib.components import Message


def test_cblock():
    cb = CBlock(value="This is some text")
    str(cb)
    repr(cb)
    assert str(cb) == "This is some text"


def test_cblpock_meta():
    cb = CBlock("asdf", meta={"x": "y"})
    assert str(cb) == "asdf"
    assert cb._meta["x"] == "y"


def test_component():
    class _ClosuredComponent(Component[str]):
        def parts(self):
            return []

        def format_for_llm(self) -> str:
            return ""

        def _parse(self, computed: ModelOutputThunk) -> str:
            return ""

    c = _ClosuredComponent()
    assert len(c.parts()) == 0


def test_parse():
    class _ChatResponse:
        def __init__(self, msg: Message) -> None:
            self.message = msg

    source = Message(role="user", content="source message")
    result = ModelOutputThunk(
        value="result value",
        meta={
            "chat_response": _ChatResponse(
                Message(role="assistant", content="assistant reply")
            )
        },
    )

    result.parsed_repr = source.parse(result)
    assert isinstance(result.parsed_repr, Message), (
        "result's parsed repr should be a message when meta includes a chat_response"
    )
    assert result.parsed_repr.role == "assistant", (
        "result's parsed repr role should be assistant"
    )
    assert result.parsed_repr.content == "assistant reply"

    result = ModelOutputThunk(value="result value")
    result.parsed_repr = source.parse(result)
    assert isinstance(result.parsed_repr, Message), (
        "result's parsed repr should be a message when source component is a message"
    )
    assert result.parsed_repr.content == "result value"


# --- CBlock edge cases ---


def test_cblock_non_string_value_raises():
    with pytest.raises(TypeError, match="should always be a string or None"):
        CBlock(value=42)  # type: ignore


def test_cblock_none_value_allowed():
    cb = CBlock(value=None)
    assert str(cb) == ""


def test_cblock_value_setter():
    cb = CBlock(value="old")
    cb.value = "new"
    assert cb.value == "new"


# --- ImageBlock.is_valid_base64_png ---


def _make_png_b64() -> str:
    img = PILImage.new("RGB", (1, 1), color="red")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def test_image_block_valid_png():
    b64 = _make_png_b64()
    assert ImageBlock.is_valid_base64_png(b64) is True


def test_image_block_invalid_base64_returns_false():
    assert ImageBlock.is_valid_base64_png("not-base64!!!") is False


def test_image_block_valid_base64_but_not_png():
    # Base64-encoded JPEG magic bytes
    jpg_magic = base64.b64encode(b"\xff\xd8\xff" + b"\x00" * 20).decode()
    assert ImageBlock.is_valid_base64_png(jpg_magic) is False


def test_image_block_data_uri_prefix_stripped():
    b64 = _make_png_b64()
    data_uri = f"data:image/png;base64,{b64}"
    assert ImageBlock.is_valid_base64_png(data_uri) is True


def test_image_block_invalid_value_raises():
    with pytest.raises(AssertionError, match="Invalid base64"):
        ImageBlock(value="not-a-png")


# --- ModelOutputThunk._copy_from ---


def test_mot_copy_from_copies_underlying_value():
    a = ModelOutputThunk(value=None)
    b = ModelOutputThunk(value="copied")
    a._copy_from(b)
    # _copy_from copies _underlying_value (not _computed), so check raw field
    assert a._underlying_value == "copied"


def test_mot_copy_from_copies_meta():
    a = ModelOutputThunk(value=None)
    b = ModelOutputThunk(value="x", meta={"key": "val"})
    a._copy_from(b)
    assert a._meta["key"] == "val"


def test_mot_copy_from_copies_tool_calls():
    a = ModelOutputThunk(value=None)
    b = ModelOutputThunk(value="x", tool_calls={"fn": None})
    a._copy_from(b)
    assert a.tool_calls == {"fn": None}


def test_mot_copy_from_copies_usage():
    a = ModelOutputThunk(value=None)
    b = ModelOutputThunk(value="x")
    b.usage = {"prompt_tokens": 10}
    a._copy_from(b)
    assert a.usage == {"prompt_tokens": 10}


if __name__ == "__main__":
    pytest.main([__file__])
