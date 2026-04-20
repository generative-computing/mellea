"""Unit tests for functional.py pure helpers — no backend, no LLM required.

Covers _parse_and_clean_image_args image preprocessing.
"""

import base64
import io

import pytest
from PIL import Image as PILImage

from mellea.core import ImageBlock
from mellea.stdlib.functional import _parse_and_clean_image_args


def _make_image_block() -> ImageBlock:
    """Return a valid ImageBlock backed by a 1x1 red PNG."""
    img = PILImage.new("RGB", (1, 1), color="red")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    return ImageBlock(value=b64)


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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
