"""Unit tests verifying WatsonxAIBackend rejects image and audio inputs.

Watsonx does not support image or audio inputs.  These tests confirm that
passing ``ImageBlock``, ``ImageUrlBlock``, ``AudioBlock``, or ``AudioUrlBlock``
raises ``ValueError`` before any network call is made, rather than silently
dropping the data.
"""

import base64
import struct

import pytest

pytest.importorskip(
    "ibm_watsonx_ai", reason="ibm_watsonx_ai not installed — install mellea[watsonx]"
)

from collections.abc import Generator
from unittest.mock import AsyncMock, MagicMock, patch

from mellea import MelleaSession
from mellea.backends.watsonx import WatsonxAIBackend
from mellea.core import AudioBlock, AudioUrlBlock, ImageBlock, ImageUrlBlock

# ---------------------------------------------------------------------------
# Minimal WAV — smallest valid base64 audio payload
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

# ---------------------------------------------------------------------------
# Minimal PNG — smallest valid base64 image payload
# ---------------------------------------------------------------------------

_MINIMAL_PNG = (
    b"\x89PNG\r\n\x1a\n"
    b"\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
    b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx"
    b"\x9cc\xf8\x0f\x00\x00\x01\x01\x00\x05\x18\xd8N\x00"
    b"\x00\x00\x00IEND\xaeB`\x82"
)
_B64_PNG = base64.b64encode(_MINIMAL_PNG).decode()


# ---------------------------------------------------------------------------
# Fixture: mocked WatsonxAIBackend wrapped in a MelleaSession
# ---------------------------------------------------------------------------


def _make_backend(monkeypatch: pytest.MonkeyPatch) -> WatsonxAIBackend:
    monkeypatch.delenv("WATSONX_API_KEY", raising=False)
    monkeypatch.delenv("WATSONX_URL", raising=False)
    monkeypatch.delenv("WATSONX_PROJECT_ID", raising=False)
    with (
        patch("mellea.backends.watsonx.Credentials"),
        patch("mellea.backends.watsonx.APIClient"),
        patch("mellea.backends.watsonx.ModelInference"),
    ):
        return WatsonxAIBackend(
            model_id="ibm/granite-4-h-small",
            base_url="https://example.com",
            project_id="test-project",
            api_key="test-key",
        )


@pytest.fixture
def watsonx_session(
    monkeypatch: pytest.MonkeyPatch,
) -> Generator[MelleaSession, None, None]:
    backend = _make_backend(monkeypatch)
    # Patch _model so generate_from_chat_context never reaches the SDK.
    mock_model = MagicMock()
    mock_model.achat = AsyncMock(return_value={})
    mock_model.achat_stream = AsyncMock(return_value=MagicMock())
    with patch.object(
        type(backend), "_model", new_callable=lambda: property(lambda self: mock_model)
    ):
        yield MelleaSession(backend)


# ---------------------------------------------------------------------------
# Image rejection tests
# ---------------------------------------------------------------------------


def test_image_block_rejected_by_watsonx(watsonx_session: MelleaSession):
    """ImageBlock raises ValueError before any network call."""
    img: list[ImageBlock | ImageUrlBlock] = [ImageBlock(_B64_PNG)]
    with pytest.raises(ValueError, match="ImageBlock"):
        watsonx_session.chat("Describe this image.", images=img)


def test_image_url_block_rejected_by_watsonx(watsonx_session: MelleaSession):
    """ImageUrlBlock raises ValueError before any network call."""
    images: list[ImageBlock | ImageUrlBlock] = [
        ImageUrlBlock("https://example.com/photo.png")
    ]
    with pytest.raises(ValueError, match="ImageBlock"):
        watsonx_session.chat("Describe this image.", images=images)


# ---------------------------------------------------------------------------
# Audio rejection tests
# ---------------------------------------------------------------------------


def test_audio_block_rejected_by_watsonx(watsonx_session: MelleaSession):
    """AudioBlock raises ValueError before any network call."""
    audio = AudioBlock(_B64_WAV, format="wav")
    with pytest.raises(ValueError, match="AudioBlock"):
        watsonx_session.chat("Transcribe this audio.", audio=[audio])


def test_audio_url_block_rejected_by_watsonx(watsonx_session: MelleaSession):
    """AudioUrlBlock raises ValueError before any network call."""
    url_block = AudioUrlBlock("https://example.com/audio.wav", format="wav")
    with pytest.raises(ValueError, match="AudioBlock"):
        watsonx_session.chat("Transcribe this audio.", audio=[url_block])
