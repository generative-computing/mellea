# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for Ollama backend vision (image) support.

Three tiers:

1. **Construction** (unit) — pure ImageBlock logic, no backend or server required.
2. **Structural payload** (unit, mocked) — verify mellea correctly embeds images
   into the Ollama conversation payload. The Ollama transport is mocked so no
   server or vision model is needed. Runs in CI unconditionally.
3. **Live e2e** (e2e) — full round-trip against a real vision-capable Ollama
   model. The assertions are structural (a thunk/message with non-empty content),
   so these are `e2e` and not `qualitative`: no assertion here can be broken by
   swapping the model version, and marking them `qualitative` would exclude them
   from CI, where `CICD=1` skips that tier. Requires `granite-vision-4.1` to be
   pulled; skipped locally with a pull command if it is not, and failed in CI,
   where the workflow is responsible for pulling it.
"""

import base64
import os
from io import BytesIO
from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch

import numpy as np
import ollama
import pytest
from PIL import Image

from mellea import MelleaSession
from mellea.backends import ModelOption
from mellea.backends.model_ids import IBM_GRANITE_VISION_4_1_4B
from mellea.core import (
    AudioBlock,
    AudioUrlBlock,
    ImageBlock,
    ImageUrlBlock,
    ModelOutputThunk,
)
from mellea.stdlib.components import Instruction, Message

# granite-vision-4.1 is not in the Ollama library, so the model is pulled from
# IBM's official GGUF repo on Hugging Face. Match on the full tag rather than a
# bare name so the check cannot be satisfied by an unrelated local alias.
_VISION_MODEL = IBM_GRANITE_VISION_4_1_4B
# Asserted rather than coerced with str(): ollama_name is `str | None`, and
# str(None) would silently make the tag the literal "None".
assert IBM_GRANITE_VISION_4_1_4B.ollama_name is not None
_VISION_MODEL_TAG = IBM_GRANITE_VISION_4_1_4B.ollama_name
_SKIP_REASON = (
    f"Vision model not pulled locally — run `ollama pull {_VISION_MODEL_TAG}`"
)

# The model's full 131072-token context window loads ~9 GB for a job that needs a
# few thousand tokens; capping it keeps the live tests near 2.5 GB.
_VISION_CONTEXT_WINDOW = 4096


# ── Shared image fixture ──────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _clear_image_cache():
    """Isolate tests from the process-wide URL -> base64 download cache."""
    from mellea.core import base as base_mod

    base_mod._image_base64_cache.clear()
    yield
    base_mod._image_base64_cache.clear()


@pytest.fixture(scope="module")
def pil_image():
    rng = np.random.default_rng(seed=42)
    data = rng.integers(0, 256, size=(150, 200, 3), dtype=np.uint8)
    img = Image.fromarray(data, "RGB")
    yield img
    del img


# ── Tier 1: Construction tests (unit, no server) ──────────────────────────────


def test_image_block_construction(pil_image: Image.Image):
    buffered = BytesIO()
    pil_image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    image_block = ImageBlock(img_str)
    assert isinstance(image_block, ImageBlock)
    assert isinstance(image_block.value, str)


def test_image_block_construction_from_pil(pil_image: Image.Image):
    image_block = ImageBlock.from_pil_image(pil_image)
    assert isinstance(image_block, ImageBlock)
    assert isinstance(image_block.value, str)
    assert ImageBlock.is_valid_base64_png(str(image_block))


# ── Tier 2: Structural payload tests (unit, offline mock) ────────────────────
#
# Verify that mellea correctly embeds ImageBlock instances into the Ollama
# conversation payload — the images=[...] field on the outgoing message dict.
# The Ollama transport (OllamaModelBackend._async_client.chat) is replaced with
# an AsyncMock so post_processing runs and populates _generate_log.prompt
# without making any network call.  No server, no vision model required.


@pytest.fixture
def mocked_session(mock_ollama_backend):
    canned = ollama.ChatResponse(
        model="granite4.1:3b",
        created_at=None,
        message=ollama.Message(role="assistant", content="no"),
        done=True,
    )
    mock_async = MagicMock()
    mock_async.chat = AsyncMock(return_value=canned)
    backend = mock_ollama_backend(model_options={ModelOption.MAX_NEW_TOKENS: 5})
    # _async_client is an event-loop-keyed property; mock it at the class level so
    # the same mock is returned regardless of which event loop _run_async_in_thread
    # creates in the background thread.
    with patch.object(
        type(backend),
        "_async_client",
        new_callable=PropertyMock,
        return_value=mock_async,
    ):
        yield MelleaSession(backend)


def test_image_block_in_instruction(
    mocked_session: MelleaSession, pil_image: Image.Image
):
    image_block: ImageBlock | ImageUrlBlock = ImageBlock.from_pil_image(pil_image)

    instr = mocked_session.instruct(
        "Is this image mainly blue? Answer yes or no.",
        images=[image_block],
        strategy=None,
    )
    assert isinstance(instr, ModelOutputThunk)

    turn = mocked_session.ctx.last_turn()
    assert turn is not None
    last_action = turn.model_input
    assert isinstance(last_action, Instruction)
    assert last_action._images is not None  # type: ignore[union-attr]
    assert len(last_action._images) > 0  # type: ignore[union-attr]
    assert last_action._images[0] == image_block  # type: ignore[union-attr]

    lp = turn.output._generate_log.prompt  # type: ignore[union-attr]
    assert isinstance(lp, list)
    assert len(lp) == 1
    prompt_msg = lp[0]
    assert isinstance(prompt_msg, dict)

    # Ollama-specific: images are embedded as a top-level list on the message dict.
    image_list = prompt_msg.get("images")
    assert isinstance(image_list, list)
    assert len(image_list) == 1
    assert image_list[0] == str(image_block)


def test_image_url_block_auto_downloaded_by_ollama(
    mocked_session: MelleaSession, pil_image: Image.Image
):
    # Ollama only accepts base64 images, so a URL image must be downloaded and
    # encoded transparently rather than rejected.
    encoded = ImageBlock.from_pil_image(pil_image).value
    url_block: ImageUrlBlock = ImageUrlBlock("https://example.com/photo.png")
    images: list[ImageBlock | ImageUrlBlock] = [url_block]

    with patch(
        "mellea.core.base._download_image_as_base64", return_value=encoded
    ) as mock_download:
        mocked_session.chat("What is in this image?", images=images)

    mock_download.assert_called_once_with("https://example.com/photo.png")

    turn = mocked_session.ctx.last_turn()
    assert turn is not None
    lp = turn.output._generate_log.prompt  # type: ignore[union-attr]
    assert isinstance(lp, list)
    prompt_msg = lp[0]
    image_list = prompt_msg.get("images")
    assert isinstance(image_list, list)
    assert len(image_list) == 1
    # The downloaded base64 (data-URI-stripped) is embedded in the payload.
    assert image_list[0] == encoded


def test_image_url_block_download_failure_raises(mocked_session: MelleaSession):
    url_block: ImageUrlBlock = ImageUrlBlock("https://example.com/photo.png")
    images: list[ImageBlock | ImageUrlBlock] = [url_block]

    with patch(
        "mellea.core.base._download_image_as_base64",
        side_effect=ValueError("Failed to download or decode image from URL"),
    ):
        with pytest.raises(ValueError, match="Failed to download"):
            mocked_session.chat("What is in this image?", images=images)


def test_audio_block_rejected_by_ollama(mocked_session: MelleaSession):
    """AudioBlock raises ValueError — Ollama does not support audio input."""
    import base64
    import struct

    silent = struct.pack("<h", 0)
    header = (
        b"RIFF"
        + struct.pack("<I", 36 + len(silent))
        + b"WAVEfmt "
        + struct.pack("<IHHIIHH", 16, 1, 1, 16000, 32000, 2, 16)
        + b"data"
        + struct.pack("<I", len(silent))
    )
    b64_wav = base64.b64encode(header + silent).decode()
    audio = AudioBlock(b64_wav, format="wav")
    with pytest.raises(ValueError, match="audio"):
        mocked_session.chat("Transcribe this.", audio=[audio])


def test_audio_url_block_rejected_by_ollama(mocked_session: MelleaSession):
    """AudioUrlBlock raises ValueError — Ollama does not support audio input."""
    url_block = AudioUrlBlock("https://example.com/audio.wav", format="wav")
    with pytest.raises(ValueError, match="audio"):
        mocked_session.chat("Transcribe this.", audio=[url_block])


def test_image_url_block_drives_real_download(
    mocked_session: MelleaSession, pil_image: Image.Image
):
    """Exercise the real `_download_image_as_base64` through the Ollama path.

    Patches only the network layer (`requests.get`) so the block's
    `resolve_base64` call, thread-offload, and payload wiring are all covered
    — this catches regressions the fully-mocked download tests cannot.
    """
    buf = BytesIO()
    pil_image.save(buf, format="PNG")
    png_bytes = buf.getvalue()

    url_block: ImageUrlBlock = ImageUrlBlock("https://example.com/photo.png")
    images: list[ImageBlock | ImageUrlBlock] = [url_block]

    class _FakeRaw:
        def read(self, amt=None, decode_content=False):
            return png_bytes if amt is None else png_bytes[:amt]

    class _FakeResponse:
        headers: dict[str, str] = {}
        raw = _FakeRaw()

        def raise_for_status(self):
            return None

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    with patch("mellea.core.base.requests.get", return_value=_FakeResponse()):
        mocked_session.chat("What is in this image?", images=images)

    turn = mocked_session.ctx.last_turn()
    assert turn is not None
    lp = turn.output._generate_log.prompt  # type: ignore[union-attr]
    assert isinstance(lp, list)
    image_list = lp[0].get("images")
    assert isinstance(image_list, list)
    assert len(image_list) == 1
    # The real helper re-encodes as base64 PNG (data-URI-stripped in the payload).
    assert base64.b64decode(image_list[0])


def test_image_block_in_chat(mocked_session: MelleaSession, pil_image: Image.Image):
    image_block = ImageBlock.from_pil_image(pil_image)
    ct = mocked_session.chat(
        "Is this image mainly blue? Answer yes or no.", images=[pil_image]
    )
    assert isinstance(ct, Message)

    turn = mocked_session.ctx.last_turn()
    assert turn is not None
    last_action = turn.model_input
    assert isinstance(last_action, Message)
    assert last_action.images is not None  # type: ignore[union-attr]
    assert len(last_action.images) > 0  # type: ignore[union-attr]
    first_image = last_action.images[0]
    assert isinstance(first_image, ImageBlock)
    assert first_image.value == image_block.value

    lp = turn.output._generate_log.prompt  # type: ignore[union-attr]
    assert isinstance(lp, list)
    assert len(lp) == 1
    prompt_msg = lp[0]
    assert isinstance(prompt_msg, dict)

    image_list = prompt_msg.get("images")
    assert isinstance(image_list, list)
    assert len(image_list) == 1
    assert image_list[0] == str(image_block)


# ── Tier 3: Live e2e ──────────────────────────────────────────────────────────
#
# Full round-trip against a real vision-capable Ollama model.  CI pulls the model
# in the "Pull models" step of .github/workflows/quality.yml; locally these skip
# unless you have pulled it yourself (the skip message gives the command).


def _pulled_ollama_models() -> set[str]:
    """Return the model names Ollama reports locally; empty if it is unreachable."""
    import requests

    host = os.environ.get("OLLAMA_HOST", "127.0.0.1")
    if ":" in host:
        host, port = host.rsplit(":", 1)
    else:
        port = os.environ.get("OLLAMA_PORT", "11434")
    base_url = f"http://{host}:{port}"
    try:
        resp = requests.get(f"{base_url}/api/tags", timeout=5)
        resp.raise_for_status()
        return {m.get("name", "") for m in resp.json().get("models", [])}
    except Exception:
        return set()


def _vision_model_pulled(pulled: set[str]) -> bool:
    """Return True if the exact vision model tag is among the pulled model names."""
    return _VISION_MODEL_TAG in pulled


def test_vision_model_tag_matches_the_pulled_tag():
    """Only the exact pinned tag counts; anything else may be a different model."""
    assert _vision_model_pulled({_VISION_MODEL_TAG})
    assert not _vision_model_pulled(set())
    assert not _vision_model_pulled({"granite4.1:3b", "granite3.2-vision:latest"})
    # A bare alias of that name could point anywhere, and a longer tag sharing
    # the prefix is a different model -- neither counts.
    assert not _vision_model_pulled({"granite-vision-4.1:latest"})
    assert not _vision_model_pulled({f"{_VISION_MODEL_TAG}-extra"})


@pytest.fixture
def vision_session(gh_run: int):
    pulled = _pulled_ollama_models()
    if not _vision_model_pulled(pulled):
        if gh_run:
            # CI pulls this model in the "Pull models" step, so a miss here is a real
            # failure -- either the pull did not happen or Ollama reports the model
            # under a name this check does not recognise. Skipping would hide both
            # behind a green tick.
            pytest.fail(
                f"{_SKIP_REASON}\nOllama reported: "
                f"{sorted(pulled) if pulled else '<no models / server unreachable>'}"
            )
        pytest.skip(_SKIP_REASON)

    from mellea import start_session

    m = start_session(
        "ollama",
        model_id=_VISION_MODEL,
        model_options={
            ModelOption.MAX_NEW_TOKENS: 5,
            ModelOption.CONTEXT_WINDOW: _VISION_CONTEXT_WINDOW,
        },
    )
    yield m
    del m


@pytest.mark.e2e
@pytest.mark.ollama
def test_vision_instruct_live_e2e(
    vision_session: MelleaSession, pil_image: Image.Image
):
    """Live vision instruct round-trip against granite-vision-4.1."""
    image_block: ImageBlock | ImageUrlBlock = ImageBlock.from_pil_image(pil_image)
    instr = vision_session.instruct(
        "Is this image mainly blue? Answer yes or no.",
        images=[image_block],
        strategy=None,
    )
    assert isinstance(instr, ModelOutputThunk)
    assert instr.value is not None
    assert len(str(instr.value)) > 0


@pytest.mark.e2e
@pytest.mark.ollama
def test_vision_chat_live_e2e(vision_session: MelleaSession, pil_image: Image.Image):
    """Live vision chat round-trip against granite-vision-4.1."""
    ct = vision_session.chat(
        "Is this image mainly blue? Answer yes or no.", images=[pil_image]
    )
    assert isinstance(ct, Message)
    assert ct.content is not None
    assert len(ct.content) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
