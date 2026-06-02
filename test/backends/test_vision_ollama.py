"""Tests for Ollama backend vision (image) support.

Three tiers:

1. **Construction** (unit) — pure ImageBlock logic, no backend or server required.
2. **Structural payload** (unit, mocked) — verify mellea correctly embeds images
   into the Ollama conversation payload. The Ollama transport is mocked so no
   server or vision model is needed. Runs in CI unconditionally.
3. **Dormant live e2e** (e2e, qualitative) — full round-trip against a real
   vision-capable Ollama model. Skipped today because granite-vision-4.1 is not
   yet in the Ollama library. Reactivates automatically once the model is pulled.
   See #1187 for the activation checklist.
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
from mellea.core import ImageBlock, ModelOutputThunk
from mellea.stdlib.components import Instruction, Message

# Ollama name for the target vision model; bump to the model_ids constant once
# IBM_GRANITE_VISION_4_1_4B is added to mellea/backends/model_ids.py.
_VISION_MODEL = "granite-vision-4.1"
_SKIP_REASON = (
    f"Vision model {_VISION_MODEL!r} not available in Ollama — "
    "see https://github.com/generative-computing/mellea/issues/1187"
)


# ── Shared image fixture ──────────────────────────────────────────────────────


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
    image_block = ImageBlock.from_pil_image(pil_image)

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
    assert last_action.images[0] == image_block.value  # type: ignore[union-attr]

    lp = turn.output._generate_log.prompt  # type: ignore[union-attr]
    assert isinstance(lp, list)
    assert len(lp) == 1
    prompt_msg = lp[0]
    assert isinstance(prompt_msg, dict)

    image_list = prompt_msg.get("images")
    assert isinstance(image_list, list)
    assert len(image_list) == 1
    assert image_list[0] == str(image_block)


# ── Tier 3: Dormant live e2e ──────────────────────────────────────────────────
#
# Full round-trip against a real vision-capable Ollama model.  Currently skipped
# because granite-vision-4.1 is not yet in the Ollama library.
#
# To activate: ensure `ollama pull granite-vision-4.1` succeeds, then remove the
# pytest.skip() call from vision_session below and add the model to the CI pull
# step in .github/workflows/quality.yml.  See #1187 for the full checklist.


def _ollama_vision_model_available() -> bool:
    """Return True if _VISION_MODEL is present in the local Ollama model list."""
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
        pulled = {m.get("name", "") for m in resp.json().get("models", [])}
        return any(_VISION_MODEL in name for name in pulled)
    except Exception:
        return False


@pytest.fixture
def vision_session():
    if not _ollama_vision_model_available():
        pytest.skip(_SKIP_REASON)

    from mellea import start_session

    m = start_session(
        "ollama", model_id=_VISION_MODEL, model_options={ModelOption.MAX_NEW_TOKENS: 5}
    )
    yield m
    del m


@pytest.mark.e2e
@pytest.mark.ollama
@pytest.mark.qualitative
def test_vision_instruct_live_e2e(
    vision_session: MelleaSession, pil_image: Image.Image
):
    """Live vision instruct round-trip; skips until granite-vision-4.1 lands on Ollama."""
    image_block = ImageBlock.from_pil_image(pil_image)
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
@pytest.mark.qualitative
def test_vision_chat_live_e2e(vision_session: MelleaSession, pil_image: Image.Image):
    """Live vision chat round-trip; skips until granite-vision-4.1 lands on Ollama."""
    ct = vision_session.chat(
        "Is this image mainly blue? Answer yes or no.", images=[pil_image]
    )
    assert isinstance(ct, Message)
    assert ct.content is not None
    assert len(ct.content) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
