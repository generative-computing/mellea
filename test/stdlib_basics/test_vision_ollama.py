import base64
import os
from io import BytesIO

import numpy as np
import pytest
from PIL import Image

from mellea import MelleaSession, start_session
from mellea.backends import ModelOption
from mellea.stdlib.base import ImageBlock, ModelOutputThunk
from mellea.stdlib.chat import Message
from mellea.stdlib.instruction import Instruction


@pytest.fixture(scope="module")
def m_session(gh_run):
    if gh_run == 1:
        m = start_session(model_options={ModelOption.MAX_NEW_TOKENS: 5})
    else:
        m = start_session(
            "ollama",
            model_id="granite3.2-vision",
            model_options={ModelOption.MAX_NEW_TOKENS: 5},
        )
    yield m
    del m


@pytest.fixture(scope="module")
def pil_image():
    width = 200
    height = 150
    random_pixel_data = np.random.randint(
        0, 256, size=(height, width, 3), dtype=np.uint8
    )
    random_image = Image.fromarray(random_pixel_data, "RGB")
    yield random_image
    del random_image


def test_image_block_construction(pil_image: Image.Image):
    # create base64 PNG string from image:
    buffered = BytesIO()
    pil_image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

    image_block = ImageBlock(img_str)
    assert isinstance(image_block, ImageBlock)
    assert isinstance(image_block._value, str)


def test_image_block_construction_from_pil(pil_image: Image.Image):
    image_block = ImageBlock.from_pil_image(pil_image)
    assert isinstance(image_block, ImageBlock)
    assert isinstance(image_block._value, str)
    assert ImageBlock.is_valid_base64_png(str(image_block))


def test_image_block_in_instruction(
    m_session: MelleaSession, pil_image: Image.Image, gh_run: int
):
    image_block = ImageBlock.from_pil_image(pil_image)

    # Set strategy=None here since we are directly comparing the object and sampling strategies tend to do a deepcopy.
    instr = m_session.instruct(
        "Is this image mainly blue? Answer yes or no.",
        images=[image_block],
        strategy=None,
    )
    assert isinstance(instr, ModelOutputThunk)

    # if not on GH
    if not gh_run == 1:
        assert "yes" in instr.value.lower() or "no" in instr.value.lower()

    # make sure you get the last action
    turn = m_session.ctx.last_turn()
    assert turn is not None
    last_action = turn.model_input
    assert isinstance(last_action, Instruction)
    assert len(last_action._images) > 0

    # first image in image list should be the same as the image block
    image0 = last_action._images[0]
    assert image0 == image_block

    # get prompt message
    lp = turn.output._generate_log.prompt
    assert isinstance(lp, list)
    assert len(lp) == 1

    # prompt message is a dict
    prompt_msg = lp[0]
    assert isinstance(prompt_msg, dict)

    # ### OLLAMA SPECIFIC TEST ####

    # get content
    image_list = prompt_msg.get("images", None)
    assert isinstance(image_list, list)
    assert len(image_list) == 1

    # get the image content
    content_img = image_list[0]
    assert isinstance(content_img, str)

    # check that the image is the same
    assert content_img == str(image_block)


def test_image_block_in_chat(
    m_session: MelleaSession, pil_image: Image.Image, gh_run: int
):
    ct = m_session.chat(
        "Is this image mainly blue? Answer yes or no.", images=[pil_image]
    )
    assert isinstance(ct, Message)

    # if not on GH
    if not gh_run == 1:
        assert "yes" in ct.content.lower() or "no" in ct.content.lower()

    # make sure you get the last action
    turn = m_session.ctx.last_turn()
    assert turn is not None
    last_action = turn.model_input
    assert isinstance(last_action, Message)
    assert len(last_action.images) > 0

    # first image in image list should be the same as the image block
    image0_str = last_action.images[0]
    assert image0_str == ImageBlock.from_pil_image(pil_image)._value

    # get prompt message
    lp = turn.output._generate_log.prompt
    assert isinstance(lp, list)
    assert len(lp) == 1

    # prompt message is a dict
    prompt_msg = lp[0]
    assert isinstance(prompt_msg, dict)

    # ### OLLAMA SPECIFIC TEST ####

    # get content
    image_list = prompt_msg.get("images", None)
    assert isinstance(image_list, list)
    assert len(image_list) == 1

    # get the image content
    content_img = image_list[0]
    assert isinstance(content_img, str)

    # check that the image is the same
    assert content_img == str(ImageBlock.from_pil_image(pil_image))


if __name__ == "__main__":
    pytest.main([__file__])
