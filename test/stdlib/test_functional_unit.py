# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for functional.py pure helpers — no backend, no LLM required.

Covers image preprocessing plus chat()/instruct() forwarding of multimodal inputs.
"""

import base64
import io
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from PIL import Image as PILImage

from mellea.core import AudioBlock, Context, ImageBlock, ModelToolCall
from mellea.stdlib.components import (
    Document,
    Instruction,
    Message,
    MObject,
    ToolMessage,
)
from mellea.stdlib.context import ChatContext, SimpleContext
from mellea.stdlib.functional import (
    _parse_and_clean_image_args,
    aact,
    achat,
    ainstruct,
    chat,
    instruct,
)


def _make_image_block() -> ImageBlock:
    """Return a valid ImageBlock backed by a 1x1 red PNG."""
    img = PILImage.new("RGB", (1, 1), color="red")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    return ImageBlock(value=b64)


def _make_audio_block() -> AudioBlock:
    """Return a valid AudioBlock backed by a short WAV payload."""
    wav_bytes = (
        b"RIFF$\x00\x00\x00WAVEfmt "
        b"\x10\x00\x00\x00\x01\x00\x01\x00"
        b"@\x1f\x00\x00\x80>\x00\x00"
        b"\x02\x00\x10\x00data\x00\x00\x00\x00"
    )
    b64 = base64.b64encode(wav_bytes).decode()
    return AudioBlock(value=b64, format="wav")


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


# --- chat() document forwarding ---


@patch("mellea.stdlib.functional.act")
def test_chat_forwards_documents_to_message(mock_act):
    """Verify that chat() passes documents through to the Message it constructs."""
    # Set up mock to return a fake assistant message and context
    assistant_msg = Message(role="assistant", content="reply")
    mock_result = MagicMock()
    mock_result.parsed_repr = assistant_msg
    mock_ctx = SimpleContext()
    mock_act.return_value = (mock_result, mock_ctx)

    backend = MagicMock()
    ctx = SimpleContext()

    chat("hello", ctx, backend, documents=["grounding text", "more context"])

    # Inspect the Message that was passed to act()
    user_message = mock_act.call_args[0][0]
    assert isinstance(user_message, Message)
    assert user_message._docs is not None
    assert len(user_message._docs) == 2
    assert all(isinstance(d, Document) for d in user_message._docs)
    assert user_message._docs[0].text == "grounding text"
    assert user_message._docs[1].text == "more context"


@patch("mellea.stdlib.functional.act")
def test_chat_no_documents_by_default(mock_act):
    """Verify that chat() passes None documents when not specified."""
    assistant_msg = Message(role="assistant", content="reply")
    mock_result = MagicMock()
    mock_result.parsed_repr = assistant_msg
    mock_act.return_value = (mock_result, SimpleContext())

    chat("hello", SimpleContext(), MagicMock())

    user_message = mock_act.call_args[0][0]
    assert isinstance(user_message, Message)
    assert user_message._docs is None


@patch("mellea.stdlib.functional.act")
def test_chat_forwards_audio_and_images(mock_act):
    """Verify that chat() passes multimodal inputs through to the Message."""
    assistant_msg = Message(role="assistant", content="reply")
    mock_result = MagicMock()
    mock_result.parsed_repr = assistant_msg
    mock_act.return_value = (mock_result, SimpleContext())

    image = PILImage.new("RGB", (1, 1), color="green")
    audio = _make_audio_block()

    chat("hello", SimpleContext(), MagicMock(), images=[image], audio=[audio])

    user_message = mock_act.call_args[0][0]
    assert isinstance(user_message, Message)
    assert user_message.audio == [audio]
    assert user_message.images is not None
    assert len(user_message.images) == 1
    assert isinstance(user_message.images[0], ImageBlock)


@patch("mellea.stdlib.functional.act")
def test_instruct_forwards_audio_to_instruction(mock_act):
    """Verify that instruct() forwards audio blocks into the Instruction."""
    mock_act.return_value = (MagicMock(), SimpleContext())
    audio = _make_audio_block()

    instruct("describe this audio", SimpleContext(), MagicMock(), audio=[audio])

    instruction = mock_act.call_args[0][0]
    assert isinstance(instruction, Instruction)
    assert instruction._audio == [audio]


@patch("mellea.stdlib.functional.act")
def test_instruct_converts_pil_images_before_forwarding(mock_act):
    """Verify that instruct() converts PIL images before building the Instruction."""
    mock_act.return_value = (MagicMock(), SimpleContext())
    image = PILImage.new("RGB", (1, 1), color="yellow")

    instruct("describe this image", SimpleContext(), MagicMock(), images=[image])

    instruction = mock_act.call_args[0][0]
    assert isinstance(instruction, Instruction)
    assert instruction._images is not None
    assert len(instruction._images) == 1
    assert isinstance(instruction._images[0], ImageBlock)


# --- achat() and ainstruct() async forwarding ---


@pytest.mark.asyncio
@patch("mellea.stdlib.functional.aact")
async def test_achat_forwards_audio(mock_aact):
    """Verify that achat() passes audio through to the Message."""
    assistant_msg = Message(role="assistant", content="reply")
    mock_result = MagicMock()
    mock_result.parsed_repr = assistant_msg
    mock_aact.return_value = (mock_result, SimpleContext())

    audio = _make_audio_block()
    await achat("hello", SimpleContext(), MagicMock(), audio=[audio])

    user_message = mock_aact.call_args[0][0]
    assert isinstance(user_message, Message)
    assert user_message.audio == [audio]


@pytest.mark.asyncio
@patch("mellea.stdlib.functional.aact")
async def test_ainstruct_forwards_audio(mock_aact):
    """Verify that ainstruct() forwards audio blocks into the Instruction."""
    mock_aact.return_value = (MagicMock(), SimpleContext())
    audio = _make_audio_block()

    await ainstruct("describe this audio", SimpleContext(), MagicMock(), audio=[audio])

    instruction = mock_aact.call_args[0][0]
    assert isinstance(instruction, Instruction)
    assert instruction._audio == [audio]


# --- act/aact accept CBlock and ModelOutputThunk (issue #356) ---


def _default_strategy(func) -> object:
    """Return the default value of the `strategy` parameter of `func`."""
    import inspect

    return inspect.signature(func).parameters["strategy"].default


def test_act_default_strategy_is_none():
    """act's default sampling strategy must be None (issue #356, decision #2)."""
    from mellea.stdlib.functional import act
    from mellea.stdlib.session import MelleaSession

    assert _default_strategy(act) is None
    assert _default_strategy(MelleaSession.act) is None


def test_instruct_default_strategy_unchanged():
    """instruct keeps its RejectionSamplingStrategy default (scope: act only)."""
    from mellea.stdlib.functional import instruct
    from mellea.stdlib.sampling import RejectionSamplingStrategy

    assert isinstance(_default_strategy(instruct), RejectionSamplingStrategy)


def _mock_backend_returning(value: str):
    """Return a mock backend whose generate_from_context yields a computed MOT."""
    from mellea.core import GenerateLog, ModelOutputThunk

    backend = MagicMock()

    async def mock_generate(action, *, ctx, **kwargs):
        output = ModelOutputThunk(value)
        output._generate_log = GenerateLog()
        return output, ctx.add(action).add(output)

    backend.generate_from_context = mock_generate
    return backend


@pytest.mark.asyncio
async def test_aact_accepts_cblock_action():
    """aact over a raw CBlock generates without sampling (issue #356)."""
    from mellea.core import CBlock

    backend = _mock_backend_returning("2")
    ctx = SimpleContext()

    out, new_ctx = await aact(CBlock("What is 1+1?"), ctx, backend, await_result=True)

    assert str(out) == "2"
    assert new_ctx is not ctx


@pytest.mark.asyncio
async def test_aact_accepts_mot_action():
    """aact over a raw ModelOutputThunk generates without sampling (issue #356)."""
    from mellea.core import ModelOutputThunk

    backend = _mock_backend_returning("4")
    ctx = SimpleContext()

    out, new_ctx = await aact(
        ModelOutputThunk("prior"), ctx, backend, await_result=True
    )

    assert str(out) == "4"
    assert new_ctx is not ctx


# --- return_sampling_results without a strategy raises ValueError ---


@pytest.mark.asyncio
async def test_aact_sampling_results_without_strategy_raises():
    """aact rejects return_sampling_results=True with no strategy (ValueError)."""
    from mellea.core import CBlock

    backend = _mock_backend_returning("ignored")
    with pytest.raises(ValueError):
        await aact(CBlock("x"), SimpleContext(), backend, return_sampling_results=True)


def test_act_sampling_results_without_strategy_raises():
    """act surfaces the same ValueError as aact when no strategy is given."""
    from mellea.core import CBlock
    from mellea.stdlib.functional import act

    backend = _mock_backend_returning("ignored")
    with pytest.raises(ValueError):
        act(CBlock("x"), SimpleContext(), backend, return_sampling_results=True)


# --- requirements-without-strategy enforcement (issue #1448) ---


@pytest.mark.asyncio
async def test_aact_raises_for_requirements_without_strategy():
    """aact rejects any requirements passed without a strategy (issue #1448).

    With `strategy=None` there is no validate/repair loop, so a requirement passed
    via the kwarg can never be validated — it would be silently dropped. aact
    raises ValueError instead. This holds regardless of the action type.
    """
    from mellea.core import CBlock
    from mellea.stdlib.requirements import Requirement

    backend = _mock_backend_returning("ignored")
    with pytest.raises(ValueError):
        await aact(
            CBlock("x"),
            SimpleContext(),
            backend,
            requirements=[Requirement("must be short")],
        )


def test_act_raises_for_requirements_without_strategy():
    """act surfaces the same ValueError as aact for requirements without a strategy."""
    from mellea.core import CBlock
    from mellea.stdlib.functional import act
    from mellea.stdlib.requirements import Requirement

    backend = _mock_backend_returning("ignored")
    with pytest.raises(ValueError):
        act(
            CBlock("x"),
            SimpleContext(),
            backend,
            requirements=[Requirement("must be short")],
        )


@pytest.mark.asyncio
async def test_aact_raises_even_when_requirements_attached_to_component():
    """aact raises for requirements-without-strategy even if attached to the action.

    aact makes no exception for requirements a component also renders into its own
    prompt — the check is purely on the `requirements` kwarg. Callers like
    `instruct`/`ainstruct` avoid the raise by not forwarding their Instruction's
    requirements when no strategy is set; the Instruction still renders them
    (issue #1448).
    """
    from mellea.stdlib.requirements import Requirement

    backend = _mock_backend_returning("ok")
    req = Requirement("must be short")
    instruction = Instruction(description="say hi", requirements=[req])

    with pytest.raises(ValueError):
        await aact(
            instruction,
            SimpleContext(),
            backend,
            requirements=instruction.requirements,
            await_result=True,
        )


@pytest.mark.asyncio
async def test_ainstruct_does_not_raise_for_requirements_without_strategy():
    """ainstruct(strategy=None, requirements=...) does not forward reqs, so no raise.

    The Instruction renders its own requirements into the prompt; ainstruct only
    forwards them to aact when a strategy is present, so passing requirements with
    `strategy=None` generates without raising (issue #1448).
    """
    from mellea.stdlib.functional import ainstruct

    backend = _mock_backend_returning("ok")
    out, _ = await ainstruct(
        "say hi",
        SimpleContext(),
        backend,
        requirements=["must be short"],
        strategy=None,
        await_result=True,
    )
    assert str(out) == "ok"


@pytest.mark.asyncio
async def test_aact_no_raise_without_requirements():
    """aact with no requirements and no strategy generates without raising."""
    from mellea.core import CBlock

    backend = _mock_backend_returning("ok")
    out, _ = await aact(CBlock("x"), SimpleContext(), backend, await_result=True)
    assert str(out) == "ok"


# --- transform()/atransform() must persist the chosen tool message in context ---


def _make_tool_message(name: str = "some_tool") -> ToolMessage:
    """Return a real ToolMessage, as `_call_tools`/`_acall_tools` would produce."""
    tool_call = ModelToolCall(name=name, func=MagicMock(), args={"arg": 1})
    return ToolMessage(
        role="tool",
        content="tool result",
        tool_output="tool result",
        name=name,
        args={"arg": 1},
        tool=tool_call,
    )


def _assert_tool_message_persisted_after(
    new_ctx: Context, prior_messages: list[Message], tool_message: ToolMessage
) -> None:
    """Assert `new_ctx` is exactly `prior_messages + [tool_message]`, by identity."""
    result = new_ctx.as_list()
    assert len(result) == len(prior_messages) + 1
    for expected, actual in zip(prior_messages, result):
        assert actual is expected
    assert result[-1] is tool_message


@patch("mellea.stdlib.functional._call_tools")
@patch("mellea.stdlib.functional.act")
def test_transform_persists_chosen_tool_message_in_context(mock_act, mock_call_tools):
    """The tool message transform() picks must survive in the returned Context.

    `Context.add` is non-mutating (it returns a new context rather than
    mutating in place), so `new_ctx.add(chosen_tool)` as a bare statement
    silently drops the tool message. Seed the context with a prior message
    so the assertion can also catch an `add` that clobbers existing history
    instead of appending to it, and confirm the tool message lands last.
    """
    from mellea.stdlib.functional import transform

    prior_message = Message("user", "prior")
    ctx = ChatContext().add(prior_message)
    # mock_act returns this same ctx unchanged, so transform() extends it directly;
    # that's what makes asserting `result[0] is prior_message` meaningful below.
    mock_act.return_value = (MagicMock(), ctx)
    tool_message = _make_tool_message()
    mock_call_tools.return_value = [tool_message]

    _, new_ctx = transform(MObject(), "transform it", ctx, MagicMock())

    _assert_tool_message_persisted_after(new_ctx, [prior_message], tool_message)


@pytest.mark.asyncio
@patch("mellea.stdlib.functional._acall_tools", new_callable=AsyncMock)
@patch("mellea.stdlib.functional.aact", new_callable=AsyncMock)
async def test_atransform_persists_chosen_tool_message_in_context(
    mock_aact, mock_acall_tools
):
    """Async counterpart of test_transform_persists_chosen_tool_message_in_context."""
    from mellea.stdlib.functional import atransform

    prior_message = Message("user", "prior")
    ctx = ChatContext().add(prior_message)
    # mock_aact returns this same ctx unchanged, so atransform() extends it directly;
    # that's what makes asserting `result[0] is prior_message` meaningful below.
    mock_aact.return_value = (MagicMock(), ctx)
    tool_message = _make_tool_message()
    mock_acall_tools.return_value = [tool_message]

    _, new_ctx = await atransform(MObject(), "transform it", ctx, MagicMock())

    _assert_tool_message_persisted_after(new_ctx, [prior_message], tool_message)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
