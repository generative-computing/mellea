import base64
import copy
import io
from typing import Any

import pytest
from PIL import Image as PILImage

from mellea.core import (
    CBlock,
    Component,
    ImageBlock,
    ImageUrlBlock,
    ModelOutputThunk,
    RawProviderResponse,
)
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
    result = ModelOutputThunk(value="result value")
    result.raw = RawProviderResponse(
        provider="ollama",
        response=_ChatResponse(Message(role="assistant", content="assistant reply")),
    )

    result.parsed_repr = source.parse(result)
    assert isinstance(result.parsed_repr, Message), (
        "result's parsed repr should be a message when raw provider is set"
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


# --- ImageUrlBlock ---


def test_image_url_block_valid_https():
    block = ImageUrlBlock("https://example.com/image.png")
    assert block.value == "https://example.com/image.png"


def test_image_url_block_valid_http():
    block = ImageUrlBlock("http://example.com/image.png")
    assert block.value == "http://example.com/image.png"


def test_image_url_block_invalid_scheme_raises():
    with pytest.raises(ValueError, match="http"):
        ImageUrlBlock("ftp://example.com/image.png")


def test_image_url_block_base64_raises():
    b64 = _make_png_b64()
    with pytest.raises(ValueError, match="http"):
        ImageUrlBlock(b64)


def test_message_accepts_image_url_block():
    from mellea.stdlib.components import Message

    block = ImageUrlBlock("https://example.com/cat.png")
    msg = Message("user", "look at this", images=[block])
    assert msg.images == [block]


def test_message_mixed_image_types():
    from mellea.stdlib.components import Message

    url_block = ImageUrlBlock("https://example.com/cat.png")
    b64_block = ImageBlock(_make_png_b64())
    msg = Message("user", "two images", images=[url_block, b64_block])
    assert len(msg.images) == 2  # type: ignore[arg-type]


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


def _make_mot_with_generation() -> ModelOutputThunk:
    mot = ModelOutputThunk(value="x")
    mot.generation.usage = {"prompt_tokens": 10}
    mot.generation.model = "test-model"
    mot.generation.provider = "test-provider"
    mot.generation.streaming = True
    mot.generation.ttfb_ms = 42.0
    return mot


def test_mot_copy_from_copies_generation():
    a = ModelOutputThunk(value=None)
    b = _make_mot_with_generation()
    a._copy_from(b)
    assert a.generation.usage == {"prompt_tokens": 10}
    assert a.generation.model == "test-model"
    assert a.generation.provider == "test-provider"
    assert a.generation.streaming is True
    assert a.generation.ttfb_ms == 42.0


def test_mot_shallow_copy_generation_mutation_does_not_bleed():
    original = _make_mot_with_generation()
    copied = copy.copy(original)
    copied.generation.model = "mutated"
    assert original.generation.model == "test-model"


def test_mot_deep_copy_clones_generation():
    original = _make_mot_with_generation()
    deepcopied = copy.deepcopy(original)
    assert deepcopied.generation is not original.generation
    assert deepcopied.generation.usage == {"prompt_tokens": 10}
    assert deepcopied.generation.model == "test-model"
    assert deepcopied.generation.provider == "test-provider"
    assert deepcopied.generation.streaming is True
    assert deepcopied.generation.ttfb_ms == 42.0


# --- RawProviderResponse default + copy semantics ---


def _make_mot_with_raw() -> ModelOutputThunk:
    mot = ModelOutputThunk(value="x")
    mot.raw.provider = "openai"
    mot.raw.response = {"choices": [{"message": {"role": "assistant", "content": "v"}}]}
    mot.raw.streamed_chunks = [{"delta": {"content": "v"}}]
    return mot


def test_raw_provider_response_default():
    mot = ModelOutputThunk(value=None)
    assert mot.raw == RawProviderResponse()
    assert mot.raw.provider is None
    assert mot.raw.response is None
    assert mot.raw.streamed_chunks is None


def test_raw_propagates_on_copy():
    original = _make_mot_with_raw()
    copied = copy.copy(original)
    assert copied.raw.provider == "openai"
    # Shallow copy: the response dict is shared.
    assert copied.raw.response is original.raw.response
    assert copied.raw.streamed_chunks is original.raw.streamed_chunks


def test_raw_shallow_copy_provider_mutation_does_not_bleed():
    original = _make_mot_with_raw()
    copied = copy.copy(original)
    copied.raw.provider = "litellm"
    assert original.raw.provider == "openai"


def test_raw_propagates_on_deepcopy():
    original = _make_mot_with_raw()
    deepcopied = copy.deepcopy(original)
    assert deepcopied.raw is not original.raw
    assert deepcopied.raw.provider == "openai"
    assert deepcopied.raw.response == original.raw.response
    assert deepcopied.raw.response is not original.raw.response
    assert deepcopied.raw.streamed_chunks == original.raw.streamed_chunks
    assert deepcopied.raw.streamed_chunks is not original.raw.streamed_chunks


def test_raw_propagates_on_copy_from():
    a = ModelOutputThunk(value=None)
    b = _make_mot_with_raw()
    a._copy_from(b)
    # _copy_from is reference assignment for raw, matching .generation semantics.
    assert a.raw is b.raw


# --- Public error / generate_log surface ---


def test_mot_generate_log_property_aliases_private_attr() -> None:
    """`mot.generate_log` returns the same object as `_generate_log`."""
    from mellea.core.base import GenerateLog

    mot = ModelOutputThunk(value="x")
    assert mot.generate_log is None
    glog = GenerateLog()
    mot._generate_log = glog
    assert mot.generate_log is glog


def test_mot_error_and_cancelled_are_independent_channels() -> None:
    """Setting `_error` must not flip `cancelled`, and vice versa."""
    soft_failed = ModelOutputThunk(value="")
    soft_failed._error = RuntimeError("backend soft-failure")
    assert soft_failed.error is not None
    assert soft_failed.cancelled is False

    cancelled = ModelOutputThunk(value="partial")
    cancelled._cancelled = True
    assert cancelled.cancelled is True
    assert cancelled.error is None


def test_mot_error_carried_by_copy_methods() -> None:
    """`_error` survives `copy`, `deepcopy`, and `_copy_from`."""
    mot = ModelOutputThunk(value="")
    err = RuntimeError("recorded")
    mot._error = err

    shallow = copy.copy(mot)
    assert shallow.error is err

    deep = copy.deepcopy(mot)
    assert isinstance(deep.error, RuntimeError)
    assert str(deep.error) == "recorded"

    target = ModelOutputThunk(value=None)
    target._copy_from(mot)
    assert target.error is err


def test_mot_thinking_public_field_round_trip():
    mot = ModelOutputThunk(value="x")
    mot.thinking = "reasoning trace"
    assert mot.thinking == "reasoning trace"


if __name__ == "__main__":
    pytest.main([__file__])


# ---------------------------------------------------------------------------
# Fix 2 — cancel_generation invokes _cancel_hook before task cancellation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cancel_generation_invokes_cancel_hook_before_task_cancel() -> None:
    """Fix 2: _cancel_hook fires and cancel_generation() returns promptly.

    Simulates a backend thread that blocks for 5 s unless the hook sets the
    event. Without the hook, cancel_generation() would only observe the asyncio
    task as CancelledError but the thread would keep running — on a slow box
    that can mean the task wrapper hangs past the 1 s timeout here.  With the
    hook, the event is set first, the thread unblocks, and the whole path
    completes within the timeout.
    """
    import asyncio
    import threading

    hook_called = threading.Event()
    thread_released = threading.Event()

    def hook() -> None:
        hook_called.set()
        thread_released.set()

    mot = ModelOutputThunk(value=None)
    mot._cancel_hook = hook  # type: ignore[attr-defined]

    # Task that blocks in a thread until thread_released is set.
    async def spin() -> None:
        await asyncio.to_thread(thread_released.wait, 5.0)

    mot._generate = asyncio.create_task(spin())  # type: ignore[attr-defined]
    await asyncio.sleep(0)  # let the task reach to_thread

    # Must return within 1 s; without the hook it would hang ~5 s.
    await asyncio.wait_for(mot.cancel_generation(), timeout=1.0)  # type: ignore[attr-defined]

    assert hook_called.is_set(), "_cancel_hook was never called"
    assert mot._cancelled is True  # type: ignore[attr-defined]


def test_cancel_hook_not_forwarded_by_copy_methods() -> None:
    """Fix 2: copied MOTs must not inherit _cancel_hook (distinct computation)."""
    import copy as copy_mod

    def _hook() -> None:
        pass

    mot = ModelOutputThunk(value="x")
    mot._cancel_hook = _hook  # type: ignore[attr-defined]

    shallow = copy_mod.copy(mot)
    assert shallow._cancel_hook is None, "__copy__ must reset _cancel_hook to None"  # type: ignore[attr-defined]

    deep = copy_mod.deepcopy(mot)
    assert deep._cancel_hook is None, "__deepcopy__ must reset _cancel_hook to None"  # type: ignore[attr-defined]

    target = ModelOutputThunk(value="original")
    target._copy_from(mot)  # type: ignore[attr-defined]
    assert target._cancel_hook is None, "_copy_from must reset _cancel_hook to None"  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_cancel_generation_hook_exception_is_suppressed() -> None:
    """Fix 2: a faulty _cancel_hook must not mask cancel_generation itself."""
    import asyncio

    def _bad_hook() -> None:
        raise RuntimeError("hook exploded")

    mot = ModelOutputThunk(value=None)
    mot._cancel_hook = _bad_hook  # type: ignore[attr-defined]

    # No _generate task — cancel_generation still runs the hook path.
    # The hook raises, but cancel_generation must complete without propagating.
    await mot.cancel_generation()  # type: ignore[attr-defined]
    assert mot._cancelled is True  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_cancel_generation_propagates_outer_cancellation() -> None:
    """Outer cancellation of the cancel_generation() task must re-raise CancelledError.

    When cancel_generation() is awaiting self._generate and the *cancel_generation*
    task is itself cancelled from outside, cur.cancelling() > 0 and the
    CancelledError must propagate — not be swallowed by the bare ``pass`` path.
    """
    import asyncio

    inner_cancelled = asyncio.Event()

    async def _absorbs_first_cancel() -> None:
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            # Signal that cancel_generation() has called .cancel() and is
            # now blocked at ``await self._generate``.
            inner_cancelled.set()
            # Absorb this cancel so cancel_generation() stays at the await.
            await asyncio.sleep(60)

    mot = ModelOutputThunk(value=None)
    mot._generate = asyncio.create_task(_absorbs_first_cancel())  # type: ignore[attr-defined]
    await asyncio.sleep(0)

    cg_task = asyncio.create_task(mot.cancel_generation())  # type: ignore[attr-defined]
    # Wait until _generate has absorbed cancel_generation()'s .cancel() call —
    # at that point cg_task is blocked at ``await self._generate``.
    await asyncio.wait_for(inner_cancelled.wait(), timeout=2.0)

    # Cancel cancel_generation() from outside (simulates asyncio.wait_for timeout
    # or an outer TaskGroup cancelling this coroutine).
    cg_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(cg_task, timeout=2.0)

    # Cleanup: stop the still-running _generate task.
    mot._generate.cancel()  # type: ignore[attr-defined]
    try:
        await asyncio.wait_for(mot._generate, timeout=1.0)  # type: ignore[attr-defined]
    except (TimeoutError, asyncio.CancelledError):
        pass
