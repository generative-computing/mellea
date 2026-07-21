# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import base64
import copy
import io
from typing import Any

import pytest
from PIL import Image as PILImage

from mellea.core import (
    AudioBlock,
    AudioUrlBlock,
    CBlock,
    Component,
    GenerateType,
    ImageBlock,
    ImageUrlBlock,
    ModelOutputThunk,
    RawProviderResponse,
    blockify,
    get_audio_from_component,
    make_image_block,
)
from mellea.core.backend import generate_walk
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
    block = ImageUrlBlock("https://example.com/cat.png")
    msg = Message("user", "look at this", images=[block])
    assert msg.images == [block]


def test_message_mixed_image_types():
    url_block = ImageUrlBlock("https://example.com/cat.png")
    b64_block = ImageBlock(_make_png_b64())
    msg = Message("user", "two images", images=[url_block, b64_block])
    assert len(msg.images) == 2  # type: ignore[arg-type]


# --- AudioBlock ---


def test_audio_block_valid_base64_audio():
    wav_b64 = base64.b64encode(b"fake audio bytes").decode()
    block = AudioBlock(wav_b64, format="wav")
    assert block.value == wav_b64
    assert block.format == "wav"


def test_audio_block_valid_data_uri_prefix():
    wav_b64 = base64.b64encode(b"fake audio bytes").decode()
    data_uri = f"data:audio/wav;base64,{wav_b64}"
    assert AudioBlock.is_valid_base64_audio(data_uri) is True


def test_audio_block_invalid_base64_raises():
    with pytest.raises(AssertionError, match="Invalid base64"):
        AudioBlock(value="not-audio", format="wav")


def test_audio_block_empty_format_raises():
    wav_b64 = base64.b64encode(b"fake audio bytes").decode()
    with pytest.raises(ValueError, match="non-empty"):
        AudioBlock(value=wav_b64, format="")


def test_audio_block_whitespace_format_raises():
    wav_b64 = base64.b64encode(b"fake audio bytes").decode()
    with pytest.raises(ValueError, match="non-empty"):
        AudioBlock(value=wav_b64, format="   ")


def test_audio_block_format_auto_detected_from_data_uri():
    wav_b64 = base64.b64encode(b"fake audio bytes").decode()
    data_uri = f"data:audio/wav;base64,{wav_b64}"
    block = AudioBlock(data_uri)
    assert block.format == "wav"
    assert block.value == data_uri


def test_audio_block_format_auto_detected_mp3():
    mp3_b64 = base64.b64encode(b"fake mp3 bytes").decode()
    data_uri = f"data:audio/mpeg;base64,{mp3_b64}"
    block = AudioBlock(data_uri)
    assert block.format == "mp3"


@pytest.mark.parametrize(
    "mime_subtype,expected_format",
    [
        ("x-wav", "wav"),
        ("wave", "wav"),
        ("mpeg", "mp3"),
        ("x-mpeg", "mp3"),
        ("x-mp3", "mp3"),
        ("x-flac", "flac"),
        ("ogg", "ogg"),
        ("flac", "flac"),
        ("wav", "wav"),
    ],
)
def test_audio_block_mime_subtype_normalisation(
    mime_subtype: str, expected_format: str
):
    b64 = base64.b64encode(b"fake audio bytes").decode()
    data_uri = f"data:audio/{mime_subtype};base64,{b64}"
    block = AudioBlock(data_uri)
    assert block.format == expected_format


def test_audio_block_missing_format_raises():
    raw_b64 = base64.b64encode(b"fake audio bytes").decode()
    with pytest.raises(ValueError, match="non-empty"):
        AudioBlock(raw_b64)


# --- AudioUrlBlock ---


def test_audio_url_block_valid_https():
    block = AudioUrlBlock("https://example.com/audio.mp3", format="mp3")
    assert block.value == "https://example.com/audio.mp3"
    assert block.format == "mp3"


def test_audio_url_block_invalid_scheme_raises():
    with pytest.raises(ValueError, match="http"):
        AudioUrlBlock("ftp://example.com/audio.mp3", format="mp3")


def test_audio_url_block_valid_http():
    block = AudioUrlBlock("http://example.com/audio.wav", format="wav")
    assert block.value == "http://example.com/audio.wav"
    assert block.format == "wav"


def test_audio_url_block_empty_format_raises():
    with pytest.raises(ValueError, match="non-empty"):
        AudioUrlBlock("https://example.com/audio.mp3", format="")


def test_audio_url_block_whitespace_format_raises():
    with pytest.raises(ValueError, match="non-empty"):
        AudioUrlBlock("https://example.com/audio.mp3", format="   ")


# --- get_audio_from_component ---


class _ComponentWithAudio(Component[str]):
    def __init__(self, audio):
        self.audio = audio

    def parts(self):
        return []

    def format_for_llm(self) -> str:
        return ""

    def _parse(self, computed: ModelOutputThunk) -> str:
        return ""


def test_get_audio_from_component_returns_audio():
    audio = [AudioBlock(base64.b64encode(b"audio").decode(), format="wav")]
    component = _ComponentWithAudio(audio)
    assert get_audio_from_component(component) == audio


def test_get_audio_from_component_returns_audio_url_block():
    audio = [AudioUrlBlock("https://example.com/audio.mp3", format="mp3")]
    component = _ComponentWithAudio(audio)
    assert get_audio_from_component(component) == audio


def test_get_audio_from_component_returns_none_for_empty_list():
    component = _ComponentWithAudio([])
    assert get_audio_from_component(component) is None


def test_get_audio_from_component_returns_none_for_none_audio():
    component = _ComponentWithAudio(None)
    assert get_audio_from_component(component) is None


def test_get_audio_from_component_returns_none_when_missing():
    class _ComponentWithoutAudio(Component[str]):
        def parts(self):
            return []

        def format_for_llm(self) -> str:
            return ""

        def _parse(self, computed: ModelOutputThunk) -> str:
            return ""

    assert get_audio_from_component(_ComponentWithoutAudio()) is None


# --- make_image_block factory ---


def _png_bytes() -> bytes:
    img = PILImage.new("RGB", (1, 1), color="blue")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_make_image_block_from_pil():
    img = PILImage.new("RGB", (1, 1), color="green")
    block = make_image_block(img)
    assert isinstance(block, ImageBlock)
    assert ImageBlock.is_valid_base64_png(str(block))


def test_make_image_block_from_url_returns_url_block():
    block = make_image_block("https://example.com/cat.png")
    assert isinstance(block, ImageUrlBlock)
    assert block.value == "https://example.com/cat.png"


def test_make_image_block_from_base64_returns_image_block():
    b64 = _make_png_b64()
    block = make_image_block(b64)
    assert isinstance(block, ImageBlock)
    assert block.value == b64


def test_make_image_block_data_uri_returns_image_block():
    data_uri = f"data:image/png;base64,{_make_png_b64()}"
    block = make_image_block(data_uri)
    assert isinstance(block, ImageBlock)


def test_make_image_block_preserves_meta():
    b64 = _make_png_b64()
    block = make_image_block(b64, meta={"alt": "a dot"})
    assert block._meta == {"alt": "a dot"}


class _FakeRaw:
    """Stand-in for `requests.Response.raw` supporting a capped `.read()`."""

    def __init__(self, body: bytes):
        self._body = body

    def read(self, amt: int | None = None, decode_content: bool = False) -> bytes:
        return self._body if amt is None else self._body[:amt]


class _FakeResponse:
    """Minimal stand-in for a `requests.Response` used as a context manager."""

    def __init__(self, body: bytes, content_length: str | None = None):
        self.raw = _FakeRaw(body)
        self.headers = (
            {} if content_length is None else {"Content-Length": content_length}
        )

    def raise_for_status(self):
        return None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


@pytest.fixture(autouse=True)
def _clear_image_cache():
    """Isolate tests from the process-wide URL -> base64 download cache."""
    import mellea.core.base as base_mod

    base_mod._image_base64_cache.clear()
    yield
    base_mod._image_base64_cache.clear()


def _patch_download(monkeypatch, fake_get):
    """Stub `requests.get` for a download test."""
    import mellea.core.base as base_mod

    monkeypatch.setattr(base_mod.requests, "get", fake_get)


def test_make_image_block_url_convert_to_base64(monkeypatch):
    _patch_download(monkeypatch, lambda url, **kw: _FakeResponse(_png_bytes()))

    block = make_image_block("https://example.com/cat.png", convert_to_base64=True)
    assert isinstance(block, ImageBlock)
    assert ImageBlock.is_valid_base64_png(str(block))


def test_make_image_block_download_failure_raises(monkeypatch):
    import mellea.core.base as base_mod

    def boom(url, **kw):
        raise base_mod.requests.RequestException("no network")

    _patch_download(monkeypatch, boom)

    with pytest.raises(ValueError, match="Failed to download"):
        make_image_block("https://example.com/cat.png", convert_to_base64=True)


def test_make_image_block_download_rejects_oversized_body(monkeypatch):
    from mellea.core import base as base_mod

    big = _png_bytes() * (base_mod._IMAGE_DOWNLOAD_MAX_BYTES + 1)
    _patch_download(monkeypatch, lambda url, **kw: _FakeResponse(big))

    with pytest.raises(ValueError, match="exceeds"):
        make_image_block("https://example.com/big.png", convert_to_base64=True)


def test_make_image_block_download_rejects_oversized_content_length(monkeypatch):
    from mellea.core import base as base_mod

    limit = base_mod._IMAGE_DOWNLOAD_MAX_BYTES
    _patch_download(
        monkeypatch,
        lambda url, **kw: _FakeResponse(_png_bytes(), content_length=str(limit + 1)),
    )

    with pytest.raises(ValueError, match="exceeds"):
        make_image_block("https://example.com/big.png", convert_to_base64=True)


def test_image_url_block_resolve_base64_caches_by_url(monkeypatch):
    """resolve_base64 downloads once per URL; a reconstructed block hits cache."""
    calls: dict[str, int] = {}

    def fake_get(url, **kw):
        calls[url] = calls.get(url, 0) + 1
        return _FakeResponse(_png_bytes())

    _patch_download(monkeypatch, fake_get)

    url = "https://example.com/cat.png"
    first = ImageUrlBlock(url).resolve_base64()
    # A freshly reconstructed block for the same URL must not re-download.
    second = ImageUrlBlock(url).resolve_base64()

    assert ImageBlock.is_valid_base64_png(first)
    assert first == second
    assert calls[url] == 1  # cache is keyed on the URL, not the instance

    # A different URL is a cache miss and triggers its own download.
    other = "https://example.com/dog.png"
    ImageUrlBlock(other).resolve_base64()
    assert calls[other] == 1


def test_image_cache_evicts_least_recently_used(monkeypatch):
    """The download cache is bounded and evicts the least-recently-used URL."""
    from mellea.core import base as base_mod

    _patch_download(monkeypatch, lambda url, **kw: _FakeResponse(_png_bytes()))
    monkeypatch.setattr(base_mod, "_IMAGE_CACHE_MAX_ENTRIES", 2)

    a, b, c = (f"https://example.com/{n}.png" for n in ("a", "b", "c"))
    base_mod._cached_download_image_as_base64(a)
    base_mod._cached_download_image_as_base64(b)
    base_mod._cached_download_image_as_base64(a)  # touch `a` so `b` is now LRU
    base_mod._cached_download_image_as_base64(c)  # evicts `b`

    assert set(base_mod._image_base64_cache) == {a, c}


def test_make_image_block_invalid_string_raises():
    with pytest.raises(ValueError, match="could not interpret"):
        make_image_block("not-a-url-or-base64!!!")


def test_make_image_block_invalid_type_raises():
    with pytest.raises(TypeError, match="expects a PIL image or a string"):
        make_image_block(12345)  # type: ignore[arg-type]


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
    mot._gen.cancel_hook = hook  # type: ignore[attr-defined]

    # Task that blocks in a thread until thread_released is set.
    async def spin() -> None:
        await asyncio.to_thread(thread_released.wait, 5.0)

    mot._gen.generate = asyncio.create_task(spin())  # type: ignore[attr-defined]
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
    mot._gen.cancel_hook = _hook  # type: ignore[attr-defined]

    shallow = copy_mod.copy(mot)
    assert shallow._gen.cancel_hook is None, "__copy__ must reset _cancel_hook to None"  # type: ignore[attr-defined]

    deep = copy_mod.deepcopy(mot)
    assert deep._gen.cancel_hook is None, "__deepcopy__ must reset _cancel_hook to None"  # type: ignore[attr-defined]

    target = ModelOutputThunk(value="original")
    target._copy_from(mot)  # type: ignore[attr-defined]
    assert target._gen.cancel_hook is None, "_copy_from must reset _cancel_hook to None"  # type: ignore[attr-defined]


def test_copy_of_uncomputed_mot_raises() -> None:
    """Copying an uncomputed MOT raises; copies are only valid post-generation."""
    uncomputed = ModelOutputThunk(value=None)
    assert not uncomputed.is_computed()

    with pytest.raises(RuntimeError):
        copy.copy(uncomputed)

    with pytest.raises(RuntimeError):
        copy.deepcopy(uncomputed)


def test_deepcopy_resets_gen_and_preserves_call() -> None:
    """Deepcopying a computed MOT yields a fresh _gen but preserves _call."""
    mot = ModelOutputThunk(value="done")
    mot._call.action = Message("user", "hi")
    mot._call.context = []
    mot._call.model_options = {"temperature": 0.5}
    mot._call.generation_id = "gen-123"
    # Dirty the in-flight machinery so a shared _gen would be observable.
    mot._gen.generate_type = GenerateType.ASYNC
    mot._gen.chunk_size = 99

    deep = copy.deepcopy(mot)

    # _gen is a distinct, fresh instance — not shared with the original.
    assert deep._gen is not mot._gen
    assert deep._gen.queue is not mot._gen.queue
    assert deep._gen.generate is None
    assert deep._gen.generate_type is GenerateType.NONE
    assert deep._gen.chunk_size == 3

    # _call is preserved.
    assert deep._call.model_options == {"temperature": 0.5}
    assert deep._call.generation_id == "gen-123"
    assert deep._call.context == []
    assert isinstance(deep._call.action, Message)


def test_copy_resets_gen_and_preserves_call() -> None:
    """Shallow-copying a computed MOT yields a fresh _gen but preserves _call."""
    mot = ModelOutputThunk(value="done")
    mot._call.action = Message("user", "hi")
    mot._call.context = []
    mot._call.model_options = {"temperature": 0.5}
    mot._call.generation_id = "gen-123"
    # Dirty the in-flight machinery so a shared _gen would be observable.
    mot._gen.generate_type = GenerateType.ASYNC
    mot._gen.chunk_size = 99

    shallow = copy.copy(mot)

    # _gen is a distinct, fresh instance — not shared with the original.
    assert shallow._gen is not mot._gen
    assert shallow._gen.queue is not mot._gen.queue
    assert shallow._gen.generate is None
    assert shallow._gen.generate_type is GenerateType.NONE
    assert shallow._gen.chunk_size == 3

    # _call is preserved.
    assert shallow._call.model_options == {"temperature": 0.5}
    assert shallow._call.generation_id == "gen-123"
    assert shallow._call.context == []
    assert isinstance(shallow._call.action, Message)


@pytest.mark.asyncio
async def test_cancel_generation_hook_exception_is_suppressed() -> None:
    """Fix 2: a faulty _cancel_hook must not mask cancel_generation itself."""
    import asyncio

    def _bad_hook() -> None:
        raise RuntimeError("hook exploded")

    mot = ModelOutputThunk(value=None)
    mot._gen.cancel_hook = _bad_hook  # type: ignore[attr-defined]

    # No _generate task — cancel_generation still runs the hook path.
    # The hook raises, but cancel_generation must complete without propagating.
    await mot.cancel_generation()  # type: ignore[attr-defined]
    assert mot._cancelled is True  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_cancel_generation_propagates_outer_cancellation() -> None:
    """Outer cancellation of the cancel_generation() task must re-raise CancelledError.

    When cancel_generation() is awaiting self._gen.generate and the *cancel_generation*
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
            # now blocked at ``await self._gen.generate``.
            inner_cancelled.set()
            # Absorb this cancel so cancel_generation() stays at the await.
            await asyncio.sleep(60)

    mot = ModelOutputThunk(value=None)
    mot._gen.generate = asyncio.create_task(_absorbs_first_cancel())  # type: ignore[attr-defined]
    await asyncio.sleep(0)

    cg_task = asyncio.create_task(mot.cancel_generation())  # type: ignore[attr-defined]
    # Wait until _generate has absorbed cancel_generation()'s .cancel() call —
    # at that point cg_task is blocked at ``await self._gen.generate``.
    await asyncio.wait_for(inner_cancelled.wait(), timeout=2.0)

    # Cancel cancel_generation() from outside (simulates asyncio.wait_for timeout
    # or an outer TaskGroup cancelling this coroutine).
    cg_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(cg_task, timeout=2.0)

    # Cleanup: stop the still-running _generate task.
    mot._gen.generate.cancel()  # type: ignore[attr-defined]
    try:
        await asyncio.wait_for(mot._gen.generate, timeout=1.0)  # type: ignore[attr-defined]
    except (TimeoutError, asyncio.CancelledError):
        pass


def test_mot_is_not_shadowed_by_cblock_in_pattern_match():
    """ModelOutputThunk must not be a CBlock subtype.

    Before #269, ModelOutputThunk inherited CBlock, so a match block with
    case CBlock() appearing before case ModelOutputThunk() would silently
    consume all MOTs — the MOT branch was unreachable.
    """
    mot = ModelOutputThunk("hello")
    cb = CBlock("hello")

    def classify(obj):
        match obj:
            case CBlock():
                return "cblock"
            case ModelOutputThunk():
                return "mot"
            case _:
                return "other"

    assert classify(mot) == "mot"
    assert classify(cb) == "cblock"
    assert not isinstance(mot, CBlock)


def test_generate_walk_computed_mot_returns_empty():
    mot = ModelOutputThunk(value="already computed")
    assert generate_walk(mot) == []


def test_generate_walk_uncomputed_mot_returns_self():
    mot = ModelOutputThunk(value=None)
    assert generate_walk(mot) == [mot]


def test_generate_walk_cblock_returns_empty():
    assert generate_walk(CBlock("text")) == []


def test_blockify_string_returns_cblock():
    result = blockify("hello")
    assert isinstance(result, CBlock)
    assert result.value == "hello"


def test_blockify_cblock_returns_unchanged():
    cb = CBlock("hello")
    assert blockify(cb) is cb


def test_blockify_component_returns_unchanged():
    class DummyComponent(Component[str]):
        def parts(self):
            return []

        def _parse(self, computed):
            return computed.value or ""

    comp = DummyComponent()
    assert blockify(comp) is comp


def test_blockify_mot_returns_unchanged():
    mot = ModelOutputThunk("hello")
    assert blockify(mot) is mot


def test_blockify_uncomputed_mot_returns_unchanged():
    mot = ModelOutputThunk(value=None)
    assert blockify(mot) is mot


# --- Context.last_turn() tests ---


def test_last_turn_empty_context():
    from mellea.stdlib.context import ChatContext

    ctx = ChatContext()
    turn = ctx.last_turn()
    assert turn is None


def test_last_turn_with_model_output_thunk():
    from mellea.stdlib.context import ChatContext

    ctx = ChatContext().add(Message("user", "hello")).add(ModelOutputThunk("response"))
    turn = ctx.last_turn()
    assert turn is not None
    assert isinstance(turn.model_input, Message)
    assert turn.model_input.role == "user"
    assert isinstance(turn.output, ModelOutputThunk)
    assert turn.output.value == "response"


def test_last_turn_with_assistant_message():
    from mellea.stdlib.context import ChatContext

    ctx = (
        ChatContext()
        .add(Message("user", "hello"))
        .add(Message("assistant", "response"))
    )
    turn = ctx.last_turn()
    assert turn is not None
    assert isinstance(turn.model_input, Message)
    assert turn.model_input.role == "user"
    assert isinstance(turn.output, Message)
    assert turn.output.role == "assistant"
    assert turn.output.content == "response"


def test_last_turn_output_only_assistant_message():
    from mellea.stdlib.context import ChatContext

    ctx = ChatContext().add(Message("assistant", "response"))
    turn = ctx.last_turn()
    assert turn is not None
    assert turn.model_input is None
    assert isinstance(turn.output, Message)
    assert turn.output.role == "assistant"
    assert turn.output.content == "response"


def test_last_turn_input_only_user_message():
    from mellea.stdlib.context import ChatContext

    ctx = ChatContext().add(Message("user", "hello"))
    turn = ctx.last_turn()
    assert turn is not None
    assert isinstance(turn.model_input, Message)
    assert turn.model_input.role == "user"
    assert turn.output is None


def test_last_turn_user_message_treated_as_input():
    from mellea.stdlib.context import ChatContext

    ctx = ChatContext().add(Message("user", "hello"))
    turn = ctx.last_turn()
    assert turn is not None
    # User message should be in model_input, not output
    assert isinstance(turn.model_input, Message)
    assert turn.model_input.role == "user"
    assert turn.output is None


def test_last_turn_system_message_treated_as_input():
    from mellea.stdlib.context import ChatContext

    ctx = ChatContext().add(Message("system", "system prompt"))
    turn = ctx.last_turn()
    assert turn is not None
    # System message should be in model_input, not output
    assert isinstance(turn.model_input, Message)
    assert turn.model_input.role == "system"
    assert turn.output is None
