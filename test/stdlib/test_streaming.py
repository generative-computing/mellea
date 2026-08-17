# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for stream() and Streamer.

Uses StreamingMockBackend — a deterministic test double that feeds tokens from a
fixed response string into a MOT queue without network or LLM calls.

Terminal state (`failed_early`, `full_text`, `final_validations`,
`streaming_failures`, `mot`) is read from the `Streamer` after iteration. Typed
`StreamEvent`s are observed via the `STREAMING_EVENT` hook.
"""

import asyncio
import time
from contextlib import contextmanager
from typing import Any

import pytest

from mellea.core.backend import Backend
from mellea.core.base import CBlock, Context, GenerateType, ModelOutputThunk
from mellea.core.requirement import (
    PartialValidationResult,
    Requirement,
    ValidationResult,
)
from mellea.plugins import hook, register, unregister
from mellea.plugins.manager import (
    disable_background_collection,
    drain_background_tasks,
    enable_background_collection,
)
from mellea.plugins.registry import _HAS_PLUGIN_FRAMEWORK
from mellea.stdlib.context import SimpleContext
from mellea.stdlib.streaming import (
    ChunkEvent,
    CompletedEvent,
    ErrorEvent,
    FullValidationEvent,
    QuickCheckEvent,
    RetryEvent,
    Streamer,
    StreamEvent,
    StreamingDoneEvent,
    stream,
)

# Tests that observe the stream via the STREAMING_EVENT / STREAMING_END hooks need
# the optional `hooks` extra (cpex); the core streaming tests below do not. Skip
# the hook-observing tests when it is not installed rather than fail on register().
_cpex_skip = pytest.mark.skipif(
    not _HAS_PLUGIN_FRAMEWORK, reason="cpex not installed — install mellea[hooks]"
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
async def _drain_fire_and_forget_tasks():
    """Drain FIRE_AND_FORGET plugin tasks so they run before the loop closes.

    STREAMING_END schedules a background task via the suite-wide `fandf`
    acceptance plugin, then returns with no further await — the loop tears down
    first and the coroutine is GC'd unawaited ("coroutine never awaited").
    Collection is global, so disable it after to leave other test files untouched.
    """
    enable_background_collection()
    yield
    await drain_background_tasks()
    disable_background_collection()


# ---------------------------------------------------------------------------
# StreamingMockBackend
# ---------------------------------------------------------------------------


async def _mock_process(mot: ModelOutputThunk, chunk: Any) -> None:
    if mot._underlying_value is None:
        mot._underlying_value = ""
    if chunk is not None:
        mot._underlying_value += chunk


async def _mock_post_process(_mot: ModelOutputThunk) -> None:
    pass


def _make_mot() -> ModelOutputThunk:
    mot = ModelOutputThunk(value=None)
    mot._call.action = CBlock("mock_action")
    mot._gen.generate_type = GenerateType.ASYNC
    mot._gen.process = _mock_process
    mot._gen.post_process = _mock_post_process
    mot._gen.chunk_size = 0
    return mot


async def _feed_tokens(mot: ModelOutputThunk, response: str, token_size: int) -> None:
    i = 0
    while i < len(response):
        token = response[i : i + token_size]
        await mot._gen.queue.put(token)
        await asyncio.sleep(0)
        i += token_size
    await mot._gen.queue.put(None)


class StreamingMockBackend(Backend):
    """Test double that streams a fixed response one token at a time.

    `token_size` controls how many characters constitute one token.
    Validation calls (via `stream_validate` / `validate`) are delegated
    to the requirements themselves — this backend does not perform any real
    inference.
    """

    def __init__(self, response: str, token_size: int = 1) -> None:
        self._response = response
        self._token_size = token_size
        self._model_id: str = "streaming-mock-model"
        self._provider: str = "streaming-mock-provider"

    async def _generate_from_context(
        self,
        action: Any,
        ctx: Context,
        *,
        format: Any = None,
        model_options: dict | None = None,
        tool_calls: bool = False,
    ) -> tuple[ModelOutputThunk, Context]:
        _ = format, model_options, tool_calls
        mot = _make_mot()
        task = asyncio.create_task(_feed_tokens(mot, self._response, self._token_size))
        _ = task
        new_ctx = ctx.add(action).add(mot)
        return mot, new_ctx

    async def _generate_from_raw(
        self, actions: Any, ctx: Any, **kwargs: Any
    ) -> tuple[list[ModelOutputThunk], dict[str, Any] | None]:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Requirement test doubles
# ---------------------------------------------------------------------------


class AlwaysUnknownReq(Requirement):
    """stream_validate always returns 'unknown'; validate returns True."""

    def format_for_llm(self) -> str:
        return "always unknown"

    async def stream_validate(
        self, chunk: str, *, backend: Any, ctx: Any
    ) -> PartialValidationResult:
        return PartialValidationResult("unknown")

    async def validate(
        self, backend: Any, ctx: Any, *, format: Any = None, model_options: Any = None
    ) -> ValidationResult:
        return ValidationResult(result=True)


class FailAfterWordsReq(Requirement):
    """Returns 'fail' once the cumulative word count reaches *threshold*.

    Each call to `stream_validate` receives a single chunk (delta) from the
    chunking strategy; the running total is maintained on the instance.
    """

    def __init__(self, threshold: int) -> None:
        super().__init__()
        self._threshold = threshold
        self._word_count = 0

    def format_for_llm(self) -> str:
        return f"fail after {self._threshold} words"

    async def stream_validate(
        self, chunk: str, *, backend: Any, ctx: Any
    ) -> PartialValidationResult:
        self._word_count += len(chunk.split())
        if self._word_count >= self._threshold:
            return PartialValidationResult("fail", reason="too many words")
        return PartialValidationResult("unknown")

    async def validate(
        self, backend: Any, ctx: Any, *, format: Any = None, model_options: Any = None
    ) -> ValidationResult:
        return ValidationResult(result=True)


class BackendRecordingReq(Requirement):
    """Records which backend was passed to stream_validate and validate."""

    def __init__(self) -> None:
        super().__init__()
        self.seen_backends: list[Any] = []

    def __copy__(self) -> "BackendRecordingReq":
        clone = BackendRecordingReq()
        clone.seen_backends = []  # fresh list — do not share with original
        return clone

    def format_for_llm(self) -> str:
        return "backend recorder"

    async def stream_validate(
        self, chunk: str, *, backend: Any, ctx: Any
    ) -> PartialValidationResult:
        _ = chunk
        self.seen_backends.append(backend)
        return PartialValidationResult("unknown")

    async def validate(
        self, backend: Any, ctx: Any, *, format: Any = None, model_options: Any = None
    ) -> ValidationResult:
        self.seen_backends.append(backend)
        return ValidationResult(result=True)


class ChunkRecordingReq(Requirement):
    """Records every chunk passed to stream_validate."""

    def __init__(self) -> None:
        super().__init__()
        self.seen_chunks: list[str] = []

    def __copy__(self) -> "ChunkRecordingReq":
        clone = ChunkRecordingReq()
        clone.seen_chunks = []  # fresh list — do not share with original
        return clone

    def format_for_llm(self) -> str:
        return "chunk recorder"

    async def stream_validate(
        self, chunk: str, *, backend: Any, ctx: Any
    ) -> PartialValidationResult:
        self.seen_chunks.append(chunk)
        return PartialValidationResult("unknown")

    async def validate(
        self, backend: Any, ctx: Any, *, format: Any = None, model_options: Any = None
    ) -> ValidationResult:
        return ValidationResult(result=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ctx() -> SimpleContext:
    return SimpleContext()


def _action() -> CBlock:
    return CBlock("prompt")


@contextmanager
def _record_events():
    """Install a temporary `streaming_event` hook that collects emitted events.

    Registers a recorder plugin for the duration of the block and unregisters it
    on exit. Yields the list each `StreamEvent` is appended to.
    """
    events: list[StreamEvent] = []

    @hook("streaming_event")
    async def _recorder(payload: Any, ctx: Any) -> Any:
        events.append(payload.event)
        return None

    register(_recorder)
    try:
        yield events
    finally:
        unregister(_recorder)


# ---------------------------------------------------------------------------
# Consumption + terminal state
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_normal_completion_calls_validate_at_stream_end() -> None:
    """All 'unknown' requirements → validate() runs at stream end; no early fail."""
    response = "Hello world. How are you. "
    backend = StreamingMockBackend(response, token_size=3)
    req = AlwaysUnknownReq()

    async with await stream(
        _action(), backend, _ctx(), requirements=[req], chunking="sentence"
    ) as streamer:
        async for _chunk in streamer:
            pass

    assert streamer.failed_early is False
    assert streamer.full_text == response
    assert len(streamer.final_validations) == 1
    assert streamer.final_validations[0].as_bool() is True
    assert streamer.streaming_failures == []
    assert streamer.mot is not None
    assert streamer.completed_normally is True


@pytest.mark.asyncio
async def test_mot_set_on_natural_completion() -> None:
    """On natural completion, `mot` holds the computed thunk with the full value."""
    response = "One. Two. "
    backend = StreamingMockBackend(response, token_size=2)

    async with await stream(
        _action(), backend, _ctx(), chunking="sentence"
    ) as streamer:
        async for _chunk in streamer:
            pass

    assert streamer.mot is not None
    assert streamer.mot.is_computed()
    assert streamer.mot.value == streamer.full_text == response


@pytest.mark.asyncio
async def test_yields_individual_chunks() -> None:
    """Each iteration yields one validated chunk, in order."""
    response = "Alpha one. Beta two. Gamma three. "
    backend = StreamingMockBackend(response, token_size=3)

    chunks: list[str] = []
    async with await stream(
        _action(), backend, _ctx(), chunking="sentence"
    ) as streamer:
        async for chunk in streamer:
            chunks.append(chunk)

    assert chunks == ["Alpha one.", "Beta two.", "Gamma three."]


@pytest.mark.asyncio
async def test_no_requirements_streams_without_validation() -> None:
    """With no requirements, chunks stream through and no final validation runs."""
    response = "Sentence one. Sentence two. "
    backend = StreamingMockBackend(response, token_size=4)

    chunks: list[str] = []
    async with await stream(
        _action(), backend, _ctx(), chunking="sentence"
    ) as streamer:
        async for chunk in streamer:
            chunks.append(chunk)

    assert streamer.failed_early is False
    assert streamer.full_text == response
    assert streamer.final_validations == []
    assert streamer.streaming_failures == []
    assert chunks == ["Sentence one.", "Sentence two."]


@pytest.mark.asyncio
async def test_raw_delta_mode_when_chunking_none() -> None:
    """chunking=None yields raw deltas verbatim, without re-chunking on boundaries."""
    # Spaces make this a real test: any chunker that split on whitespace would drop
    # the inter-word spaces, so the concatenation would not equal the response.
    response = "one two three"
    backend = StreamingMockBackend(response, token_size=2)

    chunks: list[str] = []
    async with await stream(_action(), backend, _ctx(), chunking=None) as streamer:
        async for chunk in streamer:
            chunks.append(chunk)

    # Deltas may batch, so the exact split is not pinned — but text (spaces
    # included) must survive verbatim, proving no whitespace boundary was applied.
    assert "".join(chunks) == response
    assert streamer.full_text == response
    assert any(" " in c for c in chunks)


@pytest.mark.asyncio
async def test_stream_validate_receives_individual_chunks() -> None:
    """Each stream_validate call receives exactly one chunk, in order."""
    response = "First one. Second two. Third three. "
    backend = StreamingMockBackend(response, token_size=3)
    req = ChunkRecordingReq()

    captured: list[ChunkRecordingReq] = []
    original_copy = ChunkRecordingReq.__copy__

    def _capturing_copy(self: ChunkRecordingReq) -> ChunkRecordingReq:
        clone = original_copy(self)
        captured.append(clone)
        return clone

    ChunkRecordingReq.__copy__ = _capturing_copy  # type: ignore[method-assign]
    try:
        async with await stream(
            _action(), backend, _ctx(), requirements=[req], chunking="sentence"
        ) as streamer:
            async for _chunk in streamer:
                pass
    finally:
        ChunkRecordingReq.__copy__ = original_copy  # type: ignore[method-assign]

    assert captured[0].seen_chunks == ["First one.", "Second two.", "Third three."]


@pytest.mark.asyncio
async def test_trailing_fragment_is_flushed_to_consumer() -> None:
    """A final sentence with no trailing whitespace still reaches the consumer."""
    # No trailing whitespace after the final sentence — the chunker withholds it
    # until flush at stream end.
    response = "First sentence. Second sentence."
    backend = StreamingMockBackend(response, token_size=4)
    req = ChunkRecordingReq()

    captured: list[ChunkRecordingReq] = []
    original_copy = ChunkRecordingReq.__copy__

    def _capturing_copy(self: ChunkRecordingReq) -> ChunkRecordingReq:
        clone = original_copy(self)
        captured.append(clone)
        return clone

    ChunkRecordingReq.__copy__ = _capturing_copy  # type: ignore[method-assign]
    try:
        yielded: list[str] = []
        async with await stream(
            _action(), backend, _ctx(), requirements=[req], chunking="sentence"
        ) as streamer:
            async for chunk in streamer:
                yielded.append(chunk)
    finally:
        ChunkRecordingReq.__copy__ = original_copy  # type: ignore[method-assign]

    # Both sentences reach the consumer, including the terminating one.
    assert yielded == ["First sentence.", "Second sentence."]
    # stream_validate was called on both — the flush path is not a shortcut.
    assert captured[0].seen_chunks == ["First sentence.", "Second sentence."]
    assert streamer.failed_early is False


# ---------------------------------------------------------------------------
# Early exit on validation failure
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_early_exit_on_fail() -> None:
    """A 'fail' stops the stream early, records the failure, skips final validate()."""
    response = "one two three four five six seven eight "
    backend = StreamingMockBackend(response, token_size=2)
    req = FailAfterWordsReq(threshold=3)

    async with await stream(
        _action(), backend, _ctx(), requirements=[req], chunking="word"
    ) as streamer:
        async for _chunk in streamer:
            pass

    assert streamer.failed_early is True
    assert len(streamer.streaming_failures) == 1
    _req, pvr = streamer.streaming_failures[0]
    assert pvr.success == "fail"
    assert pvr.reason == "too many words"
    assert streamer.final_validations == []


@pytest.mark.asyncio
async def test_early_exit_on_trailing_fragment() -> None:
    """A fail on the flushed fragment records a failure and skips final validate()."""

    class FailOnSecondSentence(Requirement):
        def __init__(self) -> None:
            super().__init__()
            self._count = 0

        def format_for_llm(self) -> str:
            return "fail on second sentence"

        async def stream_validate(
            self, chunk: str, *, backend: Any, ctx: Any
        ) -> PartialValidationResult:
            _ = chunk, backend, ctx
            self._count += 1
            if self._count >= 2:
                return PartialValidationResult("fail", reason="second sentence hit")
            return PartialValidationResult("unknown")

        async def validate(
            self,
            backend: Any,
            ctx: Any,
            *,
            format: Any = None,
            model_options: Any = None,
        ) -> ValidationResult:
            return ValidationResult(result=True)

    response = "First sentence. Second sentence."
    backend = StreamingMockBackend(response, token_size=4)
    req = FailOnSecondSentence()

    yielded: list[str] = []
    async with await stream(
        _action(), backend, _ctx(), requirements=[req], chunking="sentence"
    ) as streamer:
        async for chunk in streamer:
            yielded.append(chunk)

    assert streamer.failed_early is True
    assert len(streamer.streaming_failures) == 1
    # First sentence was emitted; the second (flushed fragment) failed, unemitted.
    assert yielded == ["First sentence."]
    assert streamer.final_validations == []


@pytest.mark.asyncio
async def test_multiple_chunks_in_one_batch_with_mid_batch_fail() -> None:
    """When one delta yields several chunks, a mid-batch fail stops before later ones."""

    captured: list[Any] = []

    class FailOnThird(Requirement):
        def __init__(self) -> None:
            super().__init__()
            self._count = 0
            self.seen_chunks: list[str] = []

        def __copy__(self) -> "FailOnThird":
            clone = FailOnThird()
            captured.append(clone)
            return clone

        def format_for_llm(self) -> str:
            return "fail on third chunk"

        async def stream_validate(
            self, chunk: str, *, backend: Any, ctx: Any
        ) -> PartialValidationResult:
            _ = backend, ctx
            self._count += 1
            self.seen_chunks.append(chunk)
            if self._count == 3:
                return PartialValidationResult("fail", reason="third chunk")
            return PartialValidationResult("unknown")

        async def validate(
            self,
            backend: Any,
            ctx: Any,
            *,
            format: Any = None,
            model_options: Any = None,
        ) -> ValidationResult:
            return ValidationResult(result=True)

    # Whole response arrives as one delta: split() yields four sentences at once.
    response = "One. Two. Three. Four. "
    backend = StreamingMockBackend(response, token_size=len(response))
    req = FailOnThird()

    yielded: list[str] = []
    async with await stream(
        _action(), backend, _ctx(), requirements=[req], chunking="sentence"
    ) as streamer:
        async for chunk in streamer:
            yielded.append(chunk)

    assert streamer.failed_early is True
    assert len(streamer.streaming_failures) == 1
    # First two passed and were emitted; the third failed before emission.
    assert yielded == ["One.", "Two."]
    # The fourth chunk was neither validated nor emitted: validation stopped at
    # the failing third.
    assert captured[0].seen_chunks == ["One.", "Two.", "Three."]


@pytest.mark.asyncio
async def test_cancel_generation_invoked_on_fail() -> None:
    """An early fail cancels the backend generation (mot ends computed+cancelled)."""
    response = "one two three four five six seven eight nine ten "
    backend = StreamingMockBackend(response, token_size=1)
    req = FailAfterWordsReq(threshold=2)

    async with await stream(
        _action(), backend, _ctx(), requirements=[req], chunking="word"
    ) as streamer:
        async for _chunk in streamer:
            pass

    assert streamer.failed_early is True
    # On early exit the driving MOT was cancelled; `mot` (natural-completion only)
    # stays None.
    assert streamer.mot is None
    assert streamer.completed_normally is False
    # The underlying generation was actually cancelled, not merely abandoned.
    assert streamer._mot._cancelled is True
    assert streamer._mot.is_computed() is True


class _FailOnSecondReq(Requirement):
    """Passes the first chunk, fails the second — to drive early exit mid-stream."""

    def __init__(self) -> None:
        super().__init__()
        self._count = 0

    def format_for_llm(self) -> str:
        return "fail on second"

    async def stream_validate(
        self, chunk: str, *, backend: Any, ctx: Any
    ) -> PartialValidationResult:
        _ = chunk, backend, ctx
        self._count += 1
        if self._count == 2:
            return PartialValidationResult("fail", reason="second")
        return PartialValidationResult("unknown")

    async def validate(
        self, backend: Any, ctx: Any, *, format: Any = None, model_options: Any = None
    ) -> ValidationResult:
        return ValidationResult(result=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("token_size", [1, 6, len("One. Two. Three. Four. ")])
async def test_full_text_is_chunk_exact_on_early_exit(token_size: int) -> None:
    """full_text on early exit is the accumulated text through the last EMITTED chunk.

    Chunk-exact, not delta-granular: only "One." was validated and yielded before
    the fail on "Two.", so full_text is exactly "One." regardless of how the
    response was split into deltas.
    """
    response = "One. Two. Three. Four. "
    backend = StreamingMockBackend(response, token_size=token_size)

    async with await stream(
        _action(),
        backend,
        _ctx(),
        requirements=[_FailOnSecondReq()],
        chunking="sentence",
    ) as streamer:
        async for _chunk in streamer:
            pass

    assert streamer.failed_early is True
    assert streamer.full_text == "One."


@pytest.mark.asyncio
@pytest.mark.parametrize("token_size", [1, 6, len("One. Two. Three. Four. ")])
async def test_full_text_spans_multiple_emitted_chunks_on_early_exit(
    token_size: int,
) -> None:
    """full_text accumulates every emitted chunk, not just the first.

    Failing on the third chunk emits "One." and "Two." first, so full_text is the
    concatenation of both — verifying the emitted-text cursor advances across
    chunks, regardless of delta boundaries.
    """

    class FailOnThirdReq(Requirement):
        def __init__(self) -> None:
            super().__init__()
            self._count = 0

        def format_for_llm(self) -> str:
            return "fail on third"

        async def stream_validate(
            self, chunk: str, *, backend: Any, ctx: Any
        ) -> PartialValidationResult:
            _ = chunk, backend, ctx
            self._count += 1
            if self._count == 3:
                return PartialValidationResult("fail", reason="third")
            return PartialValidationResult("unknown")

        async def validate(
            self,
            backend: Any,
            ctx: Any,
            *,
            format: Any = None,
            model_options: Any = None,
        ) -> ValidationResult:
            return ValidationResult(result=True)

    response = "One. Two. Three. Four. "
    backend = StreamingMockBackend(response, token_size=token_size)

    async with await stream(
        _action(), backend, _ctx(), requirements=[FailOnThirdReq()], chunking="sentence"
    ) as streamer:
        async for _chunk in streamer:
            pass

    assert streamer.failed_early is True
    assert streamer.full_text == "One. Two."


@pytest.mark.asyncio
async def test_full_text_through_last_emitted_chunk_on_break() -> None:
    """A caller `break` leaves full_text at the last chunk actually delivered."""
    response = "One. Two. Three. "
    backend = StreamingMockBackend(response, token_size=3)

    async with await stream(
        _action(), backend, _ctx(), chunking="sentence"
    ) as streamer:
        async for _chunk in streamer:
            break  # take only the first chunk

    assert streamer.full_text == "One."


# ---------------------------------------------------------------------------
# Cleanup contract (async with / aclose)
# ---------------------------------------------------------------------------


@_cpex_skip
@pytest.mark.asyncio
async def test_break_cancels_generation_and_fires_end() -> None:
    """`async with` + early break cancels generation and fires STREAMING_END once."""
    response = "One. Two. Three. Four. Five. "
    backend = StreamingMockBackend(response, token_size=2)

    ends: list[Any] = []

    @hook("streaming_end")
    async def _end(payload: Any, ctx: Any) -> Any:
        ends.append(payload)
        return None

    register(_end)
    try:
        async with await stream(_action(), backend, _ctx(), chunking="sentence") as s:
            async for _chunk in s:
                break
    finally:
        unregister(_end)

    assert len(ends) == 1
    assert ends[0].success is False
    # Generation was cancelled: it did not reach natural completion, so `mot`
    # stays None but the underlying stream is computed/cancelled.
    assert s.mot is None
    assert s.completed_normally is False


@_cpex_skip
@pytest.mark.asyncio
async def test_acquired_never_iterated_aclose_cancels_and_ends() -> None:
    """A Streamer acquired but never iterated still cancels + fires END on aclose().

    This is the leak the cleanup contract closes: eager generation starts at
    stream(), so an abandoned handle must still be released.
    """
    response = "One. Two. Three. "
    backend = StreamingMockBackend(response, token_size=2)

    ends: list[Any] = []
    gen_errors: list[Any] = []

    @hook("streaming_end")
    async def _end(payload: Any, ctx: Any) -> Any:
        ends.append(payload)
        return None

    @hook("generation_error")
    async def _gerr(payload: Any, ctx: Any) -> Any:
        gen_errors.append(payload)
        return None

    register(_end)
    register(_gerr)
    try:
        streamer = await stream(_action(), backend, _ctx(), chunking="sentence")
        # Never iterated.
        await streamer.aclose()
    finally:
        unregister(_end)
        unregister(_gerr)

    assert len(ends) == 1
    assert ends[0].success is False
    # The eager, in-flight generation was cancelled.
    assert len(gen_errors) >= 1


@_cpex_skip
@pytest.mark.asyncio
async def test_aclose_after_natural_completion_is_noop() -> None:
    """aclose() after full drain does not re-fire STREAMING_END."""
    response = "One. Two. "
    backend = StreamingMockBackend(response, token_size=2)

    ends: list[Any] = []

    @hook("streaming_end")
    async def _end(payload: Any, ctx: Any) -> Any:
        ends.append(payload)
        return None

    register(_end)
    try:
        streamer = await stream(_action(), backend, _ctx(), chunking="sentence")
        async for _chunk in streamer:
            pass
        await streamer.aclose()
    finally:
        unregister(_end)

    assert len(ends) == 1
    assert ends[0].success is True


@_cpex_skip
@pytest.mark.asyncio
async def test_double_aclose_fires_end_once() -> None:
    """Calling aclose() twice fires STREAMING_END exactly once."""
    response = "One. Two. Three. "
    backend = StreamingMockBackend(response, token_size=2)

    ends: list[Any] = []

    @hook("streaming_end")
    async def _end(payload: Any, ctx: Any) -> Any:
        ends.append(payload)
        return None

    register(_end)
    try:
        async with await stream(_action(), backend, _ctx(), chunking="sentence") as s:
            async for _chunk in s:
                break
        await s.aclose()  # explicit second close after context-manager exit
    finally:
        unregister(_end)

    assert len(ends) == 1


async def _feed_tokens_slowly(
    mot: ModelOutputThunk, response: str, token_size: int, delay: float
) -> None:
    i = 0
    while i < len(response):
        await mot._gen.queue.put(response[i : i + token_size])
        await asyncio.sleep(delay)
        i += token_size
    await mot._gen.queue.put(None)


class SlowStreamingMockBackend(StreamingMockBackend):
    """Streams with a real per-token delay and exposes the feed task as `_gen.generate`.

    The delay lets an external timeout land mid-stream, and setting `_gen.generate`
    means `cancel_generation()` awaits the in-flight task — the path where an
    externally cancelled task re-raises `CancelledError`.
    """

    def __init__(
        self, response: str, token_size: int = 1, delay: float = 0.005
    ) -> None:
        super().__init__(response, token_size)
        self._delay = delay

    async def _generate_from_context(
        self,
        action: Any,
        ctx: Context,
        *,
        format=None,
        model_options=None,
        tool_calls=False,
    ) -> tuple[ModelOutputThunk, Context]:
        _ = format, model_options, tool_calls
        mot = _make_mot()
        mot._gen.generate = asyncio.create_task(
            _feed_tokens_slowly(mot, self._response, self._token_size, self._delay)
        )
        new_ctx = ctx.add(action).add(mot)
        return mot, new_ctx


@_cpex_skip
@pytest.mark.asyncio
async def test_external_cancellation_mid_stream_still_finalizes() -> None:
    """An external timeout mid-stream still fires terminal events and computes the MOT.

    Wrapping consumption in `asyncio.wait_for` cancels the consuming task while a
    chunk is in flight. The cancellation must still run cleanup: STREAMING_END and
    GENERATION_ERROR fire once each, and the MOT ends computed + cancelled rather
    than stranded, before the TimeoutError propagates.
    """
    response = "One. Two. Three. Four. Five. Six. Seven. Eight. "
    backend = SlowStreamingMockBackend(response, token_size=2, delay=0.005)

    ends: list[Any] = []
    gen_errors: list[Any] = []

    @hook("streaming_end")
    async def _end(payload: Any, ctx: Any) -> Any:
        ends.append(payload)
        return None

    @hook("generation_error")
    async def _gerr(payload: Any, ctx: Any) -> Any:
        gen_errors.append(payload)
        return None

    register(_end)
    register(_gerr)
    streamer: Streamer | None = None
    try:

        async def _run() -> None:
            nonlocal streamer
            async with await stream(
                _action(), backend, _ctx(), chunking="sentence"
            ) as s:
                streamer = s
                async for _chunk in s:
                    pass

        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(_run(), timeout=0.05)
    finally:
        unregister(_end)
        unregister(_gerr)

    assert len(ends) == 1
    assert ends[0].success is False
    assert len(gen_errors) == 1
    assert streamer is not None
    assert streamer._mot.is_computed() is True
    assert streamer._mot.cancelled is True


@pytest.mark.asyncio
async def test_early_exit_does_not_deadlock() -> None:
    """A high-throughput stream that fails early must not hang.

    The response is far longer than the MOT queue (maxsize 20), so an early
    fail must not leave the producer blocked on a full queue after the consumer
    stops. A hang trips the timeout.
    """
    response = "word " * 200
    backend = StreamingMockBackend(response, token_size=5)
    req = FailAfterWordsReq(threshold=3)

    streamer: Streamer | None = None

    async def _run() -> None:
        nonlocal streamer
        async with await stream(
            _action(), backend, _ctx(), requirements=[req], chunking="word"
        ) as s:
            streamer = s
            async for _chunk in s:
                pass

    await asyncio.wait_for(_run(), timeout=5.0)
    assert streamer is not None
    assert streamer.failed_early is True


# ---------------------------------------------------------------------------
# Requirement cloning / backend routing
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_clone_isolation_across_runs() -> None:
    """Requirement instances are cloned per run; the original is never mutated."""
    req = FailAfterWordsReq(threshold=2)

    backend1 = StreamingMockBackend("one two three ", token_size=2)
    async with await stream(
        _action(), backend1, _ctx(), requirements=[req], chunking="word"
    ) as s1:
        async for _chunk in s1:
            pass

    # The original's running counter was not mutated by the first run.
    assert req._word_count == 0

    backend2 = StreamingMockBackend("four five six ", token_size=2)
    async with await stream(
        _action(), backend2, _ctx(), requirements=[req], chunking="word"
    ) as s2:
        async for _chunk in s2:
            pass

    assert s1.failed_early is True
    assert s2.failed_early is True


@pytest.mark.asyncio
async def test_validation_backend_routing() -> None:
    """validation_backend, when given, receives stream_validate + validate calls."""
    gen_backend = StreamingMockBackend("Hello world. ", token_size=3)
    val_backend = StreamingMockBackend("", token_size=1)
    req = BackendRecordingReq()

    captured: list[BackendRecordingReq] = []
    original_copy = BackendRecordingReq.__copy__

    def _capturing_copy(self: BackendRecordingReq) -> BackendRecordingReq:
        clone = original_copy(self)
        captured.append(clone)
        return clone

    BackendRecordingReq.__copy__ = _capturing_copy  # type: ignore[method-assign]
    try:
        async with await stream(
            _action(),
            gen_backend,
            _ctx(),
            requirements=[req],
            chunking="sentence",
            validation_backend=val_backend,
        ) as streamer:
            async for _chunk in streamer:
                pass
    finally:
        BackendRecordingReq.__copy__ = original_copy  # type: ignore[method-assign]

    assert streamer.failed_early is False
    # The original requirement was never called — only its per-run clone.
    assert req.seen_backends == []
    # Every recorded backend was the validation backend, not the generation one.
    assert len(captured) == 1
    assert captured[0].seen_backends
    assert all(b is val_backend for b in captured[0].seen_backends)


@pytest.mark.asyncio
async def test_requirement_copy_contract() -> None:
    """A raising __copy__ propagates from stream() before generation starts."""

    class RaisingCopyReq(Requirement):
        def __copy__(self) -> "RaisingCopyReq":
            raise RuntimeError("copy boom")

        def format_for_llm(self) -> str:
            return "raising copy"

        async def stream_validate(
            self, chunk: str, *, backend: Any, ctx: Any
        ) -> PartialValidationResult:
            return PartialValidationResult("unknown")

        async def validate(
            self,
            backend: Any,
            ctx: Any,
            *,
            format: Any = None,
            model_options: Any = None,
        ) -> ValidationResult:
            return ValidationResult(result=True)

    class CountingBackend(StreamingMockBackend):
        def __init__(self, response: str, token_size: int = 1) -> None:
            super().__init__(response, token_size)
            self.gen_calls = 0

        async def _generate_from_context(self, *args: Any, **kwargs: Any):
            self.gen_calls += 1
            return await super()._generate_from_context(*args, **kwargs)

    # Failure path: the copy fails, so generation is never started.
    fail_backend = CountingBackend("Hello. ", token_size=3)
    with pytest.raises(RuntimeError, match="copy boom"):
        await stream(_action(), fail_backend, _ctx(), requirements=[RaisingCopyReq()])
    assert fail_backend.gen_calls == 0

    # Success path: a good copy starts generation exactly once.
    ok_backend = CountingBackend("Hello. ", token_size=3)
    async with await stream(
        _action(), ok_backend, _ctx(), requirements=[AlwaysUnknownReq()]
    ) as streamer:
        async for _chunk in streamer:
            pass
    assert ok_backend.gen_calls == 1


# ---------------------------------------------------------------------------
# Error + precomputed-MOT handling
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_exception_in_stream_validate_propagates_and_cancels() -> None:
    """An exception in stream_validate propagates from the loop and cancels gen."""

    class RaisingReq(Requirement):
        def format_for_llm(self) -> str:
            return "raiser"

        async def stream_validate(
            self, chunk: str, *, backend: Any, ctx: Any
        ) -> PartialValidationResult:
            raise RuntimeError("validate boom")

        async def validate(
            self,
            backend: Any,
            ctx: Any,
            *,
            format: Any = None,
            model_options: Any = None,
        ) -> ValidationResult:
            return ValidationResult(result=True)

    backend = StreamingMockBackend("Hello world. ", token_size=3)

    streamer = await stream(
        _action(), backend, _ctx(), requirements=[RaisingReq()], chunking="sentence"
    )
    with pytest.raises(RuntimeError, match="validate boom"):
        async with streamer:
            async for _chunk in streamer:
                pass

    # The generation was cancelled during teardown, not merely finished.
    assert streamer._mot._cancelled is True
    assert streamer._mot.is_computed() is True


@pytest.mark.asyncio
async def test_rejects_precomputed_mot() -> None:
    """A backend returning an already-computed MOT raises RuntimeError.

    stream() requires streaming; a pre-computed MOT would skip the loop entirely,
    producing empty output and silently passing final validators against an empty
    string.
    """

    class PrecomputedBackend(Backend):
        _model_id: str = "precomputed-mock-model"
        _provider: str = "precomputed-mock-provider"

        async def _generate_from_context(
            self,
            action: Any,
            ctx: Any,
            *,
            format: Any = None,
            model_options: dict | None = None,
            tool_calls: bool = False,
        ) -> tuple[ModelOutputThunk, Any]:
            return ModelOutputThunk(value="already done"), ctx

        async def _generate_from_raw(
            self, actions: Any, ctx: Any, **kwargs: Any
        ) -> tuple[list[ModelOutputThunk], dict[str, Any] | None]:
            raise NotImplementedError

    with pytest.raises(RuntimeError, match="already-computed MOT"):
        await stream(_action(), PrecomputedBackend(), _ctx())


@pytest.mark.asyncio
async def test_unknown_chunking_alias_raises_value_error() -> None:
    """An unknown chunking alias string raises ValueError."""
    backend = StreamingMockBackend("Hello. ", token_size=3)
    with pytest.raises(ValueError, match="Unknown chunking alias"):
        await stream(_action(), backend, _ctx(), chunking="unknown_alias")


@pytest.mark.asyncio
async def test_cancels_peer_validators() -> None:
    """A failing stream_validate does not let a slow peer run to completion."""
    reached_final_stage = asyncio.Event()

    class _RaisingReq(Requirement):
        def format_for_llm(self) -> str:
            return "raiser"

        async def stream_validate(
            self, chunk: str, *, backend: Any, ctx: Any
        ) -> PartialValidationResult:
            raise RuntimeError("validator failed")

        async def validate(
            self,
            backend: Any,
            ctx: Any,
            *,
            format: Any = None,
            model_options: Any = None,
        ) -> ValidationResult:
            return ValidationResult(result=False)

    class _SlowReq(Requirement):
        def format_for_llm(self) -> str:
            return "slow"

        async def stream_validate(
            self, chunk: str, *, backend: Any, ctx: Any
        ) -> PartialValidationResult:
            await asyncio.sleep(5.0)
            reached_final_stage.set()
            return PartialValidationResult("pass")

        async def validate(
            self,
            backend: Any,
            ctx: Any,
            *,
            format: Any = None,
            model_options: Any = None,
        ) -> ValidationResult:
            return ValidationResult(result=True)

    backend = StreamingMockBackend("Hello world. ", token_size=2)
    streamer = await stream(
        _action(), backend, _ctx(), requirements=[_RaisingReq(), _SlowReq()]
    )
    with pytest.raises(RuntimeError, match="validator failed"):
        async with streamer:
            async for _chunk in streamer:
                pass

    await asyncio.sleep(0.05)
    assert not reached_final_stage.is_set(), "slow sibling ran to completion"


# ---------------------------------------------------------------------------
# Event emission (observed via the STREAMING_EVENT hook)
# ---------------------------------------------------------------------------


def test_stream_event_types_have_auto_timestamp() -> None:
    """All seven event types set timestamp automatically; callers do not pass it."""
    before = time.time()
    all_events = [
        ChunkEvent(text="hello", chunk_index=0, attempt=1),
        QuickCheckEvent(
            chunk_index=0,
            attempt=1,
            passed=True,
            results=[PartialValidationResult("unknown")],
        ),
        StreamingDoneEvent(attempt=1, full_text="hello"),
        FullValidationEvent(
            attempt=1, passed=True, results=[ValidationResult(result=True)]
        ),
        RetryEvent(attempt=2, reason="too long"),
        CompletedEvent(success=True, full_text="hello", attempts_used=1),
        ErrorEvent(exception_type="ValueError", detail="boom"),
    ]
    after = time.time()

    for ev in all_events:
        assert isinstance(ev, StreamEvent)
        assert before <= ev.timestamp <= after, (
            f"{type(ev).__name__} timestamp out of range"
        )


@_cpex_skip
@pytest.mark.asyncio
async def test_event_emission_order_happy_path() -> None:
    """Natural completion emits QuickCheck/Chunk pairs, then Done, FullValidation, Completed."""
    response = "One. Two. "
    backend = StreamingMockBackend(response, token_size=2)

    with _record_events() as events:
        async with await stream(
            _action(),
            backend,
            _ctx(),
            requirements=[AlwaysUnknownReq()],
            chunking="sentence",
        ) as streamer:
            async for _chunk in streamer:
                pass

    types = [type(e) for e in events]
    assert types[-1] is CompletedEvent
    assert events[-1].success is True
    assert events[-1].attempts_used == 1
    assert StreamingDoneEvent in types
    assert FullValidationEvent in types
    # Done precedes FullValidation precedes Completed.
    assert types.index(StreamingDoneEvent) < types.index(FullValidationEvent)
    assert types.index(FullValidationEvent) < types.index(CompletedEvent)

    # Two sentences → two QuickCheck/Chunk pairs, indexed in order.
    chunk_events = [e for e in events if isinstance(e, ChunkEvent)]
    qc_events = [e for e in events if isinstance(e, QuickCheckEvent)]
    assert len(chunk_events) == 2
    assert len(qc_events) == 2
    assert [e.chunk_index for e in chunk_events] == [0, 1]
    assert [e.chunk_index for e in qc_events] == [0, 1]
    assert all(e.passed for e in qc_events)

    # QuickCheckEvent precedes ChunkEvent within each pair: a chunk is validated
    # before it is emitted.
    for ci in range(2):
        assert events.index(qc_events[ci]) < events.index(chunk_events[ci])


@_cpex_skip
@pytest.mark.asyncio
async def test_streaming_done_event_carries_full_text() -> None:
    """StreamingDoneEvent.full_text matches the streamer's full_text."""
    response = "One. Two. Three. "
    backend = StreamingMockBackend(response, token_size=3)

    with _record_events() as events:
        async with await stream(
            _action(), backend, _ctx(), chunking="sentence"
        ) as streamer:
            async for _chunk in streamer:
                pass

    done = [e for e in events if isinstance(e, StreamingDoneEvent)]
    assert len(done) == 1
    assert done[0].full_text == streamer.full_text


@_cpex_skip
@pytest.mark.asyncio
async def test_event_emission_on_early_exit() -> None:
    """Early exit emits a failing QuickCheck then Completed; no Done/FullValidation."""
    response = "one two three four five "
    backend = StreamingMockBackend(response, token_size=2)
    req = FailAfterWordsReq(threshold=2)

    with _record_events() as events:
        async with await stream(
            _action(), backend, _ctx(), requirements=[req], chunking="word"
        ) as streamer:
            async for _chunk in streamer:
                pass

    types = [type(e) for e in events]
    assert StreamingDoneEvent not in types
    assert FullValidationEvent not in types
    assert types[-1] is CompletedEvent
    assert events[-1].success is False
    # The last quick-check failed.
    quick = [e for e in events if isinstance(e, QuickCheckEvent)]
    assert quick[-1].passed is False


@_cpex_skip
@pytest.mark.asyncio
async def test_no_requirements_omits_full_validation_event() -> None:
    """With no requirements, no QuickCheck or FullValidation events fire."""
    response = "One. Two. "
    backend = StreamingMockBackend(response, token_size=2)

    with _record_events() as events:
        async with await stream(
            _action(), backend, _ctx(), chunking="sentence"
        ) as streamer:
            async for _chunk in streamer:
                pass

    types = [type(e) for e in events]
    assert QuickCheckEvent not in types
    assert FullValidationEvent not in types
    assert ChunkEvent in types
    assert StreamingDoneEvent in types
    assert types[-1] is CompletedEvent
    assert events[-1].success is True


@_cpex_skip
@pytest.mark.asyncio
async def test_error_event_on_stream_validate_exception() -> None:
    """An exception in stream_validate emits ErrorEvent then Completed(success=False)."""

    class RaisingReq(Requirement):
        def format_for_llm(self) -> str:
            return "raiser"

        async def stream_validate(
            self, chunk: str, *, backend: Any, ctx: Any
        ) -> PartialValidationResult:
            raise RuntimeError("boom")

        async def validate(
            self,
            backend: Any,
            ctx: Any,
            *,
            format: Any = None,
            model_options: Any = None,
        ) -> ValidationResult:
            return ValidationResult(result=True)

    backend = StreamingMockBackend("Hello world. ", token_size=3)

    with _record_events() as events:
        streamer = await stream(
            _action(), backend, _ctx(), requirements=[RaisingReq()], chunking="sentence"
        )
        with pytest.raises(RuntimeError, match="boom"):
            async with streamer:
                async for _chunk in streamer:
                    pass

    types = [type(e) for e in events]
    assert types[-1] is CompletedEvent
    assert events[-1].success is False
    # Exactly one ErrorEvent, carrying the exception type and message.
    error_events = [e for e in events if isinstance(e, ErrorEvent)]
    assert len(error_events) == 1
    assert error_events[0].exception_type == "RuntimeError"
    assert "boom" in error_events[0].detail


if __name__ == "__main__":
    pytest.main([__file__])
