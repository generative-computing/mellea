"""Tests for stream_with_chunking() and StreamChunkingResult.

Uses StreamingMockBackend — a deterministic test double that feeds tokens from a
fixed response string into a MOT queue without network or LLM calls.

All tests are unit tests (no @pytest.mark.ollama needed).
"""

import asyncio
import time
from typing import Any
from unittest.mock import patch

import pytest

from mellea.core.backend import Backend
from mellea.core.base import CBlock, Context, GenerateType, ModelOutputThunk
from mellea.core.requirement import (
    PartialValidationResult,
    Requirement,
    ValidationResult,
)
from mellea.stdlib.context import SimpleContext
from mellea.stdlib.streaming import (
    ChunkEvent,
    CompletedEvent,
    ErrorEvent,
    FullValidationEvent,
    QuickCheckEvent,
    RetryEvent,
    StreamEvent,
    StreamingDoneEvent,
    stream_with_chunking,
)

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

    ``token_size`` controls how many characters constitute one token.
    Validation calls (via ``stream_validate`` / ``validate``) are delegated
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

    Each call to ``stream_validate`` receives a single chunk (delta) from the
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


class MutationDetectorReq(Requirement):
    """Tracks how many times stream_validate was called on this instance."""

    def __init__(self) -> None:
        super().__init__()
        self._call_count = 0

    def format_for_llm(self) -> str:
        return "mutation detector"

    async def stream_validate(
        self, chunk: str, *, backend: Any, ctx: Any
    ) -> PartialValidationResult:
        _ = chunk, backend, ctx
        self._call_count += 1
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


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_normal_completion_calls_validate_at_stream_end() -> None:
    """All 'unknown' requirements → validate() called at stream end; completed=True."""
    response = "Hello world. How are you. "
    backend = StreamingMockBackend(response, token_size=3)
    req = AlwaysUnknownReq()

    result = await stream_with_chunking(
        _action(), backend, _ctx(), requirements=[req], chunking="sentence"
    )
    await result.acomplete()

    assert result.completed is True
    assert result.full_text == response
    assert len(result.final_validations) == 1
    assert result.final_validations[0].as_bool() is True
    assert result.streaming_failures == []


@pytest.mark.asyncio
async def test_early_exit_on_fail() -> None:
    """Requirement fails mid-stream → completed=False, streaming_failures populated."""
    # 5 words to trigger failure
    response = "one two three four five six seven eight. "
    backend = StreamingMockBackend(response, token_size=2)
    req = FailAfterWordsReq(threshold=4)

    result = await stream_with_chunking(
        _action(), backend, _ctx(), requirements=[req], chunking="word"
    )
    await result.acomplete()

    assert result.completed is False
    assert len(result.streaming_failures) == 1
    _req, pvr = result.streaming_failures[0]
    assert pvr.success == "fail"
    assert pvr.reason == "too many words"
    # final_validations should be empty — final validate() skipped on early exit
    assert result.final_validations == []


@pytest.mark.asyncio
async def test_clone_isolation_across_retries() -> None:
    """Originals must not be mutated; two invocations are independent."""
    response = "Sentence one. Sentence two. "
    req = MutationDetectorReq()
    original_reqs = [req]

    backend = StreamingMockBackend(response, token_size=4)

    r1 = await stream_with_chunking(
        _action(), backend, _ctx(), requirements=original_reqs, chunking="sentence"
    )
    await r1.acomplete()

    r2 = await stream_with_chunking(
        _action(), backend, _ctx(), requirements=original_reqs, chunking="sentence"
    )
    await r2.acomplete()

    # Original requirement must never have been called — only clones are used
    assert req._call_count == 0


@pytest.mark.asyncio
async def test_validation_backend_routing() -> None:
    """stream_validate and validate receive validation_backend, not the main backend."""
    response = "One sentence. Two sentences. "
    main_backend = StreamingMockBackend(response, token_size=3)
    val_backend = StreamingMockBackend("unused", token_size=1)

    req = BackendRecordingReq()

    # Capture the cloned requirement so we can inspect which backends it saw.
    captured: list[BackendRecordingReq] = []
    original_copy = BackendRecordingReq.__copy__

    def _capturing_copy(self: BackendRecordingReq) -> BackendRecordingReq:
        clone = original_copy(self)
        captured.append(clone)
        return clone

    BackendRecordingReq.__copy__ = _capturing_copy  # type: ignore[method-assign]
    try:
        result = await stream_with_chunking(
            _action(),
            main_backend,
            _ctx(),
            requirements=[req],
            chunking="sentence",
            validation_backend=val_backend,
        )
        await result.acomplete()
    finally:
        BackendRecordingReq.__copy__ = original_copy  # type: ignore[method-assign]

    assert result.completed is True
    # The original was never called — only clones are used.
    assert req.seen_backends == []
    # The clone must have seen val_backend for every call (stream_validate + validate),
    # never main_backend. This is the actual routing assertion.
    assert len(captured) == 1
    assert len(captured[0].seen_backends) > 0
    assert all(b is val_backend for b in captured[0].seen_backends)


@pytest.mark.asyncio
async def test_early_exit_does_not_deadlock() -> None:
    """Early failure with a high-throughput stream must not hang."""
    long_response = "word " * 200
    backend = StreamingMockBackend(long_response, token_size=5)
    req = FailAfterWordsReq(threshold=3)

    result = await stream_with_chunking(
        _action(), backend, _ctx(), requirements=[req], chunking="word"
    )
    # 5-second timeout — should complete in milliseconds on success
    await asyncio.wait_for(result.acomplete(), timeout=5.0)

    assert result.completed is False


@pytest.mark.asyncio
async def test_as_thunk_correctness() -> None:
    """as_thunk is computed, value matches full_text, generation metadata preserved."""
    response = "This is a test sentence. "
    backend = StreamingMockBackend(response, token_size=4)

    result = await stream_with_chunking(_action(), backend, _ctx(), chunking="sentence")
    await result.acomplete()

    thunk = result.as_thunk
    assert thunk.is_computed()
    assert thunk.value == result.full_text == response


@pytest.mark.asyncio
async def test_as_thunk_raises_before_acomplete() -> None:
    """as_thunk raises RuntimeError if accessed before acomplete()."""
    response = "Some text. "
    backend = StreamingMockBackend(response, token_size=2)

    result = await stream_with_chunking(_action(), backend, _ctx(), chunking="sentence")

    with pytest.raises(RuntimeError, match="acomplete"):
        _ = result.as_thunk


@pytest.mark.asyncio
async def test_astream_yields_individual_chunks() -> None:
    """Consumer via astream() receives individual chunks, not accumulated text."""
    response = "First sentence. Second sentence. Third sentence. "
    backend = StreamingMockBackend(response, token_size=5)

    result = await stream_with_chunking(_action(), backend, _ctx(), chunking="sentence")

    chunks: list[str] = []
    async for chunk in result.astream():
        chunks.append(chunk)

    await result.acomplete()

    # Each chunk must be a complete sentence (not the accumulated text)
    assert len(chunks) == 3
    for chunk in chunks:
        assert chunk.endswith(".")
    # Chunks don't include inter-sentence spaces; joined with a space they appear in full_text
    assert " ".join(chunks) in result.full_text


@pytest.mark.asyncio
async def test_stream_validate_receives_individual_chunks() -> None:
    """stream_validate is called once per chunk with the chunk itself, not accumulated text."""

    class ChunkRecordingReq(Requirement):
        def __init__(self) -> None:
            self.seen_chunks: list[str] = []

        def __copy__(self) -> "ChunkRecordingReq":
            clone = ChunkRecordingReq()
            clone.seen_chunks = []
            return clone

        def format_for_llm(self) -> str:
            return "chunk recorder"

        async def stream_validate(
            self, chunk: str, *, backend: Any, ctx: Any
        ) -> PartialValidationResult:
            self.seen_chunks.append(chunk)
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

    response = "First sentence. Second sentence. Third sentence. "
    backend = StreamingMockBackend(response, token_size=4)
    req = ChunkRecordingReq()

    # Capture the cloned requirement used by the orchestrator via a side channel.
    captured: list[ChunkRecordingReq] = []
    original_copy = ChunkRecordingReq.__copy__

    def _capturing_copy(self: ChunkRecordingReq) -> ChunkRecordingReq:
        clone = original_copy(self)
        captured.append(clone)
        return clone

    ChunkRecordingReq.__copy__ = _capturing_copy  # type: ignore[method-assign]
    try:
        result = await stream_with_chunking(
            _action(), backend, _ctx(), requirements=[req], chunking="sentence"
        )
        await result.acomplete()
    finally:
        ChunkRecordingReq.__copy__ = original_copy  # type: ignore[method-assign]

    assert len(captured) == 1
    seen = captured[0].seen_chunks
    # Exact match: three separate calls, one per complete sentence,
    # each call receiving that sentence and nothing more.  Under the old
    # accumulated-text semantics, seen would have been
    # ["First sentence.", "First sentence. Second sentence.", ...] —
    # exact match against the per-chunk list is the direct regression guard.
    assert seen == ["First sentence.", "Second sentence.", "Third sentence."]


@pytest.mark.asyncio
async def test_trailing_fragment_is_flushed_to_consumer() -> None:
    """Response without trailing whitespace: final sentence reaches astream() and stream_validate."""

    class ChunkRecordingReq(Requirement):
        def __init__(self) -> None:
            self.seen_chunks: list[str] = []

        def __copy__(self) -> "ChunkRecordingReq":
            clone = ChunkRecordingReq()
            clone.seen_chunks = []
            return clone

        def format_for_llm(self) -> str:
            return "chunk recorder"

        async def stream_validate(
            self, chunk: str, *, backend: Any, ctx: Any
        ) -> PartialValidationResult:
            self.seen_chunks.append(chunk)
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

    # No trailing whitespace after the final sentence — SentenceChunker withholds it.
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
        result = await stream_with_chunking(
            _action(), backend, _ctx(), requirements=[req], chunking="sentence"
        )
        yielded: list[str] = []
        async for chunk in result.astream():
            yielded.append(chunk)
        await result.acomplete()
    finally:
        ChunkRecordingReq.__copy__ = original_copy  # type: ignore[method-assign]

    # Both sentences reach the consumer, including the terminating one without trailing whitespace.
    assert yielded == ["First sentence.", "Second sentence."]
    # stream_validate was called on both — the flush path is not a shortcut.
    assert captured[0].seen_chunks == ["First sentence.", "Second sentence."]
    assert result.completed is True


@pytest.mark.asyncio
async def test_early_exit_on_trailing_fragment() -> None:
    """A fail on the flushed fragment records a streaming failure and skips final validate()."""

    class FailOnSecondSentence(Requirement):
        def __init__(self) -> None:
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

    result = await stream_with_chunking(
        _action(), backend, _ctx(), requirements=[req], chunking="sentence"
    )
    yielded: list[str] = []
    async for chunk in result.astream():
        yielded.append(chunk)
    await result.acomplete()

    assert result.completed is False
    assert len(result.streaming_failures) == 1
    # First sentence was emitted; second (the flushed fragment) failed and wasn't emitted.
    assert yielded == ["First sentence."]
    # Early exit on fail skips final validate().
    assert result.final_validations == []


@pytest.mark.asyncio
async def test_no_requirements_streams_without_validation() -> None:
    """requirements=None → chunks produced, no validate() called."""
    response = "Chunk one. Chunk two. Chunk three. "
    backend = StreamingMockBackend(response, token_size=3)

    result = await stream_with_chunking(
        _action(), backend, _ctx(), requirements=None, chunking="sentence"
    )
    await result.acomplete()

    assert result.completed is True
    assert result.full_text == response
    assert result.final_validations == []
    assert result.streaming_failures == []


@pytest.mark.asyncio
async def test_no_requirements_events_omits_full_validation_event() -> None:
    """With no requirements, events() emits StreamingDoneEvent but
    NOT FullValidationEvent — there is nothing to validate at stream end."""
    response = "Chunk one. Chunk two. "
    backend = StreamingMockBackend(response, token_size=3)

    result = await stream_with_chunking(
        _action(), backend, _ctx(), requirements=None, chunking="sentence"
    )
    await result.acomplete()

    evts = [e async for e in result.events()]
    types = [type(e) for e in evts]

    assert StreamingDoneEvent in types
    assert FullValidationEvent not in types
    assert isinstance(evts[-1], CompletedEvent)
    assert evts[-1].success is True


@pytest.mark.asyncio
async def test_multiple_chunks_in_one_batch_with_mid_batch_fail() -> None:
    """When one astream() delta produces several complete chunks and one in
    the middle fails, earlier chunks emit, failing chunk is recorded, later
    chunks are neither validated nor emitted."""

    captured: list[Any] = []

    class FailOnNthChunk(Requirement):
        def __init__(self, n: int) -> None:
            self._n = n
            self._calls = 0
            self.seen: list[str] = []

        def __copy__(self) -> "FailOnNthChunk":
            clone = FailOnNthChunk(self._n)
            captured.append(clone)
            return clone

        def format_for_llm(self) -> str:
            return f"fail on chunk {self._n}"

        async def stream_validate(
            self, chunk: str, *, backend: Any, ctx: Any
        ) -> PartialValidationResult:
            _ = backend, ctx
            self._calls += 1
            self.seen.append(chunk)
            if self._calls == self._n:
                return PartialValidationResult("fail", reason=f"n={self._n}")
            return PartialValidationResult("unknown")

        async def validate(
            self,
            backend: Any,
            ctx: Any,
            *,
            format: Any = None,
            model_options: Any = None,
        ) -> ValidationResult:
            _ = backend, ctx, format, model_options
            return ValidationResult(result=True)

    # token_size larger than the whole response → one astream() delta delivers
    # the full text, so chunking.split produces 4 sentences in a single batch.
    response = "One. Two. Three. Four. "
    backend = StreamingMockBackend(response, token_size=100)
    req = FailOnNthChunk(n=2)

    result = await stream_with_chunking(
        _action(), backend, _ctx(), requirements=[req], chunking="sentence"
    )
    yielded: list[str] = []
    async for c in result.astream():
        yielded.append(c)
    await result.acomplete()

    assert result.completed is False
    assert len(result.streaming_failures) == 1
    # Chunk 1 was validated and emitted; chunk 2 was validated and failed
    # (NOT emitted); chunks 3 and 4 were NEITHER validated NOR emitted.
    assert yielded == ["One."]
    assert len(captured) == 1
    assert captured[0].seen == ["One.", "Two."]
    assert captured[0]._calls == 2


@pytest.mark.asyncio
async def test_cancel_generation_invoked_on_fail() -> None:
    """Early exit on 'fail' must call mot.cancel_generation() — the spec reason
    is that asyncio.Queue(maxsize=20) will block the producer if the consumer
    stops without cancelling."""

    from mellea.core.base import ModelOutputThunk

    response = "word " * 50
    backend = StreamingMockBackend(response, token_size=3)

    class FailOnFirstChunk(Requirement):
        def format_for_llm(self) -> str:
            return "fail immediately"

        async def stream_validate(
            self, chunk: str, *, backend: Any, ctx: Any
        ) -> PartialValidationResult:
            _ = chunk, backend, ctx
            return PartialValidationResult("fail", reason="nope")

        async def validate(
            self,
            backend: Any,
            ctx: Any,
            *,
            format: Any = None,
            model_options: Any = None,
        ) -> ValidationResult:
            _ = backend, ctx, format, model_options
            return ValidationResult(result=True)

    call_count = 0
    real_cancel = ModelOutputThunk.cancel_generation

    async def spy_cancel(
        self: ModelOutputThunk, error: Exception | None = None
    ) -> None:
        nonlocal call_count
        call_count += 1
        await real_cancel(self, error)

    ModelOutputThunk.cancel_generation = spy_cancel  # type: ignore[method-assign]
    try:
        result = await stream_with_chunking(
            _action(),
            backend,
            _ctx(),
            requirements=[FailOnFirstChunk()],
            chunking="word",
        )
        await asyncio.wait_for(result.acomplete(), timeout=5.0)
    finally:
        ModelOutputThunk.cancel_generation = real_cancel  # type: ignore[method-assign]

    assert result.completed is False
    assert call_count >= 1


@pytest.mark.asyncio
async def test_cancelled_flag_reflects_cancellation_state() -> None:
    """The ``cancelled`` property on ModelOutputThunk distinguishes an early-exit
    cancellation from a normal completion and propagates through ``as_thunk``."""

    # Early exit → cancelled is True, is_computed True, propagates through as_thunk.
    fail_response = "word " * 50
    fail_backend = StreamingMockBackend(fail_response, token_size=3)

    class FailImmediately(Requirement):
        def format_for_llm(self) -> str:
            return "fail immediately"

        async def stream_validate(
            self, chunk: str, *, backend: Any, ctx: Any
        ) -> PartialValidationResult:
            _ = chunk, backend, ctx
            return PartialValidationResult("fail", reason="nope")

        async def validate(
            self,
            backend: Any,
            ctx: Any,
            *,
            format: Any = None,
            model_options: Any = None,
        ) -> ValidationResult:
            _ = backend, ctx, format, model_options
            return ValidationResult(result=True)

    fail_result = await stream_with_chunking(
        _action(),
        fail_backend,
        _ctx(),
        requirements=[FailImmediately()],
        chunking="word",
    )
    await asyncio.wait_for(fail_result.acomplete(), timeout=5.0)

    assert fail_result.completed is False
    assert fail_result.as_thunk.cancelled is True
    assert fail_result.as_thunk.is_computed() is True

    # Normal completion → cancelled is False.
    ok_response = "Hello world. How are you. "
    ok_backend = StreamingMockBackend(ok_response, token_size=3)

    ok_result = await stream_with_chunking(
        _action(),
        ok_backend,
        _ctx(),
        requirements=[AlwaysUnknownReq()],
        chunking="sentence",
    )
    await ok_result.acomplete()

    assert ok_result.completed is True
    assert ok_result.as_thunk.cancelled is False
    assert ok_result.as_thunk.is_computed() is True


@pytest.mark.asyncio
async def test_unknown_chunking_alias_raises_value_error() -> None:
    """An unrecognised chunking alias raises ValueError before any backend call."""
    backend = StreamingMockBackend("hello world")
    with pytest.raises(ValueError, match="unknown_alias"):
        await stream_with_chunking(_action(), backend, _ctx(), chunking="unknown_alias")


@pytest.mark.asyncio
async def test_exception_in_stream_validate_cancels_generation() -> None:
    """Verifies the orchestrator's exception-path cleanup: if stream_validate
    raises, cancel_generation() is called and the exception surfaces to the
    consumer via astream()/acomplete() without hanging.

    This covers the cancel-on-exception path and the no-hang guarantee.
    It does not directly exercise the worst-case "producer already blocked on
    full queue" scenario (here the fail happens on chunk 1 so the queue never
    fills); the cancel_generation drain logic is covered by its own tests in
    test/core/.
    """

    from mellea.core.base import ModelOutputThunk

    class RaisingReq(Requirement):
        def format_for_llm(self) -> str:
            return "raises"

        async def stream_validate(
            self, chunk: str, *, backend: Any, ctx: Any
        ) -> PartialValidationResult:
            _ = chunk, backend, ctx
            raise ValueError("boom")

        async def validate(
            self,
            backend: Any,
            ctx: Any,
            *,
            format: Any = None,
            model_options: Any = None,
        ) -> ValidationResult:
            _ = backend, ctx, format, model_options
            return ValidationResult(result=True)

    response = "word " * 50  # enough to fill maxsize=20 queue without cleanup
    backend = StreamingMockBackend(response, token_size=3)

    call_count = 0
    real_cancel = ModelOutputThunk.cancel_generation

    async def spy_cancel(
        self: ModelOutputThunk, error: Exception | None = None
    ) -> None:
        nonlocal call_count
        call_count += 1
        await real_cancel(self, error)

    ModelOutputThunk.cancel_generation = spy_cancel  # type: ignore[method-assign]
    try:
        result = await stream_with_chunking(
            _action(), backend, _ctx(), requirements=[RaisingReq()], chunking="word"
        )
        with pytest.raises(ValueError, match="boom"):
            async for _chunk in result.astream():
                pass
        # acomplete must complete (not hang) even though the orchestration
        # task raised, because cancel_generation was called in the except path.
        await asyncio.wait_for(result.acomplete(), timeout=5.0)
    finally:
        ModelOutputThunk.cancel_generation = real_cancel  # type: ignore[method-assign]

    assert result.completed is False
    assert call_count >= 1


@pytest.mark.asyncio
async def test_acomplete_surfaces_exception_without_astream() -> None:
    """acomplete() must surface orchestrator exceptions even when the
    consumer never iterates astream().

    The alternative — only delivering the exception through the chunk queue
    — silently swallows validator failures for callers who skip astream().
    """

    class RaisingReq(Requirement):
        def format_for_llm(self) -> str:
            return "raises"

        async def stream_validate(
            self, chunk: str, *, backend: Any, ctx: Any
        ) -> PartialValidationResult:
            _ = chunk, backend, ctx
            raise ValueError("surfaced-without-astream")

        async def validate(
            self,
            backend: Any,
            ctx: Any,
            *,
            format: Any = None,
            model_options: Any = None,
        ) -> ValidationResult:
            _ = backend, ctx, format, model_options
            return ValidationResult(result=True)

    response = "word " * 50
    backend = StreamingMockBackend(response, token_size=3)

    result = await stream_with_chunking(
        _action(), backend, _ctx(), requirements=[RaisingReq()], chunking="word"
    )
    # Deliberately skip astream(). wait_for bounds any hang.
    with pytest.raises(ValueError, match="surfaced-without-astream"):
        await asyncio.wait_for(result.acomplete(), timeout=5.0)

    assert result.completed is False
    # Raise-once: a second acomplete() must not re-raise.
    await asyncio.wait_for(result.acomplete(), timeout=5.0)


@pytest.mark.asyncio
async def test_external_task_cancellation_releases_consumers() -> None:
    """External cancellation of the orchestration task must still set _done.

    If the finally cleanup itself contains an ``await`` (e.g. awaiting a
    terminator put into the chunk queue), CancelledError re-raises at that
    await and ``_done.set()`` never runs — any consumer blocked on
    ``acomplete()`` hangs forever. The cleanup must therefore end with
    synchronous operations only.
    """
    response = "word " * 200  # long enough that streaming is still in progress
    backend = StreamingMockBackend(response, token_size=2)

    result = await stream_with_chunking(
        _action(), backend, _ctx(), requirements=[AlwaysUnknownReq()], chunking="word"
    )

    assert result._orchestration_task is not None
    # Wait until the orchestration coroutine has started and hit its first
    # suspension point.  Using _orchestration_started rather than a wall-clock
    # sleep avoids the race where a fast runner drains the whole stream within
    # the sleep window, making cancel() a no-op on an already-done task.
    # _orchestration_started.wait() must precede cancel() — cancelling before
    # the first scheduling means the event is never set.
    await asyncio.wait_for(result._orchestration_started.wait(), timeout=2.0)
    assert not result._orchestration_task.done(), (
        "orchestrator already done before cancel() — test would vacuously pass"
    )

    # Same mechanism asyncio.wait_for uses on timeout.
    result._orchestration_task.cancel()

    # _done must be set by the finally cleanup. A hang would time out here.
    await asyncio.wait_for(result._done.wait(), timeout=2.0)
    assert result._done.is_set()

    # acomplete() surfaces the CancelledError via task.exception() and must
    # not hang.
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(result.acomplete(), timeout=2.0)


@pytest.mark.asyncio
async def test_external_cancellation_acomplete_raise_once() -> None:
    """Raise-once contract holds for the task-fallback path on external cancel.

    CancelledError bypasses the orchestrator's ``except Exception`` handler,
    so ``_orchestration_exception`` is never set. ``acomplete()`` surfaces the
    cancel via ``self._orchestration_task.exception()`` instead — and that
    branch must also flip ``_exception_surfaced`` so a second ``acomplete()``
    call does not raise the same exception twice.
    """
    response = "word " * 200
    backend = StreamingMockBackend(response, token_size=2)

    result = await stream_with_chunking(
        _action(), backend, _ctx(), requirements=[AlwaysUnknownReq()], chunking="word"
    )

    assert result._orchestration_task is not None
    await asyncio.wait_for(result._orchestration_started.wait(), timeout=2.0)
    assert not result._orchestration_task.done(), (
        "orchestrator already done before cancel() — test would vacuously pass"
    )
    result._orchestration_task.cancel()
    await asyncio.wait_for(result._done.wait(), timeout=2.0)

    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(result.acomplete(), timeout=2.0)

    # Second call must NOT re-raise — raise-once contract.
    await asyncio.wait_for(result.acomplete(), timeout=2.0)


@pytest.mark.asyncio
async def test_raise_once_acomplete_then_astream() -> None:
    """Regression for the raise-once stash bug: acomplete() first, astream() second.

    Prior to the fix, acomplete() cleared _orchestration_exception, so a
    subsequent astream() call dequeued the exception item, saw the stash was
    None, silently skipped it, and returned zero chunks with no error.
    """

    class RaisingReq(Requirement):
        def format_for_llm(self) -> str:
            return "raises"

        async def stream_validate(
            self, chunk: str, *, backend: Any, ctx: Any
        ) -> PartialValidationResult:
            raise ValueError("raise-once-regression")

        async def validate(
            self,
            backend: Any,
            ctx: Any,
            *,
            format: Any = None,
            model_options: Any = None,
        ) -> ValidationResult:
            return ValidationResult(result=True)

    response = "word " * 10
    backend = StreamingMockBackend(response, token_size=3)
    result = await stream_with_chunking(
        _action(), backend, _ctx(), requirements=[RaisingReq()], chunking="word"
    )

    # acomplete() sees the exception first and raises it.
    with pytest.raises(ValueError, match="raise-once-regression"):
        await asyncio.wait_for(result.acomplete(), timeout=5.0)

    # astream() must NOT re-raise (raise-once semantics).  Because the
    # exception fired before any chunk was emitted, the queue contains
    # [exc, None].  With the separate _exception_surfaced flag, astream()
    # correctly skips the exception item and terminates cleanly.  Without
    # the flag the behaviour is the same, but the guard conflates
    # "already surfaced" with "stash was never set" — the flag makes the
    # intent unambiguous.
    chunks: list[str] = []
    async for chunk in result.astream():
        chunks.append(chunk)
    assert chunks == []  # no partial chunks before the exception


@pytest.mark.asyncio
async def test_full_text_contains_only_validated_chunks_on_early_exit() -> None:
    """full_text must equal exactly what was emitted to the consumer on early exit.

    When one astream() delta produces N chunks and chunk K fails, full_text
    must contain chunks 0..K-1 only — not the failed chunk or any unvalidated
    chunks after it in the same delta.
    """

    class FailOnNthChunkText(Requirement):
        def __init__(self, n: int) -> None:
            self._n = n
            self._calls = 0

        def __copy__(self) -> "FailOnNthChunkText":
            return FailOnNthChunkText(self._n)

        def format_for_llm(self) -> str:
            return f"fail on chunk {self._n}"

        async def stream_validate(
            self, chunk: str, *, backend: Any, ctx: Any
        ) -> PartialValidationResult:
            self._calls += 1
            if self._calls == self._n:
                return PartialValidationResult("fail")
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

    # token_size > full response → single delta with 4 sentences; fail on chunk 2.
    response = "One. Two. Three. Four. "
    backend = StreamingMockBackend(response, token_size=100)
    req = FailOnNthChunkText(n=2)

    result = await stream_with_chunking(
        _action(), backend, _ctx(), requirements=[req], chunking="sentence"
    )
    yielded: list[str] = []
    async for chunk in result.astream():
        yielded.append(chunk)
    await result.acomplete()

    assert result.completed is False
    # Consumer received only chunk 1.
    assert yielded == ["One."]
    # full_text must match what the consumer received — not the raw delta.
    assert result.full_text == "One."
    # as_thunk.value must agree with full_text.
    assert result.as_thunk.value == result.full_text

    # Fail on chunk 3: two chunks emitted before early exit.  full_text must
    # preserve the original inter-sentence spacing from the token stream, not
    # the stripped chunk concatenation ("One.Two." would be wrong).
    backend2 = StreamingMockBackend(response, token_size=100)
    req2 = FailOnNthChunkText(n=3)
    result2 = await stream_with_chunking(
        _action(), backend2, _ctx(), requirements=[req2], chunking="sentence"
    )
    yielded2: list[str] = []
    async for chunk in result2.astream():
        yielded2.append(chunk)
    await result2.acomplete()

    assert result2.completed is False
    assert yielded2 == ["One.", "Two."]
    assert result2.full_text == "One. Two."
    assert result2.as_thunk.value == result2.full_text


@pytest.mark.asyncio
async def test_cancelled_flag_propagates_through_copy_methods() -> None:
    """_cancelled must survive __copy__, __deepcopy__, and _copy_from."""
    from copy import deepcopy

    mot = ModelOutputThunk(value="result")
    mot._cancelled = True

    # __copy__
    shallow = mot.__copy__()
    assert shallow._cancelled is True, "__copy__ must propagate _cancelled"

    # __deepcopy__
    deep = deepcopy(mot)
    assert deep._cancelled is True, "__deepcopy__ must propagate _cancelled"

    # _copy_from
    target = ModelOutputThunk(value="original")
    assert target._cancelled is False
    target._copy_from(mot)
    assert target._cancelled is True, "_copy_from must propagate _cancelled"

    # Sanity: default-constructed MOT has _cancelled=False.
    fresh = ModelOutputThunk(value="x")
    assert fresh._cancelled is False


# ---------------------------------------------------------------------------
# Fix 1 — setup-path backend leak: copy(req) before generate_from_context
# ---------------------------------------------------------------------------


class _PlainReq(Requirement):
    """Default shallow copy — cannot raise."""

    def format_for_llm(self) -> str:
        return "plain"

    async def stream_validate(
        self, chunk: str, *, backend: Any, ctx: Any
    ) -> PartialValidationResult:
        return PartialValidationResult("unknown")

    async def validate(
        self, backend: Any, ctx: Any, *, format: Any = None, model_options: Any = None
    ) -> ValidationResult:
        return ValidationResult(result=True)


class _RaisingCopyReq(Requirement):
    """__copy__ raises — simulates a user-defined Requirement with a faulty override."""

    def __copy__(self) -> "_RaisingCopyReq":
        raise ValueError("copy boom")

    def format_for_llm(self) -> str:
        return "raising copy"

    async def stream_validate(
        self, chunk: str, *, backend: Any, ctx: Any
    ) -> PartialValidationResult:
        return PartialValidationResult("unknown")

    async def validate(
        self, backend: Any, ctx: Any, *, format: Any = None, model_options: Any = None
    ) -> ValidationResult:
        return ValidationResult(result=True)


class _InstrumentedBackend(StreamingMockBackend):
    """Counts generate_from_context calls and exposes the last MOT produced."""

    def __init__(self, response: str, token_size: int = 1) -> None:
        super().__init__(response, token_size)
        self.generate_from_context_call_count = 0
        self.last_mot: ModelOutputThunk | None = None

    async def _generate_from_context(
        self,
        action: Any,
        ctx: Any,
        *,
        format: Any = None,
        model_options: dict | None = None,
        tool_calls: bool = False,
    ) -> tuple[ModelOutputThunk, Any]:
        self.generate_from_context_call_count += 1
        mot, new_ctx = await super()._generate_from_context(
            action,
            ctx,
            format=format,
            model_options=model_options,
            tool_calls=tool_calls,
        )
        self.last_mot = mot
        return mot, new_ctx


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "req_cls,expect_raise", [(_PlainReq, False), (_RaisingCopyReq, True)]
)
async def test_stream_with_chunking_requirement_copy_contract(
    req_cls: type, expect_raise: bool
) -> None:
    """Fix 1: copy(req) runs before generate_from_context.

    On __copy__ failure the backend is never started (call_count == 0).
    On success the backend is called exactly once.
    """
    backend = _InstrumentedBackend("Hello world. ", token_size=2)
    req = req_cls()
    if expect_raise:
        with pytest.raises(ValueError, match="copy boom"):
            await stream_with_chunking(_action(), backend, _ctx(), requirements=[req])
        # Hard invariant: reorder ensures backend never starts on copy failure.
        assert backend.generate_from_context_call_count == 0
        assert backend.last_mot is None
    else:
        result = await stream_with_chunking(
            _action(), backend, _ctx(), requirements=[req]
        )
        await result.acomplete()
        assert backend.generate_from_context_call_count == 1
        assert backend.last_mot is not None


# ---------------------------------------------------------------------------
# Fix 3 — TaskGroup cancels peer validators on first failure
# ---------------------------------------------------------------------------
# Event type construction
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


# ---------------------------------------------------------------------------
# Event emission — happy path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_event_emission_order_happy_path() -> None:
    """Happy path: QuickCheckEvent/ChunkEvent pairs, then StreamingDoneEvent,
    FullValidationEvent, CompletedEvent(success=True)."""
    response = "First sentence. Second sentence. "
    backend = StreamingMockBackend(response, token_size=4)
    req = AlwaysUnknownReq()

    result = await stream_with_chunking(
        _action(), backend, _ctx(), requirements=[req], chunking="sentence"
    )
    await result.acomplete()

    evts: list[StreamEvent] = [e async for e in result.events()]

    assert isinstance(evts[-1], CompletedEvent)
    assert evts[-1].success is True
    assert evts[-1].attempts_used == 1

    types = [type(e) for e in evts]
    assert StreamingDoneEvent in types
    assert types.index(StreamingDoneEvent) < types.index(CompletedEvent)
    assert FullValidationEvent in types
    assert types.index(FullValidationEvent) > types.index(StreamingDoneEvent)

    chunk_events = [e for e in evts if isinstance(e, ChunkEvent)]
    qc_events = [e for e in evts if isinstance(e, QuickCheckEvent)]
    assert len(chunk_events) == 2
    assert len(qc_events) == 2
    assert [e.chunk_index for e in chunk_events] == [0, 1]
    assert [e.chunk_index for e in qc_events] == [0, 1]
    assert all(e.passed for e in qc_events)

    # QuickCheckEvent fires before ChunkEvent within each pair: validation must
    # complete before the chunk is released to the consumer queue.
    for ci in range(2):
        qc_pos = evts.index(qc_events[ci])
        ch_pos = evts.index(chunk_events[ci])
        assert qc_pos < ch_pos, f"chunk {ci}: QuickCheckEvent must precede ChunkEvent"


@pytest.mark.asyncio
async def test_streaming_done_event_carries_full_text() -> None:
    """StreamingDoneEvent.full_text matches full_text on the result."""
    response = "One sentence. Two sentences. "
    backend = StreamingMockBackend(response, token_size=5)

    result = await stream_with_chunking(_action(), backend, _ctx(), chunking="sentence")
    await result.acomplete()

    evts = [e async for e in result.events()]
    done_events = [e for e in evts if isinstance(e, StreamingDoneEvent)]
    assert len(done_events) == 1
    assert done_events[0].full_text == result.full_text


# ---------------------------------------------------------------------------
# Event emission — early exit
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_event_emission_on_early_exit() -> None:
    """Early exit: QuickCheckEvent(passed=False) present; no StreamingDoneEvent
    or FullValidationEvent; CompletedEvent(success=False)."""
    response = "word " * 30
    backend = StreamingMockBackend(response, token_size=3)
    req = FailAfterWordsReq(threshold=2)

    result = await stream_with_chunking(
        _action(), backend, _ctx(), requirements=[req], chunking="word"
    )
    await result.acomplete()

    evts = [e async for e in result.events()]

    assert isinstance(evts[-1], CompletedEvent)
    assert evts[-1].success is False

    types = [type(e) for e in evts]
    assert FullValidationEvent not in types
    assert StreamingDoneEvent not in types

    fail_qc = [e for e in evts if isinstance(e, QuickCheckEvent) and not e.passed]
    assert len(fail_qc) >= 1


# ---------------------------------------------------------------------------
# Event emission — exception path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stream_with_chunking_cancels_peer_validators() -> None:
    """Fix 3: a failing stream_validate causes TaskGroup to cancel peer validators.

    One requirement raises immediately in stream_validate; the second sleeps
    for 5 s and sets a flag on completion. Without TaskGroup the slow sibling
    runs detached; with it the cancellation is observed.
    """
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
            try:
                await asyncio.sleep(5.0)
                reached_final_stage.set()
                return PartialValidationResult("pass")
            except asyncio.CancelledError:
                raise  # propagate so TaskGroup knows we were cancelled

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
    result = await stream_with_chunking(
        _action(), backend, _ctx(), requirements=[_RaisingReq(), _SlowReq()]
    )
    with pytest.raises(RuntimeError, match="validator failed"):
        await result.acomplete()

    # Give the loop a tick; the slow sibling must NOT have run to completion.
    await asyncio.sleep(0.05)
    assert not reached_final_stage.is_set(), (
        "slow sibling was not cancelled by TaskGroup"
    )


@pytest.mark.asyncio
async def test_stream_with_chunking_rejects_precomputed_mot() -> None:
    """Backend returning an already-computed MOT raises RuntimeError immediately.

    stream_with_chunking() requires streaming; a pre-computed MOT would cause
    the orchestrator loop to skip entirely, producing empty output and silently
    passing all final validators against an empty string.
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
        await stream_with_chunking(_action(), PrecomputedBackend(), _ctx())


@pytest.mark.asyncio
async def test_error_event_on_stream_validate_exception() -> None:
    """When stream_validate raises, ErrorEvent is emitted and CompletedEvent follows."""

    class RaisingReq2(Requirement):
        def format_for_llm(self) -> str:
            return "raises"

        async def stream_validate(
            self, chunk: str, *, backend: Any, ctx: Any
        ) -> PartialValidationResult:
            raise RuntimeError("test-error")

        async def validate(
            self,
            backend: Any,
            ctx: Any,
            *,
            format: Any = None,
            model_options: Any = None,
        ) -> ValidationResult:
            return ValidationResult(result=True)

    backend = StreamingMockBackend("hello world", token_size=5)
    result = await stream_with_chunking(
        _action(), backend, _ctx(), requirements=[RaisingReq2()], chunking="word"
    )
    with pytest.raises(RuntimeError, match="test-error"):
        async for _c in result.astream():
            pass
    await asyncio.wait_for(result.acomplete(), timeout=5.0)

    evts = [e async for e in result.events()]

    error_events = [e for e in evts if isinstance(e, ErrorEvent)]
    assert len(error_events) == 1
    assert error_events[0].exception_type == "RuntimeError"
    assert "test-error" in error_events[0].detail

    assert isinstance(evts[-1], CompletedEvent)
    assert evts[-1].success is False


# ---------------------------------------------------------------------------
# Concurrent astream() + events()
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_concurrent_astream_and_events() -> None:
    """astream() and events() can be consumed concurrently without interference."""
    response = "Alpha. Beta. Gamma. "
    backend = StreamingMockBackend(response, token_size=4)
    req = AlwaysUnknownReq()

    result = await stream_with_chunking(
        _action(), backend, _ctx(), requirements=[req], chunking="sentence"
    )

    async def drain_chunks() -> list[str]:
        return [c async for c in result.astream()]

    async def drain_events() -> list[StreamEvent]:
        return [e async for e in result.events()]

    chunks, evts = await asyncio.gather(drain_chunks(), drain_events())
    await result.acomplete()

    assert len(chunks) == 3
    assert isinstance(evts[-1], CompletedEvent)
    assert evts[-1].success is True

    chunk_evts = [e for e in evts if isinstance(e, ChunkEvent)]
    assert [e.chunk_index for e in chunk_evts] == list(range(len(chunks)))


# ---------------------------------------------------------------------------
# events() single-consumer guard
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_events_single_consumer_guard_raises_on_second_call() -> None:
    """events() raises RuntimeError if called a second time on the same result."""
    response = "One sentence. "
    backend = StreamingMockBackend(response, token_size=4)

    result = await stream_with_chunking(_action(), backend, _ctx(), chunking="sentence")
    await result.acomplete()

    # First drain — OK.
    async for _ in result.events():
        pass

    # Second call must raise immediately.
    with pytest.raises(RuntimeError, match="single-consumer"):
        async for _ in result.events():
            pass


# ---------------------------------------------------------------------------
# CancelledError path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cancelled_task_sets_completed_false() -> None:
    """External task cancellation must leave result.completed=False.

    CancelledError is a BaseException and bypasses except Exception, so
    the finally block is responsible for setting result.completed=False.
    Regression: without the fix, result.completed stays True and
    CompletedEvent / record_sampling_outcome lie to callers.

    Uses a backend whose token feed blocks on an asyncio.Event that is
    never set, guaranteeing the orchestrator is suspended at astream()
    when the task is cancelled.

    Requires ``await asyncio.sleep(0)`` before ``cancel()`` — see inline
    comment.  Python 3.12's C Task implementation skips the coroutine body
    entirely (including finally blocks) when cancelled before the first
    ``coro.send(None)``.
    """
    gate = asyncio.Event()  # never set — feed task blocks indefinitely
    feed_task: asyncio.Task[None] | None = None

    async def _blocking_feed(mot: ModelOutputThunk) -> None:
        await gate.wait()

    class BlockingBackend(Backend):
        _model_id: str = "blocking-mock-model"
        _provider: str = "blocking-mock-provider"

        async def _generate_from_context(
            self, action: Any, ctx: Any, **kwargs: Any
        ) -> tuple[ModelOutputThunk, Any]:
            nonlocal feed_task
            mot = _make_mot()
            feed_task = asyncio.create_task(_blocking_feed(mot))
            return mot, ctx.add(action).add(mot)

        async def _generate_from_raw(self, *args: Any, **kwargs: Any) -> Any:
            raise NotImplementedError

    result = await stream_with_chunking(
        _action(), BlockingBackend(), _ctx(), chunking="word"
    )
    assert result._orchestration_task is not None

    # Deliberately uses sleep(0) rather than _orchestration_started.wait():
    # BlockingBackend blocks indefinitely, so there is no race with the stream
    # completing before cancel().  sleep(0) is sufficient to satisfy the
    # coro.send(None) requirement: Python 3.12's C Task implementation skips
    # the coroutine body entirely (including finally blocks) when cancelled
    # before the first send.
    await asyncio.sleep(0)

    result._orchestration_task.cancel()

    try:
        await result._orchestration_task
    except BaseException:
        pass

    # Primary assertion: completed must be False after external cancellation.
    assert result.completed is False

    # The finally block must have run to completion: _done must be set and
    # acomplete() must not hang.  This is the actual failure mode the fix
    # guards against — if _done is never set, acomplete() blocks forever.
    # External cancellation surfaces as CancelledError (raise-once contract).
    assert result._done.is_set()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(result.acomplete(), timeout=2.0)

    # Clean up the blocking feed task to avoid "Task destroyed while pending".
    if feed_task is not None:
        feed_task.cancel()
        try:
            await feed_task
        except BaseException:
            pass
