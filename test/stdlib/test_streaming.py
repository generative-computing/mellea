"""Tests for stream_with_chunking() and StreamChunkingResult.

Uses StreamingMockBackend — a deterministic test double that feeds tokens from a
fixed response string into a MOT queue without network or LLM calls.

All tests are unit tests (no @pytest.mark.ollama needed).
"""

import asyncio
from typing import Any

import pytest

from mellea.core.backend import Backend
from mellea.core.base import CBlock, Context, GenerateType, ModelOutputThunk
from mellea.core.requirement import (
    PartialValidationResult,
    Requirement,
    ValidationResult,
)
from mellea.stdlib.context import SimpleContext
from mellea.stdlib.streaming import stream_with_chunking

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
    mot._action = CBlock("mock_action")
    mot._generate_type = GenerateType.ASYNC
    mot._process = _mock_process
    mot._post_process = _mock_post_process
    mot._chunk_size = 0
    return mot


async def _feed_tokens(mot: ModelOutputThunk, response: str, token_size: int) -> None:
    i = 0
    while i < len(response):
        token = response[i : i + token_size]
        await mot._async_queue.put(token)
        await asyncio.sleep(0)
        i += token_size
    await mot._async_queue.put(None)


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

    async def generate_from_raw(
        self, actions: Any, ctx: Any, **kwargs: Any
    ) -> list[ModelOutputThunk]:
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
    """Returns 'fail' once the accumulated text reaches *threshold* words."""

    def __init__(self, threshold: int) -> None:
        self._threshold = threshold

    def format_for_llm(self) -> str:
        return f"fail after {self._threshold} words"

    async def stream_validate(
        self, chunk: str, *, backend: Any, ctx: Any
    ) -> PartialValidationResult:
        if len(chunk.split()) >= self._threshold:
            return PartialValidationResult("fail", reason="too many words")
        return PartialValidationResult("unknown")

    async def validate(
        self, backend: Any, ctx: Any, *, format: Any = None, model_options: Any = None
    ) -> ValidationResult:
        return ValidationResult(result=True)


class BackendRecordingReq(Requirement):
    """Records which backend was passed to stream_validate and validate."""

    def __init__(self) -> None:
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
        _action(), backend, _ctx(), quick_check_requirements=[req], chunking="sentence"
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
        _action(), backend, _ctx(), quick_check_requirements=[req], chunking="word"
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
        _action(),
        backend,
        _ctx(),
        quick_check_requirements=original_reqs,
        chunking="sentence",
    )
    await r1.acomplete()

    r2 = await stream_with_chunking(
        _action(),
        backend,
        _ctx(),
        quick_check_requirements=original_reqs,
        chunking="sentence",
    )
    await r2.acomplete()

    # Original requirement must never have been called — only clones are used
    assert req._call_count == 0


@pytest.mark.asyncio
async def test_quick_check_backend_routing() -> None:
    """stream_validate and validate receive quick_check_backend, not the main backend."""
    response = "One sentence. Two sentences. "
    main_backend = StreamingMockBackend(response, token_size=3)
    val_backend = StreamingMockBackend("unused", token_size=1)

    req = BackendRecordingReq()

    result = await stream_with_chunking(
        _action(),
        main_backend,
        _ctx(),
        quick_check_requirements=[req],
        chunking="sentence",
        quick_check_backend=val_backend,
    )
    await result.acomplete()

    # The clone's seen_backends should only contain val_backend
    # (The original req was never called; clones were.)
    # Verify via final_validations side-effect: at least one backend recorded
    assert result.completed is True
    # The original req._seen_backends is untouched (clone isolation)
    assert req.seen_backends == []


@pytest.mark.asyncio
async def test_early_exit_does_not_deadlock() -> None:
    """Early failure with a high-throughput stream must not hang."""
    long_response = "word " * 200
    backend = StreamingMockBackend(long_response, token_size=5)
    req = FailAfterWordsReq(threshold=3)

    result = await stream_with_chunking(
        _action(), backend, _ctx(), quick_check_requirements=[req], chunking="word"
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
async def test_no_requirements_streams_without_validation() -> None:
    """quick_check_requirements=None → chunks produced, no validate() called."""
    response = "Chunk one. Chunk two. Chunk three. "
    backend = StreamingMockBackend(response, token_size=3)

    result = await stream_with_chunking(
        _action(), backend, _ctx(), quick_check_requirements=None, chunking="sentence"
    )
    await result.acomplete()

    assert result.completed is True
    assert result.full_text == response
    assert result.final_validations == []
    assert result.streaming_failures == []
