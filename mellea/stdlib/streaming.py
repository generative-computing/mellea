"""Streaming generation with per-chunk validation.

Provides :func:`stream_with_chunking`, the core orchestration primitive that
consumes a streaming :class:`~mellea.core.base.ModelOutputThunk`, applies a
:class:`~mellea.stdlib.chunking.ChunkingStrategy` to produce semantic chunks,
and runs :meth:`~mellea.core.requirement.Requirement.stream_validate` on each
chunk in parallel.  Higher-level streaming APIs build on this function.

The orchestrator emits typed :class:`StreamEvent` objects that consumers can
observe via :meth:`StreamChunkingResult.events`.  Raw validated chunks remain
available via :meth:`StreamChunkingResult.astream`.
"""

import asyncio
import time
import uuid
from collections.abc import AsyncIterator, Sequence
from copy import copy
from dataclasses import dataclass, field
from typing import Any

from ..backends.model_options import ModelOption
from ..core.backend import Backend
from ..core.base import CBlock, Component, Context, ModelOutputThunk
from ..core.requirement import PartialValidationResult, Requirement, ValidationResult
from ..core.utils import MelleaLogger
from ..plugins.manager import has_plugins, invoke_hook
from ..plugins.types import HookType
from .chunking import ChunkingStrategy, ParagraphChunker, SentenceChunker, WordChunker

_CHUNKING_ALIASES: dict[str, type[ChunkingStrategy]] = {
    "sentence": SentenceChunker,
    "word": WordChunker,
    "paragraph": ParagraphChunker,
}

# ---------------------------------------------------------------------------
# Streaming event types
# ---------------------------------------------------------------------------


@dataclass
class StreamEvent:
    """Base class for all streaming events emitted by :func:`stream_with_chunking`.

    The `timestamp` field is auto-populated at instantiation time; callers
    do not set it.  Because `timestamp` has `init=False` it is never part
    of `__init__`, so subclasses may declare additional fields in any order
    without conflict.  Any new `init=False` fields on subclasses must also
    use `field(..., init=False)`.

    Attributes:
        timestamp: Unix timestamp (seconds) at the moment the event was created.
    """

    timestamp: float = field(default_factory=time.time, init=False)


@dataclass
class ChunkEvent(StreamEvent):
    """Emitted after each validated chunk is delivered to the consumer.

    Fired after all active requirements' `stream_validate` calls return
    non-`"fail"` for this chunk and the chunk has been placed on the
    consumer queue.

    Args:
        text: The chunk text that was validated and emitted.
        chunk_index: Zero-based position of this chunk in the stream.
        attempt: Sampling attempt number (always `1` in v1).
    """

    text: str
    chunk_index: int
    attempt: int


@dataclass
class QuickCheckEvent(StreamEvent):
    """Emitted after each per-chunk streaming validation batch.

    One event per chunk, covering all active requirements in parallel.
    Not emitted when there are no `requirements`.

    Args:
        chunk_index: Zero-based position of the chunk that was validated.
        attempt: Sampling attempt number (always `1` in v1).
        passed: `True` if all active requirements returned non-`"fail"`
            for this chunk.
        results: :class:`~mellea.core.requirement.PartialValidationResult`
            from each active requirement, in the same order as the active
            slice of `requirements`.
    """

    chunk_index: int
    attempt: int
    passed: bool
    results: list[PartialValidationResult]


@dataclass
class StreamingDoneEvent(StreamEvent):
    """Emitted after all chunks have been validated and delivered to the consumer.

    Fired after the regular token stream and any trailing fragment released by
    :meth:`~mellea.stdlib.chunking.ChunkingStrategy.flush` have both been
    processed.  Only emitted on natural completion — not on early exit (a
    requirement returned `"fail"`) or on exception.

    Args:
        attempt: Sampling attempt number (always `1` in v1).
        full_text: Complete accumulated text at stream end.
    """

    attempt: int
    full_text: str


@dataclass
class FullValidationEvent(StreamEvent):
    """Emitted after the final :meth:`~mellea.core.requirement.Requirement.validate` calls complete.

    Only emitted when at least one requirement did not fail during streaming
    and the stream completed naturally.  Not emitted on early exit.

    Args:
        attempt: Sampling attempt number (always `1` in v1).
        passed: `True` if all final
            :class:`~mellea.core.requirement.ValidationResult` objects passed.
        results: :class:`~mellea.core.requirement.ValidationResult` from each
            non-failed requirement, in requirement order.
    """

    attempt: int
    passed: bool
    results: list[ValidationResult]


@dataclass
class RetryEvent(StreamEvent):
    """Reserved for future use.

    Defined for API completeness — `RetryEvent` is not emitted by the
    v1 orchestrator because v1 retry is caller-driven re-invocation of
    :func:`stream_with_chunking`.  When orchestrator-side retry is added,
    this event will fire before each re-attempt.

    Args:
        attempt: Attempt number being started (1-based).
        reason: Human-readable reason for the retry.
    """

    attempt: int
    reason: str


@dataclass
class CompletedEvent(StreamEvent):
    """Emitted when the orchestrator exits, including early-exit cases.

    Always the last event before :meth:`StreamChunkingResult.events`
    terminates.  `success` reflects :attr:`StreamChunkingResult.completed`.

    Args:
        success: `True` if the stream completed normally (no `"fail"`
            result and no unhandled exception); `False` otherwise.
        full_text: Complete accumulated text.  On early exit or exception,
            reflects whatever was accumulated before cancellation.
        attempts_used: Number of orchestrator invocations (always `1` in v1).
    """

    success: bool
    full_text: str
    attempts_used: int


@dataclass
class ErrorEvent(StreamEvent):
    """Emitted when an unhandled exception occurs in the orchestrator.

    Args:
        exception_type: Python class name of the exception
            (e.g. `"ValueError"`).
        detail: String representation of the exception.  If
            `cancel_generation()` also raised during cleanup, the cleanup
            error is appended.
    """

    exception_type: str
    detail: str


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


class StreamChunkingResult:
    """Result of a :func:`stream_with_chunking` operation.

    Provides async iteration over validated text chunks as they complete
    (:meth:`astream`), typed :class:`StreamEvent` objects via :meth:`events`,
    a blocking :meth:`acomplete` for awaiting the full result including final
    validation, and :attr:`as_thunk` for wrapping the output as a
    :class:`~mellea.core.base.ModelOutputThunk`.

    Instances are created by :func:`stream_with_chunking`; do not instantiate
    directly.

    Args:
        mot: The :class:`~mellea.core.base.ModelOutputThunk` from the backend
            generation call.
        ctx: The generation context returned alongside the MOT.
        streaming_id: UUID correlating the streaming hooks for this run.

    Attributes:
        completed: `False` if the stream exited early because a requirement
            returned `"fail"` during streaming; `True` otherwise.
        full_text: The generated text available after streaming completes.
            On natural completion, the full accumulated text.  On early exit
            (a requirement returned `"fail"`), only the validated and emitted
            portion — i.e. what consumers received via :meth:`astream`.
            Available after :meth:`acomplete` returns.
        final_validations: :class:`~mellea.core.requirement.ValidationResult`
            objects from the final :meth:`~mellea.core.requirement.Requirement.validate`
            calls on all non-failed requirements.  Available after
            :meth:`acomplete` returns.
        streaming_failures: `(Requirement, PartialValidationResult)` pairs
            for every requirement that returned `"fail"` during streaming.
    """

    def __init__(self, mot: ModelOutputThunk, ctx: Context, streaming_id: str) -> None:
        """Initialise with the MOT and context from the backend call."""
        self._mot = mot
        self._ctx = ctx
        self._streaming_id = streaming_id
        self._chunk_queue: asyncio.Queue[str | None | Exception] = asyncio.Queue()
        # If no consumer calls events(), events accumulate in this queue until
        # the result object is garbage-collected.  That is intentional — event
        # production is unconditional; consumption is opt-in.
        self._event_queue: asyncio.Queue[StreamEvent | None] = asyncio.Queue()
        self._orchestration_task: asyncio.Task[None] | None = None
        self._done = asyncio.Event()
        # Set synchronously at the very top of _orchestrate_streaming, before
        # any await, so external coordinators (e.g. cancellation tests) can wait
        # until the task is live and suspended at its first I/O point.
        # Single-use: not reset between runs; this object is not re-entrant.
        self._orchestration_started = asyncio.Event()
        # Stashed so acomplete() surfaces orchestrator failures even when the
        # consumer never iterates astream(). Cleared once consumed by
        # whichever of the two reads it first.
        self._orchestration_exception: BaseException | None = None
        # Tracks whether the exception has already been surfaced to the caller
        # (by astream OR acomplete). A separate flag rather than reusing the
        # stash slot avoids the race where acomplete() clears the stash, a
        # subsequent astream() dequeues the exception item, sees the stash is
        # None, and silently skips it — leaving the caller with zero chunks
        # and no error.
        self._exception_surfaced: bool = False
        self._events_consumed: bool = False
        # Outcome for the streaming_end hook. Recorded by the orchestrator;
        # kept separate from `_orchestration_exception`, which raise-once clears.
        self._streaming_failure_reason: str | None = None
        self._streaming_end_exception: Exception | None = None
        # streaming_end fires once even if acomplete() is called repeatedly.
        self._streaming_end_fired: bool = False

        self.completed: bool = True
        self.full_text: str = ""
        self.final_validations: list[ValidationResult] = []
        self.streaming_failures: list[tuple[Requirement, PartialValidationResult]] = []

    async def astream(self) -> AsyncIterator[str]:
        """Yield validated text chunks as they complete.

        Each yielded string is a chunk that has passed per-chunk streaming
        validation (or the stream had no requirements).  Iteration ends when
        all chunks have been yielded, whether the stream completed normally or
        was cancelled early on a `"fail"` result.

        **Single-consumer.** Chunks are delivered via an
        :class:`asyncio.Queue` that this method drains; calling
        `astream()` a second time on the same result blocks indefinitely
        because the queue is empty and the terminating `None` sentinel
        has already been consumed.  If you need the chunks after
        iteration, capture them into a list during the first pass or use
        :attr:`full_text` after :meth:`acomplete`.

        Yields:
            str: A validated text chunk from the chunking strategy.

        Raises:
            Exception: Propagates any error from the background orchestration
                task.

        Note:
            Draining `astream()` does not finalize the run; `full_text`,
            `final_validations`, `completed`, and the `STREAMING_END` hook are
            all driven by `acomplete()`. Call `acomplete()` after iterating.
        """
        while True:
            item = await self._chunk_queue.get()
            if item is None:
                return
            if isinstance(item, Exception):
                if self._exception_surfaced:
                    # Already surfaced by acomplete(); don't raise twice.
                    continue
                self._exception_surfaced = True
                self._orchestration_exception = None
                raise item
            yield item

    async def events(self) -> AsyncIterator[StreamEvent]:
        """Yield typed streaming events as they are emitted by the orchestrator.

        Each yielded object is a :class:`StreamEvent` subclass describing a
        point in the orchestration lifecycle.  Consumers can dispatch on type:

        .. code-block:: python

            async for event in result.events():
                match event:
                    case ChunkEvent():
                        print(f"chunk {event.chunk_index}: {event.text!r}")
                    case QuickCheckEvent(passed=False):
                        print(f"chunk {event.chunk_index} failed validation")
                    case CompletedEvent():
                        print(f"done — success={event.success}")

        Typical event order (natural completion with requirements):

        1. :class:`QuickCheckEvent` / :class:`ChunkEvent` pairs, one per chunk
           (validation fires first; the chunk is released to the consumer only
           after passing).  Includes any trailing fragment released by the
           chunking strategy's `flush()` method.
        2. :class:`StreamingDoneEvent` — all chunks (including flush) delivered.
        3. :class:`FullValidationEvent` — final `validate()` calls returned.
        4. :class:`CompletedEvent` — orchestrator is exiting.

        On early exit: :class:`QuickCheckEvent` (`passed=False`) is the
        last validation event, followed by :class:`CompletedEvent`.  No
        :class:`StreamingDoneEvent` or :class:`FullValidationEvent` is emitted.

        On exception: :class:`ErrorEvent` followed by :class:`CompletedEvent`.

        **Single-consumer.**  Events are delivered via a queue that this method
        drains; calling `events()` a second time raises :exc:`RuntimeError`.

        Yields:
            StreamEvent: A typed event from the orchestrator.

        Raises:
            RuntimeError: If called more than once on the same result.

        Note:
            `events()` itself never raises from the event stream.  If the
            orchestrator encounters an unhandled exception, an
            :class:`ErrorEvent` is emitted and iteration ends normally.
            Exceptions surface to the caller via :meth:`astream` (as a
            re-raised exception) or :meth:`acomplete`.
        """
        if self._events_consumed:
            raise RuntimeError(
                "events() is single-consumer; this iterator has already been drained"
            )
        self._events_consumed = True
        while True:
            item = await self._event_queue.get()
            if item is None:
                return
            yield item

    async def acomplete(self) -> None:
        """Await full completion, including final validation.

        After this method returns, :attr:`full_text`, :attr:`completed`,
        :attr:`final_validations`, and :attr:`streaming_failures` are all
        populated.  If :meth:`astream` has already been consumed to
        exhaustion, this call is effectively a no-op.

        Raises:
            Exception: Propagates the orchestrator exception if :meth:`astream`
                has not yet consumed it (raise-once — only one of `astream`
                or `acomplete` raises, whichever drains the failure marker
                first).
            asyncio.CancelledError: If the orchestration task was externally
                cancelled (e.g. via :func:`asyncio.wait_for` timeout).
        """
        await self._done.wait()

        if not self._streaming_end_fired:
            self._streaming_end_fired = True
            if has_plugins(HookType.STREAMING_END):
                from ..plugins.hooks.streaming import StreamingEndPayload

                await invoke_hook(
                    HookType.STREAMING_END,
                    StreamingEndPayload(
                        streaming_id=self._streaming_id,
                        success=self.completed,
                        failure_reason=self._streaming_failure_reason,
                        exception=self._streaming_end_exception,
                        model=self._mot.generation.model,
                        provider=self._mot.generation.provider,
                        full_text_length=len(self.full_text),
                    ),
                )

        # Raise-once: if astream() already surfaced the exception, skip.
        exc = self._orchestration_exception
        if exc is not None and not self._exception_surfaced:
            self._exception_surfaced = True
            self._orchestration_exception = None
            raise exc
        if self._orchestration_task is not None and self._orchestration_task.done():
            # Raise-once: a prior call already surfaced the exception.
            if self._exception_surfaced:
                return
            # `task.exception()` raises CancelledError on a cancelled task
            # (rather than returning it), so check cancelled status first.
            # This branch covers BaseException paths that bypass the
            # `except Exception` handler in `_orchestrate_streaming`.
            if self._orchestration_task.cancelled():
                self._exception_surfaced = True
                raise asyncio.CancelledError()
            task_exc = self._orchestration_task.exception()
            if task_exc is not None:
                self._exception_surfaced = True
                raise task_exc

    @property
    def as_thunk(self) -> ModelOutputThunk[str]:
        """Wrap the output as a computed :class:`~mellea.core.base.ModelOutputThunk`.

        Returns a new thunk with `value` set to :attr:`full_text` and
        generation metadata copied from the original MOT.  Safe to call on
        early-exit results; `value` reflects the validated and emitted
        portion (same as :attr:`full_text` — see its docstring).

        Note:
            On early exit, `cancel_generation()` forces the MOT into a
            computed state without running the backend's
            `post_processing()`.  `value` and `streaming` are
            reliable.  `parsed_repr` is set to the raw text (same as
            `value`) — consistent with normal completion for plain-text
            outputs, but for typed outputs the backend-parsed representation
            will not be available.  Telemetry fields (`generation.usage`,
            `generation.ttfb_ms`, etc.) may be `None` or reflect the
            partial state at cancellation time; usage totals are not
            recoverable.

        Returns:
            ModelOutputThunk[str]: A computed thunk containing the streamed output.

        Raises:
            RuntimeError: If called before :meth:`acomplete` has returned.
        """
        if not self._done.is_set():
            raise RuntimeError(
                "as_thunk accessed before acomplete() — await acomplete() first"
            )
        thunk = ModelOutputThunk(value=self.full_text)
        thunk._cancelled = self._mot.cancelled
        thunk.generation = copy(self._mot.generation)
        thunk.parsed_repr = thunk.value  # type: ignore[assignment]
        return thunk


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


async def _orchestrate_streaming(
    result: StreamChunkingResult,
    mot: ModelOutputThunk,
    ctx: Context,
    cloned_reqs: list[Requirement],
    chunking: ChunkingStrategy,
    val_backend: Backend,
) -> None:
    # Signal that the coroutine body is executing before the first suspension.
    # External coordinators waiting on _orchestration_started are guaranteed to
    # resume only after this task has yielded at its first real await, so a
    # subsequent cancel() always lands on a live, non-done task.
    result._orchestration_started.set()

    accumulated = ""
    emitted_end = 0  # byte offset in accumulated after the last emitted chunk
    prev_chunk_count = 0
    failed_indices: set[int] = set()
    early_exit = False
    chunk_index = 0

    async def _emit_event(
        ev: StreamEvent, *, requirements: list[Requirement] | None = None
    ) -> None:
        """Push *ev* on the consumer queue, then fire its streaming_event hook.

        For a `QuickCheckEvent`, *requirements* carries the active requirement
        instances in result order, so a subscriber can attribute each result to
        its requirement type.
        """
        await result._event_queue.put(ev)
        if has_plugins(HookType.STREAMING_EVENT):
            from ..plugins.hooks.streaming import StreamingEventPayload

            await invoke_hook(
                HookType.STREAMING_EVENT,
                StreamingEventPayload(
                    streaming_id=result._streaming_id,
                    event=ev,
                    requirements=requirements or [],
                ),
            )

    async def _process_chunk(c: str, ci: int) -> bool:
        """Validate *c*, emit events, push to consumer queue.

        Returns `True` if a `"fail"` was recorded (caller should
        trigger early exit), `False` if the chunk was validated and
        emitted successfully.
        """
        active = [
            (i, req) for i, req in enumerate(cloned_reqs) if i not in failed_indices
        ]
        pvrs: list[PartialValidationResult] = []
        if active:
            pvrs = list(
                await asyncio.gather(
                    *[
                        req.stream_validate(c, backend=val_backend, ctx=ctx)
                        for _, req in active
                    ]
                )
            )
            for (idx, req), pvr in zip(active, pvrs):
                if pvr.success == "fail":
                    failed_indices.add(idx)
                    result.streaming_failures.append((req, pvr))

            any_fail = any(pvr.success == "fail" for pvr in pvrs)
            await _emit_event(
                QuickCheckEvent(
                    chunk_index=ci, attempt=1, passed=not any_fail, results=pvrs
                ),
                requirements=[req for _, req in active],
            )

            if failed_indices:
                return True

        await result._chunk_queue.put(c)
        await _emit_event(ChunkEvent(text=c, chunk_index=ci, attempt=1))
        return False

    try:
        # Inside the try so a cancellation at this await still runs the finally.
        if has_plugins(HookType.STREAMING_ORCHESTRATION_START):
            from ..plugins.hooks.streaming import StreamingOrchestrationStartPayload

            await invoke_hook(
                HookType.STREAMING_ORCHESTRATION_START,
                StreamingOrchestrationStartPayload(streaming_id=result._streaming_id),
            )

        while not mot.is_computed():
            try:
                delta = await mot.astream()
            except RuntimeError:
                # Expected race: mot.is_computed() was False at the top of the
                # loop but the stream finished before we re-entered astream().
                # Any other RuntimeError is a real bug and must propagate.
                if mot.is_computed():
                    break
                raise

            accumulated += delta
            chunks = chunking.split(accumulated)
            new_chunks = chunks[prev_chunk_count:]
            prev_chunk_count = len(chunks)

            for c in new_chunks:
                failed = await _process_chunk(c, chunk_index)
                if failed:
                    early_exit = True
                    result.completed = False
                    await mot.cancel_generation()
                    reason = result.streaming_failures[-1][1].reason or ""
                    result._streaming_failure_reason = (
                        f"Streaming validation failed: {reason}"
                    )
                    break
                pos = accumulated.find(c, emitted_end)
                if pos >= 0:
                    emitted_end = pos + len(c)
                chunk_index += 1

            if early_exit:
                break

        # Stream ended naturally: flush any withheld trailing fragment, then
        # emit StreamingDoneEvent once all chunks (regular + flush) have been
        # validated and delivered.  If a flush chunk fails, early_exit is
        # set and StreamingDoneEvent is suppressed (same contract as the
        # regular early-exit path).  Skipped entirely on early exit.
        if not early_exit:
            for c in chunking.flush(accumulated):
                failed = await _process_chunk(c, chunk_index)
                if failed:
                    early_exit = True
                    result.completed = False
                    reason = result.streaming_failures[-1][1].reason or ""
                    result._streaming_failure_reason = (
                        f"Streaming validation failed on flush: {reason}"
                    )
                    break
                pos = accumulated.find(c, emitted_end)
                if pos >= 0:
                    emitted_end = pos + len(c)
                chunk_index += 1

            if not early_exit:
                await _emit_event(StreamingDoneEvent(attempt=1, full_text=accumulated))

        # On early exit, full_text is the portion of accumulated that was
        # actually validated and emitted to the consumer. On natural
        # completion, the full accumulated text is used.
        result.full_text = accumulated[:emitted_end] if early_exit else accumulated

        if not early_exit:
            non_failed = [
                req for i, req in enumerate(cloned_reqs) if i not in failed_indices
            ]
            if non_failed:
                vrs: list[ValidationResult] = list(
                    await asyncio.gather(
                        *[req.validate(val_backend, ctx) for req in non_failed]
                    )
                )
                result.final_validations = vrs
                all_passed = all(vr.as_bool() for vr in vrs)
                await _emit_event(
                    FullValidationEvent(attempt=1, passed=all_passed, results=list(vrs))
                )

    except Exception as exc:
        # Stash the exception before any await so acomplete() can always
        # surface it even if a subsequent await is interrupted by an
        # external CancelledError.
        result._orchestration_exception = exc
        result._streaming_end_exception = exc
        # Mark as failed immediately — before any event is enqueued — so
        # that CompletedEvent.success and result.completed are consistent
        # if the consumer observes them during ErrorEvent processing.
        result.completed = False
        result.full_text = accumulated  # best-effort partial capture
        # Only cancel generation if the stream hasn't already completed
        # (e.g. an exception from the final validate() call arrives after
        # the token stream ended naturally — cancelling an already-computed
        # MOT is a no-op at best and misleading in telemetry).
        if not mot.is_computed():
            try:
                await mot.cancel_generation(error=exc)
                error_detail = str(exc)
            except Exception as cleanup_exc:
                # Never let cleanup mask the original exception.
                error_detail = f"{exc!r} (cancel cleanup raised: {cleanup_exc!r})"
                MelleaLogger.get_logger().debug(
                    "stream_with_chunking: cancel_generation() raised during "
                    "exception cleanup (original: %r, cleanup: %r)",
                    exc,
                    cleanup_exc,
                )
        else:
            error_detail = str(exc)
        await _emit_event(
            ErrorEvent(exception_type=type(exc).__name__, detail=error_detail)
        )
        await result._chunk_queue.put(exc)
    finally:
        # CancelledError (BaseException, not Exception) bypasses the except
        # block above, so cancel_generation() may not have been called.
        # Guard here ensures the backend producer is always stopped, even on
        # external task cancellation (e.g. asyncio.wait_for timeout).
        # Also mark completion as failed for any BaseException path (e.g.
        # CancelledError) that bypassed the except block — otherwise
        # result.completed stays True and CompletedEvent / metrics lie.
        if not mot.is_computed():
            result.completed = False
            try:
                await mot.cancel_generation()
            except BaseException:
                pass

        # Shielded so a CancelledError from the hook cannot skip the terminal
        # queue bookkeeping and _done.set() below.
        if has_plugins(HookType.STREAMING_ORCHESTRATION_END):
            from ..plugins.hooks.streaming import StreamingOrchestrationEndPayload

            try:
                await invoke_hook(
                    HookType.STREAMING_ORCHESTRATION_END,
                    StreamingOrchestrationEndPayload(streaming_id=result._streaming_id),
                )
            except BaseException:
                pass
            # STREAMING_END is not fired here: it pairs with STREAMING_START
            # (caller's task, pre-spawn) and a subscriber may hold task-affine
            # state, so teardown must run on the caller's task — acomplete().

        completed_ev = CompletedEvent(
            success=result.completed, full_text=result.full_text, attempts_used=1
        )
        # Use put_nowait for the terminal bookkeeping: both queues are
        # unbounded so this can never raise QueueFull, and it eliminates
        # the await points that could be interrupted by a pending
        # CancelledError before _done.set() runs.
        result._event_queue.put_nowait(completed_ev)
        result._chunk_queue.put_nowait(None)
        result._event_queue.put_nowait(None)
        result._done.set()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def stream_with_chunking(
    action: Component[Any] | CBlock,
    backend: Backend,
    ctx: Context,
    *,
    requirements: Sequence[Requirement] | None = None,
    chunking: str | ChunkingStrategy = "sentence",
    validation_backend: Backend | None = None,
) -> StreamChunkingResult:
    """Generate a streaming response with per-chunk validation.

    Starts a backend generation with streaming enabled, consumes the
    :class:`~mellea.core.base.ModelOutputThunk`'s async stream in a single
    background task, splits the accumulated text using *chunking*, and runs
    :meth:`~mellea.core.requirement.Requirement.stream_validate` on each new
    chunk in parallel across all requirements.

    For each new complete chunk produced by the chunking strategy,
    `stream_validate` is called once per active requirement (in parallel
    via :func:`asyncio.gather`), receiving that single chunk.  Multiple
    chunks produced from one `astream()` iteration are validated
    sequentially in order, so early exit on a `"fail"` result prevents
    later chunks in the same batch from being validated or emitted to the
    consumer.

    If any requirement returns `"fail"`, the generation is cancelled
    immediately (via
    :meth:`~mellea.core.base.ModelOutputThunk.cancel_generation`) and
    :attr:`StreamChunkingResult.completed` is set to `False`.  The
    failing chunk is not emitted to the consumer; use
    :attr:`StreamChunkingResult.streaming_failures` to inspect what failed.

    When the stream ends naturally, any trailing fragment withheld by the
    chunking strategy (see :meth:`~mellea.stdlib.chunking.ChunkingStrategy.flush`)
    is released as a final chunk and run through `stream_validate` on the
    same terms as the regular chunks.  On early exit, the trailing fragment
    is discarded because the generation was cancelled mid-token.

    After the stream ends naturally, `validate()` is called on every
    requirement that did not return `"fail"` — both `"pass"` and
    `"unknown"` trigger final validation.  On early exit, no `validate()`
    call is made; :attr:`StreamChunkingResult.final_validations` remains
    empty.  Requirements are cloned (`copy(req)`) before backend generation
    begins, so the originals are never mutated and a raising `__copy__`
    cannot leak an in-flight backend task.

    The orchestrator emits typed :class:`StreamEvent` objects throughout
    execution.  Consume them via :meth:`StreamChunkingResult.events` in
    parallel with or instead of :meth:`StreamChunkingResult.astream`.

    Requirements that need context beyond the current chunk should
    accumulate it themselves across `stream_validate` calls (e.g.
    `self._seen = self._seen + chunk`).  They must not read `mot.astream()`
    directly — this orchestrator is the single consumer of the MOT stream.

    Note:
        Chunks are emitted to the consumer (via
        :meth:`StreamChunkingResult.astream`) only after every requirement's
        `stream_validate` has returned for that chunk.  A slow validator
        (for example, one that invokes an LLM) therefore adds latency to
        every chunk — the consumer sees a chunk at most as quickly as the
        slowest active validator.  This trade is deliberate in v1: it
        preserves the invariant that the consumer never sees content that
        has not been validated, which matters for UIs displaying generated
        text live.  A future fast-path mode that emits chunks to the
        consumer concurrently with validation (at the cost of that
        invariant) may be added if a concrete use case calls for it.

    Note:
        v1 retry is simple re-invocation of this function.  Plugin hooks
        (`SAMPLING_LOOP_START`, `SAMPLING_REPAIR`, etc.) do not fire
        during streaming — use :meth:`StreamChunkingResult.events` for
        observability instead.

    Args:
        action: The component or content block to generate from.
        backend: The backend used for generation and final validation.
        ctx: The generation context.
        requirements: Sequence of requirements to validate against each chunk
            during streaming.  `None` disables streaming validation (chunks
            are still produced; `validate()` is not called at stream end).
        chunking: Chunking strategy — either a :class:`~mellea.stdlib.chunking.ChunkingStrategy`
            instance or one of the string aliases `"sentence"` (default),
            `"word"`, or `"paragraph"`.
        validation_backend: Optional alternate backend for both
            `stream_validate` and final `validate` calls.  When `None`,
            *backend* is used for validation.

    Returns:
        StreamChunkingResult: A result object providing :meth:`~StreamChunkingResult.astream`
            for incremental chunk consumption, :meth:`~StreamChunkingResult.events` for
            typed streaming events, and
            :meth:`~StreamChunkingResult.acomplete` for blocking until done.

    Raises:
        ValueError: If *chunking* is a string that does not match any known
            alias (`"sentence"`, `"word"`, `"paragraph"`).
        RuntimeError: If the backend returns an already-computed
            :class:`~mellea.core.base.ModelOutputThunk` instead of a streaming
            one.  This indicates the backend is not honouring
            `ModelOption.STREAM`.

    Note:
        Any exception raised by `copy(req)` on a `requirements` entry
        propagates to the caller; no backend generation is started in that
        case.  See :class:`~mellea.core.Requirement` for the `__copy__`
        override contract.
    """
    if isinstance(chunking, str):
        cls = _CHUNKING_ALIASES.get(chunking)
        if cls is None:
            raise ValueError(
                f"Unknown chunking alias {chunking!r}. Choose from: {list(_CHUNKING_ALIASES)}"
            )
        chunking = cls()

    opts: dict[str, Any] = {ModelOption.STREAM: True}

    # Clone requirements before starting backend generation so that a raising
    # __copy__ (an advertised extension point on Requirement) cannot leave the
    # backend feeder task wedged against a full streaming queue with no consumer.
    cloned_reqs = [copy(req) for req in (requirements or [])]
    val_backend = validation_backend if validation_backend is not None else backend

    streaming_id = str(uuid.uuid4())
    if has_plugins(HookType.STREAMING_START):
        from ..plugins.hooks.streaming import StreamingStartPayload

        await invoke_hook(
            HookType.STREAMING_START,
            StreamingStartPayload(
                streaming_id=streaming_id,
                has_requirements=bool(cloned_reqs),
                requirement_count=len(cloned_reqs),
                chunking_strategy=type(chunking).__name__,
            ),
        )

    mot: ModelOutputThunk | None = None
    try:
        mot, gen_ctx = await backend.generate_from_context(
            action, ctx, model_options=opts
        )
        if mot.is_computed():
            raise RuntimeError(
                "stream_with_chunking() requires a streaming backend; the backend "
                "returned an already-computed MOT. Ensure the backend honours "
                "ModelOption.STREAM."
            )
        result = StreamChunkingResult(mot, gen_ctx, streaming_id)
        coro = _orchestrate_streaming(
            result, mot, gen_ctx, cloned_reqs, chunking, val_backend
        )
        try:
            result._orchestration_task = asyncio.create_task(coro)
        except BaseException:
            coro.close()  # prevent "coroutine was never awaited" RuntimeWarning
            raise
    except BaseException as exc:
        if mot is not None:
            try:
                await mot.cancel_generation()
            except Exception as cleanup_exc:
                MelleaLogger.get_logger().warning(
                    "stream_with_chunking: cancel_generation() raised during "
                    "setup-path cleanup (cleanup: %r)",
                    cleanup_exc,
                )
        if has_plugins(HookType.STREAMING_END):
            from ..plugins.hooks.streaming import StreamingEndPayload

            await invoke_hook(
                HookType.STREAMING_END,
                StreamingEndPayload(
                    streaming_id=streaming_id,
                    success=False,
                    exception=exc if isinstance(exc, Exception) else None,
                    model=mot.generation.model if mot is not None else None,
                    provider=mot.generation.provider if mot is not None else None,
                ),
            )
        raise

    return result
