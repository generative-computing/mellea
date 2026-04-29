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
from collections.abc import AsyncIterator, Sequence
from copy import copy
from dataclasses import dataclass, field
from typing import Any

from ..backends.model_options import ModelOption
from ..core.backend import Backend
from ..core.base import CBlock, Component, Context, ModelOutputThunk
from ..core.requirement import PartialValidationResult, Requirement, ValidationResult
from ..core.utils import MelleaLogger
from ..telemetry.metrics import (
    classify_error,
    record_error,
    record_requirement_check,
    record_requirement_failure,
    record_sampling_outcome,
)
from ..telemetry.tracing import set_span_error, set_span_status_error, trace_application
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

    The ``timestamp`` field is auto-populated at instantiation time; callers
    do not set it.  Subclasses that add fields **must** declare them before the
    ``timestamp`` field would conflict, and any new ``init=False`` fields must
    use ``field(... , init=False)`` so the dataclass does not expose them as
    constructor arguments.

    Attributes:
        timestamp: Unix timestamp (seconds) at the moment the event was created.
    """

    timestamp: float = field(default_factory=time.time, init=False)


@dataclass
class ChunkEvent(StreamEvent):
    """Emitted after each validated chunk is delivered to the consumer.

    Fired after all active requirements' ``stream_validate`` calls return
    non-``"fail"`` for this chunk and the chunk has been placed on the
    consumer queue.

    Args:
        text: The chunk text that was validated and emitted.
        chunk_index: Zero-based position of this chunk in the stream.
        attempt: Sampling attempt number (always ``1`` in v1).
    """

    text: str
    chunk_index: int
    attempt: int


@dataclass
class QuickCheckEvent(StreamEvent):
    """Emitted after each per-chunk streaming validation batch.

    One event per chunk, covering all active requirements in parallel.
    Not emitted when there are no ``quick_check_requirements``.

    Args:
        chunk_index: Zero-based position of the chunk that was validated.
        attempt: Sampling attempt number (always ``1`` in v1).
        passed: ``True`` if all active requirements returned non-``"fail"``
            for this chunk.
        results: :class:`~mellea.core.requirement.PartialValidationResult`
            from each active requirement, in the same order as the active
            slice of ``quick_check_requirements``.
    """

    chunk_index: int
    attempt: int
    passed: bool
    results: list[PartialValidationResult]


@dataclass
class StreamingDoneEvent(StreamEvent):
    """Emitted when the raw token stream ends, before final validation.

    Only emitted on natural stream completion.  Not emitted on early exit
    (generation was cancelled before the stream finished) or on exception.

    Args:
        attempt: Sampling attempt number (always ``1`` in v1).
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
        attempt: Sampling attempt number (always ``1`` in v1).
        passed: ``True`` if all final
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

    Defined for API completeness — ``RetryEvent`` is not emitted by the
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
    terminates.  ``success`` reflects :attr:`StreamChunkingResult.completed`.

    Args:
        success: ``True`` if the stream completed normally (no ``"fail"``
            result and no unhandled exception); ``False`` otherwise.
        full_text: Complete accumulated text.  On early exit or exception,
            reflects whatever was accumulated before cancellation.
        attempts_used: Number of orchestrator invocations (always ``1`` in v1).
    """

    success: bool
    full_text: str
    attempts_used: int


@dataclass
class ErrorEvent(StreamEvent):
    """Emitted when an unhandled exception occurs in the orchestrator.

    Args:
        exception_type: Python class name of the exception
            (e.g. ``"ValueError"``).
        detail: String representation of the exception.  If
            ``cancel_generation()`` also raised during cleanup, the cleanup
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

    Attributes:
        completed: ``False`` if the stream exited early because a requirement
            returned ``"fail"`` during streaming; ``True`` otherwise.
        full_text: The complete generated text accumulated during streaming.
            Available after :meth:`acomplete` returns.
        final_validations: :class:`~mellea.core.requirement.ValidationResult`
            objects from the final :meth:`~mellea.core.requirement.Requirement.validate`
            calls on all non-failed requirements.  Available after
            :meth:`acomplete` returns.
        streaming_failures: ``(Requirement, PartialValidationResult)`` pairs
            for every requirement that returned ``"fail"`` during streaming.
    """

    def __init__(self, mot: ModelOutputThunk, ctx: Context) -> None:
        """Initialise with the MOT and context from the backend call."""
        self._mot = mot
        self._ctx = ctx
        self._chunk_queue: asyncio.Queue[str | None | Exception] = asyncio.Queue()
        # If no consumer calls events(), events accumulate in this queue until
        # the result object is garbage-collected.  That is intentional — event
        # production is unconditional; consumption is opt-in.
        self._event_queue: asyncio.Queue[StreamEvent | None] = asyncio.Queue()
        self._orchestration_task: asyncio.Task[None] | None = None
        self._done = asyncio.Event()

        self.completed: bool = True
        self.full_text: str = ""
        self.final_validations: list[ValidationResult] = []
        self.streaming_failures: list[tuple[Requirement, PartialValidationResult]] = []

    async def astream(self) -> AsyncIterator[str]:
        """Yield validated text chunks as they complete.

        Each yielded string is a chunk that has passed per-chunk streaming
        validation (or the stream had no requirements).  Iteration ends when
        all chunks have been yielded, whether the stream completed normally or
        was cancelled early on a ``"fail"`` result.

        **Single-consumer.** Chunks are delivered via an
        :class:`asyncio.Queue` that this method drains; calling
        ``astream()`` a second time on the same result blocks indefinitely
        because the queue is empty and the terminating ``None`` sentinel
        has already been consumed.  If you need the chunks after
        iteration, capture them into a list during the first pass or use
        :attr:`full_text` after :meth:`acomplete`.

        Yields:
            str: A validated text chunk from the chunking strategy.

        Raises:
            Exception: Propagates any error from the background orchestration
                task.
        """
        while True:
            item = await self._chunk_queue.get()
            if item is None:
                return
            if isinstance(item, Exception):
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

        1. :class:`ChunkEvent` / :class:`QuickCheckEvent` pairs, one per chunk.
        2. :class:`StreamingDoneEvent` — raw token stream has ended.
        3. :class:`FullValidationEvent` — final ``validate()`` calls returned.
        4. :class:`CompletedEvent` — orchestrator is exiting.

        On early exit: :class:`QuickCheckEvent` (``passed=False``) is the
        last validation event, followed by :class:`CompletedEvent`.  No
        :class:`StreamingDoneEvent` or :class:`FullValidationEvent` is emitted.

        On exception: :class:`ErrorEvent` followed by :class:`CompletedEvent`.

        **Single-consumer.**  Events are delivered via a queue that this method
        drains; a second call after the first iteration completes blocks
        indefinitely.

        Yields:
            StreamEvent: A typed event from the orchestrator.

        Note:
            ``events()`` itself never raises.  If the orchestrator encounters
            an unhandled exception, an :class:`ErrorEvent` is emitted and
            iteration ends normally.  Exceptions surface to the caller via
            :meth:`astream` (as a re-raised exception) or :meth:`acomplete`.
        """
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
            BaseException: Propagates any :class:`BaseException` that escaped
                the orchestration task entirely (e.g. ``KeyboardInterrupt``).
                Ordinary :class:`Exception` types are caught by the orchestrator,
                surfaced as :class:`ErrorEvent` objects, and re-raised to
                :meth:`astream` consumers — they do **not** propagate here.
        """
        await self._done.wait()
        if self._orchestration_task is not None and self._orchestration_task.done():
            exc = self._orchestration_task.exception()
            if exc is not None:
                raise exc

    @property
    def as_thunk(self) -> ModelOutputThunk:
        """Wrap the output as a computed :class:`~mellea.core.base.ModelOutputThunk`.

        Returns a new thunk with ``value`` set to :attr:`full_text` and
        generation metadata copied from the original MOT.  Safe to call on
        early-exit results; ``value`` will reflect whatever was accumulated
        before cancellation.

        Note:
            On early exit, ``cancel_generation()`` forces the MOT into a
            computed state without running the backend's
            ``post_processing()``.  Telemetry fields on the returned thunk
            (``generation.usage``, ``generation.ttfb_ms``, etc.) may
            therefore be ``None`` or reflect the partial state at
            cancellation time.  ``value`` and ``streaming`` are reliable;
            usage totals are not.

        Returns:
            ModelOutputThunk: A computed thunk containing the streamed output.

        Raises:
            RuntimeError: If called before :meth:`acomplete` has returned.
        """
        if not self._done.is_set():
            raise RuntimeError(
                "as_thunk accessed before acomplete() — await acomplete() first"
            )
        thunk = ModelOutputThunk(value=self.full_text)
        thunk.generation = copy(self._mot.generation)
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
    accumulated = ""
    prev_chunk_count = 0
    failed_indices: set[int] = set()
    early_exit = False
    chunk_index = 0

    with trace_application("stream_with_chunking") as span:

        async def _process_chunk(c: str, ci: int) -> bool:
            """Validate *c*, emit events, push to consumer queue.

            Returns ``True`` if a ``"fail"`` was recorded (caller should
            trigger early exit), ``False`` if the chunk was validated and
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
                qc_event = QuickCheckEvent(
                    chunk_index=ci, attempt=1, passed=not any_fail, results=pvrs
                )
                await result._event_queue.put(qc_event)
                if span is not None:
                    span.add_event(
                        "quick_check",
                        {
                            "chunk_index": ci,
                            "passed": not any_fail,
                            "requirement_count": len(active),
                        },
                    )
                for (_, req), pvr in zip(active, pvrs):
                    record_requirement_check(type(req).__name__)
                    if pvr.success == "fail":
                        record_requirement_failure(type(req).__name__, pvr.reason or "")

                if failed_indices:
                    return True

            await result._chunk_queue.put(c)
            chunk_ev = ChunkEvent(text=c, chunk_index=ci, attempt=1)
            await result._event_queue.put(chunk_ev)
            if span is not None:
                span.add_event("chunk", {"chunk_index": ci, "text_length": len(c)})
            return False

        try:
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
                        if span is not None:
                            reason = result.streaming_failures[-1][1].reason or ""
                            set_span_status_error(
                                span, f"Streaming validation failed: {reason}"
                            )
                        break
                    chunk_index += 1

                if early_exit:
                    break

            # Stream ended naturally: flush any withheld trailing fragment.
            # Skipped on early exit — the generation was cancelled.
            if not early_exit:
                for c in chunking.flush(accumulated):
                    failed = await _process_chunk(c, chunk_index)
                    if failed:
                        early_exit = True
                        result.completed = False
                        if span is not None:
                            reason = result.streaming_failures[-1][1].reason or ""
                            set_span_status_error(
                                span, f"Streaming validation failed on flush: {reason}"
                            )
                        break
                    chunk_index += 1

            result.full_text = accumulated

            if not early_exit:
                streaming_done = StreamingDoneEvent(attempt=1, full_text=accumulated)
                await result._event_queue.put(streaming_done)
                if span is not None:
                    span.add_event(
                        "streaming_done", {"full_text_length": len(accumulated)}
                    )

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
                    full_val_ev = FullValidationEvent(
                        attempt=1, passed=all_passed, results=vrs
                    )
                    await result._event_queue.put(full_val_ev)
                    if span is not None:
                        span.add_event(
                            "full_validation",
                            {
                                "passed": all_passed,
                                "requirement_count": len(non_failed),
                            },
                        )

        except Exception as exc:
            # Mark as failed immediately — before any event is enqueued — so
            # that CompletedEvent.success and result.completed are consistent
            # if the consumer observes them during ErrorEvent processing.
            result.completed = False
            # Orchestrator is leaving — stop the backend producer.
            result.full_text = accumulated  # best-effort partial capture
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
            error_ev = ErrorEvent(
                exception_type=type(exc).__name__, detail=error_detail
            )
            await result._event_queue.put(error_ev)
            if span is not None:
                span.add_event(
                    "error",
                    {
                        "exception_type": error_ev.exception_type,
                        "detail": error_ev.detail,
                    },
                )
                set_span_error(span, exc)
            record_error(
                error_type=classify_error(exc),
                model=result._mot.generation.model or "unknown",
                provider=result._mot.generation.provider or "unknown",
                exception_class=type(exc).__name__,
            )
            await result._chunk_queue.put(exc)
        finally:
            # CancelledError (BaseException, not Exception) bypasses the except
            # block above, so cancel_generation() may not have been called.
            # Guard here ensures the backend producer is always stopped, even on
            # external task cancellation (e.g. asyncio.wait_for timeout).
            if not mot.is_computed():
                try:
                    await mot.cancel_generation()
                except BaseException:
                    pass

            completed_ev = CompletedEvent(
                success=result.completed, full_text=result.full_text, attempts_used=1
            )
            await result._event_queue.put(completed_ev)
            if span is not None:
                span.add_event(
                    "completed",
                    {
                        "success": result.completed,
                        "full_text_length": len(result.full_text),
                    },
                )
            record_sampling_outcome("stream_with_chunking", success=result.completed)

            await result._chunk_queue.put(None)
            await result._event_queue.put(None)
            result._done.set()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def stream_with_chunking(
    action: Component[Any] | CBlock,
    backend: Backend,
    ctx: Context,
    *,
    quick_check_requirements: Sequence[Requirement] | None = None,
    chunking: str | ChunkingStrategy = "sentence",
    quick_check_backend: Backend | None = None,
) -> StreamChunkingResult:
    """Generate a streaming response with per-chunk validation.

    Starts a backend generation with streaming enabled, consumes the
    :class:`~mellea.core.base.ModelOutputThunk`'s async stream in a single
    background task, splits the accumulated text using *chunking*, and runs
    :meth:`~mellea.core.requirement.Requirement.stream_validate` on each new
    chunk in parallel across all requirements.

    For each new complete chunk produced by the chunking strategy,
    ``stream_validate`` is called once per active requirement (in parallel
    via :func:`asyncio.gather`), receiving that single chunk.  Multiple
    chunks produced from one ``astream()`` iteration are validated
    sequentially in order, so early exit on a ``"fail"`` result prevents
    later chunks in the same batch from being validated or emitted to the
    consumer.

    If any requirement returns ``"fail"``, the generation is cancelled
    immediately (via
    :meth:`~mellea.core.base.ModelOutputThunk.cancel_generation`) and
    :attr:`StreamChunkingResult.completed` is set to ``False``.  The
    failing chunk is not emitted to the consumer; use
    :attr:`StreamChunkingResult.streaming_failures` to inspect what failed.

    When the stream ends naturally, any trailing fragment withheld by the
    chunking strategy (see :meth:`~mellea.stdlib.chunking.ChunkingStrategy.flush`)
    is released as a final chunk and run through ``stream_validate`` on the
    same terms as the regular chunks.  On early exit, the trailing fragment
    is discarded because the generation was cancelled mid-token.

    After the stream ends naturally, ``validate()`` is called on every
    requirement that did not return ``"fail"`` — both ``"pass"`` and
    ``"unknown"`` trigger final validation.  On early exit, no ``validate()``
    call is made; :attr:`StreamChunkingResult.final_validations` remains
    empty.  Requirements are cloned (``copy(req)``) before use so originals
    are never mutated.

    The orchestrator emits typed :class:`StreamEvent` objects throughout
    execution.  Consume them via :meth:`StreamChunkingResult.events` in
    parallel with or instead of :meth:`StreamChunkingResult.astream`.

    Requirements that need context beyond the current chunk should
    accumulate it themselves across ``stream_validate`` calls (e.g.
    ``self._seen = self._seen + chunk``).  They must not read ``mot.astream()``
    directly — this orchestrator is the single consumer of the MOT stream.

    Note:
        Chunks are emitted to the consumer (via
        :meth:`StreamChunkingResult.astream`) only after every requirement's
        ``stream_validate`` has returned for that chunk.  A slow validator
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
        (``SAMPLING_LOOP_START``, ``SAMPLING_REPAIR``, etc.) do not fire
        during streaming — use :meth:`StreamChunkingResult.events` for
        observability instead.

    Args:
        action: The component or content block to generate from.
        backend: The backend used for generation and final validation.
        ctx: The generation context.
        quick_check_requirements: Sequence of requirements to validate against
            each chunk during streaming.  ``None`` disables streaming validation
            (chunks are still produced; ``validate()`` is not called at stream end).
        chunking: Chunking strategy — either a :class:`~mellea.stdlib.chunking.ChunkingStrategy`
            instance or one of the string aliases ``"sentence"`` (default),
            ``"word"``, or ``"paragraph"``.
        quick_check_backend: Optional alternate backend for both
            ``stream_validate`` and final ``validate`` calls.  When ``None``,
            *backend* is used for validation.

    Returns:
        StreamChunkingResult: A result object providing :meth:`~StreamChunkingResult.astream`
            for incremental chunk consumption, :meth:`~StreamChunkingResult.events` for
            typed streaming events, and
            :meth:`~StreamChunkingResult.acomplete` for blocking until done.

    Raises:
        ValueError: If *chunking* is a string that does not match any known
            alias (``"sentence"``, ``"word"``, ``"paragraph"``).
    """
    if isinstance(chunking, str):
        cls = _CHUNKING_ALIASES.get(chunking)
        if cls is None:
            raise ValueError(
                f"Unknown chunking alias {chunking!r}. Choose from: {list(_CHUNKING_ALIASES)}"
            )
        chunking = cls()

    opts: dict[str, Any] = {ModelOption.STREAM: True}
    mot, gen_ctx = await backend.generate_from_context(action, ctx, model_options=opts)

    result = StreamChunkingResult(mot, gen_ctx)

    cloned_reqs = [copy(req) for req in (quick_check_requirements or [])]
    val_backend = quick_check_backend if quick_check_backend is not None else backend

    result._orchestration_task = asyncio.create_task(
        _orchestrate_streaming(result, mot, gen_ctx, cloned_reqs, chunking, val_backend)
    )

    return result
