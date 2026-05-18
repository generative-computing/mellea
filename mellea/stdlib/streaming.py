"""Streaming generation with per-chunk validation.

Provides :func:`stream_with_chunking`, the core orchestration primitive that
consumes a streaming :class:`~mellea.core.base.ModelOutputThunk`, applies a
:class:`~mellea.stdlib.chunking.ChunkingStrategy` to produce semantic chunks,
and runs :meth:`~mellea.core.requirement.Requirement.stream_validate` on each
chunk in parallel.  Higher-level streaming APIs build on this function.
"""

import asyncio
from collections.abc import AsyncIterator, Sequence
from copy import copy
from typing import Any

from ..backends.model_options import ModelOption
from ..core.backend import Backend
from ..core.base import CBlock, Component, Context, ModelOutputThunk
from ..core.requirement import PartialValidationResult, Requirement, ValidationResult
from ..core.utils import MelleaLogger
from .chunking import ChunkingStrategy, ParagraphChunker, SentenceChunker, WordChunker

_CHUNKING_ALIASES: dict[str, type[ChunkingStrategy]] = {
    "sentence": SentenceChunker,
    "word": WordChunker,
    "paragraph": ParagraphChunker,
}


class StreamChunkingResult:
    """Result of a :func:`stream_with_chunking` operation.

    Provides async iteration over validated text chunks as they complete
    (:meth:`astream`), a blocking :meth:`acomplete` for awaiting the full
    result including final validation, and :attr:`as_thunk` for wrapping the
    output as a :class:`~mellea.core.base.ModelOutputThunk`.

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
        self._orchestration_task: asyncio.Task[None] | None = None
        self._done = asyncio.Event()
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
                if self._exception_surfaced:
                    # Already surfaced by acomplete(); don't raise twice.
                    continue
                self._exception_surfaced = True
                self._orchestration_exception = None
                raise item
            yield item

    async def acomplete(self) -> None:
        """Await full completion, including final validation.

        After this method returns, :attr:`full_text`, :attr:`completed`,
        :attr:`final_validations`, and :attr:`streaming_failures` are all
        populated.  If :meth:`astream` has already been consumed to
        exhaustion, this call is effectively a no-op.

        Raises:
            Exception: Propagates any error from the orchestration task.
        """
        await self._done.wait()
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
            # ``task.exception()`` raises CancelledError on a cancelled task
            # (rather than returning it), so check cancelled status first.
            # This branch covers BaseException paths that bypass the
            # ``except Exception`` handler in ``_orchestrate_streaming``.
            if self._orchestration_task.cancelled():
                self._exception_surfaced = True
                raise asyncio.CancelledError()
            task_exc = self._orchestration_task.exception()
            if task_exc is not None:
                self._exception_surfaced = True
                raise task_exc

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
        thunk._cancelled = self._mot._cancelled
        thunk.generation = copy(self._mot.generation)
        return thunk


async def _orchestrate_streaming(
    result: StreamChunkingResult,
    mot: ModelOutputThunk,
    ctx: Context,
    cloned_reqs: list[Requirement],
    chunking: ChunkingStrategy,
    val_backend: Backend,
) -> None:
    accumulated = ""
    emitted_text = ""
    prev_chunk_count = 0
    failed_indices: set[int] = set()
    early_exit = False

    async def _validate_and_emit(c: str) -> bool:
        """Run stream_validate on chunk c across active requirements.

        Returns True if a failure was recorded (caller should early-exit),
        False otherwise (chunk was emitted to the consumer queue).
        """
        active = [
            (i, req) for i, req in enumerate(cloned_reqs) if i not in failed_indices
        ]
        if active:
            async with asyncio.TaskGroup() as tg:
                _tasks = [
                    tg.create_task(req.stream_validate(c, backend=val_backend, ctx=ctx))
                    for _, req in active
                ]
            pvrs: list[PartialValidationResult] = [t.result() for t in _tasks]
            for (idx, req), pvr in zip(active, pvrs):
                if pvr.success == "fail":
                    failed_indices.add(idx)
                    result.streaming_failures.append((req, pvr))

        if failed_indices:
            return True

        await result._chunk_queue.put(c)
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
                failed = await _validate_and_emit(c)
                if failed:
                    early_exit = True
                    result.completed = False
                    await mot.cancel_generation()
                    break
                emitted_text += c

            if early_exit:
                break  # break the while loop; cancel_generation() already set _computed=True

        # Stream ended naturally: flush any withheld trailing fragment and
        # run stream_validate on it. Skipped on early exit — the generation
        # was cancelled, the trailing fragment is incomplete.
        if not early_exit:
            for c in chunking.flush(accumulated):
                failed = await _validate_and_emit(c)
                if failed:
                    early_exit = True
                    result.completed = False
                    break
                emitted_text += c

        # On early exit, full_text reflects only validated-and-emitted chunks
        # so it matches exactly what the consumer received via astream().
        # On natural completion emitted_text == accumulated (every character
        # ends up in some chunk or flushed fragment), so either value is
        # equivalent; accumulated is used to preserve the original raw text.
        result.full_text = emitted_text if early_exit else accumulated

        non_failed = [
            req for i, req in enumerate(cloned_reqs) if i not in failed_indices
        ]
        if non_failed and not early_exit:
            async with asyncio.TaskGroup() as tg:
                _final_tasks = [
                    tg.create_task(req.validate(val_backend, ctx)) for req in non_failed
                ]
            result.final_validations = [t.result() for t in _final_tasks]

    except Exception as exc:
        # Orchestrator is leaving — we must stop the backend producer too,
        # otherwise mot._async_queue (maxsize=20) fills and the feeder task
        # blocks indefinitely. The spec (#891, #901) calls this out for the
        # "fail" path; the same reasoning applies to any unplanned exit.
        # Pass `exc` so the backend telemetry span records the real cause
        # rather than a generic "Generation cancelled".
        # TaskGroup wraps failures in ExceptionGroup; unwrap so telemetry and
        # the chunk queue see the original exception, not the wrapper.
        # ExceptionGroup (not BaseExceptionGroup) guarantees Exception elements.
        if isinstance(exc, ExceptionGroup) and exc.exceptions:
            reported_exc: Exception = exc.exceptions[0]
            if len(exc.exceptions) > 1:
                MelleaLogger.get_logger().warning(
                    "stream_with_chunking: %d validator(s) failed simultaneously; "
                    "reporting first, suppressing rest: %r",
                    len(exc.exceptions) - 1,
                    exc.exceptions[1:],
                )
        else:
            reported_exc = exc
        try:
            await mot.cancel_generation(error=reported_exc)
        except Exception as cleanup_exc:
            # Never let cleanup mask the original exception: log loudly and
            # continue to surface `exc` to the consumer.
            # TODO(#902): replace this log with an ErrorEvent emission.
            MelleaLogger.get_logger().warning(
                "stream_with_chunking: cancel_generation() raised during "
                "exception cleanup (original: %r, cleanup: %r)",
                reported_exc,
                cleanup_exc,
            )
        result.completed = False
        result._orchestration_exception = reported_exc
        await result._chunk_queue.put(reported_exc)
    finally:
        # CancelledError (BaseException, not Exception) bypasses the except
        # block above, so cancel_generation() may not have been called.
        # Catch only Exception here so CancelledError / KeyboardInterrupt /
        # SystemExit still propagate to the caller.
        if not mot.is_computed():
            try:
                await mot.cancel_generation()
            except Exception:
                pass
        # put_nowait + set() are synchronous — no await point, so they cannot
        # be interrupted by task cancellation. Consumers waiting on
        # _done.wait() are always released, even if the task was cancelled
        # mid-cleanup. The queue is unbounded, so QueueFull cannot occur.
        try:
            result._chunk_queue.put_nowait(None)
        except asyncio.QueueFull:
            pass
        result._done.set()


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
    empty.  Requirements are cloned (``copy(req)``) before backend generation
    begins, so the originals are never mutated and a raising ``__copy__``
    cannot leak an in-flight backend task.

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
        on retries — use the ``#902`` event types for observability instead.

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
            for incremental chunk consumption and
            :meth:`~StreamChunkingResult.acomplete` for blocking until done.

    Raises:
        ValueError: If *chunking* is a string that does not match any known
            alias (``"sentence"``, ``"word"``, ``"paragraph"``).
        RuntimeError: If the backend returns an already-computed
            :class:`~mellea.core.base.ModelOutputThunk` instead of a streaming
            one.  This indicates the backend is not honouring
            ``ModelOption.STREAM``.

    Note:
        Any exception raised by ``copy(req)`` on a ``quick_check_requirements``
        entry propagates to the caller; no backend generation is started in
        that case.  See :class:`~mellea.core.Requirement` for the ``__copy__``
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
    # backend feeder task wedged against a full _async_queue with no consumer.
    cloned_reqs = [copy(req) for req in (quick_check_requirements or [])]
    val_backend = quick_check_backend if quick_check_backend is not None else backend

    mot, gen_ctx = await backend.generate_from_context(action, ctx, model_options=opts)
    if mot.is_computed():
        raise RuntimeError(
            "stream_with_chunking() requires a streaming backend; the backend returned "
            "an already-computed MOT. Ensure the backend honours ModelOption.STREAM."
        )
    try:
        result = StreamChunkingResult(mot, gen_ctx)
        coro = _orchestrate_streaming(
            result, mot, gen_ctx, cloned_reqs, chunking, val_backend
        )
        try:
            result._orchestration_task = asyncio.create_task(coro)
        except BaseException:
            coro.close()  # prevent "coroutine was never awaited" RuntimeWarning
            raise
    except BaseException:
        try:
            await mot.cancel_generation()
        except Exception as cleanup_exc:
            MelleaLogger.get_logger().warning(
                "stream_with_chunking: cancel_generation() raised during "
                "setup-path cleanup (cleanup: %r)",
                cleanup_exc,
            )
        raise

    return result
