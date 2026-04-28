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

    async def _validate_and_emit(c: str) -> bool:
        """Run stream_validate on chunk c across active requirements.

        Returns True if a failure was recorded (caller should early-exit),
        False otherwise (chunk was emitted to the consumer queue).
        """
        active = [
            (i, req) for i, req in enumerate(cloned_reqs) if i not in failed_indices
        ]
        if active:
            pvrs: list[PartialValidationResult] = list(
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

            if early_exit:
                break

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

        result.full_text = accumulated

        non_failed = [
            req for i, req in enumerate(cloned_reqs) if i not in failed_indices
        ]
        if non_failed and not early_exit:
            result.final_validations = list(
                await asyncio.gather(
                    *[req.validate(val_backend, ctx) for req in non_failed]
                )
            )

    except Exception as exc:
        # Orchestrator is leaving — we must stop the backend producer too,
        # otherwise mot._async_queue (maxsize=20) fills and the feeder task
        # blocks indefinitely. The spec (#891, #901) calls this out for the
        # "fail" path; the same reasoning applies to any unplanned exit.
        try:
            await mot.cancel_generation()
        except Exception as cleanup_exc:
            # Never let cleanup mask the original exception: log loudly and
            # continue to surface `exc` to the consumer.
            # TODO(#902): replace this log with an ErrorEvent emission.
            MelleaLogger.get_logger().warning(
                "stream_with_chunking: cancel_generation() raised during "
                "exception cleanup (original: %r, cleanup: %r)",
                exc,
                cleanup_exc,
            )
        result.completed = False
        await result._chunk_queue.put(exc)
    finally:
        await result._chunk_queue.put(None)
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

    After the stream ends (naturally or via early exit), ``validate()`` is
    called on all requirements that did not return ``"fail"``.  Requirements
    are cloned (``copy(req)``) before use so originals are never mutated.

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
