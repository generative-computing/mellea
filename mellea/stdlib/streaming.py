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

    try:
        while not mot.is_computed():
            try:
                delta = await mot.astream()
            except RuntimeError:
                break

            accumulated += delta
            chunks = chunking.split(accumulated)
            new_chunks = chunks[prev_chunk_count:]

            if new_chunks:
                active = [
                    (i, req)
                    for i, req in enumerate(cloned_reqs)
                    if i not in failed_indices
                ]
                if active:
                    pvrs: list[PartialValidationResult] = list(
                        await asyncio.gather(
                            *[
                                req.stream_validate(
                                    accumulated, backend=val_backend, ctx=ctx
                                )
                                for _, req in active
                            ]
                        )
                    )
                    for (idx, req), pvr in zip(active, pvrs):
                        if pvr.success == "fail":
                            failed_indices.add(idx)
                            result.streaming_failures.append((req, pvr))

                if failed_indices:
                    early_exit = True
                    result.completed = False
                    await mot.cancel_generation()
                    for c in new_chunks:
                        await result._chunk_queue.put(c)
                    break

                for c in new_chunks:
                    await result._chunk_queue.put(c)
                prev_chunk_count = len(chunks)

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

    If any requirement returns ``"fail"`` during streaming validation, the
    generation is cancelled immediately (via
    :meth:`~mellea.core.base.ModelOutputThunk.cancel_generation`) and
    :attr:`StreamChunkingResult.completed` is set to ``False``.

    After the stream ends (naturally or via early exit), ``validate()`` is
    called on all requirements that did not return ``"fail"``.  Requirements
    are cloned (``copy(req)``) before use so originals are never mutated.

    ``stream_validate`` receives the *accumulated* model output so far, not
    just the current chunk.  The chunking strategy determines *when* it is
    called (at chunk boundaries).  Requirements that want delta-only
    processing track ``self._seen_len`` and slice
    ``accumulated[self._seen_len:]``.

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
