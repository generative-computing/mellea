# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Streaming generation: a single-task `async for` primitive.

`stream()` starts a streaming generation and returns a `Streamer` you consume with
`async for`. It drives token draining, chunking, and (when requirements are given)
per-chunk and final validation.

Consume inside `async with` so cleanup always runs: the stream runs on the caller's
task, and leaving the block — normally, on early `break`, or on exception —
cancels the generation and fires the `STREAMING_END` hook.

Typed `StreamEvent` objects are emitted via the `STREAMING_EVENT` hook; subscribe a
plugin to observe them (see `docs/examples/streaming/`). To iterate them directly
instead, `stream(as_events=True)` returns an `EventStreamer` that yields events in
place of chunks.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import AsyncGenerator, AsyncIterator, Sequence
from copy import copy
from dataclasses import dataclass, field
from typing import Any, Literal, overload

from ..backends.model_options import ModelOption
from ..core.backend import Backend
from ..core.base import CBlock, Component, Context, ModelOutputThunk
from ..core.requirement import PartialValidationResult, Requirement, ValidationResult
from ..plugins.manager import has_plugins, invoke_hook
from ..plugins.types import HookType
from .chunking import Chunker, ChunkingStrategy, resolve_chunking_strategy

# ---------------------------------------------------------------------------
# Streaming event types
# ---------------------------------------------------------------------------


@dataclass
class StreamEvent:
    """Base class for all streaming events emitted by `stream`.

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
    non-`"fail"` for this chunk and the chunk has been yielded to the consumer.

    Args:
        text: The chunk text that was validated and emitted.
        chunk_index: Zero-based position of this chunk in the stream.
        attempt: Sampling attempt number; currently always `1`.
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
        attempt: Sampling attempt number; currently always `1`.
        passed: `True` if all active requirements returned non-`"fail"`
            for this chunk.
        results: `PartialValidationResult` from each active requirement, in the
            same order as the active slice of `requirements`.
    """

    chunk_index: int
    attempt: int
    passed: bool
    results: list[PartialValidationResult]


@dataclass
class StreamingDoneEvent(StreamEvent):
    """Emitted after all chunks have been validated and delivered to the consumer.

    Fired after the regular token stream and any trailing fragment released by
    the chunker's `flush()` have both been processed.  Only emitted on natural
    completion — not on early exit (a requirement returned `"fail"`) or on
    exception.

    Args:
        attempt: Sampling attempt number; currently always `1`.
        full_text: Complete accumulated text at stream end.
    """

    attempt: int
    full_text: str


@dataclass
class FullValidationEvent(StreamEvent):
    """Emitted after the final `Requirement.validate` calls complete.

    Only emitted when the stream completed naturally (no requirement failed
    during streaming).  Not emitted on early exit.

    Args:
        attempt: Sampling attempt number; currently always `1`.
        passed: `True` if all final `ValidationResult` objects passed.
        results: `ValidationResult` from each requirement, in requirement order.
    """

    attempt: int
    passed: bool
    results: list[ValidationResult]


@dataclass
class RetryEvent(StreamEvent):
    """Reserved for future use.

    Defined for API completeness — `RetryEvent` is not currently emitted; today
    retry is caller-driven re-invocation of `stream`. If retry is added to
    streaming itself, this event will fire before each re-attempt.

    Args:
        attempt: Attempt number being started (1-based).
        reason: Human-readable reason for the retry.
    """

    attempt: int
    reason: str


@dataclass
class CompletedEvent(StreamEvent):
    """Emitted when the stream exits, including early-exit cases.

    Always the last `StreamEvent` on every exit path.  `success` reflects
    whether the stream completed with no `"fail"` result and no exception.

    Args:
        success: `True` if the stream completed normally (no `"fail"`
            result and no unhandled exception); `False` otherwise.
        full_text: Validated-and-emitted output.  On early exit or exception,
            reflects whatever passed validation before the stop.
        attempts_used: Number of stream attempts; currently always `1`.
    """

    success: bool
    full_text: str
    attempts_used: int


@dataclass
class ErrorEvent(StreamEvent):
    """Emitted when an unhandled exception occurs while streaming.

    Args:
        exception_type: Python class name of the exception
            (e.g. `"ValueError"`).
        detail: String representation of the exception.
    """

    exception_type: str
    detail: str


# ---------------------------------------------------------------------------
# Streamer handle
# ---------------------------------------------------------------------------


class Streamer:
    """Async-iterable handle for a `stream` call.

    Iterate the returned `Streamer` object with `async for` to receive the output
    as validated chunks, ideally inside `async with` so the stream is released on
    every exit. Each chunk is a `str` segment of the model output text, sized by the
    `chunking` strategy (or a raw model delta when `chunking` is `None`). The
    attributes below track progress and outcome. Instances are created by `stream`;
    do not instantiate directly.

    Args:
        mot: The in-flight streaming thunk from the backend generation call.
        ctx: The generation context, used for validation calls.
        chunking: Resolved chunking strategy, or `None` for raw deltas.
        requirements: Requirements to validate against; pre-copied by `stream`.
        validation_backend: Backend used for validation calls.
        event_queue: Optional queue; when set, every emitted `StreamEvent` is
            also pushed onto it. `None` for the plain chunk-iterating path.

    Attributes:
        failed_early: `True` if a requirement returned `"fail"` during streaming
            and the stream stopped before natural completion.
        completed_normally: `True` only if the stream reached its natural end,
            prior to final validation. `False` on requirement failure, an early
            `break`, or an exception — unlike `not failed_early`, which stays
            `True` after an early `break`.
        failure_reason: Human-readable reason when `failed_early` is `True`.
        streaming_failures: `(Requirement, PartialValidationResult)` pairs for
            every requirement that failed the offending chunk.
        full_text: Validated-and-emitted output. On natural completion, the full
            accumulated text; on early exit, the accumulated text through the last
            emitted chunk.
        mot: The computed thunk, set on natural completion; `None` otherwise.
        final_validations: `ValidationResult` objects from the stream-end
            `validate()` calls; empty on early exit.
        streaming_id: UUID correlating this stream's START/EVENT/END hooks.
    """

    def __init__(
        self,
        mot: ModelOutputThunk,
        ctx: Context,
        chunking: ChunkingStrategy | None,
        requirements: list[Requirement],
        validation_backend: Backend,
        streaming_id: str,
        event_queue: asyncio.Queue[StreamEvent | None] | None = None,
    ) -> None:
        """Wrap an in-flight generation; iterating the `Streamer` drives it."""
        self.failed_early: bool = False
        self.completed_normally: bool = False
        self.failure_reason: str | None = None
        self.streaming_failures: list[tuple[Requirement, PartialValidationResult]] = []
        self.full_text: str = ""
        self.mot: ModelOutputThunk | None = None
        self.final_validations: list[ValidationResult] = []
        # Correlates this stream's START/EVENT/END hooks; created in `stream()`
        # so START can fire before generation opens the backend span.
        self.streaming_id: str = streaming_id
        self._event_queue = event_queue
        # The in-flight thunk, for teardown. Held separately from the public `mot`,
        # which is only set once the stream completes.
        self._mot = mot
        self._finalized: bool = False
        self._gen: AsyncGenerator[str, None] = _drive(
            self, mot, ctx, chunking, requirements, validation_backend
        )

    def __aiter__(self) -> AsyncIterator[str]:
        """Return the generator that drives generation and yields chunks."""
        return self._gen

    async def _finalize(
        self,
        *,
        success: bool = False,
        error: Exception | None = None,
        full_text_length: int = 0,
    ) -> None:
        """Cancel the generation and fire the terminal events, at most once.

        Idempotent: the `_finalized` guard makes every call after the first a
        no-op, so callers need not coordinate. It is invoked both from the
        driver's `finally` and from `aclose()` — either may run first, or only one
        may run at all (e.g. a `Streamer` closed without being iterated) — and the
        terminal events still fire exactly once.
        """
        if self._finalized:
            return
        # Set before teardown: a cancellation mid-dispatch can curtail the remaining
        # STREAMING_END subscribers with no retry (unclosed span / unrecorded metrics).
        self._finalized = True

        try:
            # aclose() is a no-op once the stream is fully drained; cancels otherwise.
            await self._mot.aclose()
        finally:
            try:
                await _emit_event(
                    self.streaming_id,
                    CompletedEvent(
                        success=success, full_text=self.full_text, attempts_used=1
                    ),
                    event_queue=self._event_queue,
                )
            finally:
                if has_plugins(HookType.STREAMING_END):
                    from ..plugins.hooks.streaming import StreamingEndPayload

                    await invoke_hook(
                        HookType.STREAMING_END,
                        StreamingEndPayload(
                            streaming_id=self.streaming_id,
                            success=success,
                            failure_reason=self.failure_reason,
                            exception=error,
                            model=self._mot.generation.model,
                            provider=self._mot.generation.provider,
                            full_text_length=full_text_length,
                        ),
                    )

    async def aclose(self) -> None:
        """Release the stream, cancelling generation if it is still in flight.

        Runs the driver's cleanup (cancelling the backend generation and firing
        `STREAMING_END`). Safe and idempotent on every path: after natural
        completion, after an early exit/break, and on a `Streamer` that was never
        iterated — the eager generation is still cancelled in every case.

        Prefer consuming with `async with stream(...) as s:` so this runs
        automatically on every exit path; call `aclose()` directly only when not
        using the context manager.
        """
        # Closing the generator finalizes via its finally if iteration started;
        # the explicit call handles the never-iterated case.
        await self._gen.aclose()
        await self._finalize()

    async def __aenter__(self) -> Streamer:
        """Enter the async context manager, returning this `Streamer`."""
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        """Exit the context manager, releasing the stream via `aclose()`."""
        await self.aclose()


# ---------------------------------------------------------------------------
# EventStreamer handle
# ---------------------------------------------------------------------------


class EventStreamer:
    """Async-iterator handle over a stream's `StreamEvent` objects.

    Returned by `stream(..., as_events=True)`; iterate with `async for` inside
    `async with`. It is its own iterator, so `break` then iterating again
    resumes from the next queued event. Events are produced eagerly, independent
    of consumption, so the stream can run ahead of iteration; outcome attributes
    (`full_text`, `completed_normally`, …) mirror the wrapped `Streamer`. Do not
    instantiate directly.
    """

    def __init__(self) -> None:
        """Create an idle handle; `stream()` spawns the pump task."""
        self._queue: asyncio.Queue[StreamEvent | None] = asyncio.Queue()
        self._streamer: Streamer | None = None
        self._pump_task: asyncio.Task[None] | None = None
        self._ready: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        self._exhausted: bool = False  # set when the terminating None is drained

    async def _pump(
        self,
        action: Component[Any] | CBlock,
        backend: Backend,
        ctx: Context,
        *,
        chunking: str | ChunkingStrategy | None,
        requirements: Sequence[Requirement] | None,
        validation_backend: Backend | None,
    ) -> None:
        """Drive the stream to completion, funnelling its events into the queue."""
        try:
            self._streamer = await _stream(
                action,
                backend,
                ctx,
                chunking=chunking,
                requirements=requirements,
                validation_backend=validation_backend,
                event_queue=self._queue,
            )
        except BaseException as exc:
            # Resolve _ready on every setup exit so `await _ready` can't hang.
            if isinstance(exc, asyncio.CancelledError):
                self._ready.cancel()
                self._queue.put_nowait(None)
                raise
            self._ready.set_exception(exc)
            self._queue.put_nowait(None)
            return
        self._ready.set_result(None)
        try:
            async for _ in self._streamer:
                pass
        finally:
            self._queue.put_nowait(None)

    def __aiter__(self) -> AsyncIterator[StreamEvent]:
        """Return this handle as its own event iterator."""
        return self

    async def __anext__(self) -> StreamEvent:
        """Return the next queued event, stopping once the queue is drained.

        Returns:
            StreamEvent: The next event from the queue.

        Raises:
            StopAsyncIteration: Once the terminating `None` has been drained.
        """
        if self._exhausted:
            raise StopAsyncIteration
        item = await self._queue.get()
        if item is None:
            self._exhausted = True
            assert self._pump_task is not None
            # A faulted pump re-raises here; skip our own aclose() cancellation.
            if not self._pump_task.cancelled():
                await self._pump_task
            raise StopAsyncIteration
        return item

    async def __aenter__(self) -> EventStreamer:
        """Enter the async context manager, returning this `EventStreamer`."""
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        """Exit the context manager, releasing the stream via `aclose()`."""
        await self.aclose()

    async def aclose(self) -> None:
        """Cancel the pump and release the underlying stream, idempotently.

        Prefer `async with stream(..., as_events=True) as s:` so this runs
        automatically on every exit path.
        """
        if self._pump_task is not None:
            if not self._pump_task.done():
                self._pump_task.cancel()
                try:
                    await self._pump_task
                except asyncio.CancelledError:
                    # Consume the pump's own cancellation; propagate an external
                    # cancel of this task (cancelling() > 0).
                    cur = asyncio.current_task()
                    if cur is not None and cur.cancelling() > 0:
                        raise
            elif not self._pump_task.cancelled():
                # Retrieve the stored exception so it isn't "never retrieved" at GC.
                self._pump_task.exception()
        if self._streamer is not None:
            await self._streamer.aclose()

    # Outcome properties delegate to the wrapped Streamer, falling back to its
    # pre-iteration defaults until the pump has created it.

    @property
    def failed_early(self) -> bool:
        """`True` if a requirement failed during streaming and stopped it early."""
        return self._streamer.failed_early if self._streamer is not None else False

    @property
    def completed_normally(self) -> bool:
        """`True` only if the stream reached its natural end, prior to final validation."""
        return (
            self._streamer.completed_normally if self._streamer is not None else False
        )

    @property
    def failure_reason(self) -> str | None:
        """Human-readable reason when `failed_early` is `True`; else `None`."""
        return self._streamer.failure_reason if self._streamer is not None else None

    @property
    def streaming_failures(self) -> list[tuple[Requirement, PartialValidationResult]]:
        """`(Requirement, PartialValidationResult)` pairs for each failing requirement."""
        return self._streamer.streaming_failures if self._streamer is not None else []

    @property
    def full_text(self) -> str:
        """Validated-and-emitted output; the full text on natural completion."""
        return self._streamer.full_text if self._streamer is not None else ""

    @property
    def mot(self) -> ModelOutputThunk | None:
        """The computed thunk, set on natural completion; `None` otherwise."""
        return self._streamer.mot if self._streamer is not None else None

    @property
    def final_validations(self) -> list[ValidationResult]:
        """`ValidationResult` objects from the stream-end `validate()` calls."""
        return self._streamer.final_validations if self._streamer is not None else []

    @property
    def streaming_id(self) -> str | None:
        """UUID correlating this stream's START/EVENT/END hooks; `None` pre-start."""
        return self._streamer.streaming_id if self._streamer is not None else None


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


async def _emit_event(
    streaming_id: str,
    ev: StreamEvent,
    *,
    requirements: list[Requirement] | None = None,
    event_queue: asyncio.Queue[StreamEvent | None] | None = None,
) -> None:
    """Fire the STREAMING_EVENT hook for `ev`, also pushing it onto `event_queue` if set.

    For a `QuickCheckEvent`, `requirements` carries the active requirement
    instances in result order so a subscriber can attribute each result.

    Args:
        streaming_id: UUID correlating this stream's events.
        ev: The event to emit.
        requirements: Active requirements for a `QuickCheckEvent`, in result
            order; `None` for other event types.
        event_queue: Optional queue; when set, `ev` is also pushed onto it.
            `None` for the plain chunk-iterating path.
    """
    if event_queue is not None:
        event_queue.put_nowait(ev)
    if has_plugins(HookType.STREAMING_EVENT):
        from ..plugins.hooks.streaming import StreamingEventPayload

        await invoke_hook(
            HookType.STREAMING_EVENT,
            StreamingEventPayload(
                streaming_id=streaming_id, event=ev, requirements=requirements or []
            ),
        )


async def _validate_chunk(
    streamer: Streamer,
    chunk: str,
    chunk_index: int,
    requirements: list[Requirement],
    validation_backend: Backend,
    ctx: Context,
    *,
    on_flush: bool = False,
) -> bool:
    """Run every requirement's `stream_validate` on `chunk`.

    Returns `True` when `chunk` passed and may be emitted (no requirements, or
    all returned `"pass"`/`"unknown"`). Returns `False` when any requirement
    fails — every failing `(requirement, result)` is recorded on `streamer` and
    the caller should stop before yielding `chunk`. `on_flush` distinguishes a
    failure on the trailing flushed fragment (stream already ended) from a
    mid-stream one in the recorded reason.

    Args:
        streamer: The handle recording failures for the caller.
        chunk: The chunk text to validate.
        chunk_index: Zero-based position of this chunk in the stream.
        requirements: Requirements to validate against.
        validation_backend: Backend used for validation calls.
        ctx: The generation context.
        on_flush: `True` when validating the trailing flushed fragment.

    Returns:
        `True` if the chunk passed and may be emitted; `False` if it failed.
    """
    if not requirements:
        return True
    results = list(
        await asyncio.gather(
            *[
                req.stream_validate(chunk, backend=validation_backend, ctx=ctx)
                for req in requirements
            ]
        )
    )
    failures = [
        (req, r) for req, r in zip(requirements, results) if r.success == "fail"
    ]
    await _emit_event(
        streamer.streaming_id,
        QuickCheckEvent(
            chunk_index=chunk_index, attempt=1, passed=not failures, results=results
        ),
        requirements=requirements,
        event_queue=streamer._event_queue,
    )
    if not failures:
        return True
    streamer.failed_early = True
    streamer.streaming_failures.extend(failures)
    where = " on flush" if on_flush else ""
    streamer.failure_reason = (
        f"Streaming validation failed{where}: {failures[-1][1].reason or ''}"
    )
    return False


async def _drive(
    streamer: Streamer,
    mot: ModelOutputThunk,
    ctx: Context,
    chunking: ChunkingStrategy | None,
    requirements: list[Requirement],
    validation_backend: Backend,
) -> AsyncGenerator[str, None]:
    """Drive the whole stream from one generator on the caller's task.

    A caller `break`/`aclose()` delivers `GeneratorExit` to the suspended `yield`,
    so the single `finally` always runs — cleanup and STREAMING_END fire on every
    exit path (natural end, early exit, caller break, exception).

    On natural completion every requirement's `validate()` runs on the full output
    (early exit already returned, so all requirements reached the end unfailed);
    this is what checks judge/aLoRA requirements that streamed only `"unknown"`.

    Args:
        streamer: The handle recording terminal state for the caller.
        mot: The in-flight streaming thunk.
        ctx: The generation context.
        chunking: Resolved chunking strategy, or `None` for raw deltas.
        requirements: Requirements to validate against.
        validation_backend: Backend used for validation calls.

    Yields:
        str: Each validated chunk, in order.
    """
    # `accumulated` is the full raw text across deltas; the Chunker holds only the
    # pending fragment. chunking=None means yield raw deltas, no Chunker.
    accumulated = ""
    chunk_index = 0
    success = False
    error: Exception | None = None
    chunker = Chunker(chunking) if chunking is not None else None
    emitted_end = 0  # offset in `accumulated` just past the last emitted chunk

    def _snapshot_full_text(chunk: str) -> None:
        nonlocal emitted_end
        pos = accumulated.find(chunk, emitted_end)
        if pos >= 0:
            emitted_end = pos + len(chunk)
        streamer.full_text = accumulated[:emitted_end]

    try:
        async for delta in mot:
            accumulated += delta

            if chunker is None:
                new_chunks = [delta] if delta else []  # raw mode: delta is the chunk
            else:
                new_chunks = chunker.feed(delta)

            for c in new_chunks:
                if not await _validate_chunk(
                    streamer, c, chunk_index, requirements, validation_backend, ctx
                ):
                    return
                _snapshot_full_text(c)  # record before yield; a break skips past it
                await _emit_event(
                    streamer.streaming_id,
                    ChunkEvent(text=c, chunk_index=chunk_index, attempt=1),
                    event_queue=streamer._event_queue,
                )
                yield c
                chunk_index += 1

        # Flush the trailing fragment the chunker withheld (skipped in raw mode).
        if chunker is not None:
            for c in chunker.flush():
                if not await _validate_chunk(
                    streamer,
                    c,
                    chunk_index,
                    requirements,
                    validation_backend,
                    ctx,
                    on_flush=True,
                ):
                    return
                _snapshot_full_text(c)
                await _emit_event(
                    streamer.streaming_id,
                    ChunkEvent(text=c, chunk_index=chunk_index, attempt=1),
                    event_queue=streamer._event_queue,
                )
                yield c
                chunk_index += 1

        streamer.full_text = accumulated
        streamer.mot = mot
        streamer.completed_normally = True
        await _emit_event(
            streamer.streaming_id,
            StreamingDoneEvent(attempt=1, full_text=accumulated),
            event_queue=streamer._event_queue,
        )

        # Reached only on natural completion, so every requirement is still
        # unfailed and gets a full-output validate().
        if requirements:
            streamer.final_validations = list(
                await asyncio.gather(
                    *[req.validate(validation_backend, ctx) for req in requirements]
                )
            )
            await _emit_event(
                streamer.streaming_id,
                FullValidationEvent(
                    attempt=1,
                    passed=all(v.as_bool() for v in streamer.final_validations),
                    results=streamer.final_validations,
                ),
                event_queue=streamer._event_queue,
            )
        success = True
    except Exception as exc:
        # Record for the STREAMING_END span, then re-raise so the exception
        # still propagates to the caller through the `async for`.
        error = exc
        await _emit_event(
            streamer.streaming_id,
            ErrorEvent(exception_type=type(exc).__name__, detail=str(exc)),
            event_queue=streamer._event_queue,
        )
        raise
    finally:
        # Driver-side teardown on every exit path
        await streamer._finalize(
            success=success, error=error, full_text_length=len(streamer.full_text)
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def _stream(
    action: Component[Any] | CBlock,
    backend: Backend,
    ctx: Context,
    *,
    chunking: str | ChunkingStrategy | None = None,
    requirements: Sequence[Requirement] | None = None,
    validation_backend: Backend | None = None,
    event_queue: asyncio.Queue[StreamEvent | None] | None = None,
) -> Streamer:
    """Start a streaming generation and return its `Streamer`.

    The implementation behind the public `stream()`; see it for the full
    contract.

    Args:
        action: The component or content block to generate from.
        backend: Backend used for generation and, unless `validation_backend`
            is set, validation.
        ctx: The generation context.
        chunking: A `ChunkingStrategy`, a recognized alias string, or `None`
            (default) to yield raw deltas unchunked.
        requirements: Requirements validated against each chunk during
            streaming and against the full output at stream end. `None` yields
            chunks without validation.
        validation_backend: Backend for validation calls; defaults to `backend`.
        event_queue: Optional queue; when set, every emitted `StreamEvent` is
            also pushed onto it. `None` for the plain chunk-iterating path.

    Returns:
        Streamer: An async-iterable handle over the validated chunks.

    Raises:
        ValueError: If `chunking` is a string that is not a known alias.
        RuntimeError: If the backend returns an already-computed thunk instead
            of a streaming one — i.e. it is not honouring `ModelOption.STREAM`.
    """
    strategy = resolve_chunking_strategy(chunking)

    # Copy so a raising __copy__ surfaces before generation starts, and the
    # caller's requirement instances are never mutated by streaming state.
    cloned_reqs = [copy(req) for req in (requirements or [])]
    resolved_backend = validation_backend if validation_backend is not None else backend

    streaming_id = str(uuid.uuid4())
    if has_plugins(HookType.STREAMING_START):
        from ..plugins.hooks.streaming import StreamingStartPayload

        await invoke_hook(
            HookType.STREAMING_START,
            StreamingStartPayload(
                streaming_id=streaming_id,
                has_requirements=bool(cloned_reqs),
                requirement_count=len(cloned_reqs),
                chunking_strategy=type(strategy).__name__ if strategy else "none",
            ),
        )

    mot = None
    try:
        mot, gen_ctx = await backend.generate_from_context(
            action, ctx, model_options={ModelOption.STREAM: True}
        )
        if mot.is_computed():
            raise RuntimeError(
                "stream() requires a streaming backend; the backend returned an "
                "already-computed MOT. Ensure the backend honours ModelOption.STREAM."
            )
    except BaseException as exc:
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

    return Streamer(
        mot, gen_ctx, strategy, cloned_reqs, resolved_backend, streaming_id, event_queue
    )


@overload
async def stream(
    action: Component[Any] | CBlock,
    backend: Backend,
    ctx: Context,
    *,
    chunking: str | ChunkingStrategy | None = ...,
    requirements: Sequence[Requirement] | None = ...,
    validation_backend: Backend | None = ...,
    as_events: Literal[False] = ...,
) -> Streamer: ...


@overload
async def stream(
    action: Component[Any] | CBlock,
    backend: Backend,
    ctx: Context,
    *,
    chunking: str | ChunkingStrategy | None = ...,
    requirements: Sequence[Requirement] | None = ...,
    validation_backend: Backend | None = ...,
    as_events: Literal[True],
) -> EventStreamer: ...


@overload
async def stream(
    action: Component[Any] | CBlock,
    backend: Backend,
    ctx: Context,
    *,
    chunking: str | ChunkingStrategy | None = ...,
    requirements: Sequence[Requirement] | None = ...,
    validation_backend: Backend | None = ...,
    as_events: bool,
) -> Streamer | EventStreamer: ...


async def stream(
    action: Component[Any] | CBlock,
    backend: Backend,
    ctx: Context,
    *,
    chunking: str | ChunkingStrategy | None = None,
    requirements: Sequence[Requirement] | None = None,
    validation_backend: Backend | None = None,
    as_events: bool = False,
) -> Streamer | EventStreamer:
    """Start a streaming generation.

    Generation begins eagerly, before this call returns. Consume the returned
    handle inside `async with` so the stream is always released. On early exit or
    `break`, `async with` cancels the in-flight generation:

    ```python
    async with await stream(action, backend, ctx) as s:
        async for chunk in s:
            ...
    ```

    By default this yields validated `str` chunks. Each iteration yields a chunk —
    a unit produced by the `chunking` strategy, or the raw model delta when
    `chunking` is `None`. A chunk is delivered once it has passed every
    requirement's `stream_validate`; a `"fail"` stops the stream early and cancels
    the backend. On natural completion, `validate()` runs on the full output. With
    no `requirements`, chunks are yielded without validation.

    With `as_events=True`, this returns an `EventStreamer` instead: iterating it
    yields the stream's typed `StreamEvent` objects (the same events otherwise seen
    only via the `STREAMING_EVENT` hook). The terminal `CompletedEvent`/`ErrorEvent`
    come through the iterator too; an early `break` leaves them queued, so iterating
    again drains them. Each `ChunkEvent` carries its text, so no output is lost.

    Args:
        action: The component or content block to generate from.
        backend: Backend used for generation and, unless `validation_backend`
            is set, validation.
        ctx: The generation context.
        chunking: A `ChunkingStrategy`, a recognized alias string, or `None`
            (default) to yield raw deltas unchunked.
        requirements: Requirements validated against each chunk during
            streaming and against the full output at stream end. `None` yields
            chunks without validation.
        validation_backend: Backend for validation calls; defaults to `backend`.
        as_events: When `True`, return an `EventStreamer` that iterates typed
            `StreamEvent` objects; when `False` (default), return a `Streamer`
            that iterates validated `str` chunks.

    Returns:
        Streamer | EventStreamer: A `Streamer` over validated chunks, or — when
            `as_events=True` — an `EventStreamer` over the stream's events.

    Raises:
        ValueError: If `chunking` is a string that is not a known alias.
        RuntimeError: If the backend returns an already-computed thunk instead
            of a streaming one — i.e. it is not honouring `ModelOption.STREAM`.
    """
    if as_events:
        event_streamer = EventStreamer()
        event_streamer._pump_task = asyncio.create_task(
            event_streamer._pump(
                action,
                backend,
                ctx,
                chunking=chunking,
                requirements=requirements,
                validation_backend=validation_backend,
            )
        )
        try:
            await event_streamer._ready
        except BaseException:
            await event_streamer.aclose()
            raise
        return event_streamer

    return await _stream(
        action,
        backend,
        ctx,
        chunking=chunking,
        requirements=requirements,
        validation_backend=validation_backend,
    )
