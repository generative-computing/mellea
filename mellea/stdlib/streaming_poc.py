"""Validated streaming: a thin layer over `ModelOutputThunk.__aiter__`.

Functional replacement for `mellea/stdlib/streaming.py` (`stream_with_chunking` +
`StreamChunkingResult`). The design rests on two properties:

1. Plain and validated streaming share one iteration protocol (`async for`). The
   shared part (`ModelOutputThunk.__aiter__`) lives in core; chunking and
   validation layer on top here in stdlib, so core keeps no dependency on stdlib.

2. The whole stream runs on the caller's task — no background orchestration task,
   so no queues, no raise-once exception plumbing, and no
   `acomplete()`-vs-`astream()` finalization ambiguity.
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import AsyncIterator, Sequence
from copy import copy
from typing import Any

from ..backends.model_options import ModelOption
from ..core.backend import Backend
from ..core.base import CBlock, Component, Context, ModelOutputThunk
from ..core.requirement import PartialValidationResult, Requirement, ValidationResult
from ..plugins.hooks.streaming import StreamingEndPayload, StreamingStartPayload
from ..plugins.manager import has_plugins, invoke_hook
from ..plugins.types import HookType
from .chunking import ChunkingStrategy, ParagraphChunker, SentenceChunker, WordChunker

_CHUNKING_ALIASES: dict[str, type[ChunkingStrategy]] = {
    "sentence": SentenceChunker,
    "word": WordChunker,
    "paragraph": ParagraphChunker,
}


class Streamer:
    """Async-iterable handle for a `stream` call.

    Consume the chunks with `async for`, then read `failed_early` /
    `failure_reason`. On early exit, `streaming_failures` holds every
    `(requirement, result)` that failed the offending chunk and `full_text`
    holds the validated output through the last fully-passed delta. On natural
    completion, `mot` holds the computed output and `final_validations` holds the
    stream-end `validate()` results.

    Args:
        mot: The in-flight streaming thunk from the backend generation call.
        ctx: The generation context, used for validation calls.
        chunking: Resolved chunking strategy, or `None` for raw deltas.
        requirements: Requirements to validate against; pre-copied by `stream`.
        validation_backend: Backend used for validation calls.
    """

    def __init__(
        self,
        mot: ModelOutputThunk,
        ctx: Context,
        chunking: ChunkingStrategy | None,
        requirements: list[Requirement],
        validation_backend: Backend,
    ) -> None:
        """Wrap an in-flight generation; iterating the `Streamer` drives it."""
        self.failed_early: bool = False
        self.failure_reason: str | None = None
        self.streaming_failures: list[tuple[Requirement, PartialValidationResult]] = []
        self.full_text: str = ""
        self.mot: ModelOutputThunk | None = None
        self.final_validations: list[ValidationResult] = []
        # Correlates this stream's START/END hooks; unique per concurrent stream.
        self.streaming_id: str = str(uuid.uuid4())
        self._chunks: AsyncIterator[str] = _drive(
            self, mot, ctx, chunking, requirements, validation_backend
        )

    def __aiter__(self) -> AsyncIterator[str]:
        """Return the generator that drives generation and yields chunks."""
        return self._chunks


async def _validate_chunk(
    streamer: Streamer,
    chunk: str,
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
    """
    if not requirements:
        return True
    results = await asyncio.gather(
        *[
            req.stream_validate(chunk, backend=validation_backend, ctx=ctx)
            for req in requirements
        ]
    )
    failures = [
        (req, r) for req, r in zip(requirements, results) if r.success == "fail"
    ]
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
) -> AsyncIterator[str]:
    """Drive the whole stream from one generator on the caller's task.

    A caller `break`/`aclose()` delivers `GeneratorExit` to the suspended `yield`,
    so the single `finally` always runs — cleanup and STREAMING_END fire on every
    exit path (natural end, early exit, caller break, exception).

    On natural completion every requirement's `validate()` runs on the full output
    (early exit already returned, so all requirements reached the end unfailed);
    this is what checks judge/aLoRA requirements that streamed only `"unknown"`.
    """
    accumulated = ""
    prev_chunk_count = 0
    success = False
    error: Exception | None = None

    if has_plugins(HookType.STREAMING_START):
        await invoke_hook(
            HookType.STREAMING_START,
            StreamingStartPayload(
                streaming_id=streamer.streaming_id,
                has_requirements=bool(requirements),
                requirement_count=len(requirements),
                chunking_strategy=type(chunking).__name__ if chunking else "none",
            ),
        )

    try:
        async for delta in mot:
            accumulated += delta

            # chunking=None -> yield each raw delta as its own chunk.
            if chunking is None:
                new_chunks = [delta] if delta else []
            else:
                chunks = chunking.split(accumulated)
                new_chunks = chunks[prev_chunk_count:]
                prev_chunk_count = len(chunks)

            for c in new_chunks:
                if not await _validate_chunk(
                    streamer, c, requirements, validation_backend, ctx
                ):
                    return
                yield c

            # Snapshot after a delta fully passes so `full_text` excludes any
            # unvalidated chunk on early exit.
            # TODO(#1013): delta-granular, not chunk-exact; MOT-owned chunking
            # (one unit per iteration) makes it exact.
            streamer.full_text = accumulated

        # Flush the trailing fragment the chunker withheld (skipped in raw mode).
        if chunking is not None:
            for c in chunking.flush(accumulated):
                if not await _validate_chunk(
                    streamer, c, requirements, validation_backend, ctx, on_flush=True
                ):
                    return
                yield c

        # Natural completion: capture the flushed fragment the snapshot missed.
        streamer.full_text = accumulated
        streamer.mot = mot

        # Reached only on natural completion, so every requirement is still
        # unfailed and gets a full-output validate().
        if requirements:
            streamer.final_validations = list(
                await asyncio.gather(
                    *[req.validate(validation_backend, ctx) for req in requirements]
                )
            )
        success = True
    except Exception as exc:
        # Record for the STREAMING_END span, then re-raise so the exception
        # still propagates to the caller through the `async for`.
        error = exc
        raise
    finally:
        # Cancel on any early/broken exit so the backend producer never wedges
        # on a full queue; no-op once the stream is fully drained.
        if not mot.is_computed():
            await mot.cancel_generation()

        if has_plugins(HookType.STREAMING_END):
            await invoke_hook(
                HookType.STREAMING_END,
                StreamingEndPayload(
                    streaming_id=streamer.streaming_id,
                    success=success,
                    failure_reason=streamer.failure_reason,
                    exception=error,
                    model=mot.generation.model,
                    provider=mot.generation.provider,
                    full_text_length=len(accumulated),
                ),
            )


async def stream(
    action: Component[Any] | CBlock,
    backend: Backend,
    ctx: Context,
    *,
    chunking: str | ChunkingStrategy | None = "sentence",
    requirements: Sequence[Requirement] | None = None,
    validation_backend: Backend | None = None,
) -> Streamer:
    """Start a streaming generation, optionally chunked and validated per chunk.

    Consume the returned `Streamer` with `async for`. Each iteration yields a
    chunk once it has passed every requirement's `stream_validate`; a `"fail"`
    stops the stream early and cancels the backend. On natural completion,
    `validate()` runs on the full output. With no `requirements`, chunks are
    yielded without validation.

    Args:
        action: The component or content block to generate from.
        backend: Backend used for generation and, unless `validation_backend`
            is set, validation.
        ctx: The generation context.
        chunking: A `ChunkingStrategy`, one of the aliases `"sentence"`,
            `"word"`, `"paragraph"`, or `None` to yield raw deltas unchunked.
        requirements: Requirements validated against each chunk during
            streaming and against the full output at stream end. `None` yields
            chunks without validation.
        validation_backend: Backend for validation calls; defaults to `backend`.

    Returns:
        Streamer: An async-iterable handle over the validated chunks.

    Raises:
        ValueError: If `chunking` is a string that is not a known alias.
        RuntimeError: If the backend returns an already-computed thunk instead
            of a streaming one (it is not honouring `ModelOption.STREAM`).
    """
    if isinstance(chunking, str):
        cls = _CHUNKING_ALIASES.get(chunking)
        if cls is None:
            raise ValueError(f"Unknown chunking alias {chunking!r}")
        chunking = cls()

    # Copy so a raising __copy__ surfaces before generation starts, and the
    # caller's requirement instances are never mutated by streaming state.
    cloned_reqs = [copy(req) for req in (requirements or [])]
    resolved_backend = validation_backend if validation_backend is not None else backend

    mot, gen_ctx = await backend.generate_from_context(
        action, ctx, model_options={ModelOption.STREAM: True}
    )
    if mot.is_computed():
        raise RuntimeError(
            "stream() requires a streaming backend; got an already-computed MOT."
        )

    return Streamer(mot, gen_ctx, chunking, cloned_reqs, resolved_backend)
