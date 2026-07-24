# pytest: skip

"""Runnable verification harness for the validated-streaming refactor POC.

    uv run python docs/examples/streaming/validated_streaming_poc.py

Transient scaffolding, not a usage example: it drives the prototype
`mellea.stdlib.streaming_poc` module (which is not yet wired into the framework)
and uses a mock backend that pokes `ModelOutputThunk` internals, so it is marked
`skip` and excluded from CI. It exists only to exercise the POC's interface end
to end while the refactor is unplugged; delete it once the refactor lands and a
real test suite replaces it.

Runs without Ollama (deterministic mock stream, from test/core/test_astream_mock.py)
and prints one block per case:

  1. Plain streaming — `async for delta in mot` straight off the MOT (core only).
  2. Per-chunk requirement passes — natural completion + full-output validate.
  3. Per-chunk requirement fails a chunk — early exit, no final validate.
  4. Judge-style requirement (streams "unknown") — passes the stream-end validate.
  5. Judge-style requirement — fails the stream-end validate.
  6. Requirement raises mid-stream — exception propagates to the caller.

Each case also emits the real `STREAMING_START`/`STREAMING_EVENT`/`STREAMING_END`
hooks; the recorders installed at startup print them to show the single-task
event lifecycle.
"""

import asyncio
from typing import Any

from mellea.core.base import CBlock, GenerateType, ModelOutputThunk
from mellea.core.requirement import (
    PartialValidationResult,
    Requirement,
    ValidationResult,
)
from mellea.plugins import hook, register
from mellea.stdlib.streaming_poc import stream


# --- register REAL telemetry hooks (the same ones the prod plugin uses) ------
def _install_hook_recorders() -> None:
    """Subscribe to the actual STREAMING_START/END hooks and print each firing.

    This is what proves the pitch's telemetry claim: the real hooks fire at two
    clean points on ONE task. A production span subscriber would open on START,
    close on END — no cross-task detach.
    """

    @hook("streaming_start")
    async def _on_start(payload: Any, ctx: Any) -> Any:
        print(
            f"  [hook streaming_start] id={payload.streaming_id[:8]} "
            f"reqs={payload.requirement_count} chunker={payload.chunking_strategy}"
        )
        return None

    @hook("streaming_end")
    async def _on_end(payload: Any, ctx: Any) -> Any:
        print(
            f"  [hook streaming_end  ] id={payload.streaming_id[:8]} "
            f"success={payload.success} reason={payload.failure_reason!r} "
            f"exception={payload.exception!r}"
        )
        return None

    @hook("streaming_event")
    async def _on_event(payload: Any, ctx: Any) -> Any:
        print(f"  [hook streaming_event] {type(payload.event).__name__}")
        return None

    register([_on_start, _on_end, _on_event])


# --- fake MOT plumbing (from test_astream_mock.py) --------------------------
async def _mock_process(mot: ModelOutputThunk, chunk: Any) -> None:
    if mot._underlying_value is None:
        mot._underlying_value = ""
    if chunk is not None:
        mot._underlying_value += chunk


async def _mock_post_process(mot: ModelOutputThunk) -> None:
    mot.parsed_repr = mot.value  # type: ignore[assignment]


def _streaming_mot(text: str, *, tokens: int = 6) -> ModelOutputThunk:
    """A MOT preloaded to stream `text` in `tokens` slices, then a sentinel."""
    mot: ModelOutputThunk = ModelOutputThunk(value=None)
    mot._call.action = CBlock("demo")
    mot._gen.generate_type = GenerateType.ASYNC
    mot._gen.process = _mock_process
    mot._gen.post_process = _mock_post_process
    mot._gen.chunk_size = 0
    step = max(1, len(text) // tokens)
    for i in range(0, len(text), step):
        mot._gen.queue.put_nowait(text[i : i + step])
    mot._gen.queue.put_nowait(None)  # completion sentinel
    return mot


# --- 1. PLAIN streaming: async for straight off the MOT ---------------------
async def demo_plain() -> None:
    print("\n=== 1. Plain streaming — `async for delta in mot` (core, no stdlib) ===")
    mot = _streaming_mot("The robot learned to cook. It burned the toast.")
    got = ""
    async for delta in mot:  # <-- the NEW protocol; no while/is_computed loop
        got += delta
        print(f"  delta: {delta!r}")
    print(f"  computed={mot.is_computed()}  value={mot.value!r}")
    assert got == mot.value


# --- 2-6. VALIDATED streaming through stream() ------------------------------
class _NoBurntToast(Requirement):
    """Per-chunk failure: fails the instant a chunk mentions burning (no LLM).

    Its final `validate()` (run on natural completion) re-checks the full text
    deterministically via `validation_fn`, so even a per-chunk requirement gets
    the stream-end pass — matching Phase 1.
    """

    def __init__(self, desc: str) -> None:
        super().__init__(
            desc,
            validation_fn=lambda ctx: ValidationResult(
                "burn" not in str(ctx).lower(), reason="mentions burning"
            ),
        )

    async def stream_validate(
        self, chunk: str, *, backend: Any, ctx: Any
    ) -> PartialValidationResult:
        if "burn" in chunk.lower():
            return PartialValidationResult(success="fail", reason="mentions burning")
        return PartialValidationResult(success="unknown")


class _MaxLength(Requirement):
    """Judge-style requirement: can't judge per-chunk, only the full output.

    Streams `"unknown"` throughout, then checks the complete text at stream end
    via `validate()` — the exact pattern that breaks if final validation is
    dropped. Deterministic (`validation_fn`), no LLM.
    """

    def __init__(self, limit: int) -> None:
        super().__init__(
            f"at most {limit} chars",
            validation_fn=lambda ctx: ValidationResult(
                len(str(ctx)) <= limit, reason=f">{limit} chars"
            ),
        )


class _Explodes(Requirement):
    """Raises mid-stream to exercise the exception path (span closes in error)."""

    async def stream_validate(
        self, chunk: str, *, backend: Any, ctx: Any
    ) -> PartialValidationResult:
        raise RuntimeError("validator blew up")


class _FakeBackend:
    """Minimal backend: generate_from_context returns a preloaded streaming MOT."""

    def __init__(self, text: str) -> None:
        self._text = text

    async def generate_from_context(self, action, ctx, *, model_options=None):
        # The "context" passed to validate() is just the text here, so the
        # judge-style validation_fn can inspect the full output deterministically.
        return _streaming_mot(self._text), self._text


async def demo_validated(text: str, label: str, req: Requirement) -> None:
    print(f"\n=== {label} ===")
    backend = _FakeBackend(text)
    streamer = await stream(
        CBlock("demo"),
        backend,  # type: ignore[arg-type]
        ctx=None,  # type: ignore[arg-type]
        chunking="sentence",
        requirements=[req],
    )
    async for chunk in streamer:
        print(f"  chunk: {chunk!r}")
    print(f"  failed_early={streamer.failed_early}  reason={streamer.failure_reason!r}")
    print(
        f"  streaming_failures={[(type(r).__name__, p.success) for r, p in streamer.streaming_failures]}"
    )
    print(f"  full_text={streamer.full_text!r}")
    finals = [v.as_bool() for v in streamer.final_validations]
    print(f"  final_validations (stream-end validate): {finals}")


async def main() -> None:
    _install_hook_recorders()
    await demo_plain()
    await demo_validated(
        "The robot sliced a tomato. It plated the dish beautifully.",
        "2. Per-chunk req passes -> natural completion + final validate",
        _NoBurntToast("no burning"),
    )
    await demo_validated(
        "The robot sliced a tomato. Then it burned the whole kitchen down.",
        "3. Per-chunk req fails chunk 2 -> early exit, NO final validate",
        _NoBurntToast("no burning"),
    )
    await demo_validated(
        "Short and sweet.",
        "4. Judge-style req: streams 'unknown', PASSES final validate",
        _MaxLength(100),
    )
    await demo_validated(
        "This one is deliberately written to be quite a bit longer than the limit "
        "so that the stream-end validate() is the thing that catches it.",
        "5. Judge-style req: streams 'unknown', FAILS final validate",
        _MaxLength(50),
    )

    print("\n=== 6. Requirement raises mid-stream -> exception propagates ===")
    streamer = await stream(
        CBlock("demo"),
        _FakeBackend("The robot sliced a tomato. It plated the dish."),  # type: ignore[arg-type]
        ctx=None,  # type: ignore[arg-type]
        chunking="sentence",
        requirements=[_Explodes("boom")],
    )
    try:
        async for chunk in streamer:
            print(f"  chunk: {chunk!r}")
    except RuntimeError as exc:
        print(f"  caller caught: {exc!r}")


if __name__ == "__main__":
    asyncio.run(main())
