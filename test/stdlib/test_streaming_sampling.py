# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for streaming sampling: `stream(..., strategy=...)` retry/repair."""

import asyncio
from typing import Any

from mellea.core.backend import Backend
from mellea.core.base import CBlock, Context, GenerateType, ModelOutputThunk
from mellea.core.requirement import (
    PartialValidationResult,
    Requirement,
    ValidationResult,
)
from mellea.stdlib.components import Instruction
from mellea.stdlib.context import SimpleContext
from mellea.stdlib.sampling import RejectionSamplingStrategy
from mellea.stdlib.streaming import (
    ChunkEvent,
    CompletedEvent,
    RetryEvent,
    StreamEvent,
    stream,
)

# ---------------------------------------------------------------------------
# Mock backend (one response per generation call = one per attempt)
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
        await mot._gen.queue.put(response[i : i + token_size])
        await asyncio.sleep(0)
        i += token_size
    await mot._gen.queue.put(None)


class MultiResponseStreamingMockBackend(Backend):
    """Streams a different response on each generate call (one per attempt).

    The last response repeats if there are more attempts than responses.
    """

    def __init__(self, responses: list[str], token_size: int = 3) -> None:
        self._responses = list(responses)
        self._call_index = 0
        self._token_size = token_size
        self._model_id: str = "multi-mock-model"
        self._provider: str = "multi-mock-provider"

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
        response = self._responses[min(self._call_index, len(self._responses) - 1)]
        self._call_index += 1
        mot = _make_mot()
        task = asyncio.create_task(_feed_tokens(mot, response, self._token_size))
        _ = task
        return mot, ctx.add(action).add(mot)

    async def _generate_from_raw(
        self, actions: Any, ctx: Any, **kwargs: Any
    ) -> tuple[list[ModelOutputThunk], dict[str, Any] | None]:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Requirement doubles
# ---------------------------------------------------------------------------


class RequireMarkerReq(Requirement):
    """`validate()` passes iff `marker` appears in the streamed text.

    `stream_validate` stays `"unknown"` throughout, so failure surfaces only at
    stream end — the completed-but-failed trigger (drives `repair`).
    """

    def __init__(self, marker: str) -> None:
        super().__init__()
        self._marker = marker
        self._seen = ""

    def format_for_llm(self) -> str:
        return f"must contain {self._marker!r}"

    async def _stream_validate(
        self, chunk: str, *, backend: Any, ctx: Any
    ) -> PartialValidationResult:
        self._seen = self._seen + chunk  # reassign (shallow-copy safe)
        return PartialValidationResult("unknown")

    async def validate(
        self, backend: Any, ctx: Any, *, format: Any = None, model_options: Any = None
    ) -> ValidationResult:
        return ValidationResult(result=self._marker in self._seen)


class FailMidStreamReq(Requirement):
    """`stream_validate` returns `"fail"` as soon as `marker` is seen mid-stream.

    Drives the early-broken trigger (`stream_repair`).
    """

    def __init__(self, marker: str) -> None:
        super().__init__()
        self._marker = marker
        self._seen = ""

    def format_for_llm(self) -> str:
        return f"must not contain {self._marker!r}"

    async def _stream_validate(
        self, chunk: str, *, backend: Any, ctx: Any
    ) -> PartialValidationResult:
        self._seen = self._seen + chunk
        if self._marker in self._seen:
            return PartialValidationResult("fail", reason=f"saw {self._marker!r}")
        return PartialValidationResult("unknown")

    async def validate(
        self, backend: Any, ctx: Any, *, format: Any = None, model_options: Any = None
    ) -> ValidationResult:
        return ValidationResult(result=self._marker not in self._seen)


async def _run(**kwargs: Any) -> tuple[list[str], Any]:
    """Drive a plain `Streamer` to completion; return (chunks, streamer)."""
    chunks: list[str] = []
    async with await stream(**kwargs) as s:
        async for c in s:
            chunks.append(c)
    return chunks, s


async def _run_events(**kwargs: Any) -> tuple[list[StreamEvent], Any]:
    """Drive an `EventStreamer` to completion; return (events, streamer)."""
    events: list[StreamEvent] = []
    async with await stream(as_events=True, **kwargs) as s:
        async for ev in s:
            events.append(ev)
    return events, s


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


async def test_completed_fail_retries_then_succeeds() -> None:
    backend = MultiResponseStreamingMockBackend(
        ["no marker here.", "now with GOOD in it."]
    )
    chunks, s = await _run(
        action=Instruction("write"),
        backend=backend,
        ctx=SimpleContext(),
        requirements=[RequireMarkerReq("GOOD")],
        strategy=RejectionSamplingStrategy(loop_budget=2),
    )
    assert s.completed_normally is True
    # Winner is attempt 2; top-level fields reflect it.
    assert s.full_text == "now with GOOD in it."
    assert len(s.attempts) == 2
    assert s.attempts[0].completed_normally is True and s.attempts[0].success is False
    assert s.attempts[1].success is True and s.attempts[1].selected is True
    # Flat continuation: both attempts' chunks arrive in one async-for.
    assert "".join(chunks) == "no marker here.now with GOOD in it."


async def test_mid_stream_fail_retries_then_succeeds() -> None:
    backend = MultiResponseStreamingMockBackend(
        ["this has BAD stuff.", "all clean now."]
    )
    events, s = await _run_events(
        action=Instruction("write"),
        backend=backend,
        ctx=SimpleContext(),
        requirements=[FailMidStreamReq("BAD")],
        strategy=RejectionSamplingStrategy(loop_budget=2),
    )
    retries = [e for e in events if isinstance(e, RetryEvent)]
    assert len(retries) == 1 and retries[0].attempt == 2
    # The mid-stream break drives a stream_repair retry, surfaced on the event.
    assert retries[0].failed_early is True
    assert retries[0].failed_count == 1
    completed = [e for e in events if isinstance(e, CompletedEvent)]
    assert completed[-1].success is True
    assert completed[-1].attempts_used == 2
    assert s.attempts[0].failed_early is True
    assert s.attempts[1].success is True and s.attempts[1].selected is True


async def test_attempt_numbers_increment_on_chunk_events() -> None:
    backend = MultiResponseStreamingMockBackend(
        ["no marker here.", "now with GOOD in it."]
    )
    events, _s = await _run_events(
        action=Instruction("write"),
        backend=backend,
        ctx=SimpleContext(),
        requirements=[RequireMarkerReq("GOOD")],
        strategy=RejectionSamplingStrategy(loop_budget=2),
    )
    attempts_seen = sorted({e.attempt for e in events if isinstance(e, ChunkEvent)})
    assert attempts_seen == [1, 2]


async def test_budget_exhausted_reports_failure() -> None:
    backend = MultiResponseStreamingMockBackend(["still nope.", "again nope."])
    events, s = await _run_events(
        action=Instruction("write"),
        backend=backend,
        ctx=SimpleContext(),
        requirements=[RequireMarkerReq("GOOD")],
        strategy=RejectionSamplingStrategy(loop_budget=2),
    )
    # Completed-but-failed attempts drive a `repair` retry, not `stream_repair`.
    retries = [e for e in events if isinstance(e, RetryEvent)]
    assert len(retries) == 1
    assert retries[0].failed_early is False
    assert retries[0].failed_count == 1
    completed = [e for e in events if isinstance(e, CompletedEvent)]
    # Both attempts completed (they just failed validation), so `success` reflects
    # completion; the failure verdict lives in the attempts.
    assert completed[-1].success is True
    assert not any(a.success for a in s.attempts)
    assert completed[-1].attempts_used == 2
    assert len(s.attempts) == 2
    # RejectionSamplingStrategy.select_from_failure picks index 0.
    assert sum(1 for a in s.attempts if a.selected) == 1
    assert s.attempts[0].selected is True
    # Top-level fields project the selected failed attempt.
    assert s.full_text == s.attempts[0].full_text == "still nope."
    assert s.mot is s.attempts[0].mot


async def test_strategy_requirements_merge_with_call_requirements() -> None:
    # Output satisfies the call requirement ("CALL") but not the strategy's
    # ("STRAT"); merge enforces both, so the strategy requirement fails the attempt.
    backend = MultiResponseStreamingMockBackend(["has CALL only."])
    _chunks, s = await _run(
        action=Instruction("write"),
        backend=backend,
        ctx=SimpleContext(),
        requirements=[RequireMarkerReq("CALL")],
        strategy=RejectionSamplingStrategy(
            requirements=[RequireMarkerReq("STRAT")], loop_budget=1
        ),
    )
    assert len(s.attempts[0].final_validations) == 2  # both requirements validated
    assert s.attempts[0].success is False  # strategy requirement enforced via merge


async def test_no_strategy_is_single_attempt() -> None:
    backend = MultiResponseStreamingMockBackend(["only one response."])
    chunks, s = await _run(
        action=Instruction("write"),
        backend=backend,
        ctx=SimpleContext(),
        requirements=[RequireMarkerReq("GOOD")],  # fails, but no strategy = no retry
    )
    # One attempt is always recorded; on the no-strategy path it is the result.
    assert len(s.attempts) == 1
    assert s.attempts[0].completed_normally is True and s.attempts[0].selected is True
    assert s.attempts[0].success is False  # completed but validation failed
    assert s.completed_normally is True
    assert s.full_text == "only one response."
    assert "".join(chunks) == "only one response."


async def test_terminal_event_fires_once_across_retries() -> None:
    backend = MultiResponseStreamingMockBackend(
        ["no marker here.", "now with GOOD in it."]
    )
    events, s = await _run_events(
        action=Instruction("write"),
        backend=backend,
        ctx=SimpleContext(),
        requirements=[RequireMarkerReq("GOOD")],
        strategy=RejectionSamplingStrategy(loop_budget=2),
    )
    # `_finalize` runs once after the retry loop, not per attempt.
    completed = [e for e in events if isinstance(e, CompletedEvent)]
    assert len(completed) == 1
    assert completed[0].success is True
    assert completed[0].attempts_used == 2
    assert len(s.attempts) == 2


def _recording_strategy(calls: list[str]) -> RejectionSamplingStrategy:
    """A `RejectionSamplingStrategy` that records which repair method each retry uses."""

    class RecordingStrategy(RejectionSamplingStrategy):
        @staticmethod
        def stream_repair(*args: Any) -> Any:
            calls.append("stream_repair")
            return RejectionSamplingStrategy.stream_repair(*args)

        @staticmethod
        def repair(*args: Any) -> Any:
            calls.append("repair")
            return RejectionSamplingStrategy.repair(*args)

    return RecordingStrategy(loop_budget=2)


async def test_mid_stream_fail_routes_to_stream_repair() -> None:
    calls: list[str] = []
    backend = MultiResponseStreamingMockBackend(
        ["this has BAD stuff.", "all clean now."]
    )
    await _run(
        action=Instruction("write"),
        backend=backend,
        ctx=SimpleContext(),
        requirements=[FailMidStreamReq("BAD")],
        strategy=_recording_strategy(calls),
    )
    assert calls == ["stream_repair"]


async def test_completed_fail_routes_to_repair() -> None:
    calls: list[str] = []
    backend = MultiResponseStreamingMockBackend(
        ["no marker here.", "now with GOOD in it."]
    )
    await _run(
        action=Instruction("write"),
        backend=backend,
        ctx=SimpleContext(),
        requirements=[RequireMarkerReq("GOOD")],
        strategy=_recording_strategy(calls),
    )
    assert calls == ["repair"]
