"""Unit tests for Requirement.stream_validate() hook."""

import inspect
from copy import copy

import pytest

from mellea.core import PartialValidationResult, Requirement
from mellea.core.backend import Backend
from mellea.core.base import Context


@pytest.mark.asyncio
async def test_default_returns_unknown():
    req = Requirement(description="some requirement")
    result = await req.stream_validate("some chunk", backend=None, ctx=None)  # type: ignore[arg-type]
    assert result.success == "unknown"


@pytest.mark.asyncio
async def test_default_returns_partial_validation_result_instance():
    req = Requirement()
    result = await req.stream_validate("chunk", backend=None, ctx=None)  # type: ignore[arg-type]
    assert isinstance(result, PartialValidationResult)


def test_stream_validate_is_coroutine():
    req = Requirement()
    assert inspect.iscoroutinefunction(req.stream_validate)


@pytest.mark.asyncio
async def test_subclass_can_return_pass():
    class PassRequirement(Requirement):
        async def stream_validate(
            self, chunk: str, *, backend: Backend, ctx: Context
        ) -> PartialValidationResult:
            return PartialValidationResult("pass")

    req = PassRequirement(description="always passes")
    result = await req.stream_validate("any chunk", backend=None, ctx=None)  # type: ignore[arg-type]
    assert result.success == "pass"


@pytest.mark.asyncio
async def test_subclass_can_return_fail():
    class FailRequirement(Requirement):
        async def stream_validate(
            self, chunk: str, *, backend: Backend, ctx: Context
        ) -> PartialValidationResult:
            if "bad" in chunk:
                return PartialValidationResult("fail", reason="bad word detected")
            return PartialValidationResult("unknown")

    req = FailRequirement(description="no bad words")
    result = await req.stream_validate("this is bad content", backend=None, ctx=None)  # type: ignore[arg-type]
    assert result.success == "fail"
    assert result.reason == "bad word detected"

    result_unknown = await req.stream_validate("good content", backend=None, ctx=None)  # type: ignore[arg-type]
    assert result_unknown.success == "unknown"


@pytest.mark.asyncio
async def test_does_not_mutate_requirement():
    req = Requirement(description="original description")
    original_description = req.description
    original_output = req._output
    original_validation_fn = req.validation_fn

    await req.stream_validate("some chunk", backend=None, ctx=None)  # type: ignore[arg-type]

    assert req.description == original_description
    assert req._output == original_output
    assert req.validation_fn == original_validation_fn


@pytest.mark.asyncio
async def test_stream_validate_idempotent():
    req = Requirement(description="repeated calls")
    result1 = await req.stream_validate("chunk one", backend=None, ctx=None)  # type: ignore[arg-type]
    result2 = await req.stream_validate("chunk two", backend=None, ctx=None)  # type: ignore[arg-type]
    assert result1.success == "unknown"
    assert result2.success == "unknown"
    assert req._output is None


@pytest.mark.asyncio
async def test_stateful_subclass_accumulates_state():
    """Stateful subclass correctly accumulates state across stream_validate calls.

    Uses delta extraction (via _seen_len) to count only new bullet points per call —
    a pattern that genuinely requires state from prior calls.
    """

    class BulletCounter(Requirement):
        def __init__(self) -> None:
            super().__init__(description="no more than 3 bullets")
            self._seen_len = 0
            self._bullet_count = 0

        async def stream_validate(
            self, chunk: str, *, backend: Backend, ctx: Context
        ) -> PartialValidationResult:
            delta = chunk[self._seen_len :]
            self._seen_len = len(chunk)
            self._bullet_count += delta.count("\n-")
            if self._bullet_count > 3:
                return PartialValidationResult(
                    "fail", reason=f"{self._bullet_count} bullets exceeds limit"
                )
            return PartialValidationResult("unknown")

    req = BulletCounter()
    assert req._bullet_count == 0

    await req.stream_validate("intro text", backend=None, ctx=None)  # type: ignore[arg-type]
    assert req._bullet_count == 0

    await req.stream_validate("intro text\n- one\n- two", backend=None, ctx=None)  # type: ignore[arg-type]
    assert req._bullet_count == 2  # delta added 2 new bullets

    result = await req.stream_validate(
        "intro text\n- one\n- two\n- three\n- four",
        backend=None,
        ctx=None,  # type: ignore[arg-type]
    )
    assert req._bullet_count == 4  # delta added 2 more
    assert result.success == "fail"
    assert result.reason is not None and "4" in result.reason


@pytest.mark.asyncio
async def test_stateful_subclass_clone_isolation():
    """copy() of a stateful requirement gives an independent clone — orchestrator pattern."""

    class CallCounter(Requirement):
        def __init__(self) -> None:
            super().__init__(description="call counter")
            self._calls = 0

        async def stream_validate(
            self, chunk: str, *, backend: Backend, ctx: Context
        ) -> PartialValidationResult:
            self._calls += 1
            return PartialValidationResult("unknown")

    req = CallCounter()
    await req.stream_validate("a", backend=None, ctx=None)  # type: ignore[arg-type]
    await req.stream_validate("b", backend=None, ctx=None)  # type: ignore[arg-type]
    assert req._calls == 2

    # Simulate orchestrator cloning before a new attempt
    cloned = copy(req)
    assert cloned._calls == 2  # clone inherits state at clone time

    await cloned.stream_validate("c", backend=None, ctx=None)  # type: ignore[arg-type]
    assert cloned._calls == 3  # clone advances independently
    assert req._calls == 2  # original is unchanged
