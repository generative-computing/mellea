"""Unit tests for sampling/base.py static repair() logic — no backend required."""

import asyncio
from unittest.mock import Mock

import pytest

from mellea.core import (
    ComputedModelOutputThunk,
    Context,
    GenerateLog,
    ModelOutputThunk,
    Requirement,
    ValidationResult,
)
from mellea.stdlib.components import Instruction, Message
from mellea.stdlib.context import ChatContext
from mellea.stdlib.sampling import MultiTurnStrategy, RejectionSamplingStrategy
from mellea.stdlib.sampling.base import RepairTemplateStrategy, _SamplingResultSlice

# --- BaseSamplingStrategy.repair ---


def _val(passed: bool, reason: str | None = None) -> ValidationResult:
    return ValidationResult(result=passed, reason=reason)


def test_repair_instruction_builds_repair_string():
    ins = Instruction(description="Write a poem", requirements=["be concise"])
    req = Requirement(description="be concise")
    old_ctx = ChatContext()
    new_ctx = ChatContext()

    action, ctx = RepairTemplateStrategy.repair(
        old_ctx=old_ctx,
        new_ctx=new_ctx,
        past_actions=[ins],
        past_results=[
            ComputedModelOutputThunk(thunk=ModelOutputThunk(value="long text"))
        ],
        past_val=[[(req, _val(False, reason="Output was too long"))]],
    )
    assert isinstance(action, Instruction)
    assert action._repair_string is not None
    assert "Output was too long" in action._repair_string
    assert ctx is old_ctx


def test_repair_uses_req_description_when_no_reason():
    ins = Instruction(description="task")
    req = Requirement(description="must be brief")
    old_ctx = ChatContext()

    action, _ = RepairTemplateStrategy.repair(
        old_ctx=old_ctx,
        new_ctx=ChatContext(),
        past_actions=[ins],
        past_results=[ComputedModelOutputThunk(thunk=ModelOutputThunk(value="x"))],
        past_val=[[(req, _val(False))]],
    )
    assert "must be brief" in action._repair_string


def test_repair_non_instruction_returns_same_action():
    msg = Message("user", "hello")
    old_ctx = ChatContext()

    action, ctx = RepairTemplateStrategy.repair(
        old_ctx=old_ctx,
        new_ctx=ChatContext(),
        past_actions=[msg],
        past_results=[ComputedModelOutputThunk(thunk=ModelOutputThunk(value="x"))],
        past_val=[[]],
    )
    assert action is msg
    assert ctx is old_ctx


def test_repair_multiple_failures_all_listed():
    ins = Instruction(description="task")
    r1 = Requirement(description="be short")
    r2 = Requirement(description="be polite")
    old_ctx = ChatContext()

    action, _ = RepairTemplateStrategy.repair(
        old_ctx=old_ctx,
        new_ctx=ChatContext(),
        past_actions=[ins],
        past_results=[ComputedModelOutputThunk(thunk=ModelOutputThunk(value="x"))],
        past_val=[[(r1, _val(False, "too long")), (r2, _val(False, "rude tone"))]],
    )
    assert "too long" in action._repair_string
    assert "rude tone" in action._repair_string


def test_repair_passed_requirements_excluded():
    ins = Instruction(description="task")
    r_pass = Requirement(description="format ok")
    r_fail = Requirement(description="content wrong")
    old_ctx = ChatContext()

    action, _ = RepairTemplateStrategy.repair(
        old_ctx=old_ctx,
        new_ctx=ChatContext(),
        past_actions=[ins],
        past_results=[ComputedModelOutputThunk(thunk=ModelOutputThunk(value="x"))],
        past_val=[[(r_pass, _val(True)), (r_fail, _val(False, "incorrect"))]],
    )
    assert "format ok" not in action._repair_string
    assert "incorrect" in action._repair_string


# --- concurrency_budget integration tests (mocked backend) ---


@pytest.fixture(scope="function")
def mocked_context_backend():
    """Backend whose generate_from_context sleeps then returns a mocked thunk."""
    backend = Mock()

    async def mock_generate(*args, **kwargs):
        # Mirrors how BaseSamplingStrategy._subsample_iteration calls it: action
        # is positional; ctx + others are keyword.
        action = args[0] if args else kwargs["action"]
        ctx = kwargs["ctx"]
        await asyncio.sleep(0.05)
        output = ModelOutputThunk("mocked")
        output._generate_log = GenerateLog()
        return output, ctx.add(action).add(output)

    backend.generate_from_context = mock_generate
    return backend


# Module-level counter used by the "every 5th call passes" requirement below.
_validation_counter = 0


async def test_rejection_sampling_with_concurrency_early_success(
    mocked_context_backend,
):
    """With concurrency, sampling stops once any subsample succeeds.

    Validation passes only on every 5th call, so the 5th request succeeds
    and the strategy must stop before exhausting the 9-request budget.
    """
    global _validation_counter
    _validation_counter = 0

    def sometimes_pass(_ctx: Context) -> ValidationResult:
        global _validation_counter
        _validation_counter += 1
        return ValidationResult(_validation_counter % 5 == 0)

    sometimes_pass_req = Requirement("sometimes_pass", validation_fn=sometimes_pass)

    loop_budget = 3
    concurrency_budget = 3
    strategy = RejectionSamplingStrategy(
        loop_budget=loop_budget, concurrency_budget=concurrency_budget
    )

    result = await strategy.sample(
        action=Instruction(description=""),
        context=ChatContext(),
        backend=mocked_context_backend,
        requirements=[sometimes_pass_req],
    )

    # The 5th validation call is the first to succeed, so at least 5 slices
    # must be observed; the strategy must also stop before exhausting the
    # full 9-request budget.
    assert 5 <= len(result.sample_actions) < loop_budget * concurrency_budget, (
        "early success should occur on/after the 5th call but before exhausting the budget"
    )


async def test_subsample_iteration_ends_after_success(mocked_context_backend):
    """The async generator must emit exactly one slice and then StopAsyncIteration."""
    always_pass = Requirement(
        "always pass", validation_fn=lambda _ctx: ValidationResult(result=True)
    )

    strategy = RejectionSamplingStrategy(loop_budget=5, concurrency_budget=5)

    generator = strategy._subsample_iteration(
        subsample_index=0,
        iterations=2,
        action=Instruction(description=""),
        context=ChatContext(),
        backend=mocked_context_backend,
        requirements=[always_pass],
    )

    first = await generator.__anext__()
    assert isinstance(first, _SamplingResultSlice)
    assert first.success is True

    with pytest.raises(StopAsyncIteration):
        await generator.__anext__()


async def test_repair_strategy_with_concurrency(mocked_context_backend):
    """All slices are produced when every requirement fails, and the last
    iteration's action carries a repair string.
    """
    always_fail = Requirement(
        "always fail", validation_fn=lambda _ctx: ValidationResult(result=False)
    )

    loop_budget = 5
    concurrency_budget = 5
    strategy = RepairTemplateStrategy(
        loop_budget=loop_budget, concurrency_budget=concurrency_budget
    )

    result = await strategy.sample(
        action=Instruction(description=""),
        context=ChatContext(),
        backend=mocked_context_backend,
        requirements=[always_fail],
    )

    assert len(result.sample_actions) == loop_budget * concurrency_budget, (
        "forced failures should exhaust the full loop * concurrency budget"
    )
    last_ctx = result.sample_contexts[-1]
    first_node = last_ctx.view_for_generation()[0]  # type: ignore[union-attr]
    assert first_node._repair_string is not None, (
        "last batch of retries should carry a repair_string with RepairTemplateStrategy"
    )


async def test_multi_turn_strategy_with_concurrency(mocked_context_backend):
    """MultiTurnStrategy with concurrency: budget exhausted, last context is 2*loop_budget long."""
    always_fail = Requirement(
        "always fail", validation_fn=lambda _ctx: ValidationResult(result=False)
    )

    loop_budget = 2
    concurrency_budget = 3
    strategy = MultiTurnStrategy(
        loop_budget=loop_budget, concurrency_budget=concurrency_budget
    )

    result = await strategy.sample(
        action=Instruction(description=""),
        context=ChatContext(),
        backend=mocked_context_backend,
        requirements=[always_fail],
    )

    assert len(result.sample_actions) == loop_budget * concurrency_budget, (
        "forced failures should exhaust the full loop * concurrency budget"
    )
    last_ctx = result.sample_contexts[-1]
    assert len(last_ctx.view_for_generation()) == loop_budget * 2, (  # type: ignore[union-attr]
        "MultiTurnStrategy: last context should be 2*loop_budget long after repair"
    )


# --- Constructor validation ---


def test_loop_budget_must_be_positive():
    with pytest.raises(AssertionError, match="Loop budget"):
        RejectionSamplingStrategy(loop_budget=0)


def test_concurrency_budget_must_be_positive():
    with pytest.raises(AssertionError, match="Concurrency budget"):
        RejectionSamplingStrategy(concurrency_budget=0)


# --- Global vs per-call requirements ---


async def test_global_requirements_override_per_call(mocked_context_backend):
    """When `requirements` is set on the strategy, it supersedes the per-call list."""
    global_req_calls = 0
    per_call_req_calls = 0

    def global_validator(_ctx: Context) -> ValidationResult:
        nonlocal global_req_calls
        global_req_calls += 1
        return ValidationResult(result=True)

    def per_call_validator(_ctx: Context) -> ValidationResult:
        nonlocal per_call_req_calls
        per_call_req_calls += 1
        return ValidationResult(result=True)

    global_req = Requirement("global", validation_fn=global_validator)
    per_call_req = Requirement("per_call", validation_fn=per_call_validator)

    strategy = RejectionSamplingStrategy(loop_budget=1, requirements=[global_req])

    await strategy.sample(
        action=Instruction(description=""),
        context=ChatContext(),
        backend=mocked_context_backend,
        requirements=[per_call_req],
    )

    assert global_req_calls == 1, "global requirement should be evaluated"
    assert per_call_req_calls == 0, (
        "per-call requirement must be ignored when global is set"
    )


# --- Drain on early success: no slices lost ---


async def test_early_success_does_not_lose_in_flight_slices(mocked_context_backend):
    """On early success the result should still contain at least the successful slice.

    The drain in the finally block is best-effort; this test guards against the
    regression where the sentinel/cancellation logic silently dropped the winning
    slice itself.
    """
    always_pass = Requirement(
        "always_pass", validation_fn=lambda _ctx: ValidationResult(result=True)
    )
    strategy = RejectionSamplingStrategy(loop_budget=2, concurrency_budget=4)

    result = await strategy.sample(
        action=Instruction(description=""),
        context=ChatContext(),
        backend=mocked_context_backend,
        requirements=[always_pass],
    )

    assert result.success is True
    assert len(result.sample_generations) >= 1, (
        "the winning slice must be retained even after producer cancellation"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
