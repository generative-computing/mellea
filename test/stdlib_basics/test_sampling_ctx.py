import asyncio
from unittest.mock import Mock
import pytest
from mellea import start_session
from mellea.backends import Backend, ModelOption
from mellea.stdlib.base import ChatContext, GenerateLog, ModelOutputThunk, Context
from mellea.stdlib.instruction import Instruction
from mellea.stdlib.requirement import Requirement, ValidationResult
from mellea.stdlib.sampling import (
    MultiTurnStrategy,
    RejectionSamplingStrategy,
    SamplingResult,
)
from mellea.stdlib.sampling.base import _SamplingResultSlice, RepairTemplateStrategy


class TestSamplingCtxCase:
    m = start_session(
        model_options={ModelOption.MAX_NEW_TOKENS: 100}, ctx=ChatContext()
    )

    def _run_asserts_for_ctx_testing(self, res):
        assert isinstance(res, SamplingResult), "res should be a SamplingResult."

        assert isinstance(res.value, str), "Value should be set and a string."

        assert len(res.sample_generations) >= 1, (
            "sample generation should have at least one sample."
        )
        assert len(res.sample_validations) >= 1, (
            "sample validation should have at least one sample."
        )
        assert len(res.sample_validations[0]) == 3, (
            "there should be 3 validation results."
        )

    def test_ctx_for_rejection_sampling(self):
        self.m.reset()
        res = self.m.instruct(
            "Write a sentence.",
            requirements=[
                "be funny",
                "be formal",
                "use only words starting with the letter w",
            ],
            strategy=RejectionSamplingStrategy(loop_budget=3),
            return_sampling_results=True,
        )
        self._run_asserts_for_ctx_testing(res)
        assert len(self.m.ctx.as_list()) == 2, (
            "there should only be a message and a response in the ctx."
        )
        assert len(self.m.last_prompt()) == 1, (
            "Last prompt should only have only one instruction inside - independent of sampling iterations."
        )

        _, val_res = res.result_validations[0]
        # Ensure the ValidationResult has its thunk and context set. Ensure the context has
        # the correct actions / results in it.
        assert isinstance(val_res.context, Context)
        assert isinstance(val_res.thunk, ModelOutputThunk)
        assert isinstance(val_res.context.previous_node.node_data, Requirement)
        assert val_res.context.node_data is val_res.thunk

    def test_ctx_for_multiturn(self):
        self.m.reset()
        res = self.m.instruct(
            "Write a sentence.",
            requirements=[
                "be funny",
                "be formal",
                "use only words starting with the letter w",
            ],
            strategy=MultiTurnStrategy(loop_budget=3),
            return_sampling_results=True,
        )

        self._run_asserts_for_ctx_testing(res)
        assert len(self.m.ctx.as_list()) >= 2, (
            "there should be at least a message and a response in the ctx; more if the first result failed validation"
        )
        assert len(self.m.last_prompt()) == len(res.sample_generations) * 2 - 1, (
            "For n sampling iterations there should be 2n-1 prompt conversation elements in the last prompt."
        )


@pytest.fixture(scope="function")
def mocked_context_backend(sleep_time: float = 1) -> Backend:
    backend = Mock()

    async def mock_generate(*args, **kwargs):
        # We know that these will be passed as kwargs since we are invoking the generate call.
        action = kwargs.pop("action")
        ctx = kwargs.pop("ctx")

        await asyncio.sleep(sleep_time)
        output = ModelOutputThunk("mocked")
        output._generate_log = GenerateLog()
        return output, ctx.add(action).add(output)

    backend.generate_from_context = mock_generate
    return backend


# Have to define this globally for ease of use.
counter = 1


async def test_rejection_sampling_with_concurrency_early_success(
    mocked_context_backend,
):
    backend = mocked_context_backend

    ctx = ChatContext()

    def sometimes_pass(ctx: Context) -> ValidationResult:
        global counter
        if counter % 5 == 0:
            val_result = ValidationResult(True)
        else:
            val_result = ValidationResult(False)

        counter += 1
        return val_result

    sometimes_pass_req = Requirement("sometimes_pass", validation_fn=sometimes_pass)

    loop_budget = 3
    concurrency_budget = 3
    sampling_strat = RejectionSamplingStrategy(
        loop_budget=loop_budget, concurrency_budget=concurrency_budget
    )

    result = await sampling_strat.sample(
        action=Instruction(""),
        context=ctx,
        backend=backend,
        requirements=[sometimes_pass_req],
    )

    # Note: we can't necessarily assert that the 4th index has the successful result since items may be added to the
    # queue in a somewhat random order.
    assert len(result.sample_actions) < loop_budget * concurrency_budget, (
        "success on the 5th request means the total samples should be less than the budget"
    )


async def test_subsample_iteration(mocked_context_backend):
    backend = mocked_context_backend

    ctx = ChatContext()
    always_pass_req = Requirement(
        "always pass", validation_fn=lambda x: ValidationResult(result=True)
    )

    loop_budget = 5
    concurrency_budget = 5
    sampling_strat = RejectionSamplingStrategy(
        loop_budget=loop_budget, concurrency_budget=concurrency_budget
    )

    generator = sampling_strat._subsample_iteration(
        iterations=2,
        action=Instruction(""),
        context=ctx,
        backend=backend,
        requirements=[always_pass_req],
    )

    slice = await generator.__anext__()
    assert isinstance(slice, _SamplingResultSlice)

    # Generator should stop after the first successful
    with pytest.raises(StopAsyncIteration):
        await generator.__anext__()


async def test_repair_strategy_with_concurrency(mocked_context_backend):
    backend = mocked_context_backend

    ctx = ChatContext()
    req = Requirement(
        "always fail", validation_fn=lambda x: ValidationResult(result=False)
    )

    loop_budget = 5
    concurrency_budget = 5
    sampling_strat = RepairTemplateStrategy(
        loop_budget=loop_budget, concurrency_budget=concurrency_budget
    )

    result = await sampling_strat.sample(
        action=Instruction(""), context=ctx, backend=backend, requirements=[req]
    )

    assert len(result.sample_actions) == loop_budget * concurrency_budget, (
        "forced failures mean we should exhaust the loop"
    )
    assert (
        result.sample_contexts[-1].view_for_generation()[0]._repair_string is not None  # type: ignore
    ), "last batch of retries should have a repair string with `RepairTemplateStrategy`"  


async def test_mulit_turn_strategy_with_concurrency(mocked_context_backend):
    backend = mocked_context_backend

    ctx = ChatContext()
    req = Requirement(
        "always fail", validation_fn=lambda x: ValidationResult(result=False)
    )

    loop_budget = 2
    concurrency_budget = 3
    sampling_strat = MultiTurnStrategy(
        loop_budget=loop_budget, concurrency_budget=concurrency_budget
    )
    result = await sampling_strat.sample(
        action=Instruction(""), context=ctx, backend=backend, requirements=[req]
    )
    assert len(result.sample_actions) == loop_budget * concurrency_budget, (
        "forced failures mean we should exhaust the loop"
    )
    assert len(result.sample_contexts[-1].view_for_generation()) == loop_budget * 2, (  # type: ignore
        "last batch of retries should have a context length of `2*loop_budget` with `MultiTurnStrategy`"
    )


if __name__ == "__main__":
    pytest.main([__file__])
