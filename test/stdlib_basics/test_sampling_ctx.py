import pytest

from mellea import start_session
from mellea.backends import ModelOption
from mellea.stdlib.base import ChatContext, Context, ModelOutputThunk
from mellea.stdlib.requirement import Requirement
from mellea.stdlib.sampling import (
    MultiTurnStrategy,
    RejectionSamplingStrategy,
    SamplingResult,
)


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


if __name__ == "__main__":
    pytest.main([__file__])
