"""Sampling Strategies for budget forcing generation."""

from copy import deepcopy

import tqdm

from mellea.backends import Backend, BaseModelSubclass
from mellea.backends.ollama import OllamaModelBackend
from mellea.helpers.fancy_logger import FancyLogger
from mellea.stdlib import functional as mfuncs
from mellea.stdlib.base import ModelOutputThunk
from mellea.stdlib.requirement import Requirement, ValidationResult
from mellea.stdlib.sampling import RejectionSamplingStrategy, SamplingResult
from mellea.stdlib.sampling.base import Component, Context
from mellea.stdlib.sampling_algos.budget_forcing_alg import think_budget_forcing


class BudgetForcingSamplingStrategy(RejectionSamplingStrategy):
    """Budget forcing sampling class."""

    think_max_tokens: int | None
    answer_max_tokens: int | None
    start_think_token: str | None
    end_think_token: str | None
    begin_response_token: str | None
    end_response_token: str
    think_more_suffix: str | None
    answer_suffix: str | None

    def __init__(
        self,
        *,
        think_max_tokens: int | None = 4096,
        answer_max_tokens: int | None = None,
        start_think_token: str | None = "<think>",
        end_think_token: str | None = "</think>",
        begin_response_token: str | None = "",
        end_response_token: str = "",
        think_more_suffix: str | None = "",
        answer_suffix: str | None = "",
        loop_budget: int = 1,
        requirements: list[Requirement] | None,
    ):
        r"""Initialize class.

        Inherits from RejectionSamplingStrategy.

        Args:
            think_max_tokens: Number of tokens for think block
            answer_max_tokens: Number of tokens allocated for answer portion, if set to None answer tokens will be unlimited
            start_think_token: Special start of think block token defaults to '<think>'
            end_think_token: Special end of think block token defaults to '</think>'
            begin_response_token: Special begin of response block token e.g. '<response>' defaults to ""
            end_response_token: Special end of response block token e.g. '</response>' defaults to ""
            think_more_suffix: Suffix for continue thinking e.g. "\nWait let's think more carefully" to force the model to think more, defaults to "".  If set to "", no force thinking will be applied, the token budget will be become an upper bound.
            answer_suffix: Suffix to obtain final answer, default to "\nThe final answer is:"
            loop_budget: Number of times to iterate through the process. Must be greater than 0.
            requirements: List of requirements to test against. If None, test all requirements attached to the given instruction.

        Raises:
            AssertionError: If loop_budget is not greater than 0.
        """
        super().__init__(loop_budget=loop_budget, requirements=requirements)
        self.think_max_tokens = think_max_tokens
        self.answer_max_tokens = answer_max_tokens
        self.start_think_token = start_think_token
        self.end_think_token = end_think_token
        self.begin_response_token = begin_response_token
        self.end_response_token = end_response_token
        self.think_more_suffix = think_more_suffix
        self.answer_suffix = answer_suffix

    async def sample(
        self,
        action: Component,
        context: Context,
        backend: Backend,
        requirements: list[Requirement] | None,
        *,
        validation_ctx: Context | None = None,
        format: type[BaseModelSubclass] | None = None,
        model_options: dict | None = None,
        tool_calls: bool = False,
        show_progress: bool = True,
    ) -> SamplingResult:
        """This method performs a sampling operation based on the given instruction.

        Args:
            action : The action object to be sampled.
            context: The context to be passed to the sampling strategy.
            backend: The backend used for generating samples.
            requirements: List of requirements to test against (merged with global requirements).
            validation_ctx: Optional context to use for validation. If None, validation_ctx = ctx.
            format: output format for structured outputs.
            model_options: model options to pass to the backend during generation / validation.
            tool_calls: True if tool calls should be used during this sampling strategy.
            show_progress: if true, a tqdm progress bar is used. Otherwise, messages will still be sent to flog.

        Returns:
            SamplingResult: A result object indicating the success or failure of the sampling process.

        Raises:
            AssertionError: Asserts that all required components (repair, select_from_failure, validate, and generate) are provided before proceeding with the sampling.
        """
        validation_ctx = validation_ctx if validation_ctx is not None else context

        flog = FancyLogger.get_logger()

        sampled_results: list[ModelOutputThunk] = []
        sampled_scores: list[list[tuple[Requirement, ValidationResult]]] = []
        sampled_actions: list[Component] = []
        sample_contexts: list[Context] = []

        # The `logging_redirect_tqdm` approach did not work, so instead we will use the show_progress
        # flag to determine whether we should show the pbar.
        show_progress = show_progress and flog.getEffectiveLevel() <= FancyLogger.INFO

        reqs = []
        # global requirements supersede local requirements (global requirements can be defined by user)
        # Todo: re-evaluate if this makes sense
        if self.requirements is not None:
            reqs += self.requirements
        elif requirements is not None:
            reqs += requirements
        reqs = list(set(reqs))

        loop_count = 0
        loop_budget_range_iterator = (
            tqdm.tqdm(range(self.loop_budget))  # type: ignore
            if show_progress
            else range(self.loop_budget)  # type: ignore
        )

        next_action = deepcopy(action)
        next_context = context
        for _ in loop_budget_range_iterator:  # type: ignore
            loop_count += 1
            if not show_progress:
                flog.info(f"Running loop {loop_count} of {self.loop_budget}")

            # TODO
            # tool_calls is not supported for budget forcing
            assert tool_calls is False, (
                "tool_calls is not supported with budget forcing"
            )
            # TODO
            assert isinstance(backend, OllamaModelBackend), (
                "Only ollama backend supported with budget forcing"
            )
            # run a generation pass with budget forcing
            result = await think_budget_forcing(
                backend,
                next_action,
                ctx=context,
                format=format,
                tool_calls=tool_calls,
                think_max_tokens=self.think_max_tokens,
                answer_max_tokens=self.answer_max_tokens,
                start_think_token=self.start_think_token,
                end_think_token=self.end_think_token,
                think_more_suffix=self.think_more_suffix,
                answer_suffix=self.answer_suffix,
                model_options=model_options,
            )
            result_ctx = next_context

            # validation pass
            val_scores_co = mfuncs.avalidate(
                reqs=reqs,
                context=result_ctx,
                backend=backend,
                output=result,
                format=format,
                model_options=model_options,
                # tool_calls=tool_calls  # Don't support using tool calls in validation strategies.
            )
            val_scores = await val_scores_co

            # match up reqs with scores
            constraint_scores = list(zip(reqs, val_scores))

            # collect all data
            sampled_results.append(result)
            sampled_scores.append(constraint_scores)
            sampled_actions.append(next_action)
            sample_contexts.append(result_ctx)

            # if all vals are true -- break and return success
            if all(bool(s[1]) for s in constraint_scores):
                flog.info("SUCCESS")
                assert (
                    result._generate_log is not None
                )  # Cannot be None after generation.
                result._generate_log.is_final_result = True

                # SUCCESS !!!!
                return SamplingResult(
                    result_index=len(sampled_results) - 1,
                    success=True,
                    sample_generations=sampled_results,
                    sample_validations=sampled_scores,
                    sample_contexts=sample_contexts,
                    sample_actions=sampled_actions,
                )

            else:
                # log partial success and continue
                count_valid = len([s for s in constraint_scores if bool(s[1])])
                flog.info(f"FAILED. Valid: {count_valid}/{len(constraint_scores)}")

            # If we did not pass all constraints, update the instruction and try again.
            next_action, next_context = self.repair(
                next_context,
                result_ctx,
                sampled_actions,
                sampled_results,
                sampled_scores,
            )

        flog.info(
            f"Invoking select_from_failure after {len(sampled_results)} failed attempts."
        )

        # if no valid result could be determined, find a last resort.
        best_failed_index = self.select_from_failure(
            sampled_actions, sampled_results, sampled_scores
        )
        assert best_failed_index < len(sampled_results), (
            "The select_from_failure method did not return a valid result. It has to selected from failed_results."
        )

        assert (
            sampled_results[best_failed_index]._generate_log is not None
        )  # Cannot be None after generation.
        sampled_results[best_failed_index]._generate_log.is_final_result = True  # type: ignore

        return SamplingResult(
            result_index=best_failed_index,
            success=False,
            sample_generations=sampled_results,
            sample_validations=sampled_scores,
            sample_actions=sampled_actions,
            sample_contexts=sample_contexts,
        )
