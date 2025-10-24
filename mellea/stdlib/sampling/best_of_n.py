"""Best of N Sampling Strategy."""

from copy import deepcopy

import tqdm

from mellea.backends import Backend, BaseModelSubclass
from mellea.helpers.async_helpers import wait_for_all_mots
from mellea.helpers.fancy_logger import FancyLogger
from mellea.stdlib import funcs as mfuncs
from mellea.stdlib.base import CBlock, ChatContext, Component, Context, ModelOutputThunk
from mellea.stdlib.instruction import Instruction
from mellea.stdlib.requirement import Requirement, ScorerRequirement, ValidationResult
from mellea.stdlib.sampling import BaseSamplingStrategy, SamplingResult


class BestofNSamplingStrategy(BaseSamplingStrategy):
    """Sampling strategy that selects the best response from a set of samples as given by a Requirement Scorer."""

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
        assert validation_ctx is not None, "Validation context must be provided."

        flog = FancyLogger.get_logger()

        sampled_results: list[ModelOutputThunk] = []
        sampled_scores: list[list[tuple[Requirement, ValidationResult]]] = []
        sampled_actions: list[Component] = []
        sample_contexts: list[Context] = []

        successful_sampled_results: list[ModelOutputThunk] = []
        successful_sampled_scores: list[list[tuple[Requirement, ValidationResult]]] = []
        successful_sampled_actions: list[Component] = []
        successful_sample_contexts: list[Context] = []

        # The `logging_redirect_tqdm` approach did not work, so instead we will use the show_progress
        # flag to determine whether we should show the pbar.
        show_progress = show_progress and flog.getEffectiveLevel() <= FancyLogger.INFO

        reqs = []
        if self.requirements is not None:
            reqs += self.requirements
        elif requirements is not None:
            reqs += requirements

        reqs = list(set(reqs))

        # check that there is exactly one ScorerRequirement
        scorer_requirements = 0
        for req in reqs:
            # strict typecheck for scorer requirement
            if isinstance(req, ScorerRequirement):
                scorer_requirements += 1

        assert scorer_requirements == 1, (
            "BestOfNSamplingStrategy requires exactly one ScorerRequirement"
        )

        loop_count = 0
        generate_loop_budget_iterator = (
            tqdm.tqdm(range(self.loop_budget))  # type: ignore
            if show_progress
            else range(self.loop_budget)  # type: ignore
        )
        validate_loop_budget_iterator = (
            tqdm.tqdm(range(self.loop_budget))  # type: ignore
            if show_progress
            else range(self.loop_budget)  # type: ignore
        )

        next_action = deepcopy(action)
        next_context = context
        flog.info("BestofNSampling Generating Loop:")
        for _ in generate_loop_budget_iterator:  # type: ignore
            loop_count += 1
            if not show_progress:
                flog.info(f"Running loop {loop_count} of {self.loop_budget}")

            # run a generation pass
            result, result_ctx = backend.generate_from_context(
                next_action,
                ctx=next_context,
                format=format,
                model_options=model_options,
                tool_calls=tool_calls,
            )
            sampled_results.append(result)
            sampled_actions.append(next_action)
            sample_contexts.append(result_ctx)

        await wait_for_all_mots(sampled_results)

        flog.info("BestofNSampling Validation Loop:")
        for i in validate_loop_budget_iterator:
            result_ctx = sample_contexts[i]
            result = sampled_results[i]
            next_action = sampled_actions[i]

            val_scores_co = mfuncs.avalidate(
                reqs=reqs,
                context=result_ctx,
                backend=backend,
                format=format,
                model_options=model_options,
                input=next_action._description,  # type: ignore
                # tool_calls=tool_calls  # Don't support using tool calls in validation strategies.
            )
            val_scores = await val_scores_co

            # match up reqs with scores
            constraint_scores = list(zip(reqs, val_scores))

            # collect all data
            sampled_scores.append(constraint_scores)

            # check if requirements pass else repair and re-sample
            # if all vals are true, save it and continue to get next sample
            if all(bool(s[1]) for s in constraint_scores):
                flog.info("SUCCESS")
                assert (
                    result._generate_log is not None
                )  # Cannot be None after generation.
                result._generate_log.is_final_result = True

                successful_sampled_results.append(result)
                successful_sampled_scores.append(constraint_scores)
                successful_sampled_actions.append(next_action)
                successful_sample_contexts.append(result_ctx)

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

        # find max reward amongst results for which all requirements have passed
        if len(successful_sampled_scores) > 0:
            scores: list[float] = []
            scorer_preference_ordering = None

            for sample in successful_sampled_scores:
                for req, val_score in sample:
                    if isinstance(req, ScorerRequirement):
                        assert val_score._score is not None
                        scores.append(val_score._score)
                        scorer_preference_ordering = req.preference_ordering

            assert len(successful_sampled_results) == len(scores)
            assert scorer_preference_ordering is not None

            if scorer_preference_ordering == "max":
                best_result, best_score, best_context = max(
                    zip(successful_sampled_results, scores, successful_sample_contexts),
                    key=lambda x: x[1],
                )
            elif scorer_preference_ordering == "min":
                best_result, best_score, best_context = min(
                    zip(successful_sampled_results, scores, successful_sample_contexts),
                    key=lambda x: x[1],
                )
            else:
                raise NotImplementedError

            best_index = sampled_results.index(best_result)

            return SamplingResult(
                result_index=best_index,
                success=True,
                sample_generations=sampled_results,
                sample_validations=sampled_scores,
                sample_actions=sampled_actions,
                sample_contexts=sample_contexts,
            )

        # if all failures, call select from failure
        else:
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
            return SamplingResult(
                result_index=best_failed_index,
                success=False,
                sample_generations=sampled_results,
                sample_validations=sampled_scores,
                sample_actions=sampled_actions,
                sample_contexts=sample_contexts,
            )

    @staticmethod
    def select_from_failure(
        sampled_actions: list[Component],
        sampled_results: list[ModelOutputThunk],
        sampled_val: list[list[tuple[Requirement, ValidationResult]]],
    ) -> int:
        """Selects the attempt with the highest score.

        Args:
            sampled_actions: List of actions that have been executed (without success).
            sampled_results: List of (unsuccessful) generation results for these actions.
            sampled_val: List of validation results for the results.

        Returns:
            The index of the result that should be selected as `.value`.
        """
        scores: list[float | None] = []

        for sample in sampled_val:
            for req, val_score in sample:
                if isinstance(req, ScorerRequirement):
                    assert val_score._score is not None
                    scores.append(val_score._score)

        assert len(sampled_results) == len(scores)

        return scores.index(max(scores))  # type: ignore

    @staticmethod
    def repair(
        old_ctx: Context,
        new_ctx: Context,
        past_actions: list[Component],
        past_results: list[ModelOutputThunk],
        past_val: list[list[tuple[Requirement, ValidationResult]]],
    ) -> tuple[Component, Context]:
        """Adds a description of the requirements that failed to a copy of the original instruction.

        Args:
            old_ctx: The context WITHOUT the last action + output.
            new_ctx: The context including the last action + output.
            past_actions: List of actions that have been executed (without success).
            past_results: List of (unsuccessful) generation results for these actions.
            past_val: List of validation results for the results.

        Returns:
            The next action component and context to be used for the next generation attempt.
        """
        pa = past_actions[-1]
        if isinstance(pa, Instruction):
            last_failed_reqs: list[Requirement] = [
                s[0] for s in past_val[-1] if not s[1]
            ]
            last_failed_reqs_str = "* " + "\n* ".join(
                [str(r.description) for r in last_failed_reqs]
            )
            return pa.copy_and_repair(
                repair_string=f"The following requirements failed before:\n{last_failed_reqs_str}"
            ), old_ctx
        return past_actions[-1], old_ctx
