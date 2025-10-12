"""Base Sampling Strategies."""

import abc
from copy import deepcopy

import tqdm

from mellea.backends import Backend, BaseModelSubclass
from mellea.helpers.fancy_logger import FancyLogger
from mellea.stdlib import funcs as mfuncs
from mellea.stdlib.base import CBlock, ChatContext, Component, Context, ModelOutputThunk
from mellea.stdlib.chat import Message
from mellea.stdlib.instruction import Instruction
from mellea.stdlib.requirement import Requirement, ValidationResult

from .types import SamplingResult, SamplingStrategy


class BaseSamplingStrategy(SamplingStrategy):
    """Base class for multiple strategies that rejects samples based on given instructions."""

    loop_budget: int

    def __init__(
        self, *, loop_budget: int = 1, requirements: list[Requirement] | None = None
    ):
        """Initialize a new instance of the class with default parameters.

        Args:
            loop_budget: Number of times to iterate through the process. Must be greater than 0.
            validate: Function to validate the results against requirements. If None, validation is provided later through setter.
            generate: Function to generate new model output thunks. If None, generate is provided later through setter.
            requirements: List of requirements to test against. If None, test all requirements attached to the given instruction.

        Raises:
            AssertionError: If loop_budget is not greater than 0.
        """
        assert loop_budget > 0, "Loop budget must be at least 1."

        self.loop_budget = loop_budget
        self.requirements = requirements

    @staticmethod
    @abc.abstractmethod
    def repair(
        old_ctx: Context,
        new_ctx: Context,
        past_actions: list[Component],
        past_results: list[ModelOutputThunk],
        past_val: list[list[tuple[Requirement, ValidationResult]]],
    ) -> tuple[Component, Context]:
        """Repair function that is being invoked if not all requirements are fulfilled. It should return a next action component.

        Args:
            old_ctx: The context WITHOUT the last action + output.
            new_ctx: The context including the last action + output.
            past_actions: List of actions that have been executed (without success).
            past_results: List of (unsuccessful) generation results for these actions.
            past_val: List of validation results for the results.

        Returns:
            The next action component and context to be used for the next generation attempt.
        """
        ...

    @staticmethod
    @abc.abstractmethod
    def select_from_failure(
        sampled_actions: list[Component],
        sampled_results: list[ModelOutputThunk],
        sampled_val: list[list[tuple[Requirement, ValidationResult]]],
    ) -> int:
        """This function returns the index of the result that should be selected as `.value` iff the loop budget is exhausted and no success.

        Args:
            sampled_actions: List of actions that have been executed (without success).
            sampled_results: List of (unsuccessful) generation results for these actions.
            sampled_val: List of validation results for the results.

        Returns:
            The index of the result that should be selected as `.value`.
        """
        ...

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

            # run a generation pass
            result, result_ctx = backend.generate_from_context(
                next_action,
                ctx=next_context,
                format=format,
                model_options=model_options,
                tool_calls=tool_calls,
            )
            await result.avalue()

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


class RejectionSamplingStrategy(BaseSamplingStrategy):
    """Simple rejection sampling strategy that just repeats the same call on failure."""

    @staticmethod
    def select_from_failure(
        sampled_actions: list[Component],
        sampled_results: list[ModelOutputThunk],
        sampled_val: list[list[tuple[Requirement, ValidationResult]]],
    ) -> int:
        """Always returns the 0th index.

        Args:
            sampled_actions: List of actions that have been executed (without success).
            sampled_results: List of (unsuccessful) generation results for these actions.
            sampled_val: List of validation results for the results.

        Returns:
            The index of the result that should be selected as `.value`.
        """
        return 0

    @staticmethod
    def repair(
        old_ctx: Context,
        new_ctx: Context,
        past_actions: list[Component],
        past_results: list[ModelOutputThunk],
        past_val: list[list[tuple[Requirement, ValidationResult]]],
    ) -> tuple[Component, Context]:
        """Always returns the unedited, last action.

        Args:
            old_ctx: The context WITHOUT the last action + output.
            new_ctx: The context including the last action + output.
            past_actions: List of actions that have been executed (without success).
            past_results: List of (unsuccessful) generation results for these actions.
            past_val: List of validation results for the results.

        Returns:
            The next action component and context to be used for the next generation attempt.
        """
        return past_actions[-1], old_ctx


class RepairTemplateStrategy(BaseSamplingStrategy):
    """A sampling strategy that adds a repair string to the instruction object."""

    @staticmethod
    def select_from_failure(
        sampled_actions: list[Component],
        sampled_results: list[ModelOutputThunk],
        sampled_val: list[list[tuple[Requirement, ValidationResult]]],
    ) -> int:
        """Always returns the 0th index.

        Args:
            sampled_actions: List of actions that have been executed (without success).
            sampled_results: List of (unsuccessful) generation results for these actions.
            sampled_val: List of validation results for the results.

        Returns:
            The index of the result that should be selected as `.value`.
        """
        return 0

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
            # Get failed requirements and their detailed validation reasons
            failed_items = [
                (req, val) for req, val in past_val[-1] if not val.as_bool()
            ]

            # Build repair feedback using ValidationResult.reason when available
            repair_lines = []
            for req, validation in failed_items:
                if validation.reason:
                    repair_lines.append(f"* {validation.reason}")
                else:
                    # Fallback to requirement description if no reason
                    repair_lines.append(f"* {req.description}")

            repair_string = "The following requirements failed before:\n" + "\n".join(
                repair_lines
            )

            return pa.copy_and_repair(repair_string=repair_string), old_ctx
        return pa, old_ctx


class MultiTurnStrategy(BaseSamplingStrategy):
    """Rejection sampling strategy with (agentic) multi-turn repair."""

    @staticmethod
    def select_from_failure(
        sampled_actions: list[Component],
        sampled_results: list[ModelOutputThunk],
        sampled_val: list[list[tuple[Requirement, ValidationResult]]],
    ):
        """Always returns the last index. The last message from the model will always be returned if all results are failures.

        Args:
            sampled_actions: List of actions that have been executed (without success).
            sampled_results: List of (unsuccessful) generation results for these actions.
            sampled_val: List of validation results for the results.

        Returns:
            The index of the result that should be selected as `.value`.
        """
        return -1

    @staticmethod
    def repair(
        old_ctx: Context,
        new_ctx: Context,
        past_actions: list[Component],
        past_results: list[ModelOutputThunk],
        past_val: list[list[tuple[Requirement, ValidationResult]]],
    ) -> tuple[Component, Context]:
        """Returns a Message with a description of the failed requirements.

        Args:
            old_ctx: The context WITHOUT the last action + output.
            new_ctx: The context including the last action + output.
            past_actions: List of actions that have been executed (without success).
            past_results: List of (unsuccessful) generation results for these actions.
            past_val: List of validation results for the results.

        Returns:
            The next action component and context to be used for the next generation attempt.
        """
        assert isinstance(new_ctx, ChatContext), (
            " Need chat context to run agentic sampling."
        )

        last_failed_reqs: list[Requirement] = [s[0] for s in past_val[-1] if not s[1]]
        last_failed_reqs_str = "* " + "\n* ".join(
            [str(r.description) for r in last_failed_reqs]
        )
        # TODO: what to do with checks ??

        next_action = Message(
            role="user",
            content=f"The following requirements have not been met: \n{last_failed_reqs_str}\n Please try again to fulfill the requirements.",
        )

        return next_action, new_ctx
