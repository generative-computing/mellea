"""sampling methods go here."""

import abc
from collections.abc import Callable, Coroutine
from copy import deepcopy
from typing import Any

import tqdm

from mellea import LinearContext
from mellea.helpers.fancy_logger import FancyLogger
from mellea.stdlib.base import (
    CBlock,
    Component,
    Context,
    ContextTurn,
    GenerateLog,
    ModelOutputThunk,
)
from mellea.stdlib.chat import Message
from mellea.stdlib.instruction import Instruction
from mellea.stdlib.requirement import Requirement, ValidationResult


class SamplingResult(CBlock):
    """Stores the results from a sampling operation. This includes successful and failed samplings."""

    def __init__(
        self,
        result: ModelOutputThunk,
        success: bool,
        *,
        sample_generations: list[ModelOutputThunk] | None = None,
        sample_validations: list[list[tuple[Requirement, ValidationResult]]]
        | None = None,
        sample_actions: list[Component] | None = None,
    ):
        """Initialize a new instance of sampling results.

        Args:
            result: The final output or result from applying the sampling strategy.
            success: A boolean indicating whether the operation was successful.
            sample_generations: A list containing intermediate generations produced during the process.
            sample_validations: For each generation a list of tuples of a requirement and a validation result.
        """
        super().__init__(value=result.value)
        self.result = result
        self.success = success
        self.sample_generations = sample_generations
        self.sample_validations = sample_validations
        self.sample_actions = sample_actions


# TODO: JAL. Sampling strategies need to define a parallel width parameter...
#            ie, how many generation threads should run at the same time
#            the breadth of the tree rather than the depth
class SamplingStrategy(abc.ABC):
    """A SamplingStrategy class defines an abstract base class for implementing various sampling strategies.

    This class provides a template for creating concrete sampling strategies that can be used to generate model outputs based on given instructions.
    It allows setting custom validation and generation functions through properties.
    """

    # the function signature here matches that of m.validate
    validate: (
        Callable[
            [list[Requirement], Context, Any],
            Coroutine[Any, Any, list[ValidationResult]],
        ]
        | None
    ) = None

    generate: (
        Callable[[Component, Context, list[GenerateLog] | None], ModelOutputThunk]
        | None
    ) = None

    @abc.abstractmethod
    async def sample(
        self,
        action: Component,
        context: Context,
        requirements: list[Requirement],
        *,
        generate_logs: list[GenerateLog] | None = None,
        validation_ctx: Context | None = None,
    ) -> SamplingResult:
        """This method is the abstract method for sampling a given instruction.

        It must be implemented by any concrete subclasses to provide specific sampling logic.

        Args:
            action : The action object to be sampled.
            context: The context to be passed to the sampling strategy.
            requirements: The requirements to be used by the sampling strategy (merged with global requirements).
            generate_logs: Optional list of GenerateLog objects. If None, no collection happens.
            validation_ctx: Optional context to use for validation. If None, validation_ctx = ctx.
        """


class BaseSamplingStrategy(SamplingStrategy):
    """Base class for multiple strategies that rejects samples based on given instructions."""

    loop_budget: int

    def __init__(
        self,
        *,
        loop_budget: int = 1,
        validate: Callable[
            [list[Requirement], Context, Any],
            Coroutine[Any, Any, list[ValidationResult]],
        ]
        | None = None,
        generate: (
            Callable[[Component, Context, list[GenerateLog] | None], ModelOutputThunk]
            | None
        ) = None,
        requirements: list[Requirement] | None = None,
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
        self.validate = validate  # it's ok to be None here
        self.generate = generate  # it's ok to be None here
        self.requirements = requirements

    @staticmethod
    @abc.abstractmethod
    def repair(
        ctx: Context,
        past_actions: list[Component],
        past_results: list[ModelOutputThunk],
        past_val: list[list[tuple[Requirement, ValidationResult]]],
    ) -> Component:
        """
        Repair function that is being invoked if not all requirements are fulfilled. It should return a next action component.

        Args:
            ctx: The context to be passed to the sampling strategy.
            past_actions: List of actions that have been executed (without success).
            past_results: List of (unsuccessful) generation results for these actions.
            past_val: List of validation results for the results.

        Returns:
            The next action component.
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

    # TODO: JAL; this sampling strategy would use synchronous generation if
    #            we want to keep that. Still use asynchronous validation.
    async def sample(
        self,
        action: Component,
        context: Context,
        requirements: list[Requirement],
        *,
        show_progress: bool = True,
        generate_logs: list[GenerateLog] | None = None,
        validation_ctx: Context | None = None,
    ) -> SamplingResult:
        """This method performs a sampling operation based on the given instruction.

        Args:
            action : The action object to be sampled.
            context: The context to be passed to the sampling strategy.
            show_progress: if true, a tqdm progress bar is used. Otherwise, messages will still be sent to flog.
            generate_logs: If provided, the generations will be logged.
            requirements: List of requirements to test against (merged with global requirements).
            validation_ctx: Optional context to use for validation. If None, validation_ctx = ctx.

        Returns:
            SamplingResult: A result object indicating the success or failure of the sampling process.

        Raises:
            AssertionError: Asserts that all required components (repair, select_from_failure, validate, and generate) are provided before proceeding with the sampling.
        """
        assert self.validate is not None, "Validation must be provided."
        assert self.generate is not None, "Generate must be provided."

        # just to be sure to not cause issues to the OG context
        ctx = context.copy()
        validation_ctx = validation_ctx if validation_ctx is not None else context
        assert validation_ctx is not None, "Validation context must be provided."

        flog = FancyLogger.get_logger()

        sampled_results: list[ModelOutputThunk] = []
        sampled_scores: list[list[tuple[Requirement, ValidationResult]]] = []
        sampled_actions: list[Component] = []

        # The `logging_redirect_tqdm` approach did not work, so instead we will use the show_progress
        # flag to determine whether we should show the pbar.
        show_progress = show_progress and flog.getEffectiveLevel() <= FancyLogger.INFO

        reqs = []
        # global requirements supersede local requirements (global requiremenst can be defined by user)
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

        new_action = deepcopy(action)
        for _ in loop_budget_range_iterator:  # type: ignore
            loop_count += 1
            if not show_progress:
                flog.info(f"Running loop {loop_count} of {self.loop_budget}")

            # run a generation pass
            result = self.generate(new_action, ctx, generate_logs)
            await result.avalue()

            # validation pass
            val_scores_co = self.validate(reqs, validation_ctx, result)
            val_scores = await val_scores_co

            # match up reqs with scores
            constraint_scores = list(zip(reqs, val_scores))

            # collect all data
            sampled_results.append(result)
            sampled_scores.append(constraint_scores)
            sampled_actions.append(new_action)

            # if all vals are true -- break and return success
            if all(bool(s[1]) for s in constraint_scores):
                flog.info("SUCCESS")
                assert (
                    result._generate_log is not None
                )  # Cannot be None after generation.
                result._generate_log.is_final_result = True

                return SamplingResult(
                    result,
                    success=True,
                    sample_generations=sampled_results,
                    sample_validations=sampled_scores,
                )

            else:
                # log partial success and continue
                count_valid = len([s for s in constraint_scores if bool(s[1])])
                flog.info(f"FAILED. Valid: {count_valid}/{len(constraint_scores)}")

            # If we did not pass all constraints, update the instruction and try again.
            new_action = self.repair(
                ctx, sampled_actions, sampled_results, sampled_scores
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
            sampled_results[best_failed_index],
            success=False,
            sample_generations=sampled_results,
            sample_validations=sampled_scores,
            sample_actions=sampled_actions,
        )


class RejectionSamplingStrategy(BaseSamplingStrategy):
    """Simple rejection sampling strategy that just repeats the same call on failure."""

    @staticmethod
    def select_from_failure(
        sampled_actions: list[Component],
        sampled_results: list[ModelOutputThunk],
        sampled_val: list[list[tuple[Requirement, ValidationResult]]],
    ) -> int:
        # simply returns the first attempt if all loops fail
        return 0

    @staticmethod
    def repair(
        ctx: Context,
        past_actions: list[Component],
        past_results: list[ModelOutputThunk],
        past_val: list[list[tuple[Requirement, ValidationResult]]],
    ) -> Component:
        # repeat the last action again.
        return past_actions[-1]


class RepairTemplateStrategy(BaseSamplingStrategy):
    """A sampling strategy that adds a repair string to the instruction object."""

    @staticmethod
    def select_from_failure(
        sampled_actions: list[Component],
        sampled_results: list[ModelOutputThunk],
        sampled_val: list[list[tuple[Requirement, ValidationResult]]],
    ) -> int:
        # simply returns the first attempt if all loops fail
        return 0

    @staticmethod
    def repair(
        ctx: Context,
        past_actions: list[Component],
        past_results: list[ModelOutputThunk],
        past_val: list[list[tuple[Requirement, ValidationResult]]],
    ) -> Component:
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
            )
        return past_actions[-1]


class MultiTurnStrategy(BaseSamplingStrategy):
    """Rejection sampling strategy with (agentic) multi-turn repair."""

    @staticmethod
    def select_from_failure(
        sampled_actions: list[Component],
        sampled_results: list[ModelOutputThunk],
        sampled_val: list[list[tuple[Requirement, ValidationResult]]],
    ):
        # return the last assistant message even if all attempts of repair failed.
        return -1

    @staticmethod
    def repair(
        ctx: Context,
        past_actions: list[Component],
        past_results: list[ModelOutputThunk],
        past_val: list[list[tuple[Requirement, ValidationResult]]],
    ) -> Component:
        assert isinstance(ctx, LinearContext), (
            " Need linear context to run agentic sampling."
        )

        # add failed execution to chat history
        ctx.insert_turn(ContextTurn(past_actions[-1], past_results[-1]))

        last_failed_reqs: list[Requirement] = [s[0] for s in past_val[-1] if not s[1]]
        last_failed_reqs_str = "* " + "\n* ".join(
            [str(r.description) for r in last_failed_reqs]
        )
        # TODO: what to do with checks ??

        next_action = Message(
            role="user",
            content=f"The following requirements have not been met: \n{last_failed_reqs_str}\n Please try again to fulfill the requirements.",
        )

        return next_action
