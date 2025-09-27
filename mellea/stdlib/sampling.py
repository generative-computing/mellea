"""sampling methods go here."""

import abc
from copy import deepcopy

import tqdm

import mellea.stdlib.funcs as mfuncs
from mellea.backends import Backend, BaseModelSubclass
from mellea.helpers.async_helpers import wait_for_all_mots
from mellea.helpers.fancy_logger import FancyLogger
from mellea.stdlib.base import CBlock, ChatContext, Component, Context, ModelOutputThunk
from mellea.stdlib.chat import Message
from mellea.stdlib.instruction import Instruction
from mellea.stdlib.requirement import Requirement, ScorerRequirement, ValidationResult


class SamplingResult(CBlock):
    """Stores the results from a sampling operation. This includes successful and failed samplings."""

    def __init__(
        self,
        result: ModelOutputThunk,
        result_ctx: Context,
        success: bool,
        *,
        sample_generations: list[ModelOutputThunk] | None = None,
        sample_validations: list[list[tuple[Requirement, ValidationResult]]]
        | None = None,
        sample_actions: list[Component] | None = None,
        sample_contexts: list[Context] | None = None,
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
        self.result_ctx = result_ctx
        self.success = success
        self.sample_generations = sample_generations
        self.sample_validations = sample_validations
        self.sample_actions = sample_actions
        self.sample_contexts = sample_contexts


class SamplingStrategy(abc.ABC):
    """A SamplingStrategy class defines an abstract base class for implementing various sampling strategies.

    This class provides a template for creating concrete sampling strategies that can be used to generate model outputs based on given instructions.
    It allows setting custom validation and generation functions through properties.
    """

    @abc.abstractmethod
    async def sample(
        self,
        action: Component,
        context: Context,
        backend: Backend,
        requirements: list[Requirement],
        *,
        validation_ctx: Context | None = None,
        format: type[BaseModelSubclass] | None = None,
        model_options: dict | None = None,
        tool_calls: bool = False,
    ) -> SamplingResult:
        """This method is the abstract method for sampling a given instruction.

        It must be implemented by any concrete subclasses to provide specific sampling logic.

        Args:
            action : The action object to be sampled.
            context: The context to be passed to the sampling strategy.
            requirements: The requirements to be used by the sampling strategy (merged with global requirements).
            validation_ctx: Optional context to use for validation. If None, validation_ctx = ctx.
        """


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
        """
        Repair function that is being invoked if not all requirements are fulfilled. It should return a next action component.

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
        requirements: list[Requirement],
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
            show_progress: if true, a tqdm progress bar is used. Otherwise, messages will still be sent to flog.
            requirements: List of requirements to test against (merged with global requirements).
            validation_ctx: Optional context to use for validation. If None, validation_ctx = ctx.

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
            val_scores_co = mfuncs._validate(
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
                    result=result,
                    result_ctx=result_ctx,
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
            result=sampled_results[best_failed_index],
            result_ctx=sample_contexts[best_failed_index],
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
        # simply returns the first attempt if all loops fail
        return 0

    @staticmethod
    def repair(
        old_ctx: Context,
        new_ctx: Context,
        past_actions: list[Component],
        past_results: list[ModelOutputThunk],
        past_val: list[list[tuple[Requirement, ValidationResult]]],
    ) -> tuple[Component, Context]:
        # repeat the last action again.
        return past_actions[-1], old_ctx


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
        old_ctx: Context,
        new_ctx: Context,
        past_actions: list[Component],
        past_results: list[ModelOutputThunk],
        past_val: list[list[tuple[Requirement, ValidationResult]]],
    ) -> tuple[Component, Context]:
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
        return pa, old_ctx


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
        old_ctx: Context,
        new_ctx: Context,
        past_actions: list[Component],
        past_results: list[ModelOutputThunk],
        past_val: list[list[tuple[Requirement, ValidationResult]]],
    ) -> tuple[Component, Context]:
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


class BestofNSamplingStrategy(BaseSamplingStrategy):
    """
    Sampling strategy that selects the best response from a set of samples as given by a Requirement Scorer
    """

    async def sample(
        self,
        action: Component,
        context: Context,
        backend: Backend,
        requirements: list[Requirement],
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
            show_progress: if true, a tqdm progress bar is used. Otherwise, messages will still be sent to flog.
            requirements: List of requirements to test against (merged with global requirements).
            validation_ctx: Optional context to use for validation. If None, validation_ctx = ctx.

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

            val_scores_co = mfuncs._validate(
                reqs=reqs,
                context=result_ctx,
                backend=backend,
                output=result,
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

            return SamplingResult(
                best_result,
                result_ctx=best_context,
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
                sampled_results[best_failed_index],
                result_ctx=sample_contexts[best_failed_index],
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
        # select attempt with highest ScoreRequirementScore if all loops fail

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
