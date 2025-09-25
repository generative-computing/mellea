"""sampling methods go here."""

import abc
import re
from asyncio import TaskGroup  # type: ignore[attr-defined]
from collections import Counter
from collections.abc import Callable, Coroutine
from copy import deepcopy
from typing import Any

import numpy as np
import tqdm
from math_verify import ExprExtractionConfig, LatexExtractionConfig, parse, verify
from rouge_score.rouge_scorer import RougeScorer  # codespell:ignore

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
from mellea.stdlib.requirement import Requirement, ScorerRequirement, ValidationResult


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


class SamplingStrategy(abc.ABC):
    """A SamplingStrategy class defines an abstract base class for implementing various sampling strategies.

    This class provides a template for creating concrete sampling strategies that can be used to generate model outputs based on given instructions.
    It allows setting custom validation and generation functions through properties.
    """

    # the function signature here matches that of m.validate
    validate: (
        Callable[
            [list[Requirement], Context, Any, Any],
            Coroutine[Any, Any, list[ValidationResult]],
        ]
        | None
    ) = None

    generate: Callable[[Component, Context], ModelOutputThunk] | None = None

    @abc.abstractmethod
    async def sample(
        self,
        action: Component,
        context: Context,
        requirements: list[Requirement],
        *,
        validation_ctx: Context | None = None,
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
        self,
        *,
        loop_budget: int = 1,
        validate: Callable[
            [list[Requirement], Context, Any, Any],
            Coroutine[Any, Any, list[ValidationResult]],
        ]
        | None = None,
        generate: (Callable[[Component, Context], ModelOutputThunk] | None) = None,
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

    async def sample(
        self,
        action: Component,
        context: Context,
        requirements: list[Requirement],
        *,
        show_progress: bool = True,
        validation_ctx: Context | None = None,
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
            result = self.generate(new_action, ctx)
            await result.avalue()

            # validation pass
            val_scores_co = self.validate(
                reqs,
                validation_ctx,
                result,
                input=None,  # type: ignore
            )
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


class BestofNSamplingStrategy(BaseSamplingStrategy):
    """
    Sampling strategy that selects the best response from a set of samples as given by a Requirement Scorer
    """

    async def sample(
        self,
        action: Component,
        context: Context,
        requirements: list[Requirement],
        *,
        show_progress: bool = True,
        validation_ctx: Context | None = None,
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

        successful_sampled_results: list[ModelOutputThunk] = []
        successful_sampled_scores: list[list[tuple[Requirement, ValidationResult]]] = []
        successful_sampled_actions: list[Component] = []

        # sampled_val_scores: list[float] = []

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
            result = self.generate(new_action, ctx)
            await result.avalue()

            # validation pass
            # action has user turn
            val_scores_co = self.validate(
                reqs,
                validation_ctx,
                result,
                input=action._description,  # type: ignore
            )
            val_scores = await val_scores_co

            # match up reqs with scores
            constraint_scores = list(zip(reqs, val_scores))

            # collect all data
            sampled_results.append(result)
            sampled_scores.append(constraint_scores)
            sampled_actions.append(new_action)

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
                successful_sampled_actions.append(new_action)

            else:
                # log partial success and continue
                count_valid = len([s for s in constraint_scores if bool(s[1])])
                flog.info(f"FAILED. Valid: {count_valid}/{len(constraint_scores)}")

                # If we did not pass all constraints, update the instruction and try again.
                new_action = self.repair(
                    ctx, sampled_actions, sampled_results, sampled_scores
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
                best_result, best_score = max(
                    zip(successful_sampled_results, scores), key=lambda x: x[1]
                )
            elif scorer_preference_ordering == "min":
                best_result, best_score = min(
                    zip(successful_sampled_results, scores), key=lambda x: x[1]
                )
            else:
                raise NotImplementedError

            return SamplingResult(
                best_result,
                success=True,
                sample_generations=sampled_results,
                sample_validations=sampled_scores,
                sample_actions=sampled_actions,
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
                success=False,
                sample_generations=sampled_results,
                sample_validations=sampled_scores,
                sample_actions=sampled_actions,
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


class BaseMBRDSampling(RejectionSamplingStrategy):
    number_of_samples: int
    weighted: bool
    symmetric: bool

    def __init__(
        self,
        *,
        number_of_samples: int = 8,
        weighted: bool = False,
        loop_budget: int = 1,
        validate: Callable[
            [list[Requirement], Context, Any, Any],
            Coroutine[Any, Any, list[ValidationResult]],
        ]
        | None = None,
        generate: (Callable[[Component, Context], ModelOutputThunk] | None) = None,
        requirements: list[Requirement] | None = None,
    ):
        """Initialize a new instance of the class with default parameters.

        Args:
            number_of_samples: Number of samples to generate and use for majority voting
            loop_budget: Inner rejection sampling number of times to iterate through the process. Must be greater than 0.
            validate: Function to validate the results against requirements. If None, validation is provided later through setter.
            generate: Function to generate new model output thunks. If None, generate is provided later through setter.
            requirements: List of requirements to test against. If None, test all requirements attached to the given instruction.

        Raises:
            AssertionError: If loop_budget is not greater than 0.
        """
        super().__init__(
            loop_budget=loop_budget,
            validate=validate,
            generate=generate,
            requirements=requirements,
        )
        self.number_of_samples = number_of_samples
        self.weighted = weighted
        self.symmetric = False

    @abc.abstractmethod
    def compare_strings(self, ref: str, pred: str) -> float:
        """This method is the abstract method for MBRD similarity metric."""

    def maybe_apply_weighted(self, scr):
        # TODO not implemented yet
        if self.weighted:
            weights = np.asarray([1.0 for _ in range(len(scr))])
            scr = scr * weights

        return scr

    async def sample(
        self,
        action: Component,
        context: Context,
        requirements: list[Requirement],
        *,
        show_progress: bool = True,
        validation_ctx: Context | None = None,
    ) -> SamplingResult:
        # execute sampling concurrently
        tasks = []
        async with TaskGroup() as tg:
            for i in range(self.number_of_samples):
                task = tg.create_task(
                    super().sample(
                        action,
                        context,
                        requirements,
                        show_progress=show_progress,
                        validation_ctx=validation_ctx,
                    )
                )
                tasks.append(task)

        # collect results
        results = []
        for task in tasks:
            result = task.result()
            if result.success:
                output = str(result.result)
            else:
                # avoid type checker error
                assert isinstance(result.sample_generations, list)
                output = str(result.sample_generations[0].value)

            results.append((output, result))

        assert len(results) > 0

        scr = np.asarray(
            [[0.0 for _ in range(len(results))] for _ in range(len(results))]
        )
        for i in range(len(results)):
            for j in range(len(results)):
                if j == i:
                    scr[i][j] = 0.0  # self voting is 0.
                    continue

                # upper triangle
                # For sample i compute votes against all j references
                if j > i:
                    scr[i][j] = float(
                        self.compare_strings(results[j][0], results[i][0])
                    )
                    continue

                else:
                    if self.symmetric:
                        scr[i][j] = scr[j][i]
                    else:
                        scr[i][j] = float(
                            self.compare_strings(results[j][0], results[i][0])
                        )
                    continue

        # count votes
        scr = scr.sum(axis=0)

        # Apply weights
        scr = self.maybe_apply_weighted(scr)

        maxR = int(scr.argmax())

        return results[maxR][1]  # return one of the MV answers


class MajorityVotingStrategyForMath(BaseMBRDSampling):
    number_of_samples: int
    match_types: list[str]
    float_rounding: int
    strict: bool
    allow_set_relation_comp: bool
    weighted: bool
    symmetric: bool

    def __init__(
        self,
        *,
        number_of_samples: int = 8,
        float_rounding: int = 6,
        strict: bool = True,
        allow_set_relation_comp: bool = False,
        weighted: bool = False,
        loop_budget: int = 1,
        validate: Callable[
            [list[Requirement], Context, Any, Any],
            Coroutine[Any, Any, list[ValidationResult]],
        ]
        | None = None,
        generate: (Callable[[Component, Context], ModelOutputThunk] | None) = None,
        requirements: list[Requirement] | None = None,
    ):
        """Initialize a new instance of the class with default parameters.

        Args:
            number_of_samples: Number of samples to generate and use for majority voting
            float_rounding: Number of decimal places to round floats to. Defaults to 6.
            strict: Whether to enforce strict comparison mode. Defaults to True.
                - In strict mode: Variables matter and sets are not comparable with tuples
                - In non-strict mode: Variables are matched by position and sets can be compared with tuples
            allow_set_relation_comp: Whether to allow set - relation (e.g 1 < x < 2 and (1, 2)) comparison. Defaults to False.
                - If True, set - relation comparison will be allowed in all cases.
                - If False, set - relation comparison will be allowed only if the prediction is a set.
            loop_budget: Inner rejection sampling number of times to iterate through the process. Must be greater than 0.
            validate: Function to validate the results against requirements. If None, validation is provided later through setter.
            generate: Function to generate new model output thunks. If None, generate is provided later through setter.
            requirements: List of requirements to test against. If None, test all requirements attached to the given instruction.

        Raises:
            AssertionError: If loop_budget is not greater than 0.
        """
        super().__init__(
            number_of_samples=number_of_samples,
            weighted=weighted,
            loop_budget=loop_budget,
            validate=validate,
            generate=generate,
            requirements=requirements,
        )
        self.number_of_samples = number_of_samples
        # match_type: type of match latex, expr (match only so far)
        #     -  For math use "latex" or "expr" or both
        #     -  For general text similarity use "rougel"
        MATCH_TYPES = ["latex", "axpr"]
        self.match_types = MATCH_TYPES
        self.float_rounding = float_rounding
        self.strict = strict
        self.allow_set_relation_comp = allow_set_relation_comp
        self.weighted = weighted

        # Note: symmetry is not implied for certain expressions, see: https://github.com/huggingface/Math-Verify/blob/5d148cfaaf99214c2e4ffb4bc497ab042c592a7a/README.md?plain=1#L183
        self.symmetric = False

    # https://github.com/huggingface/Math-Verify/blob/5d148cfaaf99214c2e4ffb4bc497ab042c592a7a/tests/test_all.py#L36
    def compare_strings(self, ref: str, pred: str):
        """Helper function to compare strings using the math extraction metrics"""
        # Convert string match_types to ExtractionTarget objects
        extraction_targets = []
        for match_type in self.match_types:
            if match_type == "latex":
                extraction_targets.append(LatexExtractionConfig(boxed_match_priority=0))
            elif match_type == "expr":
                extraction_targets.append(ExprExtractionConfig())

        # NOTE: Math-Verify parse and verify functions don't support threaded environment due to usage of signal.alarm() in timeout mechanism. If you need to run in multithreaded environment it's recommended to set the parsing_timeout=None
        gold_parsed = parse(ref, extraction_targets, parsing_timeout=None)
        pred_parsed = parse(pred, extraction_targets, parsing_timeout=None)
        return verify(
            gold_parsed,
            pred_parsed,
            float_rounding=self.float_rounding,
            strict=self.strict,
            allow_set_relation_comp=self.allow_set_relation_comp,
            timeout_seconds=None,
        )


class MBRDRougeLStrategy(BaseMBRDSampling):
    number_of_samples: int
    match_types: list[str]
    weighted: bool
    symmetric: bool
    scorer: RougeScorer

    def __init__(
        self,
        *,
        number_of_samples: int = 8,
        weighted: bool = False,
        loop_budget: int = 1,
        validate: Callable[
            [list[Requirement], Context, Any, Any],
            Coroutine[Any, Any, list[ValidationResult]],
        ]
        | None = None,
        generate: (Callable[[Component, Context], ModelOutputThunk] | None) = None,
        requirements: list[Requirement] | None = None,
    ):
        """Initialize a new instance of the class with default parameters.

        Args:
            number_of_samples: Number of samples to generate and use for majority voting
            match_type: type of match latex, expr (match only so far)
                -  For math use "latex" or "expr" or both
                -  For general text similarity use "rougel"
            loop_budget: Inner rejection sampling number of times to iterate through the process. Must be greater than 0.
            validate: Function to validate the results against requirements. If None, validation is provided later through setter.
            generate: Function to generate new model output thunks. If None, generate is provided later through setter.
            requirements: List of requirements to test against. If None, test all requirements attached to the given instruction.

        Raises:
            AssertionError: If loop_budget is not greater than 0.
        """
        super().__init__(
            number_of_samples=number_of_samples,
            weighted=weighted,
            loop_budget=loop_budget,
            validate=validate,
            generate=generate,
            requirements=requirements,
        )
        self.number_of_samples = number_of_samples
        self.match_types = ["rougeL"]
        self.weighted = weighted
        self.symmetric = True
        self.scorer = RougeScorer(self.match_types, use_stemmer=True)

    # https://github.com/huggingface/Math-Verify/blob/5d148cfaaf99214c2e4ffb4bc497ab042c592a7a/tests/test_all.py#L36
    def compare_strings(self, ref: str, pred: str):
        """Helper function to compare strings using the math extraction metrics"""

        scr = self.scorer.score(ref, pred)[self.match_types[-1]].fmeasure
        return scr
