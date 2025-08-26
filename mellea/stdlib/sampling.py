"""sampling methods go here."""

import abc
from collections.abc import Callable
from typing import Any

import tqdm

from mellea.helpers.fancy_logger import FancyLogger
from mellea.stdlib.base import CBlock, GenerateLog, ModelOutputThunk
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


class SamplingStrategy(abc.ABC):
    """A SamplingStrategy class defines an abstract base class for implementing various sampling strategies.

    This class provides a template for creating concrete sampling strategies that can be used to generate model outputs based on given instructions.
    It allows setting custom validation and generation functions through properties.
    """

    # the function signature here matches that of m.validate
    validate: Callable[[list[Requirement], Any], list[ValidationResult]] | None = None

    generate: (
        Callable[[Instruction, list[GenerateLog] | None], ModelOutputThunk] | None
    ) = None

    @abc.abstractmethod
    def sample(
        self,
        instruction: Instruction,
        *,
        generate_logs: list[GenerateLog] | None = None,
    ) -> SamplingResult:
        """This method is the abstract method for sampling a given instruction.

        It must be implemented by any concrete subclasses to provide specific sampling logic.

        Args:
            instruction (Instruction): The instruction object to be sampled.
            generate_logs: Optional list of GenerateLog objects. If None, no collection happens.
        """


class RejectionSamplingStrategy(SamplingStrategy):
    """Sampling strategy that rejects samples based on given instructions."""

    def __init__(
        self,
        *,
        loop_budget: int = 1,
        repair: Callable[
            [
                Instruction,
                list[tuple[Requirement, ValidationResult]],
                list[Instruction],
            ],
            Instruction,
        ] = lambda i, r, h_i: i,
        select_from_failure: Callable[
            [
                Instruction,
                list[ModelOutputThunk],
                list[list[tuple[Requirement, ValidationResult]]],
            ],
            ModelOutputThunk,
        ] = lambda _, results, __: results[0],
        validate: Callable[[list[Requirement], Any], list[ValidationResult]]
        | None = None,
        generate: (
            Callable[[Instruction, list[GenerateLog] | None], ModelOutputThunk] | None
        ) = None,
        requirements: list[Requirement] | None = None,
    ):
        """Initialize a new instance of the class with default parameters.

        Args:
            loop_budget: Number of times to iterate through the process. Must be greater than 0.
            repair: Function to apply "repairs" to an instruction based on its requirements and validation results.
            select_from_failure: Function to select a model output thunk from failed attempts.
            validate: Function to validate the results against requirements. If None, validation is provided later through setter.
            generate: Function to generate new model output thunks. If None, generate is provided later through setter.
            requirements: List of requirements to test against. If None, test all requirements attached to the given instruction.

        Raises:
            AssertionError: If loop_budget is not greater than 0.
        """
        assert loop_budget > 0, "Loop budget must be at least 1."
        self.loop_budget = loop_budget
        self.repair = repair
        self.select_from_failure = select_from_failure
        self.validate = validate  # it's ok to be None here
        self.generate = generate  # it's ok to be None here
        self.requirements = requirements

    def sample(
        self,
        instruction: Instruction,
        *,
        show_progress: bool = True,
        generate_logs: list[GenerateLog] | None = None,
    ) -> SamplingResult:
        """This method performs a sampling operation based on the given instruction.

        Args:
            instruction: The Instruction object containing the instruction to generate a valid model output thunk.
            show_progress: if true, a tqdm progress bar is used. Otherwise messages will still be sent to flog.
            generate_logs: If provided, the generations will be logged.

        Returns:
            SamplingResult: A result object indicating the success or failure of the sampling process.

        Raises:
            AssertionError: Asserts that all required components (repair, select_from_failure, validate, and generate) are provided before proceeding with the sampling.
        """
        assert self.repair is not None, "Repair must be provided."
        assert self.select_from_failure is not None, (
            "Select from failure must be provided."
        )
        assert self.validate is not None, "Validation must be provided."
        assert self.generate is not None, "Generate must be provided."

        flog = FancyLogger.get_logger()

        # The `logging_redirect_tqdm` approach did not work, so instead we will use the show_progress
        # flag to determine whether we should show the pbar.
        show_progress = show_progress and flog.getEffectiveLevel() <= FancyLogger.INFO

        failed_results: list[ModelOutputThunk] = []
        failed_scores: list[list[tuple[Requirement, ValidationResult]]] = []
        failed_instructions: list[Instruction] = []

        loop_count = 0

        loop_budget_range_iterator = (
            tqdm.tqdm(range(self.loop_budget))
            if show_progress
            else range(self.loop_budget)
        )
        for _ in loop_budget_range_iterator:  # type: ignore
            loop_count += 1
            if not show_progress:
                flog.info(f"Running loop {loop_count} of {self.loop_budget}")

            # run a pass

            result = self.generate(instruction, generate_logs)

            if self.requirements is not None:
                reqs = self.requirements
            else:
                reqs = instruction.requirements
            val_scores = self.validate(reqs, result)
            constraint_scores = list(zip(reqs, val_scores))

            failed_results.append(result)
            failed_scores.append(constraint_scores)
            failed_instructions.append(instruction)

            if all(bool(s[1]) for s in constraint_scores):
                flog.info("SUCCESS")
                return SamplingResult(
                    result,
                    success=True,
                    sample_generations=failed_results,
                    sample_validations=failed_scores,
                )

            else:
                count_valid = len([s for s in constraint_scores if bool(s[1])])
                flog.info(f"FAILED. Valid: {count_valid}/{len(constraint_scores)}")
            # If we did not pass all constraints, update the instruction and try again.
            instruction = self.repair(
                instruction, constraint_scores, failed_instructions
            )

        flog.info(
            f"Invoking select_from_failure after {len(failed_results)} failed attempts."
        )
        best_failed_result = self.select_from_failure(
            instruction, failed_results, failed_scores
        )
        assert best_failed_result in failed_results, (
            "The select_from_failure method did not return a valid result. It has to selected from failed_results."
        )
        return SamplingResult(
            best_failed_result,
            success=False,
            sample_generations=failed_results,
            sample_validations=failed_scores,
        )
