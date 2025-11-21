"""Base Sampling Strategies."""

import abc
import asyncio
import math
from collections.abc import AsyncGenerator, Callable, Coroutine
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import tqdm

import mellea.stdlib.functional as mfuncs
from mellea.backends import Backend, BaseModelSubclass
from mellea.helpers.fancy_logger import FancyLogger
from mellea.stdlib.base import CBlock, ChatContext, Component, Context, ModelOutputThunk
from mellea.stdlib.chat import Message
from mellea.stdlib.instruction import Instruction
from mellea.stdlib.requirement import Requirement, ValidationResult

from .types import SamplingResult, SamplingStrategy


@dataclass
class _SamplingResultSlice:
    """Helper class for returning the result of a single sample operation."""

    success: bool
    generation: ModelOutputThunk
    validation: list[tuple[Requirement, ValidationResult]]
    context: Context
    action: Component


def _get_sampling_result(
    slices: list[_SamplingResultSlice],
    select_from_failure: Callable[
        [
            list[Component],
            list[ModelOutputThunk],
            list[list[tuple[Requirement, ValidationResult]]],
        ],
        int,
    ],
) -> SamplingResult:
    sample_generations: list[ModelOutputThunk] = []
    sample_validations: list[list[tuple[Requirement, ValidationResult]]] = []
    sample_actions: list[Component] = []
    sample_contexts: list[Context] = []

    success = False
    best_index = -1
    for i, slice in enumerate(slices):
        if slice.success and not success:
            # If a success hasn't already been found, update the status and index.
            success = True
            best_index = i

        sample_generations.append(slice.generation)
        sample_validations.append(slice.validation)
        sample_actions.append(slice.action)
        sample_contexts.append(slice.context)

    if not success:
        best_index = select_from_failure(
            sample_actions, sample_generations, sample_validations
        )

    return SamplingResult(
        result_index=best_index,
        success=success,
        sample_generations=sample_generations,
        sample_validations=sample_validations,
        sample_actions=sample_actions,
        sample_contexts=sample_contexts,
    )


class BaseSamplingStrategy(SamplingStrategy):
    """Base class for multiple strategies that rejects samples based on given instructions."""

    loop_budget: int
    concurrency_budget: int

    def __init__(
        self,
        *,
        loop_budget: int = 1,
        concurrency_budget: int = 1,
        requirements: list[Requirement] | None = None,
    ):
        """Initialize a new instance of the class with default parameters.

        Will generate at most loop_budget * concurrency_budget requests. The sampling will end at the first valid result.
        The loop budget specifies the depth of repair strategies.

        For example:
        - loop_budget = 1: no repair strategies will be used
        - loop_budget = 3 and concurrency_budget = 1: generation -> repair -> generation -> repair -> final generation
        - loop_budget = 2 and concurrency_budget = 2: each initial concurrent generation will undergo a repair strategy once and then attempt a second generation

        Args:
            loop_budget: Number of times to iterate through the process. Must be greater than 0.
            concurrency_budget: Number of concurrent generations per loop. Use the default of 1 for no-concurrent sampling. Must be greater than 0.
            requirements: List of requirements to test against. If None, test all requirements attached to the given instruction.

        Raises:
            AssertionError: If loop_budget is not greater than 0.
        """
        assert loop_budget > 0, "Loop budget must be at least 1."
        assert concurrency_budget > 0, "Concurrency budget must be at least 1."

        self.loop_budget = loop_budget
        self.concurrency_budget = concurrency_budget
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

        total_possible_generations = self.loop_budget * self.concurrency_budget
        progress_indicator = None
        if show_progress:
            progress_indicator = tqdm.tqdm(
                iterable=range(total_possible_generations),
                desc=f"{self.__class__.__name__}",
            )

        generators: list[AsyncGenerator[_SamplingResultSlice, Any]] = []
        for _ in range(self.concurrency_budget):
            generators.append(
                self._subsample_iteration(
                    iterations=self.loop_budget,
                    action=action,
                    context=context,
                    backend=backend,
                    requirements=reqs,
                    validation_ctx=validation_ctx,
                    format=format,
                    model_options=model_options,
                    tool_calls=tool_calls,
                )
            )

        async def async_generator_producer(
            generator: AsyncGenerator[_SamplingResultSlice, Any],
            queue: asyncio.Queue[_SamplingResultSlice],
        ):
            """Add items from an async generator to an async queue."""
            async for item in generator:
                await queue.put(item)

        sample_slice_queue: asyncio.Queue[_SamplingResultSlice] = asyncio.Queue()
        producer_tasks = [
            # Use tasks so that we don't need to explicitly await each generator.
            asyncio.create_task(async_generator_producer(generator, sample_slice_queue))
            for generator in generators
        ]

        slices: list[_SamplingResultSlice] = []
        while not all(task.done() for task in producer_tasks):
            sr_slice = await sample_slice_queue.get()
            slices.append(sr_slice)

            if progress_indicator:
                progress_indicator.update()

            if sr_slice.success:
                break

        # TODO: We could add a sleep here after a success to try to collect
        #       any other finished sample iterations. But this also risks ceding
        #       control to some other task / requirement validator that takes must longer
        #       to process and makes this sampling result take much longer.
        # await asyncio.sleep(.1)

        for task in producer_tasks:
            task.cancel()  # Works even if task is already done / cancelled.

        while not sample_slice_queue.empty():
            try:
                # Shouldn't have to wait here since all tasks are cancelled.
                sr_slice = sample_slice_queue.get_nowait()
                slices.append(sr_slice)

                if progress_indicator:
                    progress_indicator.update()
            except asyncio.QueueEmpty:
                # This is somewhat redundant but isn't harmful.
                break

        if progress_indicator:
            progress_indicator.close()

        s_result = _get_sampling_result(
            slices=slices, select_from_failure=self.select_from_failure
        )
        if not s_result.success:
            flog.info(
                f"Invoking select_from_failure after {len(s_result.sample_generations)} failed attempts."
            )
        else:
            flog.info("Sampling was successful.")

        assert s_result.result_index < len(s_result.sample_generations), (
            "The select_from_failure method did not return a valid result. It has to selected from failed_results."
        )

        assert s_result.result._generate_log is not None
        s_result.result._generate_log.is_final_result = True

        return s_result

    async def _subsample_iteration(
        self,
        iterations: int,
        action: Component,
        context: Context,
        backend: Backend,
        requirements: list[Requirement],
        *,
        validation_ctx: Context | None = None,
        format: type[BaseModelSubclass] | None = None,
        model_options: dict | None = None,
        tool_calls: bool = False,
    ):
        """Helper function that represents a single sampling iteration: generating a sample and validating it."""
        sampled_results: list[ModelOutputThunk] = []
        sampled_scores: list[list[tuple[Requirement, ValidationResult]]] = []
        sampled_actions: list[Component] = []
        sample_contexts: list[Context] = []

        next_action = deepcopy(action)
        next_context = context
        for _ in range(iterations):  # type: ignore
            # run a generation pass
            result, result_ctx = await backend.generate_from_context(
                action=next_action,
                ctx=next_context,
                format=format,
                model_options=model_options,
                tool_calls=tool_calls,
            )
            await result.avalue()

            # validation pass
            val_scores_co = mfuncs.avalidate(
                reqs=requirements,
                context=result_ctx,
                backend=backend,
                output=result,
                format=None,
                model_options=model_options,
                # tool_calls=tool_calls  # Don't support using tool calls in validation strategies.
            )
            val_scores = await val_scores_co

            # match up reqs with scores
            constraint_scores = list(zip(requirements, val_scores))

            success = False
            if all(bool(s[1]) for s in constraint_scores):
                success = True
                assert (
                    result._generate_log is not None
                )  # Cannot be None after generation.
                result._generate_log.is_final_result = True

            yield _SamplingResultSlice(
                success=success,
                generation=result,
                validation=constraint_scores,
                context=result_ctx,
                action=next_action,
            )

            if success:
                # End generation early.
                return

            # Have to append so that the repair strategy gets the correct info.
            sampled_results.append(result)
            sampled_scores.append(constraint_scores)
            sampled_actions.append(next_action)
            sample_contexts.append(result_ctx)

            # If we did not pass all constraints, update the instruction and try again.
            next_action, next_context = self.repair(
                next_context,
                result_ctx,
                sampled_actions,
                sampled_results,
                sampled_scores,
            )

        # End generation after all iterations are done.
        return


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
