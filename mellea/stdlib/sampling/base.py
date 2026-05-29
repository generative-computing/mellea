"""Base Sampling Strategies.

Sampling strategies control how Mellea handles validation failures during generation:

- **RejectionSamplingStrategy**: Simple retry with the same prompt. Best for non-deterministic
  failures where the same instruction might succeed on retry.

- **RepairTemplateStrategy**: Single-turn repair by modifying the instruction with validation
  feedback. Adds failure reasons to the instruction and retries. Best for simple tasks where
  feedback can be incorporated into the instruction.

- **MultiTurnStrategy**: Multi-turn conversational repair (requires ChatContext). Adds validation
  failure reasons as new user messages in the conversation, allowing iterative improvement through
  dialogue. Best for complex tasks and agentic workflows.
"""

import abc
import asyncio
from collections.abc import AsyncGenerator, Callable
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Generic

import tqdm

from ...core import (
    Backend,
    BaseModelSubclass,
    Component,
    ComputedModelOutputThunk,
    Context,
    MelleaLogger,
    Requirement,
    S,
    SamplingResult,
    SamplingStrategy,
    ValidationResult,
    log_context,
)
from ...plugins.manager import has_plugins, invoke_hook
from ...plugins.types import HookType
from ...stdlib import functional as mfuncs
from ...telemetry.context import with_context
from ..components import Instruction, Message
from ..context import ChatContext


@dataclass
class _SamplingResultSlice(Generic[S]):
    """Result of a single generate/validate iteration inside a subsample."""

    success: bool
    generation: ComputedModelOutputThunk[S]
    validation: list[tuple[Requirement, ValidationResult]]
    context: Context
    action: Component


def _get_sampling_result(
    slices: list[_SamplingResultSlice[S]],
    select_from_failure: Callable[
        [
            list[Component],
            list[ComputedModelOutputThunk],
            list[list[tuple[Requirement, ValidationResult]]],
        ],
        int,
    ],
) -> SamplingResult[S]:
    """Aggregate per-iteration slices into a SamplingResult.

    Picks the first successful slice as the result; falls back to
    ``select_from_failure`` over the collected slices when no slice succeeded.
    """
    sample_generations: list[ComputedModelOutputThunk] = []
    sample_validations: list[list[tuple[Requirement, ValidationResult]]] = []
    sample_actions: list[Component] = []
    sample_contexts: list[Context] = []

    success = False
    best_index = -1

    # Iterate over all entries to accumulate them but only take the first success.
    for i, sample_slice in enumerate(slices):
        if sample_slice.success and not success:
            success = True
            best_index = i

        sample_generations.append(sample_slice.generation)
        sample_validations.append(sample_slice.validation)
        sample_actions.append(sample_slice.action)
        sample_contexts.append(sample_slice.context)

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
    """Base class for multiple strategies that reject samples based on given instructions.

    Args:
        loop_budget (int): Maximum number of generate/validate cycles per
            concurrent subsample. Must be greater than 0. Defaults to `1`.
        concurrency_budget (int): Number of concurrent subsamples. Sampling
            generates at most `loop_budget * concurrency_budget` requests
            and stops at the first valid result. Must be greater than 0.
            Defaults to `1` (no concurrency).
        requirements (list[Requirement] | None): Global requirements evaluated
            on every sample. When set, overrides per-call requirements.

    Examples:
        - `loop_budget=1`: no repair strategies are used.
        - `loop_budget=3, concurrency_budget=1`: generate -> repair -> generate -> repair -> final generate.
        - `loop_budget=2, concurrency_budget=2`: two concurrent subsamples, each with one repair.

    Raises:
        AssertionError: If `loop_budget < 1` or `concurrency_budget < 1`.
    """

    loop_budget: int
    concurrency_budget: int

    def __init__(
        self,
        *,
        loop_budget: int = 1,
        concurrency_budget: int = 1,
        requirements: list[Requirement] | None = None,
    ):
        """Initialize BaseSamplingStrategy with budgets and optional global requirements."""
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
        past_results: list[ComputedModelOutputThunk],
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
        # TODO: For Component/ModelOutputThunk-typing to work, repair strategies should always return a Component with the same parsing
        #       as the initial action used for this sampling strategy.
        ...

    @staticmethod
    @abc.abstractmethod
    def select_from_failure(
        sampled_actions: list[Component],
        sampled_results: list[ComputedModelOutputThunk],
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
        action: Component[S],
        context: Context,
        backend: Backend,
        requirements: list[Requirement] | None,
        *,
        validation_ctx: Context | None = None,
        format: type[BaseModelSubclass] | None = None,
        model_options: dict | None = None,
        tool_calls: bool = False,
        show_progress: bool = True,
    ) -> SamplingResult[S]:
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
            SamplingResult[S]: A result object indicating the success or failure of the sampling process.

        Raises:
            AssertionError: Asserts that all required components (repair, select_from_failure, validate, and generate) are provided before proceeding with the sampling.
        """
        validation_ctx = validation_ctx if validation_ctx is not None else context

        flog = MelleaLogger.get_logger()

        with log_context(strategy=type(self).__name__, loop_budget=self.loop_budget):
            # The `logging_redirect_tqdm` approach did not work, so instead we will use the show_progress
            # flag to determine whether we should show the pbar.
            show_progress = (
                show_progress and flog.getEffectiveLevel() <= MelleaLogger.INFO
            )

            reqs = []
            # global requirements supersede local requirements (global requirements can be defined by user)
            # Todo: re-evaluate if this makes sense
            if self.requirements is not None:
                reqs += self.requirements
            elif requirements is not None:
                reqs += requirements
            reqs = list(set(reqs))

            # --- sampling_loop_start hook ---
            effective_loop_budget = self.loop_budget
            if has_plugins(HookType.SAMPLING_LOOP_START):
                from ...plugins.hooks.sampling import SamplingLoopStartPayload

                start_payload = SamplingLoopStartPayload(
                    strategy_name=type(self).__name__,
                    action=action,
                    context=context,
                    requirements=reqs,
                    loop_budget=self.loop_budget,
                )
                _, start_payload = await invoke_hook(
                    HookType.SAMPLING_LOOP_START, start_payload, backend=backend
                )
                effective_loop_budget = start_payload.loop_budget

            total_possible_generations = effective_loop_budget * self.concurrency_budget
            progress_indicator = (
                tqdm.tqdm(
                    iterable=range(total_possible_generations),
                    desc=f"{type(self).__name__}",
                )
                if show_progress
                else None
            )

            # Create `concurrency_budget` concurrent generators that all generate up to the `loop_budget` number of generations.
            generators: list[AsyncGenerator[_SamplingResultSlice[S], Any]] = [
                self._subsample_iteration(
                    subsample_index=idx,
                    iterations=effective_loop_budget,
                    action=action,
                    context=context,
                    backend=backend,
                    requirements=reqs,
                    validation_ctx=validation_ctx,
                    format=format,
                    model_options=model_options,
                    tool_calls=tool_calls,
                )
                for idx in range(self.concurrency_budget)
            ]

            # Sentinel pushed by each producer when it exhausts its generator.
            # Lets the consumer detect "all producers done" without racing
            # queue.get() against a separate completion future.
            _DONE = object()

            async def _producer(
                generator: AsyncGenerator[_SamplingResultSlice, Any],
                queue: asyncio.Queue[_SamplingResultSlice | object],
            ) -> None:
                """Drain an async generator into a shared queue, then signal completion."""
                try:
                    async for item in generator:
                        await queue.put(item)
                finally:
                    await queue.put(_DONE)

            # Create the queue that the samples are consumed from.
            slice_queue: asyncio.Queue[_SamplingResultSlice | object] = asyncio.Queue()
            producer_tasks = [
                # Use tasks to push to the queue so that we don't need to explicitly await each generator.
                asyncio.create_task(_producer(gen, slice_queue))
                for gen in generators
            ]

            slices: list[_SamplingResultSlice] = []

            # Keep track of the producers left. It's the easiest way to ensure we don't deadlock
            # if the producers finish and queue is empty at the same time.
            remaining_producers = len(producer_tasks)
            try:
                while remaining_producers > 0:
                    item = await slice_queue.get()

                    if item is _DONE:
                        remaining_producers -= 1
                        continue

                    assert isinstance(item, _SamplingResultSlice)
                    slices.append(item)

                    if progress_indicator is not None:
                        progress_indicator.update()

                    if item.success:
                        # Found a successful sample. Exit early.
                        break
            finally:
                for t in producer_tasks:
                    t.cancel()  # No-op if already done / cancelled.

                # Wait for cancellations to settle so we don't leak tasks.
                await asyncio.gather(*producer_tasks, return_exceptions=True)

                # Drain anything queued before producers were cancelled.
                while not slice_queue.empty():
                    try:
                        item = slice_queue.get_nowait()
                    except asyncio.QueueEmpty:
                        break
                    if item is _DONE:
                        continue
                    assert isinstance(item, _SamplingResultSlice)
                    slices.append(item)
                    if progress_indicator is not None:
                        progress_indicator.update()

                if progress_indicator is not None:
                    progress_indicator.close()

            s_result = _get_sampling_result(
                slices=slices, select_from_failure=self.select_from_failure
            )

            if s_result.success:
                flog.info("Sampling was successful.")
            else:
                flog.info(
                    f"Invoking select_from_failure after {len(s_result.sample_generations)} failed attempts."
                )

            assert s_result.result_index < len(s_result.sample_generations), (
                "The select_from_failure method did not return a valid result. It has to selected from failed_results."
            )
            assert s_result.result._generate_log is not None
            s_result.result._generate_log.is_final_result = True

            # --- sampling_loop_end hook (success or failure) ---
            if has_plugins(HookType.SAMPLING_LOOP_END):
                from ...plugins.hooks.sampling import SamplingLoopEndPayload

                end_payload = SamplingLoopEndPayload(
                    strategy_name=type(self).__name__,
                    success=s_result.success,
                    iterations_used=len(slices),
                    final_result=s_result.result,
                    final_action=s_result.result_action,
                    final_context=s_result.result_ctx,
                    all_results=list(s_result.sample_generations),
                    all_validations=list(s_result.sample_validations),
                    failure_reason=(
                        None
                        if s_result.success
                        else f"Budget exhausted after {len(slices)} iterations"
                    ),
                )
                await invoke_hook(
                    HookType.SAMPLING_LOOP_END, end_payload, backend=backend
                )

            return s_result

    async def _subsample_iteration(
        self,
        subsample_index: int,
        iterations: int,
        action: Component[S],
        context: Context,
        backend: Backend,
        requirements: list[Requirement],
        *,
        validation_ctx: Context | None = None,
        format: type[BaseModelSubclass] | None = None,
        model_options: dict | None = None,
        tool_calls: bool = False,
    ) -> AsyncGenerator[_SamplingResultSlice[S], Any]:
        """Run one concurrent subsample: up to `iterations` generate/validate/repair attempts.

        Yields a :class:`_SamplingResultSlice` per attempt and ends early after the first successful slice.
        `subsample_index` (0-based) identifies this subsample within the parent `sample()` call; it is used to derive a
        globally unique iteration counter for telemetry and hooks.
        """
        flog = MelleaLogger.get_logger()
        sampled_results: list[ComputedModelOutputThunk[S]] = []
        sampled_scores: list[list[tuple[Requirement, ValidationResult]]] = []
        sampled_actions: list[Component] = []

        next_action = deepcopy(action)
        next_context = context
        for i in range(iterations):
            # Globally unique across concurrent subsamples in the parent sample() call.
            current_iteration = subsample_index * iterations + i + 1

            with with_context(sampling_iteration=current_iteration):
                # run a generation pass
                result, result_ctx = await backend.generate_from_context(
                    next_action,
                    ctx=next_context,
                    format=format,
                    model_options=model_options,
                    tool_calls=tool_calls,
                )
                await result.avalue()
                result = ComputedModelOutputThunk(result)

                # Sampling strategies may use different components from the original
                # action. This might cause discrepancies in the expected parsed_repr
                # type / value. Explicitly overwrite that here.
                # TODO: See if there's a more elegant way for this so that each sampling
                # strategy doesn't have to re-implement it.
                result.parsed_repr = action.parse(result)

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

                constraint_scores = list(zip(requirements, val_scores))
                all_validations_passed = all(bool(s[1]) for s in constraint_scores)

                # --- sampling_iteration hook ---
                if has_plugins(HookType.SAMPLING_ITERATION):
                    from ...plugins.hooks.sampling import SamplingIterationPayload

                    iter_payload = SamplingIterationPayload(
                        strategy_name=type(self).__name__,
                        iteration=current_iteration,
                        action=next_action,
                        result=result,
                        validation_results=constraint_scores,
                        all_validations_passed=all_validations_passed,
                        valid_count=sum(1 for s in constraint_scores if bool(s[1])),
                        total_count=len(constraint_scores),
                    )
                    await invoke_hook(
                        HookType.SAMPLING_ITERATION, iter_payload, backend=backend
                    )

                if not all_validations_passed:
                    failed = [s for s in constraint_scores if not bool(s[1])]
                    failed_reqs = [
                        r[0].description
                        if r[0].description is not None
                        else "[no description]"
                        for r in failed
                    ]
                    stringify_failed = "\n\t - " + "\n\t - ".join(failed_reqs)
                    flog.info(
                        f"FAILED. Valid: {len(constraint_scores) - len(failed)}/{len(constraint_scores)}. Failed: {stringify_failed}"
                    )

                yield _SamplingResultSlice(
                    success=all_validations_passed,
                    generation=result,
                    validation=constraint_scores,
                    context=result_ctx,
                    action=next_action,
                )

                if all_validations_passed:
                    return

                if i == iterations - 1:
                    # Final iteration: repair output would be discarded, skip it.
                    # The generation budget has been exhausted.
                    return

                # Append failure history before computing the repair so the strategy
                # sees the most recent failed attempt.
                sampled_results.append(result)
                sampled_scores.append(constraint_scores)
                sampled_actions.append(next_action)

                next_action, next_context = self.repair(
                    next_context,
                    result_ctx,
                    sampled_actions,
                    sampled_results,
                    sampled_scores,
                )

                # --- sampling_repair hook ---
                if has_plugins(HookType.SAMPLING_REPAIR):
                    from ...plugins.hooks.sampling import SamplingRepairPayload

                    repair_payload = SamplingRepairPayload(
                        repair_type=getattr(
                            self, "_get_repair_type", lambda: "unknown"
                        )(),
                        failed_action=sampled_actions[-1],
                        failed_result=sampled_results[-1],
                        failed_validations=sampled_scores[-1],
                        repair_action=next_action,
                        repair_context=next_context,
                        repair_iteration=current_iteration,
                    )
                    await invoke_hook(
                        HookType.SAMPLING_REPAIR, repair_payload, backend=backend
                    )


class RejectionSamplingStrategy(BaseSamplingStrategy):
    """Simple rejection sampling strategy that just repeats the same call on failure."""

    @staticmethod
    def select_from_failure(
        sampled_actions: list[Component],
        sampled_results: list[ComputedModelOutputThunk],
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
        past_results: list[ComputedModelOutputThunk],
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
        sampled_results: list[ComputedModelOutputThunk],
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
        past_results: list[ComputedModelOutputThunk],
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
        sampled_results: list[ComputedModelOutputThunk],
        sampled_val: list[list[tuple[Requirement, ValidationResult]]],
    ) -> int:
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
        past_results: list[ComputedModelOutputThunk],
        past_val: list[list[tuple[Requirement, ValidationResult]]],
    ) -> tuple[Component, Context]:
        """Returns a Message with a description (and validation reasons) of the failed requirements.

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

        # Get failed requirements and their detailed validation reasons
        failed_items = [(req, val) for req, val in past_val[-1] if not val.as_bool()]

        # Build repair feedback using ValidationResult.reason when available
        repair_lines = []
        for req, validation in failed_items:
            if validation.reason:
                repair_lines.append(f"* {validation.reason}")
            else:
                # Fallback to requirement description if no reason
                repair_lines.append(f"* {req.description}")

        feedback = "\n".join(repair_lines)
        next_action = Message(
            role="user",
            content=(
                f"The following requirements have not been met:\n{feedback}\n"
                f"Please try again to fulfill the requirements."
            ),
        )

        return next_action, new_ctx
