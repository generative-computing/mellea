# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Abstract interfaces for sampling strategies and their results.

`SamplingStrategy` defines the contract for all sampling algorithms: an async
`sample` method that takes an action, context, backend, and requirements, and
returns a `SamplingResult`. `SamplingResult` records the chosen generation
alongside the full history of intermediate samples, their validation outcomes,
and associated contexts — enabling detailed post-hoc inspection of the sampling
process.
"""

import abc
import uuid
from collections.abc import Sequence
from typing import Generic, final

from .backend import Backend, BaseModelSubclass
from .base import (
    CBlock,
    Component,
    ComputedModelOutputThunk,
    Context,
    ModelOutputThunk,
    S,
    Span,
)
from .requirement import Requirement, ValidationResult

# The kinds of action a sampling strategy may operate on. Originally `Component`
# only; widened to include `CBlock` and `ModelOutputThunk` so that `act`/`aact`
# can sample over non-Component actions (see #356). Only `Component` carries
# `.parse()` semantics; the others are carried through as opaque spans. Now a
# re-export of the canonical `Span` alias (see #1439); kept as its own name
# so the historical `mellea.core.SampleActionType` import path keeps working.
SampleActionType = Span


class SamplingResult(CBlock, Generic[S]):
    """Stores the results from a sampling operation. This includes successful and failed samplings.

    Args:
        result_index (int): Index into `sample_generations` identifying the chosen final output.
        success (bool): Whether the sampling operation produced a passing result.
        sample_generations (list[ModelOutputThunk[S]] | None): All output thunks generated during sampling.
        sample_validations (list[list[tuple[Requirement, ValidationResult]]] | None): Per-generation validation
            results; each inner list contains one tuple per requirement evaluated.
        sample_actions (Sequence[SampleActionType] | None): The actions used to produce each generation.
        sample_contexts (list[Context] | None): The contexts associated with each generation.

    Attributes:
        result_index (int): Index into `sample_generations` identifying the chosen final output.
        success (bool): Whether the sampling operation produced a passing result.
        sample_generations (list[ModelOutputThunk[S]]): All output thunks generated during
            sampling; always a list (`None` input is normalised to `[]`).
        sample_validations (list[list[tuple[Requirement, ValidationResult]]]): Per-generation
            validation results; always a list (`None` input is normalised to `[]`).
        sample_actions (list[SampleActionType]): The actions used to produce each generation;
            always a list (`None` input is normalised to `[]`).
        sample_contexts (list[Context]): The contexts associated with each generation;
            always a list (`None` input is normalised to `[]`).
    """

    def __init__(
        self,
        result_index: int,
        success: bool,
        *,
        sample_generations: list[ComputedModelOutputThunk[S]] | None = None,
        sample_validations: list[list[tuple[Requirement, ValidationResult]]]
        | None = None,
        sample_actions: Sequence[SampleActionType] | None = None,
        sample_contexts: list[Context] | None = None,
    ):
        """Initialize SamplingResult with the chosen output index, success flag, and generation history."""
        if sample_generations is None:
            sample_generations = []
        if sample_validations is None:
            sample_validations = []
        # Accept any Sequence (e.g. a `list[Component]`) but store a list so the
        # attribute stays mutable and covariantly assignable at call sites.
        sample_actions = list(sample_actions) if sample_actions is not None else []
        if sample_contexts is None:
            sample_contexts = []

        assert result_index is not None
        assert (
            0 <= result_index < len(sample_generations)
            or -len(sample_generations) <= result_index < 0
        ), " result index cannot be out of range"

        super().__init__(value=sample_generations[result_index].value)

        self.result_index = result_index
        self.success = success
        self.sample_generations = sample_generations
        self.sample_validations = sample_validations
        self.sample_actions = sample_actions
        self.sample_contexts = sample_contexts

    @property
    def result(self) -> ComputedModelOutputThunk[S]:
        """The final output or result from applying the sampling strategy."""
        return self.sample_generations[self.result_index]

    @property
    def result_ctx(self) -> Context:
        """The context of the final output or result from applying the sampling strategy."""
        return self.sample_contexts[self.result_index]

    @property
    def result_action(self) -> SampleActionType:
        """The action that generated the final output or result from applying the sampling strategy."""
        return self.sample_actions[self.result_index]

    @property
    def result_validations(self) -> list[tuple[Requirement, ValidationResult]]:
        """The validation results associated with the final output or result from applying the sampling strategy."""
        return self.sample_validations[self.result_index]


class SamplingStrategy(abc.ABC):
    """A SamplingStrategy class defines an abstract base class for implementing various sampling strategies.

    This class provides a template for creating concrete sampling strategies that can be used to generate model outputs based on given instructions.
    It allows setting custom validation and generation functions through properties.

    Attributes:
        loop_budget: Maximum number of generate/validate cycles. Defaults to `1`.
        requirements: Global requirements evaluated on every sample. When set,
            overrides per-call requirements. Defaults to `None`.
    """

    loop_budget: int = 1
    requirements: list[Requirement] | None = None

    @final
    async def sample(
        self,
        action: Component[S] | CBlock | ModelOutputThunk,
        context: Context,
        backend: Backend,
        requirements: list[Requirement] | None,
        *,
        validation_ctx: Context | None = None,
        format: type[BaseModelSubclass] | None = None,
        model_options: dict | None = None,
        tool_calls: bool = False,
        **kwargs,
    ) -> SamplingResult[S]:
        """Concrete wrapper: owns the sampling lifecycle and fires loop start/end hooks.

        Mints a `sampling_id`, dispatches `sampling_loop_start` (which may modify
        `loop_budget`), delegates to `_sample`, and dispatches
        `sampling_loop_end` on every exit path — success, budget exhaustion, and
        raised exceptions.

        Args:
            action: The action object to be sampled. A `Component`, `CBlock`, or `ModelOutputThunk`.
            context: The context to be passed to the sampling strategy.
            backend: The backend used for generating samples.
            requirements: List of requirements to test against (merged with global requirements).
            validation_ctx: Optional context to use for validation. If None, validation_ctx = ctx.
            format: output format for structured outputs.
            model_options: model options to pass to the backend during generation / validation.
            tool_calls: True if tool calls should be used during this sampling strategy.
            **kwargs: Additional keyword arguments forwarded to `_sample`.

        Returns:
            SamplingResult[S]: A result object indicating the success or failure of the sampling process.

        Raises:
            ValueError: If a `SAMPLING_LOOP_START` hook returns a non-positive `loop_budget`.
        """
        from ..plugins.manager import has_plugins, invoke_hook
        from ..plugins.types import HookType

        sampling_id = str(uuid.uuid4())

        exception: BaseException | None = None
        s_result: SamplingResult | None = None

        try:
            reqs = self._merge_requirements(requirements)
            effective_loop_budget = self.loop_budget

            # --- sampling_loop_start hook ---
            if has_plugins(HookType.SAMPLING_LOOP_START):
                from ..plugins.hooks.sampling import SamplingLoopStartPayload

                start_payload = SamplingLoopStartPayload(
                    sampling_id=sampling_id,
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

            # Hooks can override loop_budget but bypass the constructor's
            # validation; reject non-positive values up front.
            if effective_loop_budget < 1:
                raise ValueError(
                    f"SAMPLING_LOOP_START hook returned non-positive loop_budget="
                    f"{effective_loop_budget}; must be >= 1."
                )

            s_result = await self._sample(
                action=action,
                context=context,
                backend=backend,
                requirements=reqs,
                effective_loop_budget=effective_loop_budget,
                validation_ctx=validation_ctx,
                format=format,
                model_options=model_options,
                tool_calls=tool_calls,
                sampling_id=sampling_id,
                **kwargs,
            )
            return s_result

        except BaseException as exc:
            exception = exc
            raise
        finally:
            # --- sampling_loop_end hook ---
            if has_plugins(HookType.SAMPLING_LOOP_END):
                from ..plugins.hooks.sampling import SamplingLoopEndPayload

                if exception is not None:
                    end_payload = SamplingLoopEndPayload(
                        sampling_id=sampling_id,
                        strategy_name=type(self).__name__,
                        success=False,
                        exception=exception,
                    )
                else:
                    assert s_result is not None
                    end_payload = SamplingLoopEndPayload(
                        sampling_id=sampling_id,
                        strategy_name=type(self).__name__,
                        success=s_result.success,
                        iterations_used=len(s_result.sample_generations),
                        final_result=s_result.result,
                        final_action=s_result.result_action,
                        final_context=s_result.result_ctx,
                        all_results=list(s_result.sample_generations),
                        all_validations=list(s_result.sample_validations),
                        failure_reason=(
                            None
                            if s_result.success
                            else f"Budget exhausted after {len(s_result.sample_generations)} iterations"
                        ),
                    )
                await invoke_hook(
                    HookType.SAMPLING_LOOP_END, end_payload, backend=backend
                )

    @abc.abstractmethod
    async def _sample(
        self,
        action: Component[S] | CBlock | ModelOutputThunk,
        context: Context,
        backend: Backend,
        requirements: list[Requirement],
        *,
        effective_loop_budget: int,
        validation_ctx: Context | None = None,
        format: type[BaseModelSubclass] | None = None,
        model_options: dict | None = None,
        tool_calls: bool = False,
        sampling_id: str,
        **kwargs,
    ) -> SamplingResult[S]:
        """Execute the sampling algorithm.

        Called by the `sample` wrapper after lifecycle setup. Subclasses must
        override this method instead of `sample`.

        Args:
            action: The action object to be sampled.
            context: The context to be passed to the sampling strategy.
            backend: The backend used for generating samples.
            requirements: Merged and deduplicated list of requirements.
            effective_loop_budget: The loop budget after hook modification (always >= 1).
            validation_ctx: Optional context to use for validation.
            format: Output format for structured outputs.
            model_options: Model options to pass to the backend.
            tool_calls: True if tool calls should be used.
            sampling_id: UUID correlating iteration/repair/end hooks to this loop.
            **kwargs: Additional keyword arguments (e.g., `show_progress`).

        Returns:
            SamplingResult[S]: A result object indicating the success or failure of the sampling process.
        """
        ...

    async def _emit_sampling_iteration(
        self,
        sampling_id: str,
        iteration: int,
        action: SampleActionType,
        result: ModelOutputThunk,
        validation_results: list[tuple[Requirement, ValidationResult]],
        backend: Backend,
        *,
        sample_index: int | None = None,
    ) -> None:
        """Emit the sampling-iteration hook payload if any plugin is registered."""
        from ..plugins.manager import has_plugins, invoke_hook
        from ..plugins.types import HookType

        if not has_plugins(HookType.SAMPLING_ITERATION):
            return

        from ..plugins.hooks.sampling import SamplingIterationPayload

        all_validations_passed = all(bool(s[1]) for s in validation_results)
        iter_payload = SamplingIterationPayload(
            sampling_id=sampling_id,
            strategy_name=type(self).__name__,
            iteration=iteration,
            sample_index=sample_index,
            action=action,
            result=result,
            validation_results=validation_results,
            all_validations_passed=all_validations_passed,
            valid_count=sum(1 for s in validation_results if bool(s[1])),
            total_count=len(validation_results),
        )
        await invoke_hook(HookType.SAMPLING_ITERATION, iter_payload, backend=backend)

    async def _emit_sampling_repair(
        self,
        sampling_id: str,
        repair_iteration: int,
        failed_action: SampleActionType,
        failed_result: ModelOutputThunk,
        failed_validations: list[tuple[Requirement, ValidationResult]],
        repair_action: SampleActionType,
        repair_context: Context,
        backend: Backend,
        *,
        sample_index: int | None = None,
    ) -> None:
        """Emit the sampling-repair hook payload if any plugin is registered."""
        from ..plugins.manager import has_plugins, invoke_hook
        from ..plugins.types import HookType

        if not has_plugins(HookType.SAMPLING_REPAIR):
            return

        from ..plugins.hooks.sampling import SamplingRepairPayload

        repair_payload = SamplingRepairPayload(
            sampling_id=sampling_id,
            repair_type=getattr(self, "_get_repair_type", lambda: "unknown")(),
            failed_action=failed_action,
            failed_result=failed_result,
            failed_validations=failed_validations,
            repair_action=repair_action,
            repair_context=repair_context,
            repair_iteration=repair_iteration,
            sample_index=sample_index,
        )
        await invoke_hook(HookType.SAMPLING_REPAIR, repair_payload, backend=backend)

    def _merge_requirements(
        self, call_requirements: list[Requirement] | None
    ) -> list[Requirement]:
        """Merge global strategy requirements with per-call requirements.

        Global requirements (set on the strategy) supersede per-call requirements.

        Args:
            call_requirements: Requirements provided at the call site.

        Returns:
            Deduplicated list of requirements to use for this sampling call.
        """
        reqs: list[Requirement] = []
        if self.requirements is not None:
            reqs += self.requirements
        elif call_requirements is not None:
            reqs += call_requirements
        return list(set(reqs))
