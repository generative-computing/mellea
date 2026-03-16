"""Abstract interfaces for sampling strategies and their results.

``SamplingStrategy`` defines the contract for all sampling algorithms: an async
``sample`` method that takes an action, context, backend, and requirements, and
returns a ``SamplingResult``. ``SamplingResult`` records the chosen generation
alongside the full history of intermediate samples, their validation outcomes,
and associated contexts — enabling detailed post-hoc inspection of the sampling
process.
"""

import abc
from typing import Generic

from .backend import Backend, BaseModelSubclass
from .base import CBlock, Component, Context, ModelOutputThunk, S
from .requirement import Requirement, ValidationResult


class SamplingResult(CBlock, Generic[S]):
    """Stores the results from a sampling operation. This includes successful and failed samplings.

    Args:
        result_index (int): Index into ``sample_generations`` identifying the chosen final output.
        success (bool): Whether the sampling operation produced a passing result.
        sample_generations (list[ModelOutputThunk[S]] | None): All output thunks generated during sampling.
        sample_validations (list[list[tuple[Requirement, ValidationResult]]] | None): Per-generation validation
            results; each inner list contains one tuple per requirement evaluated.
        sample_actions (list[Component] | None): The actions used to produce each generation.
        sample_contexts (list[Context] | None): The contexts associated with each generation.

    Attributes:
        result_index (int): Index into ``sample_generations`` identifying the chosen final output.
        success (bool): Whether the sampling operation produced a passing result.
        sample_generations (list[ModelOutputThunk[S]]): All output thunks generated during
            sampling; always a list (``None`` input is normalised to ``[]``).
        sample_validations (list[list[tuple[Requirement, ValidationResult]]]): Per-generation
            validation results; always a list (``None`` input is normalised to ``[]``).
        sample_actions (list[Component]): The actions used to produce each generation;
            always a list (``None`` input is normalised to ``[]``).
        sample_contexts (list[Context]): The contexts associated with each generation;
            always a list (``None`` input is normalised to ``[]``).
    """

    def __init__(
        self,
        result_index: int,
        success: bool,
        *,
        sample_generations: list[ModelOutputThunk[S]] | None = None,
        sample_validations: list[list[tuple[Requirement, ValidationResult]]]
        | None = None,
        sample_actions: list[Component] | None = None,
        sample_contexts: list[Context] | None = None,
    ):
        """Initialize SamplingResult with the chosen output index, success flag, and generation history."""
        if sample_generations is None:
            sample_generations = []
        if sample_validations is None:
            sample_validations = []
        if sample_actions is None:
            sample_actions = []
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
    def result(self) -> ModelOutputThunk[S]:
        """The final output or result from applying the sampling strategy."""
        return self.sample_generations[self.result_index]

    @property
    def result_ctx(self) -> Context:
        """The context of the final output or result from applying the sampling strategy."""
        return self.sample_contexts[self.result_index]

    @property
    def result_action(self) -> Component[S]:
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
    """

    @abc.abstractmethod
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
    ) -> SamplingResult[S]:
        """This method is the abstract method for sampling a given component.

        It must be implemented by any concrete subclasses to provide specific sampling logic.

        Args:
            action : The action object to be sampled.
            context: The context to be passed to the sampling strategy.
            backend: The backend used for generating samples.
            requirements: List of requirements to test against (merged with global requirements).
            validation_ctx: Optional context to use for validation. If None, validation_ctx = ctx.
            format: output format for structured outputs.
            model_options: model options to pass to the backend during generation / validation.
            tool_calls: True if tool calls should be used during this sampling strategy.

        Returns:
            SamplingResult: A result object indicating the success or failure of the sampling process.
        """
