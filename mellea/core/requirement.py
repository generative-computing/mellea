"""``Requirement`` interface for constrained and validated generation.

A ``Requirement`` pairs a human-readable description with a validation function that
inspects a ``Context`` (and optionally a backend) to determine whether a model output
meets a constraint. ``ValidationResult`` carries the pass/fail verdict along with an
optional reason, score, and the ``ModelOutputThunk`` produced during validation.
``PartialValidationResult`` provides a tri-state variant (``"pass"``, ``"fail"``,
``"unknown"``) for per-chunk streaming validation.
Helper factories such as ``default_output_to_bool`` make it easy to build requirements
without boilerplate.
"""

import re
from collections.abc import Callable
from copy import copy
from typing import Literal

from .backend import Backend, BaseModelSubclass
from .base import CBlock, Component, Context, ModelOutputThunk, TemplateRepresentation


class ValidationResult:
    """ValidationResults store the output of a Requirement's validation. They can be used to return additional info from validation functions, which is useful for sampling/repairing.

    Args:
        result (bool): Boolean indicating whether the requirement passed.
        reason (str | None): Optional human-readable explanation for the verdict.
        score (float | None): Optional numeric score returned by the validator.
        thunk (ModelOutputThunk | None): The ``ModelOutputThunk`` produced during LLM-as-a-Judge validation, if applicable.
        context (Context | None): The context associated with the validation backend call, if applicable.

    """

    def __init__(
        self,
        result: bool,
        *,
        reason: str | None = None,
        score: float | None = None,
        thunk: ModelOutputThunk | None = None,
        context: Context | None = None,
    ):
        """Initialize ValidationResult with a pass/fail boolean and optional metadata."""
        self._result = result
        self._reason = reason
        self._score = score
        self._thunk = thunk
        self._context = context

    @property
    def reason(self) -> str | None:
        """Reason for the validation result."""
        return self._reason

    @property
    def score(self) -> float | None:
        """An optional score for the validation result."""
        return self._score

    @property
    def thunk(self) -> ModelOutputThunk | None:
        """The ModelOutputThunk associated with the validation func if an llm was used to generate the final result."""
        return self._thunk

    @property
    def context(self) -> Context | None:
        """The context associated with validation if a backend was used to generate the final result."""
        return self._context

    def as_bool(self) -> bool:
        """Return a boolean value based on the validation result.

        Returns:
            bool: ``True`` if the requirement passed, ``False`` otherwise.
        """
        return self._result

    def __bool__(self) -> bool:
        """Return a boolean value based on the result."""
        return self.as_bool()

    def __repr__(self) -> str:
        """Return a developer-readable representation of the validation result."""
        return f"ValidationResult({self._result!r}, reason={self._reason!r}, score={self._score!r})"


class PartialValidationResult:
    """Tri-state result from per-chunk streaming validation.

    Unlike :class:`ValidationResult`, which stores its verdict as a private
    ``_result: bool``, this class exposes ``success`` as a public property.
    The asymmetry is intentional: the tri-state value cannot be recovered from
    a ``bool``, so a public property is the only way to distinguish ``"fail"``
    from ``"unknown"`` after construction.

    Args:
        success: Validation state — ``"pass"`` (constraint satisfied so far),
            ``"fail"`` (constraint violated, stop streaming), or
            ``"unknown"`` (insufficient data yet, continue streaming).
        reason: Optional human-readable explanation.
        score: Optional numeric confidence score.
        thunk: Optional ModelOutputThunk from the validation call.
        context: Optional context associated with the validation call.

    """

    def __init__(
        self,
        success: Literal["pass", "fail", "unknown"],
        *,
        reason: str | None = None,
        score: float | None = None,
        thunk: ModelOutputThunk | None = None,
        context: Context | None = None,
    ):
        """Initialize PartialValidationResult with a tri-state success value."""
        if success not in ("pass", "fail", "unknown"):
            raise ValueError(
                f"success must be 'pass', 'fail', or 'unknown', got {success!r}"
            )
        self._success: Literal["pass", "fail", "unknown"] = success
        self._reason = reason
        self._score = score
        self._thunk = thunk
        self._context = context

    @property
    def success(self) -> Literal["pass", "fail", "unknown"]:
        """The tri-state validation result."""
        return self._success

    @property
    def reason(self) -> str | None:
        """Reason for the validation result."""
        return self._reason

    @property
    def score(self) -> float | None:
        """An optional score for the validation result."""
        return self._score

    @property
    def thunk(self) -> ModelOutputThunk | None:
        """The ModelOutputThunk associated with the validation call, if any."""
        return self._thunk

    @property
    def context(self) -> Context | None:
        """The context associated with the validation call, if any."""
        return self._context

    def as_bool(self) -> bool:
        """Return True for ``"pass"``, False for ``"fail"`` or ``"unknown"``.

        ``"unknown"`` maps to ``False`` intentionally. In streaming contexts,
        check ``pvr.success == "unknown"`` before treating ``False`` as a definitive
        failure — ``"unknown"`` means insufficient data so far, not a constraint
        violation.

        Returns:
            bool: ``True`` if the partial result is ``"pass"``, ``False`` otherwise.
        """
        return self._success == "pass"

    def __bool__(self) -> bool:
        """Return a boolean value based on the success state."""
        return self.as_bool()

    def __repr__(self) -> str:
        """Return a developer-readable representation showing the tri-state value."""
        return f"PartialValidationResult({self._success!r}, reason={self._reason!r}, score={self._score!r})"


def default_output_to_bool(x: CBlock | str) -> bool:
    """Convert a model output string to a boolean by checking for a "yes" answer.

    Checks if the output is exactly equal to "yes" or "y" (case-insensitive). If not,
    also checks if any of the words in the output are "yes" (case-insensitive).

    Args:
        x: The model output to evaluate, as a ``CBlock`` or plain string.

    Returns:
        ``True`` if the output indicates a "yes" answer, ``False`` otherwise.
    """
    output = str(x)

    if output.upper() == "YES" or output.upper() == "Y":
        return True

    word_splits = re.split(r"\W+", output)
    if "YES" in [word.upper() for word in word_splits]:
        return True

    return False


class Requirement(Component[str]):
    """Requirements are a special type of Component used as input to the Validate step in Instruct/Validate/Repair patterns.

    Args:
        description (str | None): A natural-language description of the requirement. Sometimes included in
            ``Instruction`` prompts; use ``check_only=True`` to suppress this.
        validation_fn (Callable[[Context], ValidationResult] | None): If provided, this function is executed
            instead of LLM-as-a-Judge. The ``bool()`` of its return value defines pass/fail.
        output_to_bool (Callable[[CBlock | str], bool] | None): Translates LLM-as-a-Judge output to a boolean.
            Defaults to a "yes"-detection heuristic.
        check_only (bool): When ``True``, the requirement description is excluded from ``Instruction`` prompts.

    Attributes:
        description (str | None): A natural-language description of the requirement.
        output_to_bool (Callable[[CBlock | str], bool] | None): Function used to convert LLM-as-a-Judge
            output into a boolean pass/fail result.
        validation_fn (Callable[[Context], ValidationResult] | None): Optional custom validation
            function that bypasses the LLM-as-a-Judge strategy entirely.
        check_only (bool): When ``True``, the requirement description is excluded from ``Instruction``
            prompts to avoid influencing model output.
    """

    def __init__(
        self,
        description: str | None = None,
        validation_fn: Callable[[Context], ValidationResult] | None = None,
        *,
        output_to_bool: Callable[[CBlock | str], bool] | None = default_output_to_bool,
        check_only: bool = False,
    ):
        """Initialize Requirement with an optional description, validation function, and output converter."""
        self.description = description
        self.output_to_bool = output_to_bool
        self.validation_fn = validation_fn
        self.check_only = check_only

        # Used for validation. Do not manually populate.
        self._output: str | None = None

    async def validate(
        self,
        backend: Backend,
        ctx: Context,
        *,
        format: type[BaseModelSubclass] | None = None,
        model_options: dict | None = None,
    ) -> ValidationResult:
        """Chooses the appropriate validation strategy and applies it to the given context.

        Uses ``validation_fn`` if one was provided, otherwise falls back to LLM-as-a-Judge
        by generating a judgement response with the backend.

        Args:
            backend (Backend): The inference backend used when the LLM-as-a-Judge strategy is selected.
            ctx (Context): The context to validate, which must contain a ``ModelOutputThunk`` as its last output.
            format (type[BaseModelSubclass] | None): Optional structured output format for the judgement call.
            model_options (dict | None): Optional model options to pass to the backend during the judgement call.

        Returns:
            ValidationResult: The result of the validation, including a boolean pass/fail and optional metadata.
        """
        if self.validation_fn is not None:
            # Python validation strategy
            return self.validation_fn(ctx)
        else:
            # LLMaJ validation strategy. This includes ALora because the backend generate call will appropriately dispatch.
            assert self.output_to_bool is not None
            last_output = ctx.last_output()
            assert isinstance(last_output, ModelOutputThunk), (
                " Context has no appropriate last output"
            )

            # Create a copy of the requirement that holds the output
            # and its template gets populated with the output correctly.
            req_copy = copy(self)
            req_copy._output = last_output.value
            llm_as_a_judge_result, val_ctx = await backend.generate_from_context(
                req_copy, ctx, format=format, model_options=model_options
            )
            await llm_as_a_judge_result.avalue()

            return ValidationResult(
                result=self.output_to_bool(llm_as_a_judge_result),
                reason=llm_as_a_judge_result.value,
                thunk=llm_as_a_judge_result,
                context=val_ctx,
            )

    async def stream_validate(
        self, chunk: str, *, backend: Backend, ctx: Context
    ) -> PartialValidationResult:
        """Hook for per-chunk streaming validation.

        The default implementation returns ``PartialValidationResult("unknown")``
        — meaning insufficient data to decide yet. Subclasses override this method
        to inspect the accumulated chunk and return ``"pass"`` or ``"fail"`` early.

        This method must not mutate ``self``. The orchestrator is responsible for
        cloning the requirement before each attempt; any state needed across chunks
        must be managed externally.

        Args:
            chunk: The accumulated model output so far (not just the latest token).
            backend: The inference backend, available for backend-assisted checks.
            ctx: The current generation context.

        Returns:
            PartialValidationResult: ``"unknown"`` by default. Subclasses may return
            ``"pass"`` (constraint satisfied so far) or ``"fail"`` (constraint violated,
            streaming should be aborted). ``"pass"`` does not short-circuit the final
            ``validate()`` call; the orchestrator decides whether to skip it.
        """
        return PartialValidationResult("unknown")

    def parts(self) -> list[Component | CBlock]:
        """Returns all of the constituent parts of a Requirement.

        Returns:
            List of constituent components. Empty by default; subclasses override
            to expose their internal structure.
        """
        return []

    def format_for_llm(self) -> TemplateRepresentation | str:
        """Returns a ``TemplateRepresentation`` for LLM-as-a-Judge evaluation of this requirement.

        Populates the template with the requirement's ``description`` and the stored model
        ``_output``. Must only be called from within a ``validate`` call for this same requirement,
        after ``_output`` has been set.

        Returns:
            TemplateRepresentation | str: A ``TemplateRepresentation`` containing the description
            and the model output to be judged.
        """
        assert self._output is not None, (
            "Object protocol error: should never try to templatize a Requirement except inside of a validate call for that same requirement."
        )
        return TemplateRepresentation(
            obj=self,
            args={"description": self.description, "output": self._output},
            tools=None,
            template_order=["*", "Requirement"],
        )

    def _parse(self, computed: ModelOutputThunk) -> str:
        """Parse the model output. Returns string value for now."""
        return computed.value if computed.value is not None else ""
