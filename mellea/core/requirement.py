"""Interface for Requirements."""

import re
from collections.abc import Callable
from copy import copy

from .backend import Backend, BaseModelSubclass
from .base import CBlock, Component, Context, ModelOutputThunk, TemplateRepresentation


class ValidationResult:
    """ValidationResults store the output of a Requirement's validation. They can be used to return additional info from validation functions, which is useful for sampling/repairing."""

    def __init__(
        self,
        result: bool,
        *,
        reason: str | None = None,
        score: float | None = None,
        thunk: ModelOutputThunk | None = None,
        context: Context | None = None,
    ):
        """The result of a requirement's validation.

        A ValidationResult's result field always contains a definitive pass/fail. The other fields can be used to communicate additional information about that result.

        Args:
            result: a boolean that is true if the requirement passed
            reason: a reason for the result
            score: if your validator gives you a score back, you can add this as metadata
            thunk: if your validator utilizes a backend to generate a response, the ModelOutputThunk returned from that request
            context: if your validator utilizes a backend to generate a response, the context associated with that response
        """
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
        """Return a boolean value based on the result."""
        return self._result

    def __bool__(self) -> bool:
        """Return a boolean value based on the result."""
        return self.as_bool()


def default_output_to_bool(x: CBlock | str) -> bool:
    """Checks if a given output should be marked converted to `True`.

    Checks if the output is exactly equal to "yes" or "y" (case-insensitive). If not, it will also
    check if any of the words in the output are "yes" (case-insensitive).
    """
    output = str(x)

    if output.upper() == "YES" or output.upper() == "Y":
        return True

    word_splits = re.split(r"\W+", output)
    if "YES" in [word.upper() for word in word_splits]:
        return True

    return False


class Requirement(Component[str]):
    """Requirements are a special type of Component used as input to the Validate step in Instruct/Validate/Repair patterns."""

    def __init__(
        self,
        description: str | None = None,
        validation_fn: Callable[[Context], ValidationResult] | None = None,
        *,
        output_to_bool: Callable[[CBlock | str], bool] | None = default_output_to_bool,
        check_only: bool = False,
    ):
        """A Requirement, interpreted over a Context.

          By default, requirements are validated by the model using LLM-as-a-Judge (or a `constraint` LoRA when available). However, you can also provide a `validate` function with arbitrary behavior.

        Args:
            description: A natural-language description of the requirement. This will sometimes be included in `Instruction` prompts; if you do not want the requirement to be included in the prompt to avoid [Purple Elephant Effects](https://${PROJECT_URL}/llm-requirement-engineering-and-purple-elephants/) use check_only=True.
            validation_fn: If provided, this function will be executed instead of using LLM-as-a-Judge. The `bool()` for the function's output defines whether the requirement passes.
            output_to_bool: An `output_to_bool` may be provided so that the library can translate the LLM-as-a-judge or ALora output into a boolean value. If none is provided, we will look for 'yes' (case-insensitive) in the LLMaJ output.
            check_only: If set, then `Instruction` will not include this requirement in its prompt.
        """
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
        """Chooses the appropriate validation strategy and applies that strategy."""
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

    def parts(self):
        """Returns all of the constituent parts of a Requirement."""
        return []

    def format_for_llm(self) -> TemplateRepresentation | str:
        """Some object protocol magic happens here with management of the output."""
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
