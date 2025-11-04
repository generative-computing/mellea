"""Requirements are a special type of Component used as input to the "validate" step in Instruct/Validate/Repair design patterns."""

import inspect
import json
import re
from collections.abc import Callable
from copy import copy
from typing import Any, overload

from mellea.backends import Backend, BaseModelSubclass
from mellea.backends.adapters.adapter import AdapterType
from mellea.helpers.fancy_logger import FancyLogger
from mellea.stdlib.base import (
    CBlock,
    Component,
    Context,
    GenerateLog,
    ModelOutputThunk,
    TemplateRepresentation,
)
from mellea.stdlib.intrinsics.intrinsic import Intrinsic


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


class ValidationResult:
    """ValidationResults store the output of a Requirement's validation. They can be used to return additional info from validation functions, which is useful for sampling/repairing."""

    def __init__(
        self,
        result: bool,
        *,
        reason: str | None = None,
        score: float | None = None,
        thunk: ModelOutputThunk | None = None,
    ):
        """The result of a requirement's validation.

        A ValidationResult's result field always contains a definitive pass/fail. The other fields can be used to communicate additional information about that result.

        Args:
            result: a boolean that is true if the requirement passed
            reason: a reason for the result
            score: if your validator gives you a score back, you can add this as metadata
            thunk: if your validator utilizes a backend to generate a response, the ModelOutputThunk returned from that request
        """
        self._result = result
        self._reason = reason
        self._score = score
        self._thunk = thunk

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

    def as_bool(self) -> bool:
        """Return a boolean value based on the result."""
        return self._result

    def __bool__(self) -> bool:
        """Return a boolean value based on the result."""
        return self.as_bool()


class Requirement(Component):
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
            llm_as_a_judge_result, _ = backend.generate_from_context(
                req_copy, ctx, format=format, model_options=model_options
            )
            await llm_as_a_judge_result.avalue()

            return ValidationResult(
                result=self.output_to_bool(llm_as_a_judge_result),
                reason=llm_as_a_judge_result.value,
                thunk=llm_as_a_judge_result,
            )

    def parts(self):
        """Returns all of the constituent parts of a Requirement."""
        raise Exception(
            "Disallowing use of `parts` until we figure out exactly what it's supposed to be for"
        )

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


class LLMaJRequirement(Requirement):
    """A requirement that always uses LLM-as-a-Judge. Any available constraint ALoRA will be ignored."""

    use_aloras: bool = False


def requirement_check_to_bool(x: CBlock | str) -> bool:
    """Checks if a given output should be marked converted to `True`.

    By default, the requirement check alora outputs: `{"requirement_likelihood": 0.0}`.
    True if >.5
    """
    output = str(x)
    req_dict: dict[str, Any] = json.loads(output)

    likelihood = req_dict.get("requirement_likelihood", None)
    if likelihood is None:
        FancyLogger.get_logger().warning(
            f"could not get value from alora requirement output; looking for `requirement_likelihood` in {req_dict}"
        )
        return False

    if likelihood > 0.5:
        return True

    return False


class ALoraRequirement(Requirement, Intrinsic):
    """A requirement that always uses an (possibly specified) ALora. If an exception is thrown during the ALora execution path, `mellea` will fall back to LLMaJ. But that is the only case where LLMaJ will be used."""

    def __init__(self, description: str, intrinsic_name: str | None = None):
        """A requirement that is validated by an ALora.

        Args:
            description: See `Requirement.__init__`
            intrinsic_name: the name of the intrinsic; must match the adapter
        """
        # TODO: We may want to actually do the validation_fn here so that we can set the score.
        super().__init__(
            description, validation_fn=None, output_to_bool=requirement_check_to_bool
        )
        self.use_aloras: bool = True

        if intrinsic_name is None:
            intrinsic_name = "requirement_check"

        self.intrinsic_name = intrinsic_name
        self.adapter_types = [AdapterType.ALORA]

    @property
    def intrinsic_kwargs(self):
        """An AloraRequirement's intrinsic kwarg is always the requirement's description."""
        return {"requirement": f"{self.description}"}


class ScorerRequirement(Requirement):
    """A requirement that always returns a non-None score. The scorer must also define a preference ordering to indicate whether the goal is to maximize or minimize the score."""

    def __init__(
        self,
        description: str | None = None,
        validation_fn: Callable[[Context], ValidationResult] | None = None,
        preference_ordering: str = "max",
        *,
        output_to_bool: Callable[[CBlock | str], bool] | None = default_output_to_bool,
        check_only: bool = False,
    ):
        """A requirement that is validated by an ALora.

        Args:
            description: See `Requirement.__init__`
            validation_fn:  If provided, this function will be executed instead of using LLM-as-a-Judge. This function must return a valid score
            preference_ordering: indicates whether the goal is to maximize or minimize the score. must be either "max" or "min". Defaults to None
            output_to_bool: See `Requirement.__init__`
            check_only: See `Requirement.__init__`
        """
        super().__init__(
            description,
            validation_fn=validation_fn,
            output_to_bool=output_to_bool,
            check_only=check_only,
        )

        if preference_ordering.lower() not in ["max", "min"]:
            raise NotImplementedError
        self.preference_ordering: str = preference_ordering.lower()

    async def validate(
        self,
        backend: Backend,
        ctx: Context,
        *,
        format: type[BaseModelSubclass] | None = None,
        model_options: dict | None = None,
    ) -> ValidationResult:
        """Chooses the appropriate validation strategy and applies that strategy. Asserts that the returned ValidationResult has a valid score."""
        if self.validation_fn is not None:
            # Python validation strategy
            validation_result = self.validation_fn(ctx)
            assert validation_result._score is not None, (
                "ScorerRequirement must have a score that is not None"
            )
            return validation_result
        else:
            # LLMaJ validation strategy. This includes ALora because the backend generate call will appropriately dispatch.
            # For ScorerRequirement, provide score of 1 for result=True, 0 for result=False
            assert self.output_to_bool is not None
            last_output = ctx.last_output()
            assert isinstance(last_output, ModelOutputThunk), (
                " Context has no appropriate last output"
            )

            # Create a copy of the requirement that holds the output
            # and its template gets populated with the output correctly.
            req_copy = copy(self)
            req_copy._output = last_output.value
            llm_as_a_judge_result, _ = backend.generate_from_context(
                req_copy, ctx, format=format, model_options=model_options
            )
            await llm_as_a_judge_result.avalue()
            result = self.output_to_bool(llm_as_a_judge_result)

            return ValidationResult(
                result=result,
                reason=llm_as_a_judge_result.value,
                score=1 if result else 0,
                thunk=llm_as_a_judge_result,
            )


def reqify(r: str | Requirement) -> Requirement:
    """Maps strings to Requirements.

    This is a utility method for functions that allow you to pass in Requirements as either explicit Requirement objects or strings that you intend to be interpreted as requirements.
    """
    if type(r) is str:
        return Requirement(r)
    elif isinstance(r, Requirement):
        return r
    else:
        raise Exception(f"reqify takes a str or requirement, not {r}")


def req(*args, **kwargs) -> Requirement:
    """Shorthand for Requirement.__init__."""
    return Requirement(*args, **kwargs)


def check(*args, **kwargs) -> Requirement:
    """Shorthand for Requirement.__init__(..., check_only=True)."""
    return Requirement(*args, **kwargs, check_only=True)


@overload
def simple_validate(
    fn: Callable[[str], tuple[bool, str]],
) -> Callable[[Context], ValidationResult]: ...


@overload
def simple_validate(
    fn: Callable[[str], bool], *, reason: str | None = None
) -> Callable[[Context], ValidationResult]: ...


def simple_validate(
    fn: Callable[[str], Any], *, reason: str | None = None
) -> Callable[[Context], ValidationResult]:
    """Syntactic sugar for writing validation functions that only operate over the last output from the model (interpreted as a string).

    This is useful when your validation logic only depends upon the most recent model output. For example:

    `Requirement("Answer 'yes' or 'no'", simple_validate(lambda x: x == 'yes' or x == 'no')`

    Validation functions operate over `Context`. Often you do not care about the entire context, and just want to consider the most recent output from the model.

    Important notes:
     - this operates over the more recent _model output_, not the most recent message.
     - Model outputs are sometimes parsed into more complex types (eg by a `Formatter.parse` call or an OutputProcessor). This validation logic will interpret the most recent output as a string, regardless of whether it has a more complex parsed representation.

    Args:
        fn: the simple validation function that takes a string and returns either a bool or (bool, str)
        reason: only used if the provided function returns a bool; if the validation function fails, a static reason for that failure to give to the llm when repairing
    """

    def validate(ctx: Context) -> ValidationResult:
        o = ctx.last_output()
        if o is None or o.value is None:
            FancyLogger.get_logger().warn(
                "Last output of context was None. That might be a problem. We return validation as False to be able to continue..."
            )
            return ValidationResult(
                False
            )  # Don't pass in the static reason since the function didn't run.

        result = fn(o.value)

        # Only confirm that the result conforms to the fn type requirements here. Functions can
        # declare return types and then deviate from them.

        # Oneliner that checks the tuple actually contains (bool, str)
        if isinstance(result, tuple) and list(map(type, result)) == [bool, str]:
            return ValidationResult(result[0], reason=result[1])

        elif type(result) is bool:
            return ValidationResult(result, reason=reason)

        raise ValueError(
            f"function {fn.__name__} passed to simple_validate didn't return either bool or [bool, str]; returned {type(result)} instead"
        )

    return validate
