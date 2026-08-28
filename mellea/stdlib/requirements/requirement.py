# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Requirements are a special type of Component used as input to the "validate" step in Instruct/Validate/Repair design patterns."""

from collections.abc import Callable
from typing import Any, cast, overload

from ...backends.adapters import get_io_contract
from ...core import (
    CBlock,
    Context,
    MelleaLogger,
    ModelOutputThunk,
    Requirement,
    ValidationResult,
)
from ..components.intrinsic import Intrinsic


class LLMaJRequirement(Requirement):
    """A requirement that always uses LLM-as-a-Judge. Any available constraint ALoRA will be ignored.

    Attributes:
        use_aloras (bool): Always `False` for this class; ALoRA adapters are
            never used even if they are available.
    """

    use_aloras: bool = False


def requirement_check_to_bool(x: CBlock | ModelOutputThunk | str) -> bool:
    """Convert a `requirement-check` adapter output string to a boolean result.

    Parses the JSON output produced by the `requirement-check` adapter and
    returns `True` when the score exceeds 0.5.

    Args:
        x: Adapter output string or CBlock containing JSON with the contract
            `{"requirement_check": {"score": <float>}}`.

    Returns:
        `True` if the extracted score exceeds 0.5, `False` otherwise.

    Raises:
        json.JSONDecodeError: If `x` is not valid JSON.
        ValueError: If the parsed JSON is not an object (e.g. a list, string, or
            number).
        AdapterSchemaMismatchError: If the parsed output does not contain the
            expected `requirement_check.score` structure, or if the score is
            not a finite number in the range 0.0-1.0.  Callers that previously
            treated `False` as "requirement not met" must now catch this error
            separately.
    """
    # Delegates to the same `requirement-check` IOContract resolve_adapter() hands
    # back to call_intrinsic() — see mellea.backends.adapters.io_contracts. A second,
    # independent validation here is exactly the parallel-declaration problem #1516
    # closes; ALoraRequirement (below) is the other production consumer of this
    # capability's output alongside core.requirement_check().
    parsed = get_io_contract("requirement-check").parse(str(x))
    score = cast(dict[str, Any], parsed["requirement_check"])["score"]
    return cast(float, score) > 0.5


class ALoraRequirement(Requirement, Intrinsic):
    """A requirement validated by an ALoRA adapter; falls back to LLM-as-a-Judge only on generation error.

    If the adapter is unavailable (e.g. cannot be loaded), `mellea` uses
    LLMaJ for that requirement instead.  That is the only case where LLMaJ
    will be used.

    If the adapter generates output but the output **fails schema validation**
    (`requirement_check_to_bool` raises `AdapterSchemaMismatchError`), the
    error is surfaced on the returned `ValidationResult.error` by
    `Requirement.validate` — it is not caught here and does not trigger the
    LLMaJ fallback, which covers only generation errors.  The result fails
    closed (`bool(result)` is `False`), so schema drift is neither silently
    treated as a pass nor raised as an unexpected exception; callers can inspect
    `result.error` to tell it apart from an ordinary "requirement not met".

    Args:
        description (str): Human-readable requirement description.
        intrinsic_name (str | None): Name of the ALoRA intrinsic to use.
            Defaults to `"requirement-check"`.

    Attributes:
        use_aloras (bool): Always `True`; this class always attempts to use
            ALoRA adapters for validation.
    """

    def __init__(self, description: str, intrinsic_name: str | None = None):
        """Initialize ALoraRequirement with a description and optional intrinsic adapter name."""
        # TODO: We may want to actually do the validation_fn here so that we can set the score.
        super().__init__(
            description, validation_fn=None, output_to_bool=requirement_check_to_bool
        )
        self.use_aloras: bool = True

        if intrinsic_name is None:
            intrinsic_name = "requirement-check"

        # Initialize the other side of the inheritance tree
        Intrinsic.__init__(
            self,
            intrinsic_name=intrinsic_name,
            intrinsic_kwargs={"requirement": f"{self.description}"},
        )


def reqify(r: str | Requirement) -> Requirement:
    """Map strings to Requirements.

    This is a utility method for functions that allow you to pass in Requirements as either explicit Requirement objects or strings that you intend to be interpreted as requirements.

    Args:
        r: A `Requirement` object or a plain string description to wrap as one.

    Returns:
        A `Requirement` instance.

    Raises:
        Exception: If `r` is neither a `str` nor a `Requirement` instance.
    """
    if type(r) is str:
        return Requirement(r)
    elif isinstance(r, Requirement):
        return r
    else:
        raise Exception(f"reqify takes a str or requirement, not {r}")


def req(*args, **kwargs) -> Requirement:
    """Shorthand for `Requirement.__init__`.

    Args:
        *args: Positional arguments forwarded to `Requirement.__init__`.
        **kwargs: Keyword arguments forwarded to `Requirement.__init__`.

    Returns:
        A new `Requirement` instance.
    """
    return Requirement(*args, **kwargs)


def check(*args, **kwargs) -> Requirement:
    """Shorthand for `Requirement.__init__(..., check_only=True)`.

    Args:
        *args: Positional arguments forwarded to `Requirement.__init__`.
        **kwargs: Keyword arguments forwarded to `Requirement.__init__`.

    Returns:
        A new `Requirement` instance with `check_only=True`.
    """
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

    Returns:
        A validation function that takes a `Context` and returns a `ValidationResult`.

    Raises:
        ValueError: If `fn` returns a type other than `bool` or
            `tuple[bool, str]`.
    """

    def validate(ctx: Context) -> ValidationResult:
        o = ctx.last_output()
        if o is None or o.value is None:
            MelleaLogger.get_logger().warning(
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
