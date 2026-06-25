"""Requirements are a special type of Component used as input to the "validate" step in Instruct/Validate/Repair design patterns."""

import json
import math
from collections.abc import Callable
from typing import Any, overload

from ...core import (
    CBlock,
    Context,
    MelleaLogger,
    ModelOutputThunk,
    Requirement,
    ValidationResult,
)
from ...backends.adapters import AdapterSchemaMismatchError
from ..components.intrinsic import Intrinsic


class LLMaJRequirement(Requirement):
    """A requirement that always uses LLM-as-a-Judge. Any available constraint ALoRA will be ignored.

    Attributes:
        use_aloras (bool): Always `False` for this class; ALoRA adapters are
            never used even if they are available.
    """

    use_aloras: bool = False


def requirement_check_to_bool(x: CBlock | ModelOutputThunk | str) -> bool:
    """Convert a ``requirement-check`` adapter output string to a boolean result.
    Parses the JSON output produced by the ``requirement-check`` adapter and
    returns ``True`` when the score exceeds 0.5.

    Args:
        x: Adapter output string or CBlock containing JSON with the contract
            ``{"requirement_check": {"score": <float>}}``.

    Returns:
        ``True`` if the extracted score exceeds 0.5, ``False`` otherwise.

    Raises:
        json.JSONDecodeError: If ``x`` is not valid JSON.
        AdapterSchemaMismatchError: If the parsed output does not contain the
            expected ``requirement_check.score`` structure, or if the score is
            not a finite number in the range 0.0-1.0.  Callers that previously
            treated ``False`` as "requirement not met" must now catch this error
            separately.
    """
    output = str(x)
    req_dict: dict[str, Any] = json.loads(output)

    # Mirrors the validation in requirement_check() in core.py; Phase 2 will consolidate via IOContract.
    req_check = req_dict.get("requirement_check", None)
    if not isinstance(req_check, dict):
        raise AdapterSchemaMismatchError(
            name="requirement-check",
            observed_keys=frozenset(req_dict.keys()),
            expected_keys=frozenset({"requirement_check"}),
        )

    score = req_check.get("score", None)
    if (
        not isinstance(score, (int, float))
        or isinstance(score, bool)  # bool subclasses int; exclude it explicitly
        or not math.isfinite(score)
        or not 0.0 <= score <= 1.0
    ):
        raise AdapterSchemaMismatchError(
            name="requirement-check",
            observed_keys=frozenset(req_check.keys()),
            expected_keys=frozenset({"score"}),
        )

    return score > 0.5


class ALoraRequirement(Requirement, Intrinsic):
    """A requirement validated by an ALoRA adapter; falls back to LLM-as-a-Judge only on generation error.

    If the adapter is unavailable (e.g. cannot be loaded), ``mellea`` uses
    LLMaJ for that requirement instead.  That is the only case where LLMaJ
    will be used.

    If the adapter generates output but the output **fails schema validation**
    (``requirement_check_to_bool`` raises ``AdapterSchemaMismatchError``), the
    exception propagates to the caller — it is not caught and does not trigger
    the LLMaJ fallback.  This is intentional: schema drift should surface
    loudly rather than silently return a wrong result.

    Args:
        description (str): Human-readable requirement description.
        intrinsic_name (str | None): Name of the ALoRA intrinsic to use.
            Defaults to ``"requirement-check"``.

    Attributes:
        use_aloras (bool): Always ``True``; this class always attempts to use
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
