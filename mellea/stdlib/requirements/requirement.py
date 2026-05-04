"""Requirements are a special type of Component used as input to the "validate" step in Instruct/Validate/Repair design patterns."""

import json
from collections.abc import Callable
from typing import Any, overload

from ...core import CBlock, Context, MelleaLogger, Requirement, ValidationResult
from ..components.intrinsic import Intrinsic


class LLMaJRequirement(Requirement):
    """A requirement that always uses LLM-as-a-Judge. Any available constraint ALoRA will be ignored.

    Attributes:
        use_aloras (bool): Always ``False`` for this class; ALoRA adapters are
            never used even if they are available.
    """

    use_aloras: bool = False


def requirement_check_to_bool(x: CBlock | str) -> bool:
    """Checks if a given output should be marked converted to ``True``.

    By default, the requirement check alora outputs: ``{"requirement_likelihood": 0.0}``.
    Returns ``True`` if the likelihood value is > 0.5.

    Args:
        x: ALoRA output string or CBlock containing JSON with a
            ``requirement_likelihood`` field.

    Returns:
        True if the extracted likelihood exceeds 0.5, False otherwise.
    """
    output = str(x)
    req_dict: dict[str, Any] = json.loads(output)

    likelihood = req_dict.get("requirement_check", None)
    if not isinstance(likelihood, dict):
        MelleaLogger.get_logger().warning(
            f"could not get value from alora requirement output; looking for `requirement_check` in {req_dict}"
        )
        return False
    
    score = likelihood.get("score", None)
    if score is None:
        MelleaLogger.get_logger().warning(
            f"could not get value from alora requirement output; looking for `score` in {req_dict}"
        )
        return False

    if score > 0.5:
        return True

    return False


class ALoraRequirement(Requirement, Intrinsic):
    """A requirement validated by an ALoRA adapter; falls back to LLM-as-a-Judge only on error.

    If an exception is thrown during the ALoRA execution path, ``mellea`` will
    fall back to LLMaJ. That is the only case where LLMaJ will be used.

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
        r: A ``Requirement`` object or a plain string description to wrap as one.

    Returns:
        A ``Requirement`` instance.

    Raises:
        Exception: If ``r`` is neither a ``str`` nor a ``Requirement`` instance.
    """
    if type(r) is str:
        return Requirement(r)
    elif isinstance(r, Requirement):
        return r
    else:
        raise Exception(f"reqify takes a str or requirement, not {r}")


def req(*args, **kwargs) -> Requirement:
    """Shorthand for ``Requirement.__init__``.

    Args:
        *args: Positional arguments forwarded to ``Requirement.__init__``.
        **kwargs: Keyword arguments forwarded to ``Requirement.__init__``.

    Returns:
        A new ``Requirement`` instance.
    """
    return Requirement(*args, **kwargs)


def check(*args, **kwargs) -> Requirement:
    """Shorthand for ``Requirement.__init__(..., check_only=True)``.

    Args:
        *args: Positional arguments forwarded to ``Requirement.__init__``.
        **kwargs: Keyword arguments forwarded to ``Requirement.__init__``.

    Returns:
        A new ``Requirement`` instance with ``check_only=True``.
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
        A validation function that takes a ``Context`` and returns a ``ValidationResult``.

    Raises:
        ValueError: If ``fn`` returns a type other than ``bool`` or
            ``tuple[bool, str]``.
    """

    def validate(ctx: Context) -> ValidationResult:
        o = ctx.last_output()
        if o is None or o.value is None:
            MelleaLogger.get_logger().warn(
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
