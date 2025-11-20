from collections.abc import Callable
from typing import Optional

from mellea.stdlib.base import Context
from mellea.stdlib.requirement import Requirement, ValidationResult


def _name2str(tool_name: str | Callable) -> str:
    match tool_name:
        case tool_name if callable(tool_name):
            return tool_name.__name__
        case str():
            return tool_name
        case _:
            raise TypeError(f"Expected Callable or str but found: {type(tool_name)}")


def uses_tool(tool_name: str | Callable, check_only=False):
    """Forces the model to call a given tool.

    Args:
        tool_name: The tool that must be called; this can be either the name of the tool or the Callable for the tool.
        check_only: Propagates to the Requirement.

    Use `tool_choice` if the OpenAI `tool_choice` model option is supported by your model and inference engine.
    """
    tool_name = _name2str(tool_name)

    def _validate(ctx: Context):
        output = ctx.last_output()
        if output.tool_calls is None:
            return ValidationResult(result=False, reason="There were no tool calls.")
        return ValidationResult(result=tool_name in output.tool_calls)

    return Requirement(
        description=f"Use the {tool_name} tool.",
        validation_fn=_validate,
        check_only=check_only,
    )


def tool_arg_validator(
    description: str,
    tool_name: str | Callable | None,
    arg_name: str,
    validation_fn: Callable,
    check_only: bool = False,
) -> Requirement:
    """A requirement that passes only if `validation_fn` returns a True value for the *value* of the `arg_name` argument to `tool_name`.
    If `tool_name` is not specified, then this requirement is enforced for *every* tool that

    Args:
        description: The Requirement description.
        tool_name: The (optional) tool name for .
        arg_name: The argument to check.
        validation_fn: A validation function for validating the value of the `arg_name` argument.

    Todo:
        1. should this be a requirement?
        2. should this be done automatically when the user provides asserts in their function body?
    """
    if tool_name:
        tool_name = _name2str(tool_name)

    def _validate(ctx: Context):
        output = ctx.last_output()
        if tool_name:
            if tool_name not in output.tool_calls:
                return ValidationResult(
                    result=False, reason=f"Tool {tool_name} was not called."
                )
            if arg_name not in output.tool_calls[tool_name].args:
                return ValidationResult(
                    result=False,
                    reason=f"Tool {tool_name} did not call argument {arg_name}",
                )
            arg_value = output.tool_calls[tool_name].args[arg_name]
            validate_result = validation_fn(arg_value)
            if validate_result:
                return ValidationResult(result=True)
            else:
                return ValidationResult(
                    result=False,
                    reason=f"Valiudation did not pass for {tool_name}.{arg_name}. Arg value: {arg_value}. Argument validation result: {validate_result}",
                )
        else:
            for tool in ctx.last_output().tool_calls.keys():
                if arg_name in output.tool_calls[tool].args:
                    arg_value = output.tool_calls[tool].args[arg_name]
                    validate_result = validation_fn(arg_value)
                    if not validate_result:
                        return ValidationResult(
                            result=False,
                            reason=f"Valiudation did not pass for {tool_name}.{arg_name}. Arg value: {arg_value}. Argument validation result: {validate_result}",
                        )
            return ValidationResult(result=True)

    return Requirement(
        description=description, validation_fn=_validate, check_only=check_only
    )
