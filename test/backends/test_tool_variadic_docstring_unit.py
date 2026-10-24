# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for variadic parameters and docstring section parsing in tool schemas.

Tool calls pass arguments by keyword only, so a model can never fill `*args`, and
`**kwargs` has no fixed names to describe. Both are left out of the schema. The
docstring parser only treats a whole line such as `Args:` as a section header, so
parameters named `args`, `returns`, `yields` or `raises` keep their descriptions.

See: https://github.com/generative-computing/mellea/issues/1517
"""

import pytest
from pydantic import ValidationError

from mellea.backends.tools import (
    MelleaTool,
    convert_function_to_ollama_tool,
    validate_tool_arguments,
)


def _params(func):
    """Return the serialized `parameters` block for a callable."""
    schema = convert_function_to_ollama_tool(func, func.__name__).model_dump(
        exclude_none=True
    )
    return schema["function"]["parameters"]


def varargs_tool(a: int, *args: int, **kwargs: str) -> str:
    """Docs.

    Args:
        a: first
    """
    return f"a={a} args={args} kwargs={kwargs}"


def test_variadic_params_left_out_of_schema():
    """`*args` and `**kwargs` must not appear as properties or in `required`."""
    params = _params(varargs_tool)
    assert params["required"] == ["a"]
    assert set(params["properties"]) == {"a"}


def test_kwargs_tool_receives_extra_args_in_lenient_mode():
    """Extra arguments still reach `**kwargs` on the path the backends use."""
    tool = MelleaTool.from_callable(varargs_tool)
    validated = validate_tool_arguments(tool, {"a": 1, "extra": "x"}, strict=False)
    assert tool.run(**validated) == "a=1 args=() kwargs={'extra': 'x'}"


def test_extra_args_rejected_in_strict_mode():
    """Strict validation rejects arguments the schema does not list."""
    tool = MelleaTool.from_callable(varargs_tool)
    with pytest.raises(ValidationError):
        validate_tool_arguments(tool, {"a": 1, "extra": "x"}, strict=True)


def test_param_named_args_keeps_its_description():
    """A parameter line `args: ...` is not mistaken for the `Args:` header."""

    def run_cmd(args: str, retries: int = 1) -> str:
        """Run a command.

        Args:
            args: the command line to run
            retries: how many times to retry
        """
        return args

    props = _params(run_cmd)["properties"]
    assert props["args"]["description"] == "the command line to run"
    assert props["retries"]["description"] == "how many times to retry"


def test_undocumented_args_param_gets_no_other_text():
    """An undocumented `args` parameter does not pick up the Args section text."""

    def run_cmd(args: str, retries: int = 1) -> str:
        """Run a command.

        Args:
            retries: how many times to retry
        """
        return args

    props = _params(run_cmd)["properties"]
    assert props["args"].get("description", "") == ""
    assert props["retries"]["description"] == "how many times to retry"


def test_params_named_like_section_headers_keep_later_descriptions():
    """Parameters named `returns` or `raises` do not end the Args section early."""

    def check(a: int, returns: str, raises: bool, b: int) -> str:
        """Check something.

        Args:
            a: first
            returns: what to return
            raises: whether to raise
            b: last
        """
        return returns

    props = _params(check)["properties"]
    assert {k: v["description"] for k, v in props.items()} == {
        "a": "first",
        "returns": "what to return",
        "raises": "whether to raise",
        "b": "last",
    }
