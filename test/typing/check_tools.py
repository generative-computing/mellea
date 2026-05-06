"""Mypy overload-resolution checks for MelleaTool and @tool decorator."""

from typing import assert_type

from mellea.backends.tools import MelleaTool, tool


# Test basic tool decorator without arguments
@tool
def simple_tool(x: int, y: str) -> bool:
    """A simple tool."""
    return True


def check_simple_tool_return() -> None:
    """Verify @tool decorator preserves return type."""
    result = simple_tool.run(1, "test")
    assert_type(result, bool)


# Test tool decorator with name argument
@tool(name="custom_name")
def named_tool(value: float) -> str:
    """A tool with custom name."""
    return "result"


def check_named_tool_return() -> None:
    """Verify @tool(name=...) decorator preserves return type."""
    result = named_tool.run(3.14)
    assert_type(result, str)


# Test tool with default arguments
@tool
def tool_with_defaults(required: int, optional: str = "default") -> dict[str, int]:
    """A tool with default arguments."""
    return {"value": required}


def check_tool_with_defaults_return() -> None:
    """Verify tools with default arguments preserve return type."""
    result = tool_with_defaults.run(42)
    assert_type(result, dict[str, int])


def check_tool_with_defaults_optional() -> None:
    """Verify tools with default arguments can be called with optional params."""
    result = tool_with_defaults.run(42, "custom")
    assert_type(result, dict[str, int])


# Test MelleaTool.from_callable
def plain_function(a: str, b: int) -> list[str]:
    """A plain function to wrap."""
    return [a] * b


def check_from_callable_isinstance() -> None:
    """Verify MelleaTool.from_callable returns MelleaTool instance."""
    wrapped = MelleaTool.from_callable(plain_function)
    assert isinstance(wrapped, MelleaTool)


# Test MelleaTool.from_callable with custom name
def check_from_callable_with_name_isinstance() -> None:
    """Verify MelleaTool.from_callable with name returns MelleaTool instance."""
    wrapped = MelleaTool.from_callable(plain_function, name="custom")
    assert isinstance(wrapped, MelleaTool)


# Test tool as function (not decorator)
def another_function(x: bool) -> int:
    """Another function."""
    return 1 if x else 0


def check_tool_as_function_return() -> None:
    """Verify tool() as function call preserves return type."""
    wrapped = tool(another_function)
    result = wrapped.run(True)
    assert_type(result, int)


# Test tool as function with name
def check_tool_as_function_with_name_return() -> None:
    """Verify tool(func, name=...) preserves return type."""
    wrapped = tool(another_function, name="bool_to_int")
    result = wrapped.run(False)
    assert_type(result, int)


# Test complex return type
@tool
def complex_return_tool(data: list[int]) -> tuple[int, str, bool]:
    """Tool with complex return type."""
    return (len(data), "result", True)


def check_complex_return_type() -> None:
    """Verify complex return types are preserved."""
    result = complex_return_tool.run([1, 2, 3])
    assert_type(result, tuple[int, str, bool])


# Test no-argument tool
@tool
def no_arg_tool() -> str:
    """Tool with no arguments."""
    return "done"


def check_no_arg_tool_return() -> None:
    """Verify no-argument tools work correctly."""
    result = no_arg_tool.run()
    assert_type(result, str)


# Test that tool decorator returns MelleaTool instance
def check_tool_decorator_returns_melleatool() -> None:
    """Verify @tool returns a MelleaTool instance."""
    assert isinstance(simple_tool, MelleaTool)
    assert isinstance(named_tool, MelleaTool)
    assert isinstance(tool_with_defaults, MelleaTool)
    assert isinstance(complex_return_tool, MelleaTool)
    assert isinstance(no_arg_tool, MelleaTool)


# Test overload resolution for tool() function
def check_tool_overload_with_func() -> None:
    """Verify tool(func) overload returns MelleaTool."""

    def sample_func(x: int) -> str:
        return str(x)

    result = tool(sample_func)
    assert isinstance(result, MelleaTool)
    # Verify the return type is preserved
    output = result.run(42)
    assert_type(output, str)


def check_tool_overload_without_func() -> None:
    """Verify tool() overload returns decorator."""
    decorator = tool(name="custom")

    # decorator should be callable that takes a function and returns MelleaTool
    def sample_func(x: int) -> str:
        return str(x)

    result = decorator(sample_func)
    assert isinstance(result, MelleaTool)
    # Verify the return type is preserved
    output = result.run(42)
    assert_type(output, str)
