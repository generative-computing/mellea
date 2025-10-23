"""Tests for Python code verifiers - basic functionality and edge cases."""

import pytest

from mellea.stdlib.base import ModelOutputThunk, Context, ChatContext
from mellea.stdlib.reqlib.python import (
    HasPythonCodeListing,
    PythonCodeParses,
    PythonValidImports,
    PythonExecutesWithoutError,
    PythonHasFunctionDef,
    PythonHasClassDef,
    PythonMatchesExamples,
    extract_python_code,
)


def from_model(s: str) -> Context:
    """Helper to create a context with model output."""
    ctx = ChatContext()
    ctx = ctx.add(ModelOutputThunk(value=s, meta={"test": True}))
    return ctx


# region: Basic test contexts

VALID_PYTHON_MARKDOWN_CTX = from_model(
    """
Here's a simple Python function:

```python
def hello_world():
    print("Hello, world!")
    return 42
```

This function prints a greeting.
"""
)

VALID_PYTHON_GENERIC_BLOCK_CTX = from_model(
    """
```
def greet(name):
    return f"Hello, {name}!"
```
"""
)

VALID_PYTHON_PLAIN_CTX = from_model(
    """
def add(a, b):
    return a + b
"""
)

INVALID_SYNTAX_CTX = from_model(
    """
```python
def broken_function(
    print("missing closing paren")
```
"""
)

PYTHON_WITH_IMPORTS_CTX = from_model(
    """
```python
import os
import sys
from pathlib import Path

def get_home():
    return Path.home()
```
"""
)

PYTHON_WITH_INVALID_IMPORTS_CTX = from_model(
    """
```python
import nonexistent_package_xyz
import another_fake_module

def foo():
    pass
```
"""
)

PYTHON_EXECUTABLE_CTX = from_model(
    """
```python
def multiply(x, y):
    return x * y

if __name__ == "__main__":
    result = multiply(3, 4)
    print(f"Result: {result}")
```
"""
)

PYTHON_INFINITE_LOOP_CTX = from_model(
    """
```python
while True:
    pass
```
"""
)

PYTHON_WITH_CLASS_CTX = from_model(
    """
```python
class Calculator:
    def add(self, a, b):
        return a + b

    def subtract(self, a, b):
        return a - b
```
"""
)

NO_CODE_CTX = from_model(
    """
This is just a text response with no code at all.
"""
)

# endregion

# region: Edge case contexts

MULTIPLE_CODE_BLOCKS_CTX = from_model(
    """
Here's the correct version:
```python
def good():
    return "correct"
```

And here's a test for it:
```python
def test_good():
    assert good() == "correct"
```
"""
)

# Edge case: Bad code shown first, then good code
BAD_THEN_GOOD_CTX = from_model(
    """
Here's a wrong approach (don't use this):
```python
def add(a, b):
    return str(a) + str(b)  # Wrong! Concatenates
```

Here's the correct way:
```python
def add(a, b):
    return a + b
```
"""
)

# Edge case: Evolution (simple then improved)
EVOLUTION_CTX = from_model(
    """
Let's start simple:
```python
def add(a, b):
    return a + b
```

Now with type hints and docstring:
```python
def add(a: int, b: int) -> int:
    '''Add two numbers.'''
    return a + b
```
"""
)

MIXED_LANGUAGE_CTX = from_model(
    """
```javascript
function hello() {
    console.log("Hello");
}
```

```python
def greet():
    print("Hello")
```
"""
)

COMPLEX_CODE_CTX = from_model(
    '''
```python
"""Module docstring."""

def fibonacci(n: int) -> list[int]:
    """
    Calculate fibonacci sequence.

    Args:
        n: Number of terms

    Returns:
        List of fibonacci numbers
    """
    # Initialize first two terms
    if n <= 0:
        return []
    elif n == 1:
        return [0]

    fib = [0, 1]
    for i in range(2, n):
        fib.append(fib[i-1] + fib[i-2])
    return fib
```
'''
)

MULTILINE_STRING_CTX = from_model(
    '''
```python
def format_message(name, age):
    message = f"""
    Hello {name}!
    You are {age} years old.
    """
    return message.strip()
```
'''
)

NESTED_FUNCTIONS_CTX = from_model(
    """
```python
def outer(x):
    def inner(y):
        return x + y
    return inner

add_five = outer(5)
result = add_five(3)
```
"""
)

DECORATORS_CTX = from_model(
    """
```python
def my_decorator(func):
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper

@my_decorator
def greet(name):
    return f"Hello {name}"
```
"""
)

COMPREHENSIONS_CTX = from_model(
    """
```python
def process_data():
    squares = [x**2 for x in range(10)]
    evens = {x: x**2 for x in range(10) if x % 2 == 0}
    return squares, evens
```
"""
)

EXCEPTION_HANDLING_CTX = from_model(
    """
```python
def safe_divide(a, b):
    try:
        return a / b
    except ZeroDivisionError:
        return None
    except TypeError as e:
        raise ValueError(f"Invalid types: {e}")
    finally:
        print("Division attempted")
```
"""
)

CONTEXT_MANAGER_CTX = from_model(
    """
```python
class MyContext:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

def use_context():
    with MyContext() as ctx:
        pass
```
"""
)

ASYNC_CODE_CTX = from_model(
    """
```python
import asyncio

async def fetch_data():
    await asyncio.sleep(1)
    return "data"

async def main():
    result = await fetch_data()
    return result
```
"""
)

GENERATOR_CTX = from_model(
    """
```python
def count_up_to(n):
    i = 0
    while i < n:
        yield i
        i += 1
```
"""
)

LAMBDA_CTX = from_model(
    """
```python
add = lambda x, y: x + y
squares = list(map(lambda x: x**2, range(5)))
```
"""
)

ADVANCED_TYPE_HINTS_CTX = from_model(
    """
```python
from typing import Union, Optional, List, Dict

def process(data: Union[str, int], config: Optional[Dict[str, any]] = None) -> List[str]:
    return [str(data)]
```
"""
)

CODE_IN_STRING_CTX = from_model(
    """
```python
def example():
    code_snippet = "def bad( syntax error"
    return code_snippet
```
"""
)

EMPTY_FUNCTION_CTX = from_model(
    """
```python
def placeholder():
    pass
```
"""
)

ONLY_IMPORTS_CTX = from_model(
    """
```python
import os
import sys
from pathlib import Path

BASE_DIR = Path(__file__).parent
```
"""
)

COMPLEX_CLASS_CTX = from_model(
    """
```python
class Calculator:
    def __init__(self):
        self._value = 0

    @property
    def value(self):
        return self._value

    @staticmethod
    def add(a, b):
        return a + b

    @classmethod
    def create(cls):
        return cls()
```
"""
)

RUNTIME_ERROR_CTX = from_model(
    """
```python
def will_fail():
    undefined_variable = nonexistent_var + 1
    return undefined_variable

# Actually call it to trigger the error
result = will_fail()
```
"""
)

RELATIVE_IMPORT_CTX = from_model(
    """
```python
from . import sibling_module
from ..parent import something

def use_imports():
    pass
```
"""
)

# endregion

# region: Basic extraction tests


def test_extract_python_from_markdown():
    code = extract_python_code(VALID_PYTHON_MARKDOWN_CTX.last_output().value)
    assert code is not None
    assert "def hello_world():" in code


def test_extract_python_from_generic_block():
    code = extract_python_code(VALID_PYTHON_GENERIC_BLOCK_CTX.last_output().value)
    assert code is not None
    assert "def greet(name):" in code


def test_extract_python_from_plain_text():
    code = extract_python_code(VALID_PYTHON_PLAIN_CTX.last_output().value)
    assert code is not None
    assert "def add(a, b):" in code


def test_extract_python_no_code():
    code = extract_python_code(NO_CODE_CTX.last_output().value)
    assert code is None


def test_extract_multiple_code_blocks():
    """Should extract the main function, not the test."""
    code = extract_python_code(MULTIPLE_CODE_BLOCKS_CTX.last_output().value)
    assert code is not None
    assert "def good" in code
    assert "test_good" not in code  # Should not extract the test


def test_extract_bad_then_good():
    """Should extract the GOOD code even though bad code comes first."""
    code = extract_python_code(BAD_THEN_GOOD_CTX.last_output().value)
    assert code is not None
    assert "return a + b" in code
    assert "str(a)" not in code  # Should avoid the bad version


def test_extract_evolution():
    """Should extract the more complete version when code evolves."""
    code = extract_python_code(EVOLUTION_CTX.last_output().value)
    assert code is not None
    # Should prefer the longer, more complete version with type hints
    assert "int" in code or "Add two numbers" in code  # Either is fine


def test_extract_from_mixed_languages():
    """Should extract Python code, not JavaScript."""
    code = extract_python_code(MIXED_LANGUAGE_CTX.last_output().value)
    assert code is not None
    assert "def greet" in code
    assert "function hello" not in code


# endregion

# region: HasPythonCodeListing tests


def test_has_python_code_listing_valid():
    requirement = HasPythonCodeListing()
    result = requirement.validation_fn(VALID_PYTHON_MARKDOWN_CTX)
    assert result.as_bool() is True
    assert "def hello_world():" in result.reason


def test_has_python_code_listing_invalid():
    requirement = HasPythonCodeListing()
    result = requirement.validation_fn(NO_CODE_CTX)
    assert result.as_bool() is False


# endregion

# region: PythonCodeParses tests


def test_python_code_parses_valid():
    requirement = PythonCodeParses()
    result = requirement.validation_fn(VALID_PYTHON_MARKDOWN_CTX)
    assert result.as_bool() is True


def test_python_code_parses_invalid():
    requirement = PythonCodeParses()
    result = requirement.validation_fn(INVALID_SYNTAX_CTX)
    assert result.as_bool() is False
    assert "Syntax error" in result.reason or "Parse error" in result.reason


def test_python_code_parses_no_code():
    requirement = PythonCodeParses()
    result = requirement.validation_fn(NO_CODE_CTX)
    assert result.as_bool() is False


def test_complex_code_with_docstrings():
    req = PythonCodeParses()
    result = req.validation_fn(COMPLEX_CODE_CTX)
    assert result.as_bool() is True


def test_multiline_strings():
    req = PythonCodeParses()
    result = req.validation_fn(MULTILINE_STRING_CTX)
    assert result.as_bool() is True


def test_nested_functions():
    req = PythonCodeParses()
    result = req.validation_fn(NESTED_FUNCTIONS_CTX)
    assert result.as_bool() is True


def test_decorators():
    req = PythonCodeParses()
    result = req.validation_fn(DECORATORS_CTX)
    assert result.as_bool() is True


def test_comprehensions():
    req = PythonCodeParses()
    result = req.validation_fn(COMPREHENSIONS_CTX)
    assert result.as_bool() is True


def test_exception_handling():
    req = PythonCodeParses()
    result = req.validation_fn(EXCEPTION_HANDLING_CTX)
    assert result.as_bool() is True


def test_context_managers():
    req = PythonCodeParses()
    result = req.validation_fn(CONTEXT_MANAGER_CTX)
    assert result.as_bool() is True


def test_async_code():
    req = PythonCodeParses()
    result = req.validation_fn(ASYNC_CODE_CTX)
    assert result.as_bool() is True


def test_generators():
    req = PythonCodeParses()
    result = req.validation_fn(GENERATOR_CTX)
    assert result.as_bool() is True


def test_lambda_functions():
    req = PythonCodeParses()
    result = req.validation_fn(LAMBDA_CTX)
    assert result.as_bool() is True


def test_advanced_type_hints():
    req = PythonCodeParses()
    result = req.validation_fn(ADVANCED_TYPE_HINTS_CTX)
    assert result.as_bool() is True


def test_code_in_string():
    """Code with syntax errors in strings should still parse."""
    req = PythonCodeParses()
    result = req.validation_fn(CODE_IN_STRING_CTX)
    assert result.as_bool() is True


def test_relative_imports_parse():
    """Relative imports should parse fine."""
    req = PythonCodeParses()
    result = req.validation_fn(RELATIVE_IMPORT_CTX)
    assert result.as_bool() is True


def test_runtime_error_code_parses():
    """Code with runtime errors should still parse successfully."""
    req = PythonCodeParses()
    result = req.validation_fn(RUNTIME_ERROR_CTX)
    assert result.as_bool() is True


# endregion

# region: PythonValidImports tests


def test_python_valid_imports_stdlib():
    requirement = PythonValidImports()
    result = requirement.validation_fn(PYTHON_WITH_IMPORTS_CTX)
    assert result.as_bool() is True


def test_python_valid_imports_invalid():
    requirement = PythonValidImports()
    result = requirement.validation_fn(PYTHON_WITH_INVALID_IMPORTS_CTX)
    assert result.as_bool() is False
    assert "nonexistent_package_xyz" in result.reason


# endregion

# region: PythonExecutesWithoutError tests


def test_python_executes_without_error_valid():
    requirement = PythonExecutesWithoutError(timeout=2, allow_unsafe_execution=True)
    result = requirement.validation_fn(PYTHON_EXECUTABLE_CTX)
    assert result.as_bool() is True


def test_python_executes_without_error_timeout():
    requirement = PythonExecutesWithoutError(timeout=1, allow_unsafe_execution=True)
    result = requirement.validation_fn(PYTHON_INFINITE_LOOP_CTX)
    assert result.as_bool() is False
    assert "timed out" in result.reason.lower()


def test_python_executes_without_error_syntax():
    requirement = PythonExecutesWithoutError()
    result = requirement.validation_fn(INVALID_SYNTAX_CTX)
    assert result.as_bool() is False


def test_runtime_error_code_fails_execution():
    """Code with runtime errors should fail execution test."""
    req = PythonExecutesWithoutError(timeout=2, allow_unsafe_execution=True)
    result = req.validation_fn(RUNTIME_ERROR_CTX)
    assert result.as_bool() is False


def test_safe_mode_default():
    """Safe mode should be default and not execute code."""
    req = PythonExecutesWithoutError()
    result = req.validation_fn(PYTHON_EXECUTABLE_CTX)
    assert result.as_bool() is True
    assert "safe mode" in result.reason


def test_safe_mode_syntax_error():
    """Safe mode should catch syntax errors."""
    req = PythonExecutesWithoutError()
    result = req.validation_fn(INVALID_SYNTAX_CTX)
    assert result.as_bool() is False


def test_unsafe_execution_with_flag():
    """Unsafe execution should work when explicitly enabled."""
    req = PythonExecutesWithoutError(allow_unsafe_execution=True, timeout=2)
    result = req.validation_fn(PYTHON_EXECUTABLE_CTX)
    assert result.as_bool() is True


def test_unsafe_execution_with_import_restrictions():
    """Import restrictions should block unauthorized imports."""
    req = PythonExecutesWithoutError(
        allow_unsafe_execution=True,
        allowed_imports=["math", "json"]
    )
    result = req.validation_fn(PYTHON_WITH_INVALID_IMPORTS_CTX)
    assert result.as_bool() is False
    assert "Unauthorized imports" in result.reason


def test_unsafe_execution_with_allowed_imports():
    """Allowed imports should pass validation."""
    req = PythonExecutesWithoutError(
        allow_unsafe_execution=True,
        allowed_imports=["os", "sys", "pathlib"]
    )
    result = req.validation_fn(PYTHON_WITH_IMPORTS_CTX)
    assert result.as_bool() is True


# endregion

# region: PythonHasFunctionDef tests


def test_python_has_function_def_valid():
    requirement = PythonHasFunctionDef()
    result = requirement.validation_fn(VALID_PYTHON_MARKDOWN_CTX)
    assert result.as_bool() is True
    assert "hello_world" in result.reason


def test_python_has_function_def_invalid():
    requirement = PythonHasFunctionDef()
    result = requirement.validation_fn(from_model("```python\nx = 5\n```"))
    assert result.as_bool() is False


def test_decorators_have_functions():
    has_func = PythonHasFunctionDef()
    assert has_func.validation_fn(DECORATORS_CTX).as_bool() is True


def test_empty_function():
    req = PythonCodeParses()
    result = req.validation_fn(EMPTY_FUNCTION_CTX)
    assert result.as_bool() is True

    has_func = PythonHasFunctionDef()
    assert has_func.validation_fn(EMPTY_FUNCTION_CTX).as_bool() is True


def test_only_imports_no_functions():
    req = PythonCodeParses()
    result = req.validation_fn(ONLY_IMPORTS_CTX)
    assert result.as_bool() is True

    has_func = PythonHasFunctionDef()
    assert has_func.validation_fn(ONLY_IMPORTS_CTX).as_bool() is False


# endregion

# region: PythonHasClassDef tests


def test_python_has_class_def_valid():
    requirement = PythonHasClassDef()
    result = requirement.validation_fn(PYTHON_WITH_CLASS_CTX)
    assert result.as_bool() is True
    assert "Calculator" in result.reason


def test_python_has_class_def_invalid():
    requirement = PythonHasClassDef()
    result = requirement.validation_fn(VALID_PYTHON_MARKDOWN_CTX)
    assert result.as_bool() is False


def test_context_managers_have_classes():
    has_class = PythonHasClassDef()
    assert has_class.validation_fn(CONTEXT_MANAGER_CTX).as_bool() is True


def test_complex_class():
    req = PythonCodeParses()
    result = req.validation_fn(COMPLEX_CLASS_CTX)
    assert result.as_bool() is True

    has_class = PythonHasClassDef()
    assert has_class.validation_fn(COMPLEX_CLASS_CTX).as_bool() is True


# endregion

# region: Integration tests


def test_full_validation_pipeline():
    """Test chaining multiple validators."""
    ctx = PYTHON_WITH_CLASS_CTX

    # Should have code
    has_code = HasPythonCodeListing()
    assert has_code.validation_fn(ctx).as_bool() is True

    # Should parse
    parses = PythonCodeParses()
    assert parses.validation_fn(ctx).as_bool() is True

    # Should have class
    has_class = PythonHasClassDef()
    assert has_class.validation_fn(ctx).as_bool() is True


def test_chained_validation_complex_code():
    """Test all validators on complex real-world code."""
    ctx = COMPLEX_CODE_CTX

    has_code = HasPythonCodeListing()
    assert has_code.validation_fn(ctx).as_bool() is True

    parses = PythonCodeParses()
    assert parses.validation_fn(ctx).as_bool() is True

    has_func = PythonHasFunctionDef()
    assert has_func.validation_fn(ctx).as_bool() is True


# endregion

# region: PythonMatchesExamples tests

FACTORIAL_CODE_CTX = from_model(
    """
```python
def factorial(n):
    if n == 0:
        return 1
    return n * factorial(n - 1)
```
"""
)

INCORRECT_FACTORIAL_CODE_CTX = from_model(
    """
```python
def factorial(n):
    # Incorrect implementation - off by one
    if n == 0:
        return 0
    return n * factorial(n - 1)
```
"""
)

FIBONACCI_CODE_CTX = from_model(
    """
```python
def fibonacci(n):
    if n <= 0:
        return []
    elif n == 1:
        return [0]

    fib = [0, 1]
    for i in range(2, n):
        fib.append(fib[i-1] + fib[i-2])
    return fib
```
"""
)

ADD_FUNCTION_CTX = from_model(
    """
```python
def add(a, b):
    return a + b
```
"""
)


def test_matches_examples_correct():
    """Test that correct function passes all examples."""
    req = PythonMatchesExamples(
        function_name="factorial",
        examples=[
            ({"n": 0}, 1),
            ({"n": 1}, 1),
            ({"n": 5}, 120),
        ]
    )
    result = req.validation_fn(FACTORIAL_CODE_CTX)
    assert result.as_bool() is True
    assert "All 3 examples passed" in result.reason


def test_matches_examples_incorrect():
    """Test that incorrect function fails examples."""
    req = PythonMatchesExamples(
        function_name="factorial",
        examples=[
            ({"n": 0}, 1),  # Will fail - function returns 0
            ({"n": 5}, 120),
        ]
    )
    result = req.validation_fn(INCORRECT_FACTORIAL_CODE_CTX)
    assert result.as_bool() is False
    assert "Failed" in result.reason


def test_matches_examples_function_not_found():
    """Test behavior when function doesn't exist."""
    req = PythonMatchesExamples(
        function_name="nonexistent",
        examples=[({"n": 5}, 120)]
    )
    result = req.validation_fn(FACTORIAL_CODE_CTX)
    assert result.as_bool() is False
    assert "not found" in result.reason


def test_matches_examples_list_output():
    """Test with functions that return lists."""
    req = PythonMatchesExamples(
        function_name="fibonacci",
        examples=[
            ({"n": 0}, []),
            ({"n": 1}, [0]),
            ({"n": 5}, [0, 1, 1, 2, 3]),
        ]
    )
    result = req.validation_fn(FIBONACCI_CODE_CTX)
    assert result.as_bool() is True


def test_matches_examples_multiple_args():
    """Test with functions that take multiple arguments."""
    req = PythonMatchesExamples(
        function_name="add",
        examples=[
            ({"a": 2, "b": 3}, 5),
            ({"a": 0, "b": 0}, 0),
            ({"a": -1, "b": 1}, 0),
        ]
    )
    result = req.validation_fn(ADD_FUNCTION_CTX)
    assert result.as_bool() is True


def test_matches_examples_runtime_error():
    """Test behavior when function raises exception."""
    req = PythonMatchesExamples(
        function_name="factorial",
        examples=[
            ({"n": -1}, 1),  # Will cause infinite recursion
        ]
    )
    # This should catch the exception and report failure
    result = req.validation_fn(FACTORIAL_CODE_CTX)
    assert result.as_bool() is False


def test_matches_examples_no_code():
    """Test behavior with no code."""
    req = PythonMatchesExamples(
        function_name="foo",
        examples=[({"x": 1}, 1)]
    )
    result = req.validation_fn(NO_CODE_CTX)
    assert result.as_bool() is False


# endregion


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
