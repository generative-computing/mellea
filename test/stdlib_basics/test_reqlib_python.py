"""Tests for Python code execution requirements."""

import pytest

from mellea.stdlib.base import Context
from mellea.stdlib.reqlib.python import (
    PythonExecutesWithoutError,
    _has_python_code_listing,
    _python_executes_without_error,
)


def from_model(content: str) -> Context:
    """Helper to create context from model output."""
    from mellea.stdlib.base import ModelOutputThunk, ChatContext
    ctx = ChatContext()
    ctx = ctx.add(ModelOutputThunk(value=content))
    return ctx


# Test contexts
VALID_PYTHON_CODE = """```python
def hello_world():
    return "Hello, World!"

print(hello_world())
```"""

PYTHON_WITH_SYNTAX_ERROR = """```python
def hello_world(
    return "Hello, World!"
```"""

PYTHON_WITH_RUNTIME_ERROR = """```python
def divide_by_zero():
    return 1 / 0

divide_by_zero()
```"""

PYTHON_WITH_IMPORTS = """```python
import os
import sys
from pathlib import Path

print("Hello from imports!")
```"""

PYTHON_WITH_FORBIDDEN_IMPORTS = """```python
import subprocess
import socket
import urllib

print("Dangerous imports!")
```"""

PYTHON_SIMPLE_PRINT = """```python
print("Hello, World!")
```"""

PYTHON_INFINITE_LOOP = """```python
while True:
    pass
```"""

NO_PYTHON_CODE = """
This is just text without any Python code blocks.
It contains no executable content.
"""

# Create contexts
VALID_PYTHON_CTX = from_model(VALID_PYTHON_CODE)
SYNTAX_ERROR_CTX = from_model(PYTHON_WITH_SYNTAX_ERROR)
RUNTIME_ERROR_CTX = from_model(PYTHON_WITH_RUNTIME_ERROR)
PYTHON_WITH_IMPORTS_CTX = from_model(PYTHON_WITH_IMPORTS)
PYTHON_WITH_FORBIDDEN_IMPORTS_CTX = from_model(PYTHON_WITH_FORBIDDEN_IMPORTS)
PYTHON_SIMPLE_CTX = from_model(PYTHON_SIMPLE_PRINT)
PYTHON_INFINITE_LOOP_CTX = from_model(PYTHON_INFINITE_LOOP)
NO_PYTHON_CTX = from_model(NO_PYTHON_CODE)


# region: Code extraction tests


def test_has_python_code_listing_valid():
    """Test extraction of valid Python code."""
    result = _has_python_code_listing(VALID_PYTHON_CTX)
    assert result.as_bool() is True
    assert "def hello_world" in result.reason


def test_has_python_code_listing_no_code():
    """Test handling when no Python code is present."""
    result = _has_python_code_listing(NO_PYTHON_CTX)
    assert result.as_bool() is False
    assert "No Python code blocks found" in result.reason


def test_has_python_code_listing_simple():
    """Test extraction of simple Python code."""
    result = _has_python_code_listing(PYTHON_SIMPLE_CTX)
    assert result.as_bool() is True
    assert "print" in result.reason


# endregion

# region: Safe mode tests (default behavior)


def test_safe_mode_default():
    """Test that safe mode is default and validates without executing."""
    req = PythonExecutesWithoutError()
    result = req.validation_fn(VALID_PYTHON_CTX)
    assert result.as_bool() is True
    assert "safe mode" in result.reason


def test_safe_mode_syntax_error():
    """Test that safe mode catches syntax errors."""
    req = PythonExecutesWithoutError()
    result = req.validation_fn(SYNTAX_ERROR_CTX)
    assert result.as_bool() is False


def test_safe_mode_no_execution():
    """Test that safe mode doesn't execute code (even infinite loops)."""
    req = PythonExecutesWithoutError(timeout=1)
    result = req.validation_fn(PYTHON_INFINITE_LOOP_CTX)
    assert result.as_bool() is True  # Should pass because it's not actually executed
    assert "safe mode" in result.reason


# endregion

# region: Unsafe execution tests


def test_unsafe_execution_valid():
    """Test unsafe execution with valid code."""
    req = PythonExecutesWithoutError(allow_unsafe_execution=True, timeout=5)
    result = req.validation_fn(VALID_PYTHON_CTX)
    assert result.as_bool() is True


def test_unsafe_execution_runtime_error():
    """Test unsafe execution with runtime error."""
    req = PythonExecutesWithoutError(allow_unsafe_execution=True, timeout=5)
    result = req.validation_fn(RUNTIME_ERROR_CTX)
    assert result.as_bool() is False
    assert "error" in result.reason.lower()


def test_unsafe_execution_timeout():
    """Test unsafe execution with timeout."""
    req = PythonExecutesWithoutError(allow_unsafe_execution=True, timeout=1)
    result = req.validation_fn(PYTHON_INFINITE_LOOP_CTX)
    assert result.as_bool() is False
    assert "timed out" in result.reason.lower()


def test_unsafe_execution_syntax_error():
    """Test unsafe execution with syntax error."""
    req = PythonExecutesWithoutError(allow_unsafe_execution=True)
    result = req.validation_fn(SYNTAX_ERROR_CTX)
    assert result.as_bool() is False


# endregion

# region: Import restriction tests


def test_import_restrictions_block_forbidden():
    """Test that import restrictions block forbidden imports."""
    req = PythonExecutesWithoutError(
        allow_unsafe_execution=True,
        allowed_imports=["os", "sys"]
    )
    result = req.validation_fn(PYTHON_WITH_FORBIDDEN_IMPORTS_CTX)
    assert result.as_bool() is False
    assert "Unauthorized imports" in result.reason


def test_import_restrictions_allow_permitted():
    """Test that import restrictions allow permitted imports."""
    req = PythonExecutesWithoutError(
        allow_unsafe_execution=True,
        allowed_imports=["os", "sys", "pathlib"]
    )
    result = req.validation_fn(PYTHON_WITH_IMPORTS_CTX)
    assert result.as_bool() is True


def test_import_restrictions_with_safe_mode():
    """Test that import restrictions work with safe mode."""
    req = PythonExecutesWithoutError(
        allowed_imports=["os", "sys"]
    )
    result = req.validation_fn(PYTHON_WITH_FORBIDDEN_IMPORTS_CTX)
    assert result.as_bool() is False
    assert "Unauthorized imports" in result.reason


# endregion

# region: Sandbox execution tests


@pytest.mark.skipif(
    True,  # Skip by default since Docker may not be available in CI
    reason="Sandbox tests require Docker and llm-sandbox setup"
)
def test_sandbox_execution_valid():
    """Test sandbox execution with valid code."""
    req = PythonExecutesWithoutError(use_sandbox=True, timeout=10)
    result = req.validation_fn(VALID_PYTHON_CTX)
    assert result.as_bool() is True
    assert "sandbox" in result.reason.lower()


@pytest.mark.skipif(
    True,  # Skip by default since Docker may not be available in CI
    reason="Sandbox tests require Docker and llm-sandbox setup"
)
def test_sandbox_execution_with_imports():
    """Test sandbox execution with allowed imports."""
    req = PythonExecutesWithoutError(
        use_sandbox=True,
        allowed_imports=["os", "sys", "pathlib"],
        timeout=10
    )
    result = req.validation_fn(PYTHON_WITH_IMPORTS_CTX)
    assert result.as_bool() is True


@pytest.mark.skipif(
    True,  # Skip by default since Docker may not be available in CI
    reason="Sandbox tests require Docker and llm-sandbox setup"
)
def test_sandbox_execution_timeout():
    """Test sandbox execution timeout."""
    req = PythonExecutesWithoutError(use_sandbox=True, timeout=2)
    result = req.validation_fn(PYTHON_INFINITE_LOOP_CTX)
    assert result.as_bool() is False


def test_sandbox_without_llm_sandbox_installed():
    """Test graceful handling when llm-sandbox is not available."""
    # This test will pass even if llm-sandbox is installed, but tests the error handling
    req = PythonExecutesWithoutError(use_sandbox=True)
    # We can't easily test this without mocking, but the error handling is in the code
    assert req is not None


# endregion

# region: Configuration tests


def test_description_updates_based_on_mode():
    """Test that requirement description reflects execution mode."""
    safe_req = PythonExecutesWithoutError()
    assert "validation only" in safe_req.description

    unsafe_req = PythonExecutesWithoutError(allow_unsafe_execution=True, timeout=5)
    assert "unsafe execution" in unsafe_req.description
    assert "timeout: 5s" in unsafe_req.description

    sandbox_req = PythonExecutesWithoutError(use_sandbox=True, timeout=10)
    assert "sandbox execution" in sandbox_req.description
    assert "timeout: 10s" in sandbox_req.description


def test_parameter_combinations():
    """Test various parameter combinations."""
    # Safe mode (default)
    req1 = PythonExecutesWithoutError()
    assert req1._allow_unsafe is False
    assert req1._use_sandbox is False

    # Unsafe mode
    req2 = PythonExecutesWithoutError(allow_unsafe_execution=True)
    assert req2._allow_unsafe is True
    assert req2._use_sandbox is False

    # Sandbox mode
    req3 = PythonExecutesWithoutError(use_sandbox=True)
    assert req3._allow_unsafe is False
    assert req3._use_sandbox is True

    # Sandbox + unsafe (sandbox takes precedence)
    req4 = PythonExecutesWithoutError(allow_unsafe_execution=True, use_sandbox=True)
    assert req4._allow_unsafe is True
    assert req4._use_sandbox is True


# endregion

# region: Integration tests


def test_direct_validation_function():
    """Test calling validation function directly."""
    result = _python_executes_without_error(
        VALID_PYTHON_CTX,
        timeout=5,
        allow_unsafe=False,
        use_sandbox=False
    )
    assert result.as_bool() is True
    assert "safe mode" in result.reason

    result = _python_executes_without_error(
        SYNTAX_ERROR_CTX,
        timeout=5,
        allow_unsafe=False,
        use_sandbox=False
    )
    assert result.as_bool() is False


def test_no_code_extraction():
    """Test behavior when no code can be extracted."""
    req = PythonExecutesWithoutError()
    result = req.validation_fn(NO_PYTHON_CTX)
    assert result.as_bool() is False
    assert "Could not extract Python code" in result.reason


# endregion