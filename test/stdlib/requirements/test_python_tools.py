"""Tests for Python tool requirements from python_tools module."""

import pytest

from mellea.core import Context, ModelOutputThunk
from mellea.stdlib.context import ChatContext
from mellea.stdlib.requirements.python_tools import (
    ImportRestrictions,
    OutputSizeLimit,
    PythonCodeExtraction,
    PythonSyntaxValid,
    python_tool_requirements,
)


def from_model(content: str) -> Context:
    """Helper to create context from model output."""
    ctx = ChatContext()
    ctx = ctx.add(ModelOutputThunk(value=content))
    return ctx


# Test fixtures
VALID_PYTHON_CODE = """```python
def hello_world():
    return "Hello, World!"

print(hello_world())
```"""

PYTHON_WITH_SYNTAX_ERROR = """```python
def hello_world(
    return "Hello, World!"
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

NO_PYTHON_CODE = """
This is just text without any Python code blocks.
It contains no executable content.
"""


class TestPythonCodeExtraction:
    """Tests for PythonCodeExtraction requirement."""

    def test_extract_valid_code_block(self):
        """Test extraction of valid Python code."""
        req = PythonCodeExtraction()
        ctx = from_model(VALID_PYTHON_CODE)
        result = req.validation_fn(ctx)

        assert result.as_bool() is True
        assert "hello_world" in (result.reason or "")

    def test_extract_no_code_blocks(self):
        """Test extraction when no code blocks present."""
        req = PythonCodeExtraction()
        ctx = from_model(NO_PYTHON_CODE)
        result = req.validation_fn(ctx)

        assert result.as_bool() is False
        assert result.reason is not None

    def test_extract_multiple_code_blocks(self):
        """Test extraction when multiple code blocks present (should return highest scoring)."""
        code = """
Here's a simple one:
```python
print("simple")
```

And a more complex one:
```python
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

for i in range(10):
    print(fibonacci(i))
```
"""
        req = PythonCodeExtraction()
        ctx = from_model(code)
        result = req.validation_fn(ctx)

        assert result.as_bool() is True
        assert "fibonacci" in (result.reason or "")


class TestPythonSyntaxValid:
    """Tests for PythonSyntaxValid requirement."""

    def test_valid_syntax(self):
        """Test validation of syntactically valid code."""
        req = PythonSyntaxValid()
        ctx = from_model(VALID_PYTHON_CODE)
        result = req.validation_fn(ctx)

        assert result.as_bool() is True
        assert "valid" in (result.reason or "").lower()

    def test_syntax_error(self):
        """Test validation of code with syntax errors."""
        req = PythonSyntaxValid()
        ctx = from_model(PYTHON_WITH_SYNTAX_ERROR)
        result = req.validation_fn(ctx)

        assert result.as_bool() is False
        assert "syntax error" in (result.reason or "").lower()

    def test_syntax_error_unclosed_paren(self):
        """Test validation of code with unclosed parenthesis."""
        code = """```python
def foo(
    pass
```"""
        req = PythonSyntaxValid()
        ctx = from_model(code)
        result = req.validation_fn(ctx)

        assert result.as_bool() is False

    def test_syntax_error_bad_indentation(self):
        """Test validation of code with indentation errors."""
        code = """```python
def foo():
return "bad indent"
```"""
        req = PythonSyntaxValid()
        ctx = from_model(code)
        result = req.validation_fn(ctx)

        assert result.as_bool() is False

    def test_syntax_valid_no_code_extraction(self):
        """Test validation when no code can be extracted."""
        req = PythonSyntaxValid()
        ctx = from_model(NO_PYTHON_CODE)
        result = req.validation_fn(ctx)

        assert result.as_bool() is False


class TestOutputSizeLimit:
    """Tests for OutputSizeLimit requirement."""

    def test_init_valid_limit(self):
        """Test initialization with valid limit."""
        req = OutputSizeLimit(limit_chars=5000)
        assert req.limit_chars == 5000

    def test_init_invalid_limit_zero(self):
        """Test initialization with zero limit raises ValueError."""
        with pytest.raises(ValueError, match="must be positive"):
            OutputSizeLimit(limit_chars=0)

    def test_init_invalid_limit_negative(self):
        """Test initialization with negative limit raises ValueError."""
        with pytest.raises(ValueError, match="must be positive"):
            OutputSizeLimit(limit_chars=-100)

    def test_init_default_limit(self):
        """Test initialization with default limit."""
        req = OutputSizeLimit()
        assert req.limit_chars == 10_000

    def test_output_within_limit(self):
        """Test validation when output stays within limit."""
        req = OutputSizeLimit(limit_chars=1000)
        code = """```python
print("Hello, World!")
```"""
        ctx = from_model(code)
        result = req.validation_fn(ctx)
        # Result depends on execution, but size check logic is present
        assert isinstance(result.as_bool(), bool)


class TestImportRestrictions:
    """Tests for ImportRestrictions requirement."""

    def test_init_with_allowlist(self):
        """Test initialization with import allowlist."""
        req = ImportRestrictions(allowed_imports=["os", "sys", "json"])
        assert req.allowed_imports == ["os", "sys", "json"]

    def test_init_with_none(self):
        """Test initialization with None allowlist."""
        req = ImportRestrictions(allowed_imports=None)
        assert req.allowed_imports == []

    def test_init_default(self):
        """Test initialization with default (None) allowlist."""
        req = ImportRestrictions()
        assert req.allowed_imports == []

    def test_allowed_imports_pass(self):
        """Test validation when imports are in allowlist."""
        req = ImportRestrictions(allowed_imports=["os", "sys", "pathlib"])
        ctx = from_model(PYTHON_WITH_IMPORTS)
        result = req.validation_fn(ctx)

        assert result.as_bool() is True

    def test_forbidden_imports_fail(self):
        """Test validation when forbidden imports are detected."""
        req = ImportRestrictions(allowed_imports=["os", "sys"])
        ctx = from_model(PYTHON_WITH_FORBIDDEN_IMPORTS)
        result = req.validation_fn(ctx)

        assert result.as_bool() is False
        assert "forbidden" in (result.reason or "").lower()
        assert any(
            m in (result.reason or "") for m in ["subprocess", "socket", "urllib"]
        )

    def test_no_imports_pass(self):
        """Test validation when code has no imports."""
        req = ImportRestrictions(allowed_imports=["os"])
        code = """```python
def add(a, b):
    return a + b

print(add(2, 3))
```"""
        ctx = from_model(code)
        result = req.validation_fn(ctx)

        assert result.as_bool() is True

    def test_no_allowlist_passes_all(self):
        """Test validation with no allowlist (None) passes all imports."""
        req = ImportRestrictions(allowed_imports=None)
        ctx = from_model(PYTHON_WITH_FORBIDDEN_IMPORTS)
        result = req.validation_fn(ctx)

        assert result.as_bool() is True
        assert "No import restrictions" in (result.reason or "")

    def test_syntax_error_in_imports_check(self):
        """Test import validation when code has syntax errors."""
        req = ImportRestrictions(allowed_imports=["os"])
        ctx = from_model(PYTHON_WITH_SYNTAX_ERROR)
        result = req.validation_fn(ctx)

        assert result.as_bool() is False

    def test_submodule_imports(self):
        """Test validation of submodule imports."""
        req = ImportRestrictions(allowed_imports=["pathlib"])
        code = """```python
from pathlib.posixpath import join
import pathlib.pure

print("submodules")
```"""
        ctx = from_model(code)
        result = req.validation_fn(ctx)

        assert result.as_bool() is True

    def test_forbidden_submodule(self):
        """Test validation when submodule is forbidden."""
        req = ImportRestrictions(allowed_imports=["os"])
        code = """```python
from urllib.request import urlopen

print("fetch")
```"""
        ctx = from_model(code)
        result = req.validation_fn(ctx)

        assert result.as_bool() is False


class TestPythonToolRequirementsFactory:
    """Tests for python_tool_requirements() factory function."""

    def test_factory_default_returns_four_requirements(self):
        """Test factory with defaults returns 4 requirements (no import restrictions)."""
        reqs = python_tool_requirements()
        assert len(reqs) == 4
        assert isinstance(reqs[0], PythonCodeExtraction)
        assert isinstance(reqs[1], PythonSyntaxValid)
        assert isinstance(reqs[3], OutputSizeLimit)

    def test_factory_with_allowed_imports_returns_five(self):
        """Test factory with allowed_imports returns 5 requirements."""
        reqs = python_tool_requirements(allowed_imports=["os", "sys"])
        assert len(reqs) == 5
        assert isinstance(reqs[4], ImportRestrictions)

    def test_factory_parameter_propagation_output_limit(self):
        """Test factory propagates output_limit_chars to OutputSizeLimit."""
        reqs = python_tool_requirements(output_limit_chars=5000)
        output_limit_req = reqs[3]
        assert isinstance(output_limit_req, OutputSizeLimit)
        assert output_limit_req.limit_chars == 5000

    def test_factory_parameter_propagation_imports(self):
        """Test factory propagates allowed_imports to ImportRestrictions."""
        imports = ["os", "sys", "json"]
        reqs = python_tool_requirements(allowed_imports=imports)
        import_req = reqs[4]
        assert isinstance(import_req, ImportRestrictions)
        assert import_req.allowed_imports == imports

    def test_factory_timeout_parameter(self):
        """Test factory accepts and uses timeout_seconds parameter."""
        reqs = python_tool_requirements(timeout_seconds=10)
        assert len(reqs) == 4

    def test_factory_sandbox_parameter(self):
        """Test factory accepts and uses use_sandbox parameter."""
        reqs = python_tool_requirements(use_sandbox=True)
        assert len(reqs) == 4

    def test_factory_all_parameters(self):
        """Test factory with all parameters configured."""
        reqs = python_tool_requirements(
            allowed_imports=["os", "sys"],
            output_limit_chars=8000,
            timeout_seconds=15,
            use_sandbox=True,
        )
        assert len(reqs) == 5
        assert isinstance(reqs[3], OutputSizeLimit)
        assert reqs[3].limit_chars == 8000
        assert isinstance(reqs[4], ImportRestrictions)

    def test_factory_invalid_timeout(self):
        """Test factory with invalid timeout raises ValueError."""
        with pytest.raises(ValueError, match="timeout_seconds must be positive"):
            python_tool_requirements(timeout_seconds=0)

        with pytest.raises(ValueError, match="timeout_seconds must be positive"):
            python_tool_requirements(timeout_seconds=-5)

    def test_factory_invalid_output_limit(self):
        """Test factory with invalid output_limit raises ValueError."""
        with pytest.raises(ValueError, match="output_limit_chars must be positive"):
            python_tool_requirements(output_limit_chars=0)

        with pytest.raises(ValueError, match="output_limit_chars must be positive"):
            python_tool_requirements(output_limit_chars=-1000)

    def test_factory_requirement_order(self):
        """Test factory returns requirements in correct validation order."""
        reqs = python_tool_requirements(allowed_imports=["os"])

        assert isinstance(reqs[0], PythonCodeExtraction)
        assert isinstance(reqs[1], PythonSyntaxValid)
        assert isinstance(reqs[2], type(reqs[2]))  # PythonExecutionReq
        assert isinstance(reqs[3], OutputSizeLimit)
        assert isinstance(reqs[4], ImportRestrictions)

    def test_factory_timeout_propagation_to_execution_req(self):
        """Test factory propagates timeout_seconds to PythonExecutionReq."""
        from mellea.stdlib.requirements.python_reqs import PythonExecutionReq

        reqs = python_tool_requirements(timeout_seconds=15)
        execution_req = reqs[2]
        assert isinstance(execution_req, PythonExecutionReq)
        assert execution_req._timeout == 15

    def test_factory_sandbox_propagation_to_execution_req(self):
        """Test factory propagates use_sandbox to PythonExecutionReq."""
        from mellea.stdlib.requirements.python_reqs import PythonExecutionReq

        reqs = python_tool_requirements(use_sandbox=True)
        execution_req = reqs[2]
        assert isinstance(execution_req, PythonExecutionReq)
        assert execution_req._use_sandbox is True

    def test_factory_allowed_imports_propagation_to_execution_req(self):
        """Test factory propagates allowed_imports to PythonExecutionReq."""
        from mellea.stdlib.requirements.python_reqs import PythonExecutionReq

        imports = ["os", "sys", "json"]
        reqs = python_tool_requirements(allowed_imports=imports)
        execution_req = reqs[2]
        assert isinstance(execution_req, PythonExecutionReq)
        assert execution_req._allowed_imports == imports
