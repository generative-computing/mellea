"""Tests for Python tool requirements from python_tools module."""

import pytest

from mellea.core import Context, ModelOutputThunk, ModelToolCall
from mellea.core.base import AbstractMelleaTool
from mellea.stdlib.context import ChatContext
from mellea.stdlib.requirements.python_reqs import _extract_code_from_tool_call
from mellea.stdlib.requirements.python_tools import (
    ImportRestrictions,
    OutputSizeLimit,
    PythonCodeExtraction,
    PythonSyntaxValid,
    python_code_generation_requirements,
)


def from_model(content: str) -> Context:
    """Helper to create context from model output."""
    ctx = ChatContext()
    ctx = ctx.add(ModelOutputThunk(value=content))
    return ctx


class MockPythonTool(AbstractMelleaTool):
    """Mock Python tool for testing."""

    name: str = "python_executor"

    def run(self, code: str) -> str:
        """Mock execution."""
        return f"executed: {code}"

    @property
    def as_json_tool(self) -> dict[str, str]:
        """Return tool schema."""
        return {"name": self.name, "description": "Mock Python tool"}


class MockBashTool(AbstractMelleaTool):
    """Mock Bash tool for testing."""

    name: str = "bash_executor"

    def run(self, command: str) -> str:
        """Mock execution."""
        return f"executed: {command}"

    @property
    def as_json_tool(self) -> dict[str, str]:
        """Return tool schema."""
        return {"name": self.name, "description": "Mock Bash tool"}


def from_model_with_tool_calls(
    text_content: str | None = None,
    tool_calls_dict: dict[str, ModelToolCall] | None = None,
) -> Context:
    """Helper to create context with both text and tool_calls.

    Args:
        text_content: Optional text content for the model output.
        tool_calls_dict: Optional dict mapping tool names to ModelToolCall objects.

    Returns:
        Context with ModelOutputThunk containing both text and tool_calls.
    """
    ctx = ChatContext()
    ctx = ctx.add(ModelOutputThunk(value=text_content, tool_calls=tool_calls_dict))
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
        req = OutputSizeLimit(limit_chars=1000, execution_tier="local_unsafe")
        code = """```python
print("Hello, World!")
```"""
        ctx = from_model(code)
        result = req.validation_fn(ctx)
        # Should pass: "Hello, World!" is much less than 1000 chars
        assert result.as_bool() is True

    def test_output_exceeds_limit(self):
        """Test validation when output exceeds limit."""
        req = OutputSizeLimit(limit_chars=10, execution_tier="local_unsafe")
        code = """```python
print("Hello, World! This is a long message.")
```"""
        ctx = from_model(code)
        result = req.validation_fn(ctx)
        # Should fail: output is more than 10 chars
        assert result.as_bool() is False
        assert "exceeds" in (result.reason or "").lower()


class TestPythonExecutionReqWithOutputLimit:
    """Tests for PythonExecutionReq with max_output_chars parameter."""

    def test_execution_req_with_output_limit_passes_small_output(self):
        """Test PythonExecutionReq with max_output_chars passes when output is small."""
        from mellea.stdlib.requirements.python_reqs import PythonExecutionReq

        req = PythonExecutionReq(execution_tier="local_unsafe", max_output_chars=1000)
        ctx = from_model("```python\nprint('hello')\n```")
        result = req.validation_fn(ctx)
        assert result.as_bool() is True

    def test_execution_req_with_output_limit_fails_large_output(self):
        """Test PythonExecutionReq with max_output_chars fails when output exceeds limit."""
        from mellea.stdlib.requirements.python_reqs import PythonExecutionReq

        req = PythonExecutionReq(execution_tier="local_unsafe", max_output_chars=50)
        # Generate output larger than 50 chars
        ctx = from_model("```python\nprint('x' * 100)\n```")
        result = req.validation_fn(ctx)
        assert result.as_bool() is False
        assert "Output size" in (result.reason or "")
        assert "exceeds" in (result.reason or "").lower()

    def test_execution_req_without_output_limit(self):
        """Test PythonExecutionReq without max_output_chars ignores output size."""
        from mellea.stdlib.requirements.python_reqs import PythonExecutionReq

        req = PythonExecutionReq(
            execution_tier="local_unsafe",
            max_output_chars=None,  # No limit
        )
        # Generate large output
        ctx = from_model("```python\nprint('x' * 10000)\n```")
        result = req.validation_fn(ctx)
        # Should pass because output size is not checked
        assert result.as_bool() is True

    def test_execution_req_output_limit_static_tier_skipped(self):
        """Test PythonExecutionReq static tier skips output check even if max_output_chars set."""
        from mellea.stdlib.requirements.python_reqs import PythonExecutionReq

        req = PythonExecutionReq(
            execution_tier="static",
            max_output_chars=100,  # Set but ignored in static tier
        )
        # Generate output that would exceed limit
        ctx = from_model("```python\nprint('x' * 1000)\n```")
        result = req.validation_fn(ctx)
        # Static tier skips execution, so output check is also skipped
        assert result.as_bool() is True

    def test_execution_req_invalid_max_output_chars(self):
        """Test PythonExecutionReq raises ValueError for invalid max_output_chars."""
        from mellea.stdlib.requirements.python_reqs import PythonExecutionReq

        with pytest.raises(ValueError, match="max_output_chars must be positive"):
            PythonExecutionReq(execution_tier="local_unsafe", max_output_chars=0)

        with pytest.raises(ValueError, match="max_output_chars must be positive"):
            PythonExecutionReq(execution_tier="local_unsafe", max_output_chars=-100)

    def test_execution_req_description_includes_output_limit(self):
        """Test PythonExecutionReq description includes output limit when set."""
        from mellea.stdlib.requirements.python_reqs import PythonExecutionReq

        req = PythonExecutionReq(execution_tier="local_unsafe", max_output_chars=5000)
        # Description should include the output limit for non-static tiers
        assert "5000" in req.description

    def test_execution_req_description_no_limit_static_tier(self):
        """Test PythonExecutionReq description excludes output limit info for static tier."""
        from mellea.stdlib.requirements.python_reqs import PythonExecutionReq

        req = PythonExecutionReq(execution_tier="static", max_output_chars=5000)
        # Static tier should not mention output limit in description since it doesn't execute
        assert "Output limit" not in req.description


class TestImportRestrictions:
    """Tests for ImportRestrictions requirement."""

    def test_init_with_allowlist(self):
        """Test initialization with import allowlist."""
        req = ImportRestrictions(allowed_imports=["os", "sys", "json"])
        assert req.allowed_imports == ["os", "sys", "json"]

    def test_init_with_none(self):
        """Test initialization with None allowlist."""
        req = ImportRestrictions(allowed_imports=None)
        assert req.allowed_imports is None

    def test_init_default(self):
        """Test initialization with default (None) allowlist."""
        req = ImportRestrictions()
        assert req.allowed_imports is None

    def test_init_with_empty_list(self):
        """Test initialization with empty allowlist (blocks all imports)."""
        req = ImportRestrictions(allowed_imports=[])
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

    def test_empty_allowlist_blocks_all(self):
        """Test validation with empty allowlist blocks all imports."""
        req = ImportRestrictions(allowed_imports=[])
        ctx = from_model(PYTHON_WITH_IMPORTS)
        result = req.validation_fn(ctx)

        assert result.as_bool() is False
        assert "forbidden" in (result.reason or "").lower()

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

    def test_relative_import_forbidden(self):
        """Test validation catches relative-only imports like 'from . import x'."""
        req = ImportRestrictions(allowed_imports=["os"])
        code = """```python
from . import subprocess as sp

print("relative import")
```"""
        ctx = from_model(code)
        result = req.validation_fn(ctx)

        assert result.as_bool() is False
        assert "subprocess" in (result.reason or "")


class TestPythonToolRequirementsFactory:
    """Tests for python_code_generation_requirements() factory function."""

    def test_factory_default_returns_four_requirements(self):
        """Test factory with defaults returns 4 requirements (including NoImportRestrictions).

        The factory always returns 4 requirements: the last is either
        ImportRestrictions (if allowed_imports is provided) or NoImportRestrictions
        (if allowed_imports is None), providing semantic clarity in the bundle.
        """
        from mellea.stdlib.requirements.python_reqs import PythonExecutionReq
        from mellea.stdlib.requirements.python_tools import NoImportRestrictions

        reqs = python_code_generation_requirements()
        assert len(reqs) == 4
        assert isinstance(reqs[0], PythonCodeExtraction)
        assert isinstance(reqs[1], PythonSyntaxValid)
        assert isinstance(reqs[2], PythonExecutionReq)
        assert isinstance(reqs[3], NoImportRestrictions)

    def test_factory_with_allowed_imports_returns_four(self):
        """Test factory with allowed_imports returns 4 requirements with ImportRestrictions."""
        reqs = python_code_generation_requirements(allowed_imports=["os", "sys"])
        assert len(reqs) == 4
        assert isinstance(reqs[3], ImportRestrictions)

    def test_factory_parameter_propagation_output_limit(self):
        """Test factory propagates output_limit_chars to PythonExecutionReq.

        Output limit is now merged into PythonExecutionReq via max_output_chars,
        eliminating double execution. This test verifies the parameter reaches
        the execution requirement.
        """
        from mellea.stdlib.requirements.python_reqs import PythonExecutionReq

        reqs = python_code_generation_requirements(output_limit_chars=5000)
        exec_req = reqs[2]
        assert isinstance(exec_req, PythonExecutionReq)
        assert exec_req._max_output_chars == 5000

    def test_factory_parameter_propagation_imports(self):
        """Test factory propagates allowed_imports to ImportRestrictions."""
        imports = ["os", "sys", "json"]
        reqs = python_code_generation_requirements(allowed_imports=imports)
        import_req = reqs[3]
        assert isinstance(import_req, ImportRestrictions)
        assert import_req.allowed_imports == imports

    def test_factory_timeout_parameter(self):
        """Test factory accepts and uses timeout_seconds parameter."""
        reqs = python_code_generation_requirements(timeout_seconds=10)
        assert len(reqs) == 4

    def test_factory_sandbox_parameter(self):
        """Test factory accepts and uses use_sandbox parameter."""
        reqs = python_code_generation_requirements(use_sandbox=True)
        assert len(reqs) == 4

    def test_factory_all_parameters(self):
        """Test factory with all parameters configured."""
        from mellea.stdlib.requirements.python_reqs import PythonExecutionReq

        reqs = python_code_generation_requirements(
            allowed_imports=["os", "sys"],
            output_limit_chars=8000,
            timeout_seconds=15,
            use_sandbox=True,
        )
        assert len(reqs) == 4
        exec_req = reqs[2]
        assert isinstance(exec_req, PythonExecutionReq)
        assert exec_req._max_output_chars == 8000
        assert isinstance(reqs[3], ImportRestrictions)

    def test_factory_invalid_timeout(self):
        """Test factory with invalid timeout raises ValueError."""
        with pytest.raises(ValueError, match="timeout_seconds must be positive"):
            python_code_generation_requirements(timeout_seconds=0)

        with pytest.raises(ValueError, match="timeout_seconds must be positive"):
            python_code_generation_requirements(timeout_seconds=-5)

    def test_factory_invalid_output_limit(self):
        """Test factory with invalid output_limit raises ValueError."""
        with pytest.raises(ValueError, match="output_limit_chars must be positive"):
            python_code_generation_requirements(output_limit_chars=0)

        with pytest.raises(ValueError, match="output_limit_chars must be positive"):
            python_code_generation_requirements(output_limit_chars=-1000)

    def test_factory_requirement_order(self):
        """Test factory returns requirements in correct validation order."""
        from mellea.stdlib.requirements.python_reqs import PythonExecutionReq

        reqs = python_code_generation_requirements(allowed_imports=["os"])

        assert len(reqs) == 4
        assert isinstance(reqs[0], PythonCodeExtraction)
        assert isinstance(reqs[1], PythonSyntaxValid)
        assert isinstance(reqs[2], PythonExecutionReq)
        assert isinstance(reqs[3], ImportRestrictions)

    def test_factory_execution_req_description_varies_by_mode(self):
        """Test that PythonExecutionReq description reflects execution mode.

        The requirement description changes based on configuration (sandbox vs. static),
        confirming that different modes are being set up. This is observable through
        the requirement's description attribute without checking internal state.
        """
        from mellea.stdlib.requirements.python_reqs import PythonExecutionReq

        # Default mode (static analysis)
        reqs_default = python_code_generation_requirements()
        exec_req_default = reqs_default[2]
        assert isinstance(exec_req_default, PythonExecutionReq)
        assert "validation only" in exec_req_default.description.lower()

        # Sandbox mode
        reqs_sandbox = python_code_generation_requirements(use_sandbox=True)
        exec_req_sandbox = reqs_sandbox[2]
        assert isinstance(exec_req_sandbox, PythonExecutionReq)
        assert "sandbox" in exec_req_sandbox.description.lower()

    def test_factory_output_limit_description_reflects_configured_limit(self):
        """Test PythonExecutionReq description includes output limit when configured.

        The requirement description is built from configuration parameters and
        provides observable feedback about the limit being enforced.
        """
        from mellea.stdlib.requirements.python_reqs import PythonExecutionReq

        reqs = python_code_generation_requirements(
            output_limit_chars=5000, use_sandbox=True
        )
        exec_req = reqs[2]
        assert isinstance(exec_req, PythonExecutionReq)
        # The description should reflect the configured output limit (only for non-static tiers)
        assert "5000" in exec_req.description

    def test_factory_import_restrictions_description_reflects_allowed_list(self):
        """Test ImportRestrictions description includes configured allowed imports.

        When allowed_imports is provided, the requirement description shows which
        imports are allowed, providing observable evidence the config was applied.
        """
        reqs = python_code_generation_requirements(
            allowed_imports=["os", "sys", "json"]
        )
        import_req = reqs[3]
        assert isinstance(import_req, ImportRestrictions)
        # Description should reflect the allowed list
        assert "os" in import_req.description
        assert "sys" in import_req.description


class TestToolCallsCodeExtraction:
    """Tests for Python code extraction from tool_calls."""

    def test_extract_code_from_python_tool_by_name(self):
        """Test extraction from tool with 'python' in name."""
        python_tool = MockPythonTool()
        tool_call = ModelToolCall(
            name="python_executor", func=python_tool, args={"code": "print('hello')"}
        )

        result = _extract_code_from_tool_call(tool_call)
        assert result == "print('hello')"

    def test_extract_code_ignores_non_python_tools(self):
        """Test extraction ignores tools without 'python' in name."""
        bash_tool = MockBashTool()
        tool_call = ModelToolCall(
            name="bash_executor", func=bash_tool, args={"command": "echo hello"}
        )

        result = _extract_code_from_tool_call(tool_call)
        assert result is None

    def test_extract_code_case_insensitive_python_match(self):
        """Test tool name matching is case-insensitive."""
        python_tool = MockPythonTool()
        tool_call = ModelToolCall(
            name="PYTHON_Interpreter",
            func=python_tool,
            args={"code": "print('case-insensitive')"},
        )

        result = _extract_code_from_tool_call(tool_call)
        assert result == "print('case-insensitive')"

    def test_extract_code_python_runner_keyword(self):
        """Test extraction from tool with 'python' + 'runner' keyword."""
        python_tool = MockPythonTool()
        tool_call = ModelToolCall(
            name="python_runner", func=python_tool, args={"code": "print('runner')"}
        )

        result = _extract_code_from_tool_call(tool_call)
        assert result == "print('runner')"

    def test_extract_code_tries_field_names_in_order(self):
        """Test that common field names are tried in priority order."""
        python_tool = MockPythonTool()

        # Test 'code' field (highest priority)
        tool_call = ModelToolCall(
            name="python", func=python_tool, args={"code": "print('code')"}
        )
        assert _extract_code_from_tool_call(tool_call) == "print('code')"

        # Test 'script' field (second priority)
        tool_call = ModelToolCall(
            name="python", func=python_tool, args={"script": "print('script')"}
        )
        assert _extract_code_from_tool_call(tool_call) == "print('script')"

        # Test 'command' field (third priority)
        tool_call = ModelToolCall(
            name="python", func=python_tool, args={"command": "print('command')"}
        )
        assert _extract_code_from_tool_call(tool_call) == "print('command')"

        # Test 'source' field (fourth priority)
        tool_call = ModelToolCall(
            name="python", func=python_tool, args={"source": "print('source')"}
        )
        assert _extract_code_from_tool_call(tool_call) == "print('source')"

    def test_extract_code_returns_none_for_missing_fields(self):
        """Test extraction returns None when no code fields are present."""
        python_tool = MockPythonTool()
        tool_call = ModelToolCall(
            name="python", func=python_tool, args={"other_field": "value"}
        )

        result = _extract_code_from_tool_call(tool_call)
        assert result is None

    def test_extract_code_returns_none_for_non_string_values(self):
        """Test extraction ignores non-string field values."""
        python_tool = MockPythonTool()
        tool_call = ModelToolCall(
            name="python",
            func=python_tool,
            args={"code": 123, "script": None, "command": ["list"]},
        )

        result = _extract_code_from_tool_call(tool_call)
        assert result is None

    def test_extract_code_strips_whitespace(self):
        """Test extracted code is stripped of leading/trailing whitespace."""
        python_tool = MockPythonTool()
        tool_call = ModelToolCall(
            name="python_executor",
            func=python_tool,
            args={"code": "  \n  print('hello')  \n  "},
        )

        result = _extract_code_from_tool_call(tool_call)
        assert result == "print('hello')"

    def test_extract_code_ignores_empty_strings(self):
        """Test extraction ignores empty or whitespace-only strings."""
        python_tool = MockPythonTool()

        tool_call = ModelToolCall(name="python", func=python_tool, args={"code": ""})
        assert _extract_code_from_tool_call(tool_call) is None

        tool_call = ModelToolCall(
            name="python", func=python_tool, args={"code": "   \n  \t  "}
        )
        assert _extract_code_from_tool_call(tool_call) is None

    def test_extraction_from_tool_calls_in_context(self):
        """Test extraction of code from tool_calls in ModelOutputThunk."""
        python_tool = MockPythonTool()
        tool_call = ModelToolCall(
            name="python_executor",
            func=python_tool,
            args={"code": "x = 1 + 1\nprint(x)"},
        )
        tool_calls_dict = {"python_executor": tool_call}

        ctx = from_model_with_tool_calls(
            text_content=None, tool_calls_dict=tool_calls_dict
        )
        req = PythonCodeExtraction()
        result = req.validation_fn(ctx)

        assert result.as_bool() is True
        assert "x = 1 + 1" in (result.reason or "")

    def test_text_code_takes_priority_over_tool_calls(self):
        """Test that text code blocks are prioritized over tool_calls."""
        python_tool = MockPythonTool()
        tool_call = ModelToolCall(
            name="python_executor",
            func=python_tool,
            args={"code": "print('from tool')"},
        )
        tool_calls_dict = {"python_executor": tool_call}

        text_content = "```python\nprint('from text')\n```"
        ctx = from_model_with_tool_calls(
            text_content=text_content, tool_calls_dict=tool_calls_dict
        )
        req = PythonCodeExtraction()
        result = req.validation_fn(ctx)

        assert result.as_bool() is True
        # Should extract text code, not tool_calls code
        assert "from text" in (result.reason or "")
        assert "from tool" not in (result.reason or "")

    def test_tool_calls_fallback_when_no_text_code(self):
        """Test that tool_calls are used when no text code blocks exist."""
        python_tool = MockPythonTool()
        tool_call = ModelToolCall(
            name="python_interpreter",
            func=python_tool,
            args={
                "code": "def factorial(n):\n    return n * factorial(n-1) if n > 1 else 1"
            },
        )
        tool_calls_dict = {"python_interpreter": tool_call}

        text_content = "Here's a function to compute factorials:"
        ctx = from_model_with_tool_calls(
            text_content=text_content, tool_calls_dict=tool_calls_dict
        )
        req = PythonCodeExtraction()
        result = req.validation_fn(ctx)

        assert result.as_bool() is True
        assert "factorial" in (result.reason or "")

    def test_code_priority_ordering(self):
        """Test that code blocks are prioritized: python_text > tool_call > generic_text.

        Scoring breakdown:
        - ```python block: base_score + 10 (highest)
        - tool_call code: base_score + 5 (middle)
        - generic ``` block: base_score + 0 (lowest)

        This test verifies the ordering by providing multiple code sources
        with similar complexity, ensuring the highest-priority source wins.
        """
        python_tool = MockPythonTool()

        # Generic text block (lowest priority): simple one-liner
        # Score: min(1, 10) + 0 = 1
        generic_code = "x = 1"

        # Tool call code (middle priority): similar complexity
        # Score: min(1, 10) + 5 = 6
        tool_call = ModelToolCall(
            name="python_executor", func=python_tool, args={"code": "y = 2"}
        )

        # Python text block (highest priority): similar complexity
        # Score: min(1, 10) + 10 = 11
        python_text = "```python\nz = 3\n```"

        text_content = f"{generic_code}\n\n{python_text}"
        ctx = from_model_with_tool_calls(
            text_content=text_content, tool_calls_dict={"python_executor": tool_call}
        )

        req = PythonCodeExtraction()
        result = req.validation_fn(ctx)

        assert result.as_bool() is True
        # Should extract the python_text block (z = 3), not generic_code or tool_call
        assert "z = 3" in (result.reason or "")
        assert "x = 1" not in (result.reason or "")
        assert "y = 2" not in (result.reason or "")

    def test_generic_text_blocks_fallback_tier(self):
        """Test that generic text blocks (score + 0) are extracted when available.

        The code extraction hierarchy is:
        1. Text blocks marked ```python (score + 10)
        2. Generic text blocks that look like Python (score + 0) - patterns: def/class/import/print
        3. Tool call code (score + 5) - only used if both 1 and 2 are absent

        When both generic text blocks and tool_calls are present, generic blocks
        take priority because they're extracted as part of text processing before
        tool_calls are considered. Tool calls are only checked if all_blocks is empty.
        """
        python_tool = MockPythonTool()

        # Generic text block (looks like Python): contains "import" keyword
        text_content = """
Some explanation text
```
import os
```
More text here
"""

        # Tool call code (would score +5 if used)
        tool_call = ModelToolCall(
            name="python_executor", func=python_tool, args={"code": "import sys"}
        )

        ctx = from_model_with_tool_calls(
            text_content=text_content, tool_calls_dict={"python_executor": tool_call}
        )

        req = PythonCodeExtraction()
        result = req.validation_fn(ctx)

        assert result.as_bool() is True
        # Generic text block is extracted (it's detected as Python-like)
        assert "import os" in (result.reason or "")
        # Tool call is NOT used because generic text block was found
        assert "import sys" not in (result.reason or "")

    def test_syntax_validation_with_tool_calls_code(self):
        """Test syntax validation works with code from tool_calls."""
        python_tool = MockPythonTool()
        tool_call = ModelToolCall(
            name="python_executor",
            func=python_tool,
            args={"code": "def valid_func():\n    return 42"},
        )
        tool_calls_dict = {"python_executor": tool_call}

        ctx = from_model_with_tool_calls(
            text_content=None, tool_calls_dict=tool_calls_dict
        )
        req = PythonSyntaxValid()
        result = req.validation_fn(ctx)

        assert result.as_bool() is True

    def test_syntax_validation_catches_errors_in_tool_calls_code(self):
        """Test syntax validation catches errors in tool_calls code."""
        python_tool = MockPythonTool()
        tool_call = ModelToolCall(
            name="python_executor",
            func=python_tool,
            args={"code": "def invalid_func(\n    return 42"},
        )
        tool_calls_dict = {"python_executor": tool_call}

        ctx = from_model_with_tool_calls(
            text_content=None, tool_calls_dict=tool_calls_dict
        )
        req = PythonSyntaxValid()
        result = req.validation_fn(ctx)

        assert result.as_bool() is False
        assert "syntax error" in (result.reason or "").lower()

    def test_import_validation_with_tool_calls_code(self):
        """Test import restrictions work with code from tool_calls."""
        python_tool = MockPythonTool()
        tool_call = ModelToolCall(
            name="python_executor",
            func=python_tool,
            args={"code": "import os\nimport sys\nprint(os.getcwd())"},
        )
        tool_calls_dict = {"python_executor": tool_call}

        ctx = from_model_with_tool_calls(
            text_content=None, tool_calls_dict=tool_calls_dict
        )
        req = ImportRestrictions(allowed_imports=["os", "sys"])
        result = req.validation_fn(ctx)

        assert result.as_bool() is True

    def test_import_validation_catches_forbidden_in_tool_calls(self):
        """Test import restrictions catch forbidden imports in tool_calls code."""
        python_tool = MockPythonTool()
        tool_call = ModelToolCall(
            name="python_executor",
            func=python_tool,
            args={"code": "import subprocess\nprint('dangerous')"},
        )
        tool_calls_dict = {"python_executor": tool_call}

        ctx = from_model_with_tool_calls(
            text_content=None, tool_calls_dict=tool_calls_dict
        )
        req = ImportRestrictions(allowed_imports=["os"])
        result = req.validation_fn(ctx)

        assert result.as_bool() is False
        assert "subprocess" in (result.reason or "")
