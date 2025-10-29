"""Requirements for Python code generation validation."""

import ast
import subprocess
import sys
import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mellea.helpers.fancy_logger import FancyLogger
from mellea.stdlib.base import Context
from mellea.stdlib.requirement import Requirement, ValidationResult

logger = FancyLogger.get_logger()

# region execution backends


@dataclass
class ExecutionResult:
    """Result of code execution."""

    success: bool
    message: str | None = None
    error: str | None = None
    skipped: bool = False


class ExecutionBackend(ABC):
    """Abstract backend for executing Python code."""

    @abstractmethod
    def execute(self, code: str, timeout: int) -> ExecutionResult:
        """Execute code and return result."""


class SafeBackend(ExecutionBackend):
    """Safe backend that validates but does not execute code."""

    def __init__(self, allowed_imports: list[str] | None = None):
        """Initialize with optional import restrictions."""
        self.allowed_imports = allowed_imports

    def execute(self, code: str, timeout: int) -> ExecutionResult:
        """Validate code syntax and imports without executing."""
        try:
            ast.parse(code)
        except SyntaxError as e:
            return ExecutionResult(success=False, error=str(e))

        if self.allowed_imports and not _check_allowed_imports(
            code, self.allowed_imports
        ):
            return ExecutionResult(success=False, error="Unauthorized imports detected")

        return ExecutionResult(
            success=True,
            skipped=True,
            message="Code validated but not executed (safe mode)",
        )


class UnsafeBackend(ExecutionBackend):
    """Unsafe backend that executes code directly with subprocess."""

    def __init__(self, allowed_imports: list[str] | None = None):
        """Initialize with optional import restrictions."""
        self.allowed_imports = allowed_imports

    def execute(self, code: str, timeout: int) -> ExecutionResult:
        """Execute code with subprocess after checking imports."""
        if self.allowed_imports and not _check_allowed_imports(
            code, self.allowed_imports
        ):
            return ExecutionResult(success=False, error="Unauthorized imports detected")

        return self._execute_subprocess(code, timeout)

    def _execute_subprocess(self, code: str, timeout: int) -> ExecutionResult:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(code)
            temp_file = f.name

        try:
            result = subprocess.run(
                [sys.executable, temp_file],
                capture_output=True,
                text=True,
                timeout=timeout,
            )

            if result.returncode == 0:
                return ExecutionResult(
                    success=True, message="Code executed successfully"
                )
            else:
                return ExecutionResult(
                    success=False,
                    error=f"Execution failed with error: {result.stderr[:200]}",
                )
        except subprocess.TimeoutExpired:
            return ExecutionResult(
                success=False, error=f"Execution timed out after {timeout} seconds"
            )
        except Exception as e:
            return ExecutionResult(success=False, error=f"Execution error: {e!s}")
        finally:
            try:
                Path(temp_file).unlink()
            except Exception:
                pass


class LLMSandboxBackend(ExecutionBackend):
    """Backend using llm-sandbox for secure Docker-based execution."""

    def __init__(self, allowed_imports: list[str] | None = None):
        """Initialize with optional import restrictions."""
        self.allowed_imports = allowed_imports

    def execute(self, code: str, timeout: int) -> ExecutionResult:
        """Execute code using llm-sandbox."""
        if self.allowed_imports and not _check_allowed_imports(
            code, self.allowed_imports
        ):
            return ExecutionResult(success=False, error="Unauthorized imports detected")

        try:
            from llm_sandbox import SandboxSession
        except ImportError:
            return ExecutionResult(
                success=False,
                error="llm-sandbox not installed. Install with: uv add 'llm-sandbox[docker]'",
            )

        try:
            with SandboxSession(
                lang="python", verbose=False, keep_template=False
            ) as session:
                result = session.run(code, timeout=timeout)

                if result.exit_code == 0:
                    return ExecutionResult(
                        success=True, message="Code executed successfully in sandbox"
                    )
                else:
                    return ExecutionResult(
                        success=False,
                        error=f"Sandbox execution failed: {result.stderr[:200] if result.stderr else 'Unknown error'}",
                    )

        except Exception as e:
            return ExecutionResult(
                success=False, error=f"Sandbox execution error: {e!s}"
            )


def _check_allowed_imports(code: str, allowed_imports: list[str]) -> bool:
    """Check if code only uses allowed imports."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                base_module = alias.name.split(".")[0]
                if base_module not in allowed_imports:
                    return False
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                base_module = node.module.split(".")[0]
                if base_module not in allowed_imports:
                    return False
    return True


# endregion

# region code extraction


def _score_code_block(code: str, context_text: str = "") -> int:
    """Score a code block to determine if it's likely the main answer."""
    score = 0
    lines = code.split("\n")

    # Longer blocks generally better
    score += min(len(lines), 10)

    # Prefer complete functions/classes
    if "def " in code or "class " in code:
        score += 5

    # Prefer blocks with actual logic
    if any(keyword in code for keyword in ["if ", "for ", "while ", "try:", "with "]):
        score += 3

    # Avoid pure imports or comments
    non_trivial_lines = [
        line.strip()
        for line in lines
        if line.strip() and not line.strip().startswith(("#", "import ", "from "))
    ]
    if len(non_trivial_lines) < 2:
        score -= 5

    return score


def _has_python_code_listing(ctx: Context) -> ValidationResult:
    """Extract Python code from context."""
    last_output = ctx.last_output()
    if last_output is None or last_output.value is None:
        return ValidationResult(result=False, reason="No output found in context")

    content = last_output.value

    # Look for code blocks with python specifier
    import re

    # Pattern for ```python ... ``` blocks
    python_blocks = re.findall(r"```python\s*\n(.*?)\n```", content, re.DOTALL)

    # Pattern for generic ``` blocks
    generic_blocks = re.findall(r"```\s*\n(.*?)\n```", content, re.DOTALL)

    all_blocks = []

    # Add python blocks with high priority
    for block in python_blocks:
        all_blocks.append(
            (block.strip(), _score_code_block(block.strip(), content) + 10)
        )

    # Add generic blocks if they look like Python
    for block in generic_blocks:
        block = block.strip()
        if block and any(
            keyword in block
            for keyword in ["def ", "class ", "import ", "print(", "if __name__"]
        ):
            all_blocks.append((block, _score_code_block(block, content)))

    if not all_blocks:
        return ValidationResult(result=False, reason="No Python code blocks found")

    # Return the highest scoring block
    best_block = max(all_blocks, key=lambda x: x[1])
    return ValidationResult(result=True, reason=best_block[0])


# endregion

# region execution validation


def _python_executes_without_error(
    ctx: Context,
    timeout: int = 5,
    allow_unsafe: bool = False,
    allowed_imports: list[str] | None = None,
    use_sandbox: bool = False,
) -> ValidationResult:
    """Validate that Python code executes without raising exceptions."""
    extraction_result = _has_python_code_listing(ctx)
    if not extraction_result.as_bool():
        return ValidationResult(
            result=False, reason="Could not extract Python code for execution"
        )

    code = extraction_result.reason
    assert code is not None

    backend: ExecutionBackend
    if use_sandbox:
        backend = LLMSandboxBackend(allowed_imports=allowed_imports)
    elif allow_unsafe:
        backend = UnsafeBackend(allowed_imports=allowed_imports)
    else:
        backend = SafeBackend(allowed_imports=allowed_imports)

    result = backend.execute(code, timeout)
    return ValidationResult(
        result=result.success, reason=result.message or result.error
    )


class PythonExecutesWithoutError(Requirement):
    """Verifies that Python code runs without raising exceptions."""

    def __init__(
        self,
        timeout: int = 5,
        allow_unsafe_execution: bool = False,
        allowed_imports: list[str] | None = None,
        use_sandbox: bool = False,
    ):
        """Initialize execution validator.

        Args:
            timeout: Maximum seconds to allow code to run before timing out.
            allow_unsafe_execution: If True, execute code directly with subprocess (unsafe).
            allowed_imports: List of allowed import modules when using execution.
            use_sandbox: If True, use llm-sandbox for secure Docker-based execution.
        """
        self._timeout = timeout
        self._allow_unsafe = allow_unsafe_execution
        self._allowed_imports = allowed_imports
        self._use_sandbox = use_sandbox

        if allow_unsafe_execution and not use_sandbox:
            logger.warning(
                "⚠️ UNSAFE: Executing untrusted code directly. Only use with trusted sources!"
            )

        if use_sandbox and allow_unsafe_execution:
            execution_mode = f"sandbox execution (timeout: {timeout}s)"
        elif allow_unsafe_execution:
            execution_mode = f"unsafe execution (timeout: {timeout}s)"
        elif use_sandbox:
            execution_mode = f"sandbox execution (timeout: {timeout}s)"
        else:
            execution_mode = "validation only"

        super().__init__(
            description=f"The Python code should execute without errors ({execution_mode}).",
            validation_fn=lambda ctx: _python_executes_without_error(
                ctx,
                self._timeout,
                self._allow_unsafe,
                self._allowed_imports,
                self._use_sandbox,
            ),
            check_only=True,
        )


# endregion
