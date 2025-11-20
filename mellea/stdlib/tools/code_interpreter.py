from dataclasses import dataclass
from abc import ABC, abstractmethod
import ast
import tempfile
import subprocess
import sys
from pathlib import Path
from mellea.helpers.fancy_logger import FancyLogger
from mellea.stdlib.base import Context
from mellea.stdlib.requirement import Requirement, ValidationResult

logger = FancyLogger.get_logger()


@dataclass
class ExecutionResult:
    """Result of code execution."""

    success: bool
    message: str | None = None
    error: str | None = None
    skipped: bool = False


class ExecutionEnvironment(ABC):
    """Abstract environment for executing Python code."""

    def __init__(self, allowed_imports: list[str] | None = None):
        """Initialize with optional import restrictions.

        Args:
            allowed_imports: List of allowed import modules. None means any import is allowed.
        """
        self.allowed_imports = allowed_imports

    @abstractmethod
    def execute(self, code: str, timeout: int) -> ExecutionResult:
        """Execute code and return result."""


class StaticAnalysisEnvironment(ExecutionEnvironment):
    """Safe environment that validates but does not execute code."""

    def execute(self, code: str, timeout: int) -> ExecutionResult:
        """Validate code syntax and imports without executing."""
        try:
            ast.parse(code)
        except SyntaxError as e:
            return ExecutionResult(success=False, error=str(e))

        if self.allowed_imports:
            unauthorized = _get_unauthorized_imports(code, self.allowed_imports)
            if unauthorized:
                return ExecutionResult(
                    success=False,
                    error=f"Unauthorized imports detected: {', '.join(unauthorized)}",
                )

        return ExecutionResult(
            success=True,
            skipped=True,
            message="Code validated but not executed (safe mode)",
        )


class UnsafeEnvironment(ExecutionEnvironment):
    """Unsafe environment that executes code directly with subprocess."""

    def execute(self, code: str, timeout: int) -> ExecutionResult:
        """Execute code with subprocess after checking imports."""
        if self.allowed_imports:
            unauthorized = _get_unauthorized_imports(code, self.allowed_imports)
            if unauthorized:
                return ExecutionResult(
                    success=False,
                    error=f"Unauthorized imports detected: {', '.join(unauthorized)}",
                )

        return self._execute_subprocess(code, timeout)

    def _execute_subprocess(self, code: str, timeout: int) -> ExecutionResult:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(code)
            temp_file = f.name

        try:
            # Execute code using the same Python interpreter and environment as the current process
            # This ensures the code has access to all installed packages and dependencies
            result = subprocess.run(
                [sys.executable, temp_file],
                capture_output=True,
                text=True,
                timeout=timeout,
            )

            if result.returncode == 0:
                message = "Code executed successfully"
                if result.stdout.strip():
                    message += f"\nOutput: {result.stdout.strip()}"
                return ExecutionResult(success=True, message=message)
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


class LLMSandboxEnvironment(ExecutionEnvironment):
    """Environment using llm-sandbox for secure Docker-based execution."""

    def execute(self, code: str, timeout: int) -> ExecutionResult:
        """Execute code using llm-sandbox."""
        if self.allowed_imports:
            unauthorized = _get_unauthorized_imports(code, self.allowed_imports)
            if unauthorized:
                return ExecutionResult(
                    success=False,
                    error=f"Unauthorized imports detected: {', '.join(unauthorized)}",
                )

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
                    message = "Code executed successfully in sandbox"
                    if (
                        hasattr(result, "stdout")
                        and result.stdout
                        and result.stdout.strip()
                    ):
                        message += f"\nOutput: {result.stdout.strip()}"
                    return ExecutionResult(success=True, message=message)
                else:
                    if result.stderr:
                        error_msg = f"Sandbox execution failed: {result.stderr[:200]}"
                    else:
                        # Log unknown error details for debugging
                        logger.warning(
                            f"Sandbox execution failed without stderr. Exit code: {result.exit_code}, "
                            f"Available attributes: {[attr for attr in dir(result) if not attr.startswith('_')]}"
                        )
                        error_msg = f"Sandbox execution failed with exit code {result.exit_code} (no error details available)"
                    return ExecutionResult(success=False, error=error_msg)

        except Exception as e:
            return ExecutionResult(
                success=False, error=f"Sandbox execution error: {e!s}"
            )


def _get_unauthorized_imports(code: str, allowed_imports: list[str]) -> list[str]:
    """Get list of unauthorized imports used in code."""
    unauthorized: list[str] = []
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return unauthorized

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                base_module = alias.name.split(".")[0]
                if (
                    base_module not in allowed_imports
                    and base_module not in unauthorized
                ):
                    unauthorized.append(base_module)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                base_module = node.module.split(".")[0]
                if (
                    base_module not in allowed_imports
                    and base_module not in unauthorized
                ):
                    unauthorized.append(base_module)
    return unauthorized


def _check_allowed_imports(code: str, allowed_imports: list[str]) -> bool:
    """Check if code only uses allowed imports."""
    return len(_get_unauthorized_imports(code, allowed_imports)) == 0


def code_interpreter(code: str):
    """Executes python code.
    
    Args:
        code: The Python code to execute.
    """
    exec_env = LLMSandboxEnvironment(allowed_imports=None)
    exec_env.execute(code, 60)


def local_code_interpreter(code: str):
    """Executes python code in the cwd
    
    Args:
        code: The Python code to execute.
    """
    exec_env = UnsafeEnvironment(allowed_imports=None)
    exec_env.execute(code, 60)    