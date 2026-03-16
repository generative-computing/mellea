"""Code interpreter tool and execution environments for agentic workflows.

Provides ``ExecutionResult`` (capturing stdout, stderr, success, and optional static
analysis output) and three concrete ``ExecutionEnvironment`` implementations:
``StaticAnalysisEnvironment`` (parse and import-check only, no execution),
``UnsafeEnvironment`` (subprocess execution in the current Python environment), and
``LLMSandboxEnvironment`` (Docker-isolated execution via ``llm-sandbox``). All
environments support an optional ``allowed_imports`` allowlist. The top-level
``code_interpreter`` and ``local_code_interpreter`` functions are ready to be wrapped
as ``MelleaTool`` instances for ReACT or other agentic loops.
"""

import ast
import subprocess
import sys
import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ...core import FancyLogger

logger = FancyLogger.get_logger()


@dataclass
class ExecutionResult:
    """Result of code execution.

    Code execution can be aborted prior to spinning up an interpreter (e.g., if prohibited imports are used).
    In these cases, the `success` flag is set to False and the `skipped` flag is set to True.

    If code is executed, then `success` is set to true iff the exit code is 0, and the `stdout` and `stderr` outputs
    are set to non-None values.

    We also use the `ExecutionResult` object to communicate the result of static and dynamic analyses. Those are passed back
    using the `analysis_result` field.

    TODO: should we also be trying to pass back the value of the final expression evaluated, or the value of locals() and globals()?

    Args:
        success (bool): ``True`` if execution succeeded (exit code 0 or
            static-analysis passed); ``False`` otherwise.
        stdout (str | None): Captured standard output, or ``None`` if
            execution was skipped.
        stderr (str | None): Captured standard error, or ``None`` if
            execution was skipped.
        skipped (bool): ``True`` when execution was not attempted.
        skip_message (str | None): Explanation of why execution was skipped.
        analysis_result (Any | None): Optional payload from static-analysis
            environments.

    """

    success: bool

    stdout: str | None

    stderr: str | None

    """ Indicates whether execution was skipped. """
    skipped: bool = False

    """ If execution is skipped, this message indicates why. """
    skip_message: str | None = None

    """ Used for returning results from static analyses. """
    analysis_result: Any | None = None

    def to_validationresult_reason(self) -> str:
        """Maps an ExecutionResult to a ValidationResult reason.

        TODO: Downstream use of this method is really hacky. A far better solution is for `ExecutionResult` to implement the `ValidationResult` interface.
        """
        assert self.skip_message is not None or (
            self.stderr is not None and self.stdout is not None
        ), (
            "Every ExecutionResult should have either a skip_message or a stdout/stderr stream."
        )
        if self.skip_message:
            reason = self.skip_message
        else:
            if self.success:
                assert self.stdout is not None
                reason = self.stdout
            else:
                assert self.stderr is not None
                reason = self.stderr
        return reason


class ExecutionEnvironment(ABC):
    """Abstract environment for executing Python code.

    Args:
        allowed_imports (list[str] | None): Allowlist of top-level module names
            that generated code may import. ``None`` disables the import check.

    """

    def __init__(self, allowed_imports: list[str] | None = None):
        """Initialize ExecutionEnvironment with an optional import allowlist."""
        self.allowed_imports = allowed_imports

    @abstractmethod
    def execute(self, code: str, timeout: int) -> ExecutionResult:
        """Execute the given code and return the result.

        Args:
            code (str): The Python source code to execute.
            timeout (int): Maximum number of seconds to allow the code to run.

        Returns:
            ExecutionResult: Execution outcome including stdout, stderr, and
            success flag.
        """


class StaticAnalysisEnvironment(ExecutionEnvironment):
    """Safe environment that validates but does not execute code."""

    def execute(self, code: str, timeout: int) -> ExecutionResult:
        """Validate code syntax and imports without executing.

        Args:
            code (str): The Python source code to validate.
            timeout (int): Ignored for static analysis; present for interface
                compatibility.

        Returns:
            ExecutionResult: Result with ``skipped=True`` and the parsed AST in
            ``analysis_result`` on success, or a syntax-error description on
            failure.
        """
        try:
            parse_tree = ast.parse(code)
        except SyntaxError as e:
            return ExecutionResult(
                success=False,
                stdout=None,
                stderr=None,
                skipped=True,
                skip_message="Parse failed.",
                analysis_result=e,
            )

        if self.allowed_imports:
            unauthorized = _get_unauthorized_imports(code, self.allowed_imports)
            if unauthorized:
                return ExecutionResult(
                    success=False,
                    stdout=None,
                    stderr=None,
                    skipped=True,
                    skip_message=f"Unauthorized imports detected: {', '.join(unauthorized)}",
                )

        return ExecutionResult(
            success=True,
            stdout=None,
            stderr=None,
            skipped=True,
            skip_message="Code parses successful; the parse result is in the analysis_result field of the ExecutionResult object. The static analysis execution environment does not execute code. To execute code, use one of the other execution environments.",
            analysis_result=parse_tree,
        )


class UnsafeEnvironment(ExecutionEnvironment):
    """Unsafe environment that executes code directly with subprocess."""

    def execute(self, code: str, timeout: int) -> ExecutionResult:
        """Execute code with subprocess after checking imports.

        Args:
            code (str): The Python source code to execute.
            timeout (int): Maximum number of seconds before the subprocess is
                killed and a timeout result is returned.

        Returns:
            ExecutionResult: Execution outcome with captured stdout/stderr and
            success flag, or a skipped result if imports are unauthorized or an
            unexpected error occurs.
        """
        if self.allowed_imports:
            unauthorized = _get_unauthorized_imports(code, self.allowed_imports)
            if unauthorized:
                return ExecutionResult(
                    success=False,
                    stdout=None,
                    stderr=None,
                    skipped=True,
                    skip_message=f"Unauthorized imports detected: {', '.join(unauthorized)}",
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
            return ExecutionResult(
                success=result.returncode == 0,
                stdout=result.stdout.strip(),
                stderr=result.stderr.strip(),
            )
        except subprocess.TimeoutExpired:
            return ExecutionResult(
                success=False,
                stdout=None,
                stderr=None,
                skipped=True,
                skip_message="Execution timed out.",
            )
        except Exception as e:
            return ExecutionResult(
                success=False,
                stdout=None,
                stderr=None,
                skipped=True,
                skip_message=f"Exception encountered in Mellea process (*not* the code interpreter process) when trying to run code_interpreter: {e!s}",
            )
        finally:
            try:
                Path(temp_file).unlink()
            except Exception:
                pass


class LLMSandboxEnvironment(ExecutionEnvironment):
    """Environment using llm-sandbox for secure Docker-based execution."""

    def execute(self, code: str, timeout: int) -> ExecutionResult:
        """Execute code using llm-sandbox in an isolated Docker container.

        Checks the import allowlist first, then delegates to a ``SandboxSession``
        from the ``llm-sandbox`` package. Returns a skipped result if
        ``llm-sandbox`` is not installed.

        Args:
            code (str): The Python source code to execute.
            timeout (int): Maximum number of seconds to allow the sandboxed
                process to run.

        Returns:
            ExecutionResult: Execution outcome with stdout/stderr and success
            flag, or a skipped result on import violation or sandbox error.
        """
        if self.allowed_imports:
            unauthorized = _get_unauthorized_imports(code, self.allowed_imports)
            if unauthorized:
                return ExecutionResult(
                    success=False,
                    stdout=None,
                    stderr=None,
                    skipped=True,
                    skip_message=f"Unauthorized imports detected: {', '.join(unauthorized)}",
                )

        try:
            from llm_sandbox import SandboxSession
        except ImportError:
            return ExecutionResult(
                success=False,
                stdout=None,
                stderr=None,
                skipped=True,
                skip_message="llm-sandbox not installed. Install with: pip install 'mellea[sandbox]'",
            )

        try:
            with SandboxSession(
                lang="python", verbose=False, keep_template=False
            ) as session:
                result = session.run(code, timeout=timeout)

                return ExecutionResult(
                    success=result.exit_code == 0,
                    stdout=result.stdout.strip(),
                    stderr=result.stderr.strip(),
                )
        except Exception as e:
            return ExecutionResult(
                success=False,
                stdout=None,
                stderr=None,
                skipped=True,
                skip_message=f"Sandbox execution error: {e!s}",
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


def code_interpreter(code: str) -> ExecutionResult:
    """Executes python code.

    Args:
        code: The Python code to execute.

    Returns:
        An ``ExecutionResult`` with stdout, stderr, and a success flag.
    """
    exec_env = LLMSandboxEnvironment(allowed_imports=None)
    return exec_env.execute(code, 60)


def local_code_interpreter(code: str) -> ExecutionResult:
    """Executes python code in the cwd.

    Args:
        code: The Python code to execute.

    Returns:
        An ``ExecutionResult`` with stdout, stderr, and a success flag.
    """
    exec_env = UnsafeEnvironment(allowed_imports=None)
    return exec_env.execute(code, 60)
