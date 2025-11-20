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
from mellea.stdlib.tools.code_interpreter import LLMSandboxEnvironment, UnsafeEnvironment, ExecutionEnvironment, StaticAnalysisEnvironment

logger = FancyLogger.get_logger()


# region code extraction


def _score_code_block(code: str) -> int:
    """Score a code block to determine if it's likely the main answer.

    Scoring metrics:
    - Length bonus: +1 per line (capped at 10) - longer blocks are generally more substantial
    - Function/class bonus: +5 - indicates complete, structured code
    - Control flow bonus: +3 - presence of if/for/while/try/with suggests meaningful logic
    - Non-trivial content penalty: -5 if fewer than 2 executable lines (filters out import-only or comment-heavy blocks)

    Returns:
        int: Score indicating likelihood this is the primary code block to execute.
    """
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

    # Penalize blocks that are mostly imports/comments without actual logic
    # We want at least 2 lines of executable code to consider it a meaningful code block
    # This helps filter out import-only blocks or heavily commented trivial snippets
    # TODO: Consider using comment-to-code ratio in future iterations
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
        all_blocks.append((block.strip(), _score_code_block(block.strip()) + 10))

    # Add generic blocks if they look like Python
    for block in generic_blocks:
        block = block.strip()
        if block and any(
            keyword in block
            for keyword in ["def ", "class ", "import ", "print(", "if __name__"]
        ):
            all_blocks.append((block, _score_code_block(block)))

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
    """Validate that Python code executes without raising exceptions.

    First extracts the highest-scoring Python code block from the context,
    then validates/executes it based on the specified execution mode.
    """
    extraction_result = _has_python_code_listing(ctx)
    if not extraction_result.as_bool():
        return ValidationResult(
            result=False,
            reason=f"Could not extract Python code for execution: {extraction_result.reason}",
        )

    code = extraction_result.reason
    assert code is not None

    environment: ExecutionEnvironment
    if use_sandbox:
        environment = LLMSandboxEnvironment(allowed_imports=allowed_imports)
    elif allow_unsafe:
        environment = UnsafeEnvironment(allowed_imports=allowed_imports)
    else:
        environment = StaticAnalysisEnvironment(allowed_imports=allowed_imports)

    result = environment.execute(code, timeout)
    return ValidationResult(
        result=result.success, reason=result.message or result.error
    )


class PythonExecutionReq(Requirement):
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
            allowed_imports: List of allowed import modules when using execution. None means any import is allowed.
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
