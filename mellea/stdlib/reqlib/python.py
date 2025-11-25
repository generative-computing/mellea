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


class SafeEnvironment(ExecutionEnvironment):
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


# endregion

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
        environment = SafeEnvironment(allowed_imports=allowed_imports)

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

# region additional verifiers from PR


def extract_python_code(text: str) -> str | None:
    """Extract Python code from markdown code blocks or plain text.

    Uses intelligent extraction strategy:
    1. Finds all ```python...``` blocks
    2. Scores each block (prefers longer, non-test code after positive cues)
    3. Returns highest-scoring block
    4. Falls back to generic blocks or raw text

    Returns None if no Python code found.
    """
    import re

    # Try explicit python code blocks first
    python_block_pattern = r"```python\s*\n(.*?)```"
    matches = re.findall(python_block_pattern, text, re.DOTALL)

    if matches:
        if len(matches) == 1:
            return matches[0].strip()

        # Multiple blocks - need to be smart about which one
        best_block = None
        best_score = -999

        # Find positions of each match to get context
        for match in matches:
            # Get text before this code block for context
            match_pos = text.find(f"```python\n{match}")
            context_before = text[max(0, match_pos - 200) : match_pos]

            score = _score_code_block(match) + (
                5 if "correct" in context_before.lower() else 0
            )

            if score > best_score:
                best_score = score
                best_block = match

        return best_block.strip() if best_block else matches[0].strip()

    # Try generic code blocks
    generic_block_pattern = r"```\s*\n(.*?)```"
    matches = re.findall(generic_block_pattern, text, re.DOTALL)
    if matches:
        # Check if any look like Python
        for match in matches:
            candidate = match.strip()
            if any(
                keyword in candidate
                for keyword in [
                    "def ",
                    "class ",
                    "import ",
                    "from ",
                    "if ",
                    "for ",
                    "while ",
                ]
            ):
                return candidate

    # If no code blocks, check if entire text looks like Python
    stripped_text = text.strip()
    if any(
        keyword in stripped_text for keyword in ["def ", "class ", "import ", "from "]
    ):
        return stripped_text

    return None


class HasPythonCodeListing(Requirement):
    """Verifies that the output contains a valid Python code listing."""

    def __init__(self):
        """Initialize the Python code listing validator."""
        super().__init__(
            description="The result should contain a Python code listing in markdown format or as plain code.",
            validation_fn=lambda ctx: self._validate_has_code(ctx),
            check_only=True,
        )

    def _validate_has_code(self, ctx: Context) -> ValidationResult:
        """Validate that context contains extractable Python code."""
        last_output = ctx.last_output()
        if last_output is None or last_output.value is None:
            return ValidationResult(result=False, reason="No output found in context")

        code = extract_python_code(last_output.value)
        if code is None:
            return ValidationResult(
                result=False, reason="No Python code block found in output"
            )

        return ValidationResult(
            result=True,
            reason=code,  # Return extracted code for downstream use
        )


class PythonCodeParses(Requirement):
    """Verifies that the Python code is syntactically valid."""

    def __init__(self):
        """Initialize the Python code parser validator."""
        super().__init__(
            description="The Python code should be syntactically valid and parseable.",
            validation_fn=lambda ctx: self._validate_parses(ctx),
            check_only=True,
        )

    def _validate_parses(self, ctx: Context) -> ValidationResult:
        """Validate that extracted Python code is syntactically valid using AST."""
        # First extract the code
        has_code = HasPythonCodeListing()
        extraction_result = has_code._validate_has_code(ctx)
        if not extraction_result.as_bool():
            return ValidationResult(
                result=False,
                reason=extraction_result.reason or "Could not extract Python code",
            )

        code = extraction_result.reason  # Code is stored in reason field
        assert code is not None

        try:
            ast.parse(code)
            return ValidationResult(
                result=True, reason="Python code parses successfully"
            )
        except SyntaxError as e:
            return ValidationResult(
                result=False, reason=f"Syntax error at line {e.lineno}: {e.msg}"
            )
        except Exception as e:
            return ValidationResult(result=False, reason=f"Parse error: {e!s}")


class PythonValidImports(Requirement):
    """Verifies that all import statements reference available packages."""

    def __init__(self, venv_path: str | None = None):
        """Initialize import validator.

        Args:
            venv_path: Optional path to virtual environment to check imports against.
                      If None, checks against current Python environment.
        """
        self._venv_path = venv_path
        super().__init__(
            description=f"All import statements should use packages available in {'specified venv' if venv_path else 'current environment'}.",
            validation_fn=lambda ctx: self._validate_imports(ctx),
            check_only=True,
        )

    def _validate_imports(self, ctx: Context) -> ValidationResult:
        """Validate that all imports in Python code are available."""
        # First extract and parse the code
        has_code = HasPythonCodeListing()
        extraction_result = has_code._validate_has_code(ctx)
        if not extraction_result.as_bool():
            return ValidationResult(
                result=False,
                reason="Could not extract Python code for import validation",
            )

        code = extraction_result.reason
        assert code is not None

        # Check if code parses
        try:
            ast.parse(code)
        except SyntaxError:
            return ValidationResult(
                result=False, reason="Code has syntax errors, cannot validate imports"
            )

        modules = self._get_imported_modules(code)
        if not modules:
            # No imports is valid
            return ValidationResult(result=True, reason="No imports to validate")

        unavailable_modules = []
        for module in modules:
            if not self._is_module_available(module):
                unavailable_modules.append(module)

        if unavailable_modules:
            return ValidationResult(
                result=False,
                reason=f"Unavailable modules: {', '.join(unavailable_modules)}",
            )

        return ValidationResult(
            result=True, reason=f"All imports valid: {', '.join(modules)}"
        )

    def _get_imported_modules(self, code: str) -> list[str]:
        """Extract all imported module names from Python code."""
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return []

        modules = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    modules.append(alias.name.split(".")[0])  # Get top-level module
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    modules.append(node.module.split(".")[0])

        return list(set(modules))  # Remove duplicates

    def _is_module_available(self, module_name: str) -> bool:
        """Check if a module is available in the system or specified venv."""
        if self._venv_path:
            # Check in specified venv using pip list
            try:
                result = subprocess.run(
                    [
                        f"{self._venv_path}/bin/python",
                        "-m",
                        "pip",
                        "list",
                        "--format=freeze",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                installed = [
                    line.split("==")[0].lower() for line in result.stdout.split("\n")
                ]
                return module_name.lower() in installed
            except Exception:
                return False
        else:
            # Check in current environment
            import importlib.util

            return importlib.util.find_spec(module_name) is not None


# Alias for backwards compatibility
PythonExecutesWithoutError = PythonExecutionReq


class PythonHasFunctionDef(Requirement):
    """Verifies that Python code contains at least one function definition."""

    def __init__(self):
        """Initialize the function definition validator."""
        super().__init__(
            description="The Python code should define at least one function.",
            validation_fn=lambda ctx: self._validate_function_def(ctx),
            check_only=True,
        )

    def _validate_function_def(self, ctx: Context) -> ValidationResult:
        """Validate that Python code contains at least one function definition."""
        has_code = HasPythonCodeListing()
        extraction_result = has_code._validate_has_code(ctx)
        if not extraction_result.as_bool():
            return ValidationResult(
                result=False, reason="Could not extract Python code"
            )

        code = extraction_result.reason
        assert code is not None

        try:
            tree = ast.parse(code)
        except SyntaxError:
            return ValidationResult(result=False, reason="Code has syntax errors")

        function_names = []
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                function_names.append(node.name)

        if function_names:
            return ValidationResult(
                result=True,
                reason=f"Found {len(function_names)} function(s): {', '.join(function_names)}",
            )
        else:
            return ValidationResult(
                result=False, reason="No function definitions found in code"
            )


class PythonHasClassDef(Requirement):
    """Verifies that Python code contains at least one class definition."""

    def __init__(self):
        """Initialize the class definition validator."""
        super().__init__(
            description="The Python code should define at least one class.",
            validation_fn=lambda ctx: self._validate_class_def(ctx),
            check_only=True,
        )

    def _validate_class_def(self, ctx: Context) -> ValidationResult:
        """Validate that Python code contains at least one class definition."""
        has_code = HasPythonCodeListing()
        extraction_result = has_code._validate_has_code(ctx)
        if not extraction_result.as_bool():
            return ValidationResult(
                result=False, reason="Could not extract Python code"
            )

        code = extraction_result.reason
        assert code is not None

        try:
            tree = ast.parse(code)
        except SyntaxError:
            return ValidationResult(result=False, reason="Code has syntax errors")

        class_names = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                class_names.append(node.name)

        if class_names:
            return ValidationResult(
                result=True,
                reason=f"Found {len(class_names)} class(es): {', '.join(class_names)}",
            )
        else:
            return ValidationResult(
                result=False, reason="No class definitions found in code"
            )


class PythonMatchesExamples(Requirement):
    """Verifies that generated Python function produces correct outputs for given examples.

    This is a lightweight functional correctness checker that tests the generated
    function against specific input/output examples.
    """

    def __init__(self, function_name: str, examples: list[tuple[dict, Any]]):
        """Initialize example-based correctness validator.

        Args:
            function_name: Name of the function to test
            examples: List of (input_kwargs, expected_output) tuples.
                     For example: [
                         ({"n": 5}, 120),  # factorial(5) should return 120
                         ({"n": 0}, 1),     # factorial(0) should return 1
                     ]
        """
        self._function_name = function_name
        self._examples = examples
        super().__init__(
            description=f"The function '{function_name}' should produce correct outputs for {len(examples)} test examples.",
            validation_fn=lambda ctx: self._validate_examples(ctx),
            check_only=True,
        )

    def _validate_examples(self, ctx: Context) -> ValidationResult:
        """Validate that Python function produces correct outputs for given examples."""
        has_code = HasPythonCodeListing()
        extraction_result = has_code._validate_has_code(ctx)
        if not extraction_result.as_bool():
            return ValidationResult(
                result=False, reason="Could not extract Python code"
            )

        code = extraction_result.reason
        assert code is not None

        # Check if code parses
        try:
            ast.parse(code)
        except SyntaxError as e:
            return ValidationResult(result=False, reason=f"Code has syntax errors: {e}")

        # Execute code in isolated namespace
        namespace: dict[str, Any] = {}
        try:
            exec(code, namespace)
        except Exception as e:
            return ValidationResult(
                result=False, reason=f"Code execution failed: {e!s}"
            )

        # Check if function exists
        if self._function_name not in namespace:
            return ValidationResult(
                result=False,
                reason=f"Function '{self._function_name}' not found in code",
            )

        func = namespace[self._function_name]
        if not callable(func):
            return ValidationResult(
                result=False, reason=f"'{self._function_name}' is not callable"
            )

        # Test all examples
        failed_examples = []
        for i, (inputs, expected) in enumerate(self._examples):
            try:
                result = func(**inputs)
                if result != expected:
                    failed_examples.append(
                        f"Example {i + 1}: {self._function_name}({inputs}) = {result}, expected {expected}"
                    )
            except Exception as e:
                failed_examples.append(
                    f"Example {i + 1}: {self._function_name}({inputs}) raised {type(e).__name__}: {e!s}"
                )

        if failed_examples:
            return ValidationResult(
                result=False,
                reason=f"Failed {len(failed_examples)}/{len(self._examples)} examples:\n"
                + "\n".join(failed_examples),
            )

        return ValidationResult(
            result=True, reason=f"All {len(self._examples)} examples passed", score=1.0
        )


class PythonMatchesDocstring(Requirement):
    """Verifies that generated Python code matches a docstring specification.

    This validator uses an LLM to generate test cases from a docstring/specification,
    then validates the generated code against those test cases.

    The tests are generated once when you call generate_tests() and cached for reuse
    across multiple validation calls during rejection sampling.

    Example usage:
        ```python
        with start_session("ollama", "granite3.3:8b") as session:
            verifier = PythonMatchesDocstring(
                "Calculate factorial of n",
                num_tests=5
            )
            # Generate tests once (uses active session's LLM)
            verifier.generate_tests()

            # Now use in rejection sampling (no additional LLM calls)
            result = session.instruct(
                "Write a factorial function",
                requirements=[verifier]
            )
        ```
    """

    def __init__(
        self,
        docstring: str,
        function_name: str | None = None,
        num_tests: int = 5,
        tests: list[tuple[dict, Any]] | None = None,
    ):
        """Initialize docstring-based correctness validator.

        Args:
            docstring: The specification/docstring describing expected behavior.
            function_name: Name of the function to test. If None, will infer from code.
            num_tests: Number of test cases to generate (default: 5).
            tests: Pre-generated tests. If provided, skips LLM generation.
                  Format: list of (input_dict, expected_output) tuples.
        """
        self._docstring = docstring
        self._function_name = function_name
        self._num_tests = num_tests
        self._cached_tests: list[tuple[dict, Any]] | None = tests

        super().__init__(
            description=f"The code should match the specification: {docstring[:100]}{'...' if len(docstring) > 100 else ''}",
            validation_fn=lambda ctx: self._validate(ctx),
            check_only=True,
        )

    def generate_tests(self, temperature: float = 0.3):
        """Generate test cases from the docstring using the active LLM session.

        This should be called once before using the verifier. Requires an active session.

        Args:
            temperature: LLM temperature for test generation (default: 0.3 for consistency).

        Raises:
            RuntimeError: If no active session is found.
        """
        if self._cached_tests is not None:
            return  # Already have tests

        from mellea.stdlib.session import get_session

        session = get_session()

        # Infer function name if needed (use placeholder for generation)
        func_name = self._function_name or "function"

        test_prompt = f"""Given this function specification, generate {self._num_tests} diverse test cases.

Specification:
{self._docstring}

Function name: {func_name}

Generate test cases as a JSON array. Each test has "inputs" (dict of param names to values) and "expected_output".

Example format:
[
  {{"inputs": {{"n": 5}}, "expected_output": 120}},
  {{"inputs": {{"n": 0}}, "expected_output": 1}}
]

Focus on:
1. Normal cases (typical inputs)
2. Edge cases (boundaries, empty values)
3. Different data types if applicable

Output ONLY the JSON array, no other text."""

        # Generate using session.instruct (handles async properly)
        result = session.instruct(
            test_prompt, model_options={"temperature": temperature}
        )

        # Parse JSON response
        import json

        test_json_str = result.value
        assert test_json_str is not None

        # Extract JSON from markdown blocks if present
        if "```json" in test_json_str:
            test_json_str = test_json_str.split("```json")[1].split("```")[0].strip()
        elif "```" in test_json_str:
            test_json_str = test_json_str.split("```")[1].split("```")[0].strip()

        # Fix common Python->JSON inconsistencies
        test_json_str = test_json_str.replace(": None", ": null")
        test_json_str = test_json_str.replace(": True", ": true")
        test_json_str = test_json_str.replace(": False", ": false")

        try:
            test_data = json.loads(test_json_str)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Failed to parse test JSON. LLM output:\n{test_json_str[:500]}\nError: {e}"
            )

        # Convert to expected format with flexible key names
        self._cached_tests = []
        for i, test in enumerate(test_data):
            # Skip tests that expect errors (we only test success cases)
            if "expected_error" in test or "error" in test:
                continue

            # Handle various possible key names
            inputs = test.get("inputs") or test.get("input") or test.get("args") or {}
            expected = (
                test.get("expected_output")
                or test.get("output")
                or test.get("expected")
                or test.get("result")
            )

            if expected is None:
                # Skip if no expected output (might be an error case)
                continue

            # Skip if expected output looks like an error message
            if isinstance(expected, str) and expected.lower().startswith("error"):
                continue

            self._cached_tests.append((inputs, expected))

        if not self._cached_tests:
            raise ValueError(
                f"No valid test cases found. LLM response:\n{test_json_str[:500]}"
            )

    def _validate(self, ctx: Context) -> ValidationResult:
        """Validation function."""
        if self._cached_tests is None:
            return ValidationResult(
                result=False,
                reason="Tests not generated. Call generate_tests() first or provide tests in constructor.",
            )

        return self._python_matches_docstring(ctx, self._cached_tests)

    def _python_matches_docstring(
        self, ctx: Context, tests: list[tuple[dict, Any]]
    ) -> ValidationResult:
        """Validate that Python code matches a docstring specification using provided tests."""
        # Extract the code
        has_code = HasPythonCodeListing()
        extraction_result = has_code._validate_has_code(ctx)
        if not extraction_result.as_bool():
            return ValidationResult(
                result=False, reason="Could not extract Python code from output"
            )

        code = extraction_result.reason
        assert code is not None

        # Parse code to find function name if not provided
        function_name = self._function_name
        if function_name is None:
            try:
                tree = ast.parse(code)
                functions = [
                    node.name
                    for node in ast.walk(tree)
                    if isinstance(node, ast.FunctionDef)
                ]
                if not functions:
                    return ValidationResult(
                        result=False, reason="No function found in generated code"
                    )
                function_name = functions[0]  # Use first function found
            except SyntaxError:
                return ValidationResult(
                    result=False,
                    reason="Code has syntax errors, cannot identify function",
                )

        # Run the tests using existing logic
        examples_verifier = PythonMatchesExamples(function_name, tests)
        return examples_verifier._validate_examples(ctx)


# endregion
