"""Requirements for Python code generation validation."""

import ast
import importlib.util
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Optional

from mellea.stdlib.base import Context
from mellea.stdlib.requirement import Requirement, ValidationResult

# region code extraction


def _score_code_block(code: str, context_text: str = "") -> int:
    """Score a code block to determine if it's likely the main answer.

    Returns higher score for blocks that:
    - Are longer (more substantial)
    - Contain function/class definitions (not just tests)
    - Don't look like test code
    - Appear after positive language cues
    """
    score = 0

    # Length bonus (longer is usually the main implementation)
    score += len(code) // 50  # 1 point per ~50 chars

    # Has function/class definitions (not just calls)
    if "def " in code:
        score += 10
    if "class " in code:
        score += 10

    # Penalty for test code indicators
    test_indicators = ["unittest", "pytest", "assert ", "test_", "TestCase", "def test"]
    if any(indicator in code for indicator in test_indicators):
        score -= 20

    # Bonus if preceded by positive language
    if context_text:
        positive_phrases = ["here's the", "correct", "solution", "final", "here is the"]
        # Check text before this code block
        if any(phrase in context_text.lower() for phrase in positive_phrases):
            score += 5

    # Penalty for being preceded by negative language
    if context_text:
        negative_phrases = ["wrong", "bad", "don't", "avoid", "incorrect", "won't work"]
        if any(phrase in context_text.lower() for phrase in negative_phrases):
            score -= 15

    return score


def extract_python_code(text: str) -> str | None:
    """Extract Python code from markdown code blocks or plain text.

    Uses intelligent extraction strategy:
    1. Finds all ```python...``` blocks
    2. Scores each block (prefers longer, non-test code after positive cues)
    3. Returns highest-scoring block
    4. Falls back to generic blocks or raw text

    Returns None if no Python code found.
    """
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

            score = _score_code_block(match, context_before)

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


def _has_python_code_listing(ctx: Context) -> ValidationResult:
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


class HasPythonCodeListing(Requirement):
    """Verifies that the output contains a valid Python code listing."""

    def __init__(self):
        super().__init__(
            description="The result should contain a Python code listing in markdown format or as plain code.",
            validation_fn=_has_python_code_listing,
            check_only=True,
        )


# endregion

# region syntax validation


def _python_code_parses(ctx: Context) -> ValidationResult:
    """Validate that extracted Python code is syntactically valid using AST."""
    # First extract the code
    extraction_result = _has_python_code_listing(ctx)
    if not extraction_result.as_bool():
        return ValidationResult(
            result=False,
            reason=extraction_result.reason or "Could not extract Python code",
        )

    code = extraction_result.reason  # Code is stored in reason field
    assert code is not None

    try:
        ast.parse(code)
        return ValidationResult(result=True, reason="Python code parses successfully")
    except SyntaxError as e:
        return ValidationResult(
            result=False, reason=f"Syntax error at line {e.lineno}: {e.msg}"
        )
    except Exception as e:
        return ValidationResult(result=False, reason=f"Parse error: {e!s}")


class PythonCodeParses(Requirement):
    """Verifies that the Python code is syntactically valid."""

    def __init__(self):
        super().__init__(
            description="The Python code should be syntactically valid and parseable.",
            validation_fn=_python_code_parses,
            check_only=True,
        )


# endregion

# region import validation


def get_imported_modules(code: str) -> list[str]:
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


def is_module_available(module_name: str, venv_path: str | None = None) -> bool:
    """Check if a module is available in the system or specified venv."""
    if venv_path:
        # Check in specified venv using pip list
        try:
            result = subprocess.run(
                [f"{venv_path}/bin/python", "-m", "pip", "list", "--format=freeze"],
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
        return importlib.util.find_spec(module_name) is not None


def _python_valid_imports(
    ctx: Context, venv_path: str | None = None
) -> ValidationResult:
    """Validate that all imports in Python code are available."""
    # First extract and parse the code
    extraction_result = _has_python_code_listing(ctx)
    if not extraction_result.as_bool():
        return ValidationResult(
            result=False, reason="Could not extract Python code for import validation"
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

    modules = get_imported_modules(code)
    if not modules:
        # No imports is valid
        return ValidationResult(result=True, reason="No imports to validate")

    unavailable_modules = []
    for module in modules:
        if not is_module_available(module, venv_path):
            unavailable_modules.append(module)

    if unavailable_modules:
        return ValidationResult(
            result=False,
            reason=f"Unavailable modules: {', '.join(unavailable_modules)}",
        )

    return ValidationResult(
        result=True, reason=f"All imports valid: {', '.join(modules)}"
    )


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
            validation_fn=lambda ctx: _python_valid_imports(ctx, self._venv_path),
            check_only=True,
        )


# endregion

# region execution validation


def _python_executes_without_error(ctx: Context, timeout: int = 5) -> ValidationResult:
    """Validate that Python code executes without raising exceptions."""
    # First extract the code
    extraction_result = _has_python_code_listing(ctx)
    if not extraction_result.as_bool():
        return ValidationResult(
            result=False, reason="Could not extract Python code for execution"
        )

    code = extraction_result.reason
    assert code is not None

    # Create temporary file and execute
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(code)
        temp_file = f.name

    try:
        result = subprocess.run(
            [sys.executable, temp_file], capture_output=True, text=True, timeout=timeout
        )

        if result.returncode == 0:
            return ValidationResult(result=True, reason="Code executed successfully")
        else:
            return ValidationResult(
                result=False,
                reason=f"Execution failed with error: {result.stderr[:200]}",
            )
    except subprocess.TimeoutExpired:
        return ValidationResult(
            result=False, reason=f"Execution timed out after {timeout} seconds"
        )
    except Exception as e:
        return ValidationResult(result=False, reason=f"Execution error: {e!s}")
    finally:
        # Clean up temp file
        try:
            Path(temp_file).unlink()
        except Exception:
            pass


class PythonExecutesWithoutError(Requirement):
    """Verifies that Python code runs without raising exceptions."""

    def __init__(self, timeout: int = 5):
        """Initialize execution validator.

        Args:
            timeout: Maximum seconds to allow code to run before timing out.
        """
        self._timeout = timeout
        super().__init__(
            description=f"The Python code should execute without errors (timeout: {timeout}s).",
            validation_fn=lambda ctx: _python_executes_without_error(
                ctx, self._timeout
            ),
            check_only=True,
        )


# endregion

# region structural validation


def _python_has_function_def(ctx: Context) -> ValidationResult:
    """Validate that Python code contains at least one function definition."""
    extraction_result = _has_python_code_listing(ctx)
    if not extraction_result.as_bool():
        return ValidationResult(result=False, reason="Could not extract Python code")

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


class PythonHasFunctionDef(Requirement):
    """Verifies that Python code contains at least one function definition."""

    def __init__(self):
        super().__init__(
            description="The Python code should define at least one function.",
            validation_fn=_python_has_function_def,
            check_only=True,
        )


# endregion

# region completeness validation


def _python_has_class_def(ctx: Context) -> ValidationResult:
    """Validate that Python code contains at least one class definition."""
    extraction_result = _has_python_code_listing(ctx)
    if not extraction_result.as_bool():
        return ValidationResult(result=False, reason="Could not extract Python code")

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


class PythonHasClassDef(Requirement):
    """Verifies that Python code contains at least one class definition."""

    def __init__(self):
        super().__init__(
            description="The Python code should define at least one class.",
            validation_fn=_python_has_class_def,
            check_only=True,
        )


# endregion

# region correctness validation


def _python_matches_examples(
    ctx: Context, function_name: str, examples: list[tuple[dict, Any]]
) -> ValidationResult:
    """Validate that Python function produces correct outputs for given examples.

    Args:
        ctx: Context containing the code
        function_name: Name of the function to test
        examples: List of (input_kwargs, expected_output) tuples

    Returns:
        ValidationResult indicating if all examples passed
    """
    extraction_result = _has_python_code_listing(ctx)
    if not extraction_result.as_bool():
        return ValidationResult(result=False, reason="Could not extract Python code")

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
        return ValidationResult(result=False, reason=f"Code execution failed: {e!s}")

    # Check if function exists
    if function_name not in namespace:
        return ValidationResult(
            result=False, reason=f"Function '{function_name}' not found in code"
        )

    func = namespace[function_name]
    if not callable(func):
        return ValidationResult(
            result=False, reason=f"'{function_name}' is not callable"
        )

    # Test all examples
    failed_examples = []
    for i, (inputs, expected) in enumerate(examples):
        try:
            result = func(**inputs)
            if result != expected:
                failed_examples.append(
                    f"Example {i + 1}: {function_name}({inputs}) = {result}, expected {expected}"
                )
        except Exception as e:
            failed_examples.append(
                f"Example {i + 1}: {function_name}({inputs}) raised {type(e).__name__}: {e!s}"
            )

    if failed_examples:
        return ValidationResult(
            result=False,
            reason=f"Failed {len(failed_examples)}/{len(examples)} examples:\n"
            + "\n".join(failed_examples),
        )

    return ValidationResult(
        result=True, reason=f"All {len(examples)} examples passed", score=1.0
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
            validation_fn=lambda ctx: _python_matches_examples(
                ctx, self._function_name, self._examples
            ),
            check_only=True,
        )


# endregion
