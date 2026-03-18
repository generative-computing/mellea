"""Helper functions for Qiskit code validation.

This module provides utilities for extracting code from markdown and validating
Qiskit code against migration rules using the flake8-qiskit-migration plugin.
"""

import ast
import re

try:
    from flake8_qiskit_migration.plugin import Plugin
except ImportError:
    raise ImportError(
        "flake8-qiskit-migration is required for this example. "
        "Run with: uv run docs/examples/instruct_validate_repair/qiskit_code_validation/qiskit_code_validation.py"
    )


def extract_code_from_markdown(text: str) -> str:
    """Extract code from markdown code block.

    Handles both fenced code blocks (```python or ```) and returns the code content.
    If no code block is found, returns the original text.

    Args:
        text: Text potentially containing markdown code blocks

    Returns:
        Extracted code or original text if no code block found
    """
    # Pattern for fenced code blocks with optional language identifier
    # Matches ```python code``` or ``` code ```
    pattern = r"```(?:python|py)?\s*(.*?)```"

    matches = re.findall(pattern, text, re.DOTALL)

    if matches:
        # Return the first code block found
        return matches[0].strip()

    # If no code block found, return original text stripped
    return text.strip()


def validate_qiskit_migration(md_code: str) -> tuple[bool, str]:
    """Validate code against Qiskit migration rules using flake8-qiskit-migration plugin.

    This function is used as a post-condition validator to check if the generated
    code passes all QKT (Qiskit) migration rules.

    Args:
        md_code: Python code (potentially in markdown format) to validate

    Returns:
        Tuple of (is_valid, error_message) where error_message retains QKT rule codes
        for the repair loop.
    """
    try:
        code = extract_code_from_markdown(md_code)
        tree = ast.parse(code)
        plugin = Plugin(tree)
        errors = list(plugin.run())

        if not errors:
            return True, ""
        else:
            error_messages = []
            for _line, _col, message, _error_type in errors:
                error_messages.append(message)
            error_str = "\n".join(error_messages)
            print(f"Validation failed with {len(errors)} error(s):\n{error_str}")
            return False, error_str

    except SyntaxError as e:
        print(f"Syntax error during validation: {e}")
        return False, f"Invalid Python syntax: {e}"
    except Exception as e:
        print(f"Unexpected validation error: {e}")
        return False, f"Validation error: {e}"


def validate_input_code(prompt: str) -> tuple[bool, str]:
    """Validate any Qiskit code contained in the user's prompt.

    This is used as a pre-condition validation to check if the prompt
    contains code that needs to be fixed or improved.

    Args:
        prompt: User's input prompt (may contain code blocks)

    Returns:
        Tuple of (is_valid, error_message)
    """
    # Try to extract code from the prompt
    extracted_code = extract_code_from_markdown(prompt)

    # If no code block found (extracted == original), skip validation
    if extracted_code == prompt.strip():
        return True, ""

    # Code block found, validate it
    is_valid, error_msg = validate_qiskit_migration(extracted_code)

    if not is_valid:
        # Return the raw fix instructions — no wrapper prefix.
        # The caller (generate_validated_qiskit_code) frames these in the instruct prompt.
        return False, error_msg

    return True, ""
