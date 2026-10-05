# pytest: skip
"""Helper functions for Qiskit code validation.

This module provides utilities for extracting code from markdown and validating
Qiskit code against migration rules using the flake8-qiskit-migration plugin,
against a benchmark problem's test suite, and against LintQ.
"""

import ast
import ctypes
import json
import os
import re
import subprocess
import tempfile
import threading
import traceback
from typing import Any

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


def validate_correctness(problem: dict[str, Any], code: str) -> tuple[bool, str]:
    """Run `code` against a benchmark problem's own test suite.

    Adapted from the Qiskit HumanEval test harness (Copyright IBM, Apache-2.0).
    Supports Qiskit HumanEval (`check(candidate)`) and QuantumKatas
    (`test_<entry_point>()`). Warning: this `exec()`s untrusted code; it is not a sandbox.

    Args:
        problem: Problem dict with `prompt`, `test` and `entry_point` keys
        code: Candidate solution

    Returns:
        Tuple of (is_valid, error_message)
    """
    # Code-stub prompts are prepended; natural-language prompts (QHE hard) are not.
    prompt = problem["prompt"]
    is_stub = prompt.lstrip().startswith(("from ", "import ", "def ", "#"))
    entry = problem["entry_point"]
    program = f"{prompt if is_stub else ''}\n{code}\n{problem['test']}"

    # Mellea validates on a worker thread, so the timeout is raised via the C API.
    tid = ctypes.c_ulong(threading.get_ident())
    timer = threading.Timer(
        30,
        ctypes.pythonapi.PyThreadState_SetAsyncExc,
        (tid, ctypes.py_object(TimeoutError)),
    )
    cwd = os.getcwd()
    os.environ["MPLBACKEND"] = "Agg"
    with tempfile.TemporaryDirectory() as tmp:
        os.chdir(tmp)  # some tasks write files
        timer.start()
        try:
            ns: dict[str, Any] = {}
            exec(program, ns)
            if entry not in ns:
                return False, f"Entry point `{entry}` not defined"
            test = ns.get("check") or ns.get(f"test_{entry}")
            if test is None:
                return False, f"No check() or test_{entry}() in the test code"
            if "check" in ns:
                test(ns[entry])
            else:
                test()
            return True, ""
        except TimeoutError:
            print("Validation failed: Execution timed out after 30s")
            return False, "Execution timed out after 30s"
        except AssertionError as e:
            print(f"Validation failed: Test assertion failed: {e}")
            return False, f"Test assertion failed: {e}"
        except Exception as e:
            print(f"{type(e).__name__}: {e}\n{traceback.format_exc()}")
            return False, f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
        finally:
            timer.cancel()
            os.chdir(cwd)


def validate_lintq(code: str) -> tuple[bool, str]:
    """Run the LintQ CodeQL queries over `code`.

    Requires the `codeql` CLI and `LINTQ_DIR` pointing at a LintQ checkout (see README.md).

    Args:
        code: Python source to analyse

    Returns:
        Tuple of (is_valid, error_message), one `[rule-id] message` line per finding
    """
    lintq = os.environ["LINTQ_DIR"]
    with tempfile.TemporaryDirectory() as tmp:
        os.makedirs(f"{tmp}/src")
        with open(f"{tmp}/src/candidate.py", "w") as f:
            f.write(code)
        for cmd in (
            [
                "database",
                "create",
                f"{tmp}/db",
                "--language=python",
                f"--source-root={tmp}/src",
            ],
            [
                "database",
                "analyze",
                f"{tmp}/db",
                f"{lintq}/LintQ-all.qls",
                "--format=sarifv2.1.0",
                f"--output={tmp}/out.sarif",
                f"--additional-packs={lintq}/qlint/codeql/src:{lintq}/qlint/codeql/lib",
            ],
        ):
            subprocess.run(["codeql", *cmd], check=True, capture_output=True)
        with open(f"{tmp}/out.sarif") as f:
            results = [r for run in json.load(f)["runs"] for r in run["results"]]

    if not results:
        return True, ""
    findings = "\n".join(f"[{r['ruleId']}] {r['message']['text']}" for r in results)
    print(f"Validation failed: LintQ warnings:\n{findings}")
    return False, f"LintQ warnings:\n{findings}"
