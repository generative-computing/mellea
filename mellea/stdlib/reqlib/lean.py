from __future__ import annotations

import os
import subprocess
import uuid
import re
import unicodedata
from difflib import unified_diff
from typing import List, Dict, Any, Tuple
from mellea.stdlib.requirement import Requirement, ValidationResult, simple_validate

def extract_lean_code(text: str) -> str | None:
    """Extract Python code from markdown code blocks or plain text."""
    python_block_pattern = r"```lean(?:4|)\s*\n(.*?)```"
    matches = re.findall(python_block_pattern, text, re.DOTALL)

    if matches and len(matches) == 1:
            print("Extracted code:\n", matches[0].strip())
            return matches[0].strip()

    return None

class HasLeanCode(Requirement):
    """
    Requirement that ensures the model output contains valid Lean code.

    This requirement extracts Lean code from the text (e.g. markdown code blocks)
    and validates that at least one block of Lean source was detected.
    It does not verify the correctness or safety of the code — only that
    some Lean code is present.

    Notes
    -----
    - This class uses `extract_lean_code` to find code blocks.
    """
    def __init__(self):
        """Check that the output contains lean code."""
        def val_func(x):
            x = extract_lean_code(x)
            return isinstance(x, str)

        super().__init__(
            description="The output should contain lean code",
            validation_fn=simple_validate(lambda x: isinstance(extract_lean_code(x), str)),
            check_only=True,
        )

def strip_comments_and_collect_strings(code: str):
    """Normalize and remove comments from lean code for safety check."""
    if code is None: return None
    code = unicodedata.normalize("NFKC", code)

    def remove_block_comments(s: str):
        while True:
            start = s.find("/-")
            if start == -1:
                break
            end = s.find("-/", start + 2)
            if end == -1:
                s = s[:start]
                break
            s = s[:start] + s[end+2:]
        return s

    code_noblock = remove_block_comments(code)

    strings = []
    out = []
    i = 0
    n = len(code_noblock)
    while i < n:
        ch = code_noblock[i]
        if ch == '"':
            j = i + 1
            buf = []
            while j < n:
                if code_noblock[j] == '\\':
                    if j+1 < n:
                        buf.append(code_noblock[j:j+2])
                        j += 2
                    else:
                        j += 1
                elif code_noblock[j] == '"':
                    j += 1
                    break
                else:
                    buf.append(code_noblock[j])
                    j += 1
            strings.append(''.join(buf))
            out.append(' ')
            i = j
            continue
        elif code_noblock[i:i+2] == '--':
            nl = code_noblock.find('\n', i+2)
            if nl == -1:
                break
            i = nl + 1
            out.append('\n')
            continue
        else:
            out.append(ch)
            i += 1
    cleaned = ''.join(out)
    return cleaned, strings

def scan_lean_for_danger(lean_code: str) -> Dict:
    """
    Static heuristic scanner.
    Returns a dict with keys:
      - safe (bool)
      - findings (list of dict: {type, pattern, context_line, lineno})
    """
    stripped = strip_comments_and_collect_strings(lean_code)
    if stripped is None: return None
    cleaned, strings = stripped
    findings = []

    suspect_patterns = {
        "run_cmd": r'\brun_cmd\b',
        "hash_eval": r'(?m)^\s*#(eval|check|reduce)\b',
        "io_module": r'\bIO\b|\bSystem\b|\bProcess\b|\bFS\b|\bLean\.Elab\b|\bLean\.Meta\b',
        "unsafe": r'\bunsafe\b|\bpartial\s+def\b',
        "attrs_elab": r'@\[[^\]]*(command_elab|builtin_attribute|implemented_by|always_inline)[^\]]*\]',
        "macro_elab": r'\bmacro\b|\belab\b|\belab_rules\b',
        "main_io": r'\bdef\s+main\s*:\s*IO\b|\bdef\s+main\s*:\s*IO\s+Unit\b',
        "import_danger": r'^\s*import\s+.*\b(IO|System|Lean\.Elab|Lean\.Meta|Process|FS)\b',
        "open_danger": r'^\s*open\s+.*\b(IO|System|Process|FS|Lean\.Elab|Lean\.Meta)\b',
    }

    for kind, pat in suspect_patterns.items():
        for m in re.finditer(pat, cleaned, flags=re.MULTILINE):
            start = m.start()
            lineno = cleaned.count('\n', 0, start) + 1
            line = cleaned.splitlines()[lineno-1] if lineno-1 < len(cleaned.splitlines()) else ""
            findings.append({
                "type": kind,
                "pattern": m.group(0),
                "lineno": lineno,
                "line": line.strip()
            })

    shell_keywords = ["rm ", "rm -rf", "sudo ", "curl ", "wget ", "sh -c", "bash -c", "|", "&&", ";", "nc ", "nc -l", "curl(", "wget("]
    for s in strings:
        low = s.lower()
        for kw in shell_keywords:
            if kw in low:
                idx = lean_code.find(s)
                lineno = lean_code.count('\n', 0, idx) + 1 if idx != -1 else None
                findings.append({
                    "type": "suspicious_string",
                    "pattern": kw,
                    "lineno": lineno,
                    "string_prefix": (s[:120] + ('...' if len(s) > 120 else ''))
                })
                break

    if findings:
        safe = False
    else:
        safe = True

    return {
        "safe": safe,
        "findings": findings
    }

class LeanCodeClearOfUnsafePrimitives(Requirement):
    """
    Requirement that checks Lean code for unsafe or potentially dangerous constructs.

    This includes static detection of expressions that could trigger file system or
    process access, such as `IO`, `System`, `run_cmd`, or `unsafe` definitions.

    Validation
    -----------
    The validation function calls `scan_lean_for_danger(extract_lean_code(x))`
    and passes if `"safe"` is True in the returned dictionary.

    Returns
    -------
    ValidationResult
        True if no unsafe patterns are found, False otherwise.
    """
    def __init__(self):
        """Check that the output is free of unsafe commands and primitives."""
        def val_func(x):
            x = extract_lean_code(x)
            x = scan_lean_for_danger(x)
            if not isinstance(x, dict): return False
            return x.get("safe", False)

        super().__init__(
            description="The output should be free of unsafe commands and primitives.",
            validation_fn=simple_validate(val_func),
            check_only=True,
        )

def detect_lean_cheating_core(lean_code: str) -> Dict[str, Any]:
    """
    Scan Lean source for cheating constructs, excluding metaprogramming, compiled artifacts, and IO/run_cmd.

    Checks for:
      - `sorry`, `admit` (and patterns like `by sorry`, `have ... := sorry`)
      - `axiom`, `postulate`, `constant`
      - `unsafe`, `partial def`

    Returns a dict:
      {
        "safe": bool,            # True if no cheating constructs found
        "findings": [            # list of matches with context
           { "type": str, "pattern": str, "lineno": int, "line": str }
        ]
      }
    """
    stripped = strip_comments_and_collect_strings(lean_code)
    if stripped is None: return None
    cleaned, strings = stripped
    findings: List[Dict[str, Any]] = []

    # Patterns (word-boundary aware where sensible)
    patterns = {
        "sorry": r'\bsorry\b',
        "admit": r'\badmit\b',
        "axiom": r'\baxiom\b',
        "postulate": r'\bpostulate\b',
        "constant": r'\bconstant\b',
        "unsafe": r'\bunsafe\b',
        "partial_def": r'\bpartial\s+def\b',
        # compound patterns
        "by_sorry": r'(?m)\bby\s+sorry\b',
        "by_admit": r'(?m)\bby\s+admit\b',
        "have_sorry": r'(?m)\bhave\b[^\n]*:=\s*sorry\b',
        "let_sorry": r'(?m)\blet\b[^\n]*:=\s*sorry\b',
        "show_sorry": r'(?m)\bshow\b[^\n]*:=\s*sorry\b',
    }

    # Run search for each pattern
    for kind, pat in patterns.items():
        for m in re.finditer(pat, cleaned):
            start = m.start()
            lineno = cleaned.count('\n', 0, start) + 1
            # line extraction (safe even if lineno out of range)
            lines = cleaned.splitlines()
            line_text = lines[lineno-1].strip() if 0 <= lineno-1 < len(lines) else ""
            findings.append({
                "type": kind,
                "pattern": m.group(0),
                "lineno": lineno,
                "line": line_text
            })

    safe = len(findings) == 0
    return {"safe": safe, "findings": findings}

class LeanCodeProvesWithoutCheating(Requirement):
    def __init__(self):
        """Check that lean code in the output does not cheat in the proof."""
        def val_func(x):
            x = extract_lean_code(x)
            x = detect_lean_cheating_core(x)
            if not isinstance(x, dict): return False
            return x.get("safe", False)

        super().__init__(
            description="The output should contain lean code that constitutes a proof without cheating.",
            validation_fn=simple_validate(val_func),
            check_only=True,
        )

def verify_lean_code(code: str, lean_project_path: str = None) -> tuple[bool, str]:
    if code is None: return False, "No code found"

    danger_check_results = scan_lean_for_danger(code)
    try:
        assert danger_check_results["safe"]
    except:
        return False, f"Code deemed not safe. Findings: {danger_check_results["findings"]}"

    if lean_project_path is None:
        lean_project_path = os.environ.get("LEAN_PROJECT_PATH")

    if lean_project_path is None:
        raise ValueError(
            "lean_project_path not provided. Either pass it as an argument or set LEAN_PROJECT_PATH environment variable."
        )

    temp_dir = os.path.join(lean_project_path, "temp_verify")
    os.makedirs(temp_dir, exist_ok=True)
    filename = f"temp_{uuid.uuid4().hex}.lean"
    lean_file = os.path.join(temp_dir, filename)

    with open(lean_file, "w") as f:
        f.write(code)

    try:
        relative_path = os.path.relpath(lean_file, lean_project_path)
        result = subprocess.run(
            ["lake", "env", "lean", relative_path],
            cwd=lean_project_path,
            capture_output=True,
            text=True
        )
        print(result.returncode == 0, result.stdout + result.stderr)
        return result.returncode == 0, result.stdout + result.stderr
    finally:
        os.remove(lean_file)

class LeanCodeVerifies(Requirement):
    """
    Requirement that ensures extracted Lean code typechecks and verifies successfully.

    This validator writes the extracted Lean source to a temporary file within a
    provided Lean project, then invokes Lean through the Lake build tool to
    check that the code compiles and verifies.

    Parameters
    ----------
    lean_project_path : str
        Absolute path to a Lean project root (must contain a valid `lakefile.lean`).

    Validation
    ----------
    - Extracts Lean code using `extract_lean_code`.
    - Calls `verify_lean_code`, which performs:
        1. Static safety check (`scan_lean_for_danger`).
        2. On-disk compilation via `lake env lean`.
    - Validation passes if Lean returns exit code 0.

    Returns
    -------
    ValidationResult
        True if verification succeeds, False otherwise.

    Raises
    ------
    ValueError
        If `lean_project_path` is not provided and not set in `LEAN_PROJECT_PATH`.
    """
    def __init__(self, lean_project_path):
        """Check that the output contains lean code that can be verified."""
        def val_func(x):
            x = extract_lean_code(x)
            x = verify_lean_code(x, lean_project_path)
            if not isinstance(x, tuple): return False
            return x

        super().__init__(
            description="The output should contain lean code that can be verified",
            validation_fn=simple_validate(val_func),
            check_only=True,
        )

def check_num_lines(lean_code: str) -> int:
    """Check the number of non-comment, non-empty lines in the Lean code."""
    stripped = strip_comments_and_collect_strings(lean_code)
    if stripped is None: return None
    cleaned, _ = stripped
    nonempty_lines = [
        line for line in cleaned.splitlines()
        if line.strip()  # non-empty
    ]
    return len(nonempty_lines)

class LeanCodeWithinLengthLimit(Requirement):
    """
    Requirement that enforces a maximum number of Lean source lines.

    This is a structural constraint ensuring model outputs stay concise and
    within a given length bound.

    Parameters
    ----------
    max_num_lines : int
        The maximum allowed number of non-empty, non-comment lines of Lean code.

    Validation
    ----------
    - Extracts Lean code via `extract_lean_code`.
    - Counts non-empty lines after stripping comments using
      `strip_comments_and_collect_strings`.
    - Passes if the count is less than or equal to `max_num_lines`.

    Returns
    -------
    ValidationResult
        True if within the limit, False otherwise.
    """
    def __init__(self, max_num_lines: int):
        """Check that the output contains lean code that is within the specified number of lines."""
        def val_func(x):
            x = extract_lean_code(x)
            x = check_num_lines(x)
            if not isinstance(x, int): return False
            return x <= max_num_lines

        super().__init__(
            description=f"The output should contain lean code with at most {max_num_lines} lines.",
            validation_fn=simple_validate(val_func),
            check_only=True,
        )

def verify_lean_edit(original: str, edited: str) -> bool:
    if not isinstance(original, str) or not isinstance(edited, str):
        return None

    # quick check: 'sorry' must be gone
    if 'sorry' in edited:
        print("'sorry' still present.")
        return False

    # find theorem start in each file
    def find_theorem_prefix(s: str):
        m = re.search(r'\btheorem\b', s)
        if not m:
            return None
        start = m.start()
        # find first ':=' after theorem start
        idx_assign = s.find(':=', start)
        if idx_assign == -1:
            return None
        # keep everything *before* the ':=' as the theorem header/prefix
        prefix = s[:idx_assign].strip()
        # also return the rest-of-file after the ':=' if caller needs it
        return prefix

    orig_prefix = find_theorem_prefix(original)
    edit_prefix = find_theorem_prefix(edited)
    if orig_prefix is None or edit_prefix is None:
        print("Theorem not found in one of the versions.")
        return False

    # normalize prefixes (whitespace-insensitive)
    def norm_whitespace(s: str):
        return re.sub(r'\s+', ' ', s).strip()

    if norm_whitespace(orig_prefix) != norm_whitespace(edit_prefix):
        print("Theorem statement changed.")
        # show a helpful diff for debugging
        print("ORIGINAL HEADER:\n", orig_prefix)
        print("EDITED  HEADER:\n", edit_prefix)
        return False

    # Now compare the rest of the code *excluding proof content*.
    # We'll remove everything from the first ':=' after 'theorem' to the end,
    # leaving only imports + theorem header + any surrounding declarations.
    def remove_proof_part(s: str):
        m = re.search(r'\btheorem\b', s)
        if not m:
            return s
        idx_assign = s.find(':=', m.start())
        if idx_assign == -1:
            return s
        # keep everything before ':=' and everything before the theorem start (i.e., top-level prelude)
        before_theorem = s[:m.start()]
        theorem_header = s[m.start():idx_assign]
        return (before_theorem + theorem_header).strip()

    orig_noproof = remove_proof_part(original)
    edit_noproof = remove_proof_part(edited)

    # normalize lines: strip, remove blank lines
    def normalize_lines(s: str):
        lines = [line.strip() for line in s.strip().splitlines()]
        return [l for l in lines if l]

    orig_lines = normalize_lines(orig_noproof)
    edit_lines = normalize_lines(edit_noproof)

    # ignore import lines (they may be added)
    orig_nonimport = [l for l in orig_lines if not l.startswith('import')]
    edit_nonimport = [l for l in edit_lines if not l.startswith('import')]

    # compute a diff and ignore trivial whitespace-only differences
    diff = list(unified_diff(orig_nonimport, edit_nonimport, lineterm=''))
    real_diff = [
        d for d in diff
        if d.startswith(('+', '-')) and not d in ('+', '-')
    ]

    if real_diff:
        print("Code structure changed outside proof:")
        print("\n".join(real_diff))
        return False

    print("Verification passed.")
    return True

class LeanCodePreservesTheorem(Requirement):
    def __init__(self, original):
        """Check that the output contains lean code that keeps the given theorem unchanged."""
        def val_func(x):
            x = extract_lean_code(x)
            x = verify_lean_edit(original, x)
            if not isinstance(x, bool): return False
            return x

        super().__init__(
            description=f"The output should contain lean code that keeps the given theorem unchanged.",
            validation_fn=simple_validate(val_func),
            check_only=True,
        )
