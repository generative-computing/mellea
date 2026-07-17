"""Extensible bash security pattern detection framework.

Defines abstract base class and concrete implementations for bash security checks.
New patterns can be added without modifying core validation logic.
"""

from abc import ABC, abstractmethod

from ._bash_audit import record_bash_violation
from ._bash_guardrails import COMMAND_RULES, SHELL_OPERATOR_RULES


class BashSecurityPattern(ABC):
    """Base class for pattern-based security checks.

    Each pattern detects a specific class of dangerous usage (e.g., shell operators,
    code execution paths, dangerous commands). Patterns can be composed and registered
    in a central registry for modular validation.
    """

    @abstractmethod
    def check(self, argv: list[str]) -> tuple[bool, str]:
        """Check if a command violates this security pattern.

        Args:
            argv: Tokenized command arguments (from shlex.split()).

        Returns:
            Tuple of (is_dangerous, reason_message). If is_dangerous is True,
            reason_message explains why the pattern was violated.
        """


class DangerousCommandPattern(BashSecurityPattern):
    """Detects usage of dangerous commands like sudo, passwd, useradd, etc.

    Uses COMMAND_RULES from _bash_guardrails for authoritative definitions.
    """

    # Default fallback for patterns without specific per-command metadata
    category = "dangerous_command"
    severity = "HIGH"

    def check(self, argv: list[str]) -> tuple[bool, str]:
        """Check if command is in the dangerous commands list."""
        if not argv:
            return False, ""

        cmd = argv[0].split("/")[-1]  # Get basename

        if cmd in COMMAND_RULES:
            # Special case: interactive shells are only dangerous with -i flag
            if cmd in ("bash", "sh", "zsh", "ksh", "tcsh"):
                if any(arg in ("-i", "--interactive", "-l", "--login") for arg in argv):
                    return True, f"Interactive shell '{cmd}' is not allowed"
            else:
                return True, f"Command '{cmd}' is not allowed"

        return False, ""

    def get_metadata(self, argv: list[str]) -> tuple[str, str]:
        """Get category and severity for a dangerous command.

        Args:
            argv: Tokenized command arguments.

        Returns:
            Tuple of (category, severity) strings from COMMAND_RULES.
        """
        if not argv:
            return "unknown", "MEDIUM"

        cmd = argv[0].split("/")[-1]
        if cmd in COMMAND_RULES:
            rule = COMMAND_RULES[cmd]
            return rule.category.value, rule.severity.value.upper()

        return "unknown", "MEDIUM"


class ShellOperatorPattern(BashSecurityPattern):
    """Detects shell operators: |, >, &&, ;, etc.

    These operators require shell interpretation and can enable complex attacks.
    Detected after shlex.split(), so they appear as standalone tokens or prefixes.
    """

    category = "shell_operator"
    severity = "HIGH"

    def check(self, argv: list[str]) -> tuple[bool, str]:
        """Check for shell operators in argv."""
        if not argv:
            return False, ""

        shell_operators = {"<", ">", "|", ";", "&", "&&", "||", ">>", ">&", "<<", "|&"}

        for arg in argv:
            # Exact match: standalone operators like "&&", "|"
            # Exception: standalone ";" is safe (used by find -exec, etc.)
            if arg in shell_operators:
                if arg == ";":
                    # Standalone semicolon is safe when passed via argv (not shell interpretation)
                    continue
                rule = SHELL_OPERATOR_RULES.get(arg)
                reason = rule.rationale if rule else "Shell operator is not allowed"
                return True, reason

            # Prefix match: operators with content like ">&2", ">file"
            for op in shell_operators:
                if arg.startswith(op) and len(arg) > len(op):
                    # Skip if this is the semicolon substring check (handled below)
                    if op == ";":
                        continue
                    rule = SHELL_OPERATOR_RULES.get(op)
                    reason = rule.rationale if rule else "Shell operator is not allowed"
                    return True, reason

            # Semicolon: reject any arg containing semicolon (except standalone `;` from find -exec)
            # After shlex.split(), escaped semicolons become bare `;` tokens, making them safe
            # because they're passed to subprocess as argv elements, not shell strings.
            if ";" in arg and arg != ";":
                return True, "Command chaining (;) is not allowed"

        return False, ""


class CommandSubstitutionPattern(BashSecurityPattern):
    """Detects command substitution: $(cmd), `cmd`, ${var}, etc.

    These patterns allow arbitrary code execution and bypass argv parsing.
    """

    category = "code_execution"
    severity = "CRITICAL"

    def check(self, argv: list[str]) -> tuple[bool, str]:
        """Check for command substitution patterns."""
        if not argv:
            return False, ""

        for arg in argv:
            if "`" in arg or "$(" in arg:
                return True, "Command substitution is not allowed"
            if "${" in arg:
                return True, "Variable expansion is not allowed"

        return False, ""


class CodeExecutionPattern(BashSecurityPattern):
    """Detects interpreter code execution paths: python -c, bash -c, etc.

    These flags cause interpreters to treat arguments as source code, bypassing argv parsing.
    """

    category = "code_execution"
    severity = "CRITICAL"

    def check(self, argv: list[str]) -> tuple[bool, str]:
        """Check for interpreter indirection (code execution flags)."""
        if not argv:
            return False, ""

        cmd = argv[0].split("/")[-1]

        code_execution_interpreters = {
            "python": ("-c", "-m"),
            "python3": ("-c", "-m"),
            "python2": ("-c", "-m"),
            "perl": ("-e", "-E"),
            "ruby": ("-e", "-E"),
            "node": ("-e", "--eval"),
            "bash": ("-c",),
            "sh": ("-c",),
            "zsh": ("-c",),
            "ksh": ("-c",),
            "tcsh": ("-c",),
        }

        if cmd in code_execution_interpreters:
            dangerous_flags = code_execution_interpreters[cmd]
            if any(arg in dangerous_flags for arg in argv):
                return (
                    True,
                    f"Interpreter code execution ('{cmd} {' '.join(dangerous_flags)}') is not allowed",
                )

        return False, ""


class DestructiveGitPattern(BashSecurityPattern):
    """Detects dangerous git operations: push --force, reset --hard, clean -f, etc.

    These operations have high regret cost (lost commits, data loss).
    """

    category = "destructive"
    severity = "HIGH"

    def check(self, argv: list[str]) -> tuple[bool, str]:
        """Check for destructive git operations."""
        if not argv or argv[0].split("/")[-1] != "git":
            return False, ""

        # git push --force (including variants like --force-with-lease, --force-if-includes)
        if "push" in argv:
            for arg in argv:
                if arg == "-f" or arg.startswith("--force"):
                    return True, "Destructive git operation is not allowed"

        # git reset --hard
        if "reset" in argv and "--hard" in argv:
            return True, "Destructive git operation is not allowed"

        # git clean -f/-d (including long forms like --force and --force-*)
        if "clean" in argv:
            for arg in argv:
                if arg in ("-f", "-d", "-fd", "-df"):
                    return True, "Destructive git operation is not allowed"
                # Long form flags: --force, --force-*, --directory
                if arg.startswith("--force") or arg == "--directory":
                    return True, "Destructive git operation is not allowed"
                # Short combined flags like -fd, -ddf, etc. (not starting with --)
                if arg.startswith("-") and not arg.startswith("--"):
                    if "f" in arg or "d" in arg:
                        return True, "Destructive git operation is not allowed"

        return False, ""


class DestructiveRmPattern(BashSecurityPattern):
    """Detects destructive rm operations: rm -rf, rm -r, etc.

    Recursive deletion is the highest-regret operation for filesystem safety.
    """

    category = "destructive"
    severity = "HIGH"

    def check(self, argv: list[str]) -> tuple[bool, str]:
        """Check for destructive rm patterns."""
        if not argv or argv[0].split("/")[-1] != "rm":
            return False, ""

        if any(flag in argv for flag in ("-r", "-rf", "--recursive")):
            return True, "rm with -r or -rf flag is not allowed"

        return False, ""


class DangerousPackageManagerPattern(BashSecurityPattern):
    """Detects dangerous package manager operations with risky flags.

    Package managers with force/recursive flags can perform destructive operations
    or bypass important safety checks.
    """

    category = "destructive"
    severity = "HIGH"

    def check(self, argv: list[str]) -> tuple[bool, str]:
        """Check for dangerous package manager operations."""
        if not argv:
            return False, ""

        cmd = argv[0].split("/")[-1]
        if cmd not in ("apt", "yum"):
            return False, ""

        # Check for dangerous flags: -f (force), -r (recursive)
        dangerous_flags = {"-rf", "-f", "--force", "-r", "--recursive", "--force-all"}
        if any(flag in argv for flag in dangerous_flags):
            return True, f"Command '{cmd}' with dangerous flags is not allowed"

        return False, ""


# Registry of all security patterns. New patterns can be added here.
SECURITY_PATTERNS: list[BashSecurityPattern] = [
    DangerousCommandPattern(),
    ShellOperatorPattern(),
    CommandSubstitutionPattern(),
    CodeExecutionPattern(),
    DestructiveGitPattern(),
    DestructiveRmPattern(),
    DangerousPackageManagerPattern(),
]


def check_all_patterns(
    argv: list[str],
    working_dir: str | None = None,
    allowed_paths: list[str] | None = None,
) -> tuple[bool, str]:
    """Check command against all registered security patterns.

    Args:
        argv: Tokenized command arguments.
        working_dir: Working directory context for audit trail.
        allowed_paths: Allowed paths context for audit trail.

    Returns:
        Tuple of (is_dangerous, reason_message) from the first matching pattern,
        or (False, "") if all patterns pass.
    """
    for pattern in SECURITY_PATTERNS:
        is_dangerous, reason = pattern.check(argv)
        if is_dangerous:
            pattern_name = type(pattern).__name__
            # Use per-command metadata if available (e.g., from COMMAND_RULES)
            if hasattr(pattern, "get_metadata"):
                category, severity = pattern.get_metadata(argv)
            else:
                category = getattr(pattern, "category", "unknown")
                severity = getattr(pattern, "severity", "MEDIUM")
            record_bash_violation(
                command=" ".join(argv),
                argv=argv,
                pattern_name=pattern_name,
                category=category,
                severity=severity,
                reason=reason,
                working_dir=working_dir,
                allowed_paths=allowed_paths,
            )
            return True, reason
    return False, ""


def get_pattern_names() -> list[str]:
    """Get names of all registered security patterns.

    Returns:
        List of pattern class names.
    """
    return [type(pattern).__name__ for pattern in SECURITY_PATTERNS]
