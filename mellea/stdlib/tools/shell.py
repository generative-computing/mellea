"""Bash shell command execution tool and execution environments for agentic workflows.

Provides ``BashEnvironment`` (abstract base for bash execution) and two concrete
implementations: ``StaticBashEnvironment`` (parse and safety-check only, no execution)
and ``_LocalBashEnvironment`` (subprocess execution in the current shell). All
environments enforce a conservative safety denylist (sudo, rm -rf, git push --force,
system paths, interactive shells). Write operations may also be constrained by
``working_dir`` and ``allowed_paths``.

The top-level ``bash_executor`` (recommended entry point) executes commands locally
with denylist safety checks. Bash executor runs with access to the host environment;
isolation must be provided by the application layer (containers, VMs).

The function is ready to be wrapped as a ``MelleaTool`` instance for ReACT or
other agentic loops.

Security note: The denylist covers inline code execution (e.g., bash -c, python -e) and
dangerous commands in argv. However, it does not prevent execution of pre-existing
script files (e.g., bash script.sh, python script.py), which can execute arbitrary
code from the file. For untrusted inputs, ensure that script files are either absent
or come from a trusted source.
"""

import shlex
import subprocess
from abc import ABC, abstractmethod
from pathlib import Path

from ...core import MelleaLogger
from ._bash_audit import record_bash_violation
from ._bash_patterns import check_all_patterns
from .interpreter import ExecutionResult

logger = MelleaLogger.get_logger()

# Commands that are always dangerous
DANGEROUS_COMMANDS = {
    # Privilege escalation
    "sudo",
    "su",
    "doas",
    # Interactive shells
    "bash",
    "sh",
    "zsh",
    "ksh",
    "tcsh",
    # User/group/password changes
    "passwd",
    "visudo",
    "chsh",
    "chfn",
    "useradd",
    "userdel",
    "usermod",
    "groupadd",
    "groupdel",
    "groupmod",
}

# Safe wrapper commands that can invoke nested commands (e.g., env, timeout).
# Only commands in this set are allowed to have dangerous commands as nested arguments.
# For all other commands, dangerous commands are rejected regardless of nesting.
SAFE_WRAPPER_COMMANDS = {
    "env",  # set environment vars
    "timeout",  # limit execution time
    "nice",  # set process priority
    "nohup",  # ignore SIGHUP
    "stdbuf",  # modify buffering
    "unbuffer",  # alias for stdbuf
    "ionice",  # set I/O priority
}

# System paths that are dangerous to write to
# Includes both standard Linux paths and macOS /private equivalents
# (on macOS, many system paths are symlinks to /private/*)
# NOTE: On macOS, /var/folders/* resolves to /private/var/folders/* (user temp dirs).
# We don't block the entire /private/var tree because that would block legitimate
# writes to temp directories. Instead, we block specific dangerous subdirectories.
DANGEROUS_PATHS = {
    "/",
    "/bin",
    "/sbin",
    "/usr",
    "/lib",
    "/lib64",
    "/etc",
    "/sys",
    "/proc",
    "/boot",
    "/root",
    "/var/log",
    "/var/www",
    # macOS /private equivalents (specific paths, not entire /private/var)
    "/private/etc",
    "/private/var/log",
    "/private/var/www",
    "/private/var/db",
    "/private/var/root",
}

# Dangerous flags that make commands unsafe
DANGEROUS_FLAGS = {"-rf", "-r", "--recursive", "--force", "-f", "--force-all"}

# Maximum output size (10KB per stream)
MAX_OUTPUT_SIZE = 10 * 1024

# Commands that perform filesystem writes (including permissions, links, disk writes)
WRITE_COMMANDS = {
    "rm",  # Delete files
    "touch",  # Create/update files
    "cp",  # Copy files
    "mv",  # Move/rename files
    "mkdir",  # Create directories
    "mkfifo",  # Create named pipes
    "mknod",  # Create special files
    "tee",  # Write to files
    "chmod",  # Change file permissions
    "chown",  # Change file owner
    "chgrp",  # Change file group
    "ln",  # Create symbolic/hard links
    "dd",  # Copy and convert files (byte-level writes)
    "install",  # Install files
    "truncate",  # Truncate file to size
}


def _is_dangerous_command(argv: list[str]) -> tuple[bool, str]:
    """Check if a command is dangerous based on argv analysis.

    Args:
        argv: Tokenized command arguments.

    Returns:
        A tuple of (is_dangerous, reason_message).
    """
    if not argv:
        return False, ""

    # First, check against the pattern framework (comprehensive pattern registry)
    # The pattern framework provides extensible, well-tested security checks
    is_dangerous, reason = check_all_patterns(argv)
    if is_dangerous:
        return True, reason

    cmd = argv[0].split("/")[-1]  # Get basename of command

    # Check for shell metacharacters that would need shell interpretation
    # After shlex.split(), these characters in argv indicate shell operators (not quoted strings)
    # These are only dangerous if they're standalone tokens (e.g., argv[i] == ">>").
    # Substring matches would cause false positives for legitimate patterns like "a&&b".
    shell_operators = {"<", ">", "|", ";", "&", "&&", "||", ">>", ">&", "<<", "|&"}

    for arg in argv:
        # Check for redirect/pipe/logic operators (these are shell operators).
        # Operators can appear as:
        # 1. Standalone tokens: arg == "&&" (caught by exact match)
        # 2. Operator with argument: arg == ">&2" or arg == ">file" (start with operator)
        # We don't check for substring matches (e.g., "a&&b") to avoid false positives
        # for legitimate patterns like regex or AWK/sed code.

        # Exact match first (standalone operators like "&&", "|", etc.)
        if arg in shell_operators:
            return True, f"Shell operator '{arg}' is not allowed"

        # Check if argument starts with a shell operator (e.g., ">&2", ">file", "2>&1")
        for op in shell_operators:
            if arg.startswith(op) and len(arg) > len(op):
                # Token starts with operator and has additional content (e.g., ">&2", ">file")
                # This is a shell redirection/operator usage
                return True, f"Shell operator '{op}' is not allowed"

        # Note: semicolon is already in shell_operators, but we check for it separately
        # as a substring because semicolon can be dangerous even in patterns.
        # Unlike && or ||, semicolon rarely appears legitimately in arguments.
        if ";" in arg:
            return True, "Command chaining (;) is not allowed"

    # Check for command substitution (backticks, $(...))
    for arg in argv:
        if "`" in arg or "$(" in arg:
            return True, "Command substitution is not allowed"
        # Check for variable expansion patterns
        if "${" in arg:
            return True, "Variable expansion is not allowed"

    # Check for interpreter indirection (code execution via -c, -e, etc.)
    # These allow arbitrary code execution and bypass argv parsing
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

    # Check if any argument is a dangerous command (e.g., env sudo, timeout sudo)
    # Only check positional arguments that are not paths or flag values.
    # Known value-taking flags that consume the next argument (space-separated only).
    # NOTE: -i / --input are intentionally not included. env(1)'s -i takes NO value;
    # other commands' -i/--input (if present) should not mask dangerous commands.
    flag_value_flags = {
        "-c",
        "--config",
        "-f",
        "--file",
        "-o",
        "--output",
        "-d",
        "--dir",
        "-p",
        "--path",
        "-t",
        "--timeout",
        "-w",
        "--wait",
    }
    for i, arg in enumerate(argv[1:], start=1):
        # Skip if this argument is the value for a preceding flag (space-separated)
        # E.g., in "timeout -t 10 sudo", skip "10" (it's the value for -t)
        # But don't skip "sudo" when the flag uses = notation (e.g., --kill-after=1)
        if i > 1 and argv[i - 1] in flag_value_flags:
            continue
        # Skip if argument contains / (it's a path, not a command name)
        if "/" in arg:
            continue
        # Skip if argument starts with - (it's a flag)
        if arg.startswith("-"):
            continue

        arg_cmd = arg.split("/")[-1]

        # Only allow dangerous commands as nested arguments if the top-level
        # command is in SAFE_WRAPPER_COMMANDS. This prevents bypasses like
        # "env -i sudo" or "timeout -t 10 sudo". Without the wrapper allowlist,
        # these would pass validation despite being dangerous.
        if arg_cmd in DANGEROUS_COMMANDS:
            if cmd not in SAFE_WRAPPER_COMMANDS or arg_cmd not in (
                "bash",
                "sh",
                "zsh",
                "ksh",
                "tcsh",
            ):
                # Allow shells as arguments in safe wrappers (e.g., timeout bash script.sh)
                # but reject sudo/su/etc as nested commands
                return True, f"Command '{arg_cmd}' is not allowed as an argument"
            # For shells in safe wrappers, check if they have code-execution flags (-c, -e, etc.)
            # that bypass the denylist (e.g., env bash -c <payload>)
            shell_code_exec_flags = {"-c", "-e"}
            if arg_cmd in ("bash", "sh", "zsh", "ksh", "tcsh"):
                if any(flag in argv for flag in shell_code_exec_flags):
                    return (
                        True,
                        f"Interpreter code execution ('{arg_cmd} -c' or '{arg_cmd} -e' in nested argument) is not allowed",
                    )

        # Check for dangerous nested commands that aren't in DANGEROUS_COMMANDS but
        # have dangerous flags (e.g., "env rm -rf" or "timeout rm -rf").
        # Only apply this check if the wrapper is in SAFE_WRAPPER_COMMANDS.
        if cmd in SAFE_WRAPPER_COMMANDS and arg_cmd == "rm":
            # Check if rm has dangerous flags anywhere in argv after this command
            if any(flag in argv for flag in ["-r", "-rf", "--recursive"]):
                return (
                    True,
                    "Command 'rm' with dangerous flags is not allowed as an argument",
                )

    # Check for dangerous commands
    if cmd in DANGEROUS_COMMANDS:
        if cmd in ("bash", "sh", "zsh", "ksh", "tcsh"):
            if any(arg in ("-i", "--interactive", "-l", "-login") for arg in argv):
                return True, f"Interactive shell '{cmd}' is not allowed"
        else:
            return True, f"Command '{cmd}' is not allowed"

    # Check for dangerous git operations
    if cmd == "git":
        # Check for destructive git operations: push --force, reset --hard, clean -f/-d
        has_destructive_op = False

        # git push --force: check for exact tokens (not substrings)
        if "push" in argv and any(arg == "--force" or arg == "-f" for arg in argv):
            has_destructive_op = True

        # git reset --hard: both must be exact tokens
        if "reset" in argv and "--hard" in argv:
            has_destructive_op = True

        # git clean -f/-d: check for these dangerous flags (exact match or combined like -fd)
        # Avoid false positives: --dry-run should not match -d (use exact or combined check)
        if "clean" in argv:
            for arg in argv:
                # Exact matches: -f, -d, -fd, -df, etc.
                if arg in ("-f", "-d", "-fd", "-df"):
                    has_destructive_op = True
                    break
                # Also check arg startswith for combined flags containing d or f
                # but NOT for things like --dry-run (those start with --)
                if arg.startswith("-") and not arg.startswith("--"):
                    # Short flags: -f, -d, or combinations like -fd, -ddf, etc.
                    if "f" in arg or "d" in arg:
                        has_destructive_op = True
                        break

        if has_destructive_op:
            return True, "Destructive git operation is not allowed"

    # Check for dangerous rm patterns
    if cmd == "rm":
        if "-r" in argv or "-rf" in argv or "--recursive" in argv:
            return True, "rm with -r or -rf flag is not allowed"

    # Check for dangerous flags in specific commands where they are truly dangerous.
    # Note: We don't check cp/mv/make here because:
    # - cp -r: standard way to copy directories recursively
    # - mv -r: standard way to move directories recursively
    # - make -f: standard way to specify a makefile
    # These are not "dangerous" operations in themselves.
    # The real danger is rm -rf (covered above) and git --force (covered above).
    for flag in DANGEROUS_FLAGS:
        if flag in argv:
            # Only apply DANGEROUS_FLAGS check to apt/yum (package managers)
            # These with -f or -r can indeed be risky
            if cmd in ("apt", "yum"):
                return True, f"Command '{cmd}' with '{flag}' flag is not allowed"

    return False, ""


def _is_path_within(path_str: str, allowed_root: str) -> bool:
    """Return whether a resolved path is equal to or nested under an allowed root."""
    if path_str == allowed_root:
        return True
    allowed_root_prefix = (
        allowed_root if allowed_root.endswith("/") else allowed_root + "/"
    )
    return path_str.startswith(allowed_root_prefix)


def _normalize_allowed_path(allowed_path: str) -> str:
    """Normalize an allowlisted path for string-prefix containment checks."""
    return str(Path(allowed_path).expanduser().resolve(strict=False))


def _resolve_allowed_paths(allowed_paths: list[str]) -> list[str]:
    """Normalize allowed path roots for prefix-based containment checks."""
    normalized_paths: list[str] = []
    for allowed_path in allowed_paths:
        try:
            normalized_paths.append(_normalize_allowed_path(allowed_path))
        except Exception:
            logger.warning("Skipping invalid allowed path: %s", allowed_path)
    return normalized_paths


def _is_default_safe_write_path(path_str: str) -> bool:
    """Return whether a path is in a default safe write location."""
    home_dir = str(Path.home())
    return path_str.startswith(("/tmp", "/private/tmp")) or _is_path_within(
        path_str, home_dir
    )


def _check_dangerous_paths(
    argv: list[str], allowed_paths: list[str] | None = None
) -> tuple[bool, str]:
    """Check if command targets dangerous or disallowed filesystem paths.

    Args:
        argv: Tokenized command arguments.
        allowed_paths: Optional additional resolved path roots where writes are allowed.

    Returns:
        A tuple of (has_dangerous_paths, reason_message).
    """
    if not argv or argv[0].split("/")[-1] not in WRITE_COMMANDS:
        return False, ""

    cmd = argv[0].split("/")[-1]
    resolved_allowed_paths = _resolve_allowed_paths(allowed_paths or [])

    # For ln (symlink/hardlink), check both source and target
    # ln can create symlinks pointing outside allowed paths (e.g., ln -s /etc/passwd /allowed/path/link)
    if cmd == "ln":
        source_idx = None
        target_idx = None
        for i, arg in enumerate(argv[1:], start=1):
            if not arg.startswith("-"):
                if source_idx is None:
                    source_idx = i
                elif target_idx is None:
                    target_idx = i
                    break
        # For ln, validate that the symlink *target* (not the source) is in allowed paths
        # The source path is what the link points to (can be arbitrary)
        if target_idx is not None and target_idx < len(argv):
            try:
                target_arg = argv[target_idx]
                if target_arg.startswith(("/", "~")):
                    resolved_target = str(
                        Path(target_arg).expanduser().resolve(strict=False)
                    )
                else:
                    resolved_target = str(
                        Path(target_arg).expanduser().resolve(strict=False)
                    )
            except Exception as e:
                logger.warning(f"Cannot resolve ln target '{target_arg}': {e}")
                return (
                    True,
                    f"Cannot validate symlink target '{target_arg}': path resolution failed ({type(e).__name__})",
                )
            # Check target against dangerous paths and allowed paths
            for danger_path in DANGEROUS_PATHS:
                if resolved_target == danger_path or resolved_target.startswith(
                    danger_path + "/"
                ):
                    return (
                        True,
                        f"Symlink target '{resolved_target}' points to dangerous path",
                    )
            if resolved_allowed_paths:
                if not any(
                    _is_path_within(resolved_target, allowed_root)
                    for allowed_root in resolved_allowed_paths
                ):
                    return (
                        True,
                        f"Symlink target '{target_arg}' is outside explicitly allowed paths",
                    )

    for arg in argv[1:]:
        if arg.startswith("-"):
            continue

        # Handle key=value format (e.g., of=/boot/vmlinuz for dd command)
        path_to_check = arg
        if "=" in arg:
            # Extract the value part for commands like dd (of=..., if=...)
            _, path_part = arg.split("=", 1)
            # Only check if the value looks like a path
            if path_part.startswith(("/", "~")):
                path_to_check = path_part

        try:
            # Resolve all paths consistently to catch symlink bypasses
            if path_to_check.startswith(("/", "~")):
                resolved_arg = str(
                    Path(path_to_check).expanduser().resolve(strict=False)
                )
            else:
                # For relative paths, resolve against current directory
                resolved_arg = str(
                    Path(path_to_check).expanduser().resolve(strict=False)
                )
        except Exception as e:
            # Fail closed: if we can't resolve a path, deny it.
            # This prevents attackers from bypassing checks via crafted paths that fail to resolve.
            logger.warning(
                f"Cannot resolve path '{path_to_check}' in dangerous paths check: {e}"
            )
            return (
                True,
                f"Cannot validate path '{path_to_check}': path resolution failed ({type(e).__name__})",
            )

        for danger_path in DANGEROUS_PATHS:
            if resolved_arg == danger_path or resolved_arg.startswith(
                danger_path + "/"
            ):
                return True, f"Writing to '{resolved_arg}' is not allowed"

        if resolved_allowed_paths:
            if not any(
                _is_path_within(resolved_arg, allowed_root)
                for allowed_root in resolved_allowed_paths
            ):
                return (
                    True,
                    f"Path '{path_to_check}' is outside explicitly allowed paths: {', '.join(allowed_paths or [])}",
                )

    return False, ""


def _check_working_dir_restriction(
    argv: list[str], working_dir: str | None
) -> tuple[bool, str]:
    """Check if command respects working directory restriction.

    Args:
        argv: Tokenized command arguments.
        working_dir: Allowed working directory, or None for no restriction.

    Returns:
        A tuple of (violates_restriction, reason_message).
    """
    if not working_dir:
        return False, ""

    if not argv or argv[0].split("/")[-1] not in WRITE_COMMANDS:
        return False, ""

    try:
        allowed_path_str = str(Path(working_dir).expanduser().resolve())
    except Exception as e:
        # Fail closed: if we can't resolve working_dir, deny all writes in this directory.
        logger.warning(f"Cannot resolve working_dir '{working_dir}': {e}")
        return (
            True,
            f"Cannot validate working directory: working_dir '{working_dir}' is not resolvable ({type(e).__name__})",
        )

    # Ensure the allowed path ends with / for prefix matching
    if not allowed_path_str.endswith("/"):
        allowed_path_str_prefix = allowed_path_str + "/"
    else:
        allowed_path_str_prefix = allowed_path_str

    for arg in argv[1:]:
        if arg.startswith("-"):
            continue

        # Try to resolve all paths (both absolute and relative)
        try:
            # For relative paths, resolve them relative to working_dir, not caller's cwd
            if arg.startswith(("/", "~")):
                resolved_path = str(Path(arg).expanduser().resolve())
                is_relative = False
            else:
                resolved_path = str(
                    Path(working_dir, arg).expanduser().resolve(strict=False)
                )
                is_relative = True

            # Check if path is allowed: in working_dir, /tmp, or /private/tmp (macOS)
            is_in_tmp = resolved_path.startswith(("/tmp", "/private/tmp"))
            is_in_working_dir = (
                resolved_path == allowed_path_str
                or resolved_path.startswith(allowed_path_str_prefix)
            )

            # For relative paths: must be within working_dir (not just /tmp)
            # For absolute paths: can be in working_dir OR /tmp
            if is_relative:
                # Relative paths must stay within working_dir
                if not is_in_working_dir:
                    return (
                        True,
                        f"Path '{arg}' is outside allowed directory '{working_dir}'",
                    )
            else:
                # Absolute paths can be in working_dir or /tmp
                if not (is_in_tmp or is_in_working_dir):
                    return (
                        True,
                        f"Path '{arg}' is outside allowed directory '{working_dir}'",
                    )
        except Exception as e:
            # Fail closed: if we can't resolve an argument path, deny it.
            # This prevents attackers from bypassing checks via crafted paths that fail to resolve.
            logger.warning(
                f"Cannot resolve argument path '{arg}' in working_dir check: {e}"
            )
            return (
                True,
                f"Cannot validate path '{arg}': path resolution failed ({type(e).__name__})",
            )

    return False, ""


def _truncate_output(output: str, max_size: int = MAX_OUTPUT_SIZE) -> tuple[str, bool]:
    """Truncate output if it exceeds max size.

    Args:
        output: The output string to potentially truncate.
        max_size: Maximum allowed size in bytes.

    Returns:
        A tuple of (truncated_output, was_truncated). The output string is clean
        (no truncation message). The caller is responsible for appending any
        truncation message.
    """
    if len(output) > max_size:
        return output[:max_size], True
    return output, False


class BashEnvironment(ABC):
    """Abstract environment for executing bash commands.

    Args:
        allowed_paths (list[str] | None): Optional explicit write allowlist. When
            provided, write-target paths must fall under one of these roots in
            addition to passing the default dangerous-path checks.
        working_dir (str | None): Optional directory restriction for write
            operations. This is a host path where the command executes. When
            specified, writes must remain within this directory or ``/tmp``.
        timeout (int): Maximum number of seconds to allow command execution.

    Note:
        Subclass ``StaticBashEnvironment`` returns ``success=True, skipped=True``
        to indicate that validation passed but the command was intentionally not
        executed (analysis-only mode). Consumers that branch on ``success`` should
        check ``skipped`` first to handle this state correctly.

    """

    def __init__(
        self,
        allowed_paths: list[str] | None = None,
        working_dir: str | None = None,
        timeout: int = 60,
    ):
        """Initialize BashEnvironment with optional path allowlist and timeout."""
        self.allowed_paths = allowed_paths or []
        self.working_dir = working_dir
        self.timeout = timeout

    def _validate_command(self, command: str) -> ExecutionResult | list[str]:
        """Parse and validate a command before execution.

        The shared validation step performs argv parsing, rejects dangerous shell
        constructs, applies path safety checks, and enforces ``allowed_paths`` and
        ``working_dir`` restrictions for write operations.

        Args:
            command: The bash command string to validate.

        Returns:
            Either the validated argv list or a skipped ``ExecutionResult``
            describing why validation failed.
        """
        try:
            argv = shlex.split(command)
        except ValueError as e:
            return ExecutionResult(
                success=False,
                stdout=None,
                stderr=None,
                skipped=True,
                skip_message=f"Failed to parse command: {e!s}",
            )

        if not argv:
            return ExecutionResult(
                success=False,
                stdout=None,
                stderr=None,
                skipped=True,
                skip_message="Empty command",
            )

        is_dangerous, reason = _is_dangerous_command(argv)
        if is_dangerous:
            record_bash_violation(
                command=" ".join(argv),
                argv=argv,
                pattern_name="DangerousCommandPattern",
                category="UNKNOWN",
                severity="HIGH",
                reason=reason,
                working_dir=self.working_dir,
                allowed_paths=self.allowed_paths,
            )
            return ExecutionResult(
                success=False,
                stdout=None,
                stderr=None,
                skipped=True,
                skip_message=reason,
            )

        has_dangerous, reason = _check_dangerous_paths(argv, self.allowed_paths)
        if has_dangerous:
            record_bash_violation(
                command=" ".join(argv),
                argv=argv,
                pattern_name="DangerousPathPattern",
                category="UNKNOWN",
                severity="HIGH",
                reason=reason,
                working_dir=self.working_dir,
                allowed_paths=self.allowed_paths,
            )
            return ExecutionResult(
                success=False,
                stdout=None,
                stderr=None,
                skipped=True,
                skip_message=reason,
            )

        violates_restriction, reason = _check_working_dir_restriction(
            argv, self.working_dir
        )
        if violates_restriction:
            record_bash_violation(
                command=" ".join(argv),
                argv=argv,
                pattern_name="WorkingDirRestrictionPattern",
                category="UNKNOWN",
                severity="MEDIUM",
                reason=reason,
                working_dir=self.working_dir,
                allowed_paths=self.allowed_paths,
            )
            return ExecutionResult(
                success=False,
                stdout=None,
                stderr=None,
                skipped=True,
                skip_message=reason,
            )

        return argv

    @abstractmethod
    def execute(self, command: str) -> ExecutionResult:
        """Execute the given bash command and return the result.

        Args:
            command (str): The bash command to execute.

        Returns:
            ExecutionResult: Execution outcome including stdout, stderr, and
            success flag.
        """


class StaticBashEnvironment(BashEnvironment):
    """Safe environment that validates but does not execute bash commands.

    Returns ``success=True, skipped=True`` when validation passes (command is
    syntactically valid and passes all safety checks), indicating the command
    would be safe to execute but this environment intentionally does not run it.
    Returns ``success=False, skipped=True`` when validation fails (safety check
    rejection or parse error).
    """

    def execute(self, command: str) -> ExecutionResult:
        """Parse and validate command without executing.

        Args:
            command (str): The bash command to validate.

        Returns:
            ExecutionResult: Result with ``skipped=True`` and parsed argv in
            ``analysis_result`` on success, or a safety-check failure on rejection.
        """
        validated = self._validate_command(command)
        if isinstance(validated, ExecutionResult):
            return validated

        argv = validated

        return ExecutionResult(
            success=True,
            stdout=None,
            stderr=None,
            skipped=True,
            skip_message="Command passes safety checks; static analysis environment does not execute commands. To execute, use bash_executor().",
            analysis_result=argv,
        )


class _LocalBashEnvironment(BashEnvironment):
    """Environment that executes bash commands directly with subprocess.

    This is the primary execution environment for bash_executor(). Commands execute
    in the current process with access to the host environment (working directory,
    PATH, git repos, installed tools, environment variables).

    Safety model: Denylist-based (not isolation-based). The conservative denylist
    covers dangerous commands, shell operators, code execution paths, and writes to
    system directories. This is sufficient for typical agentic workflows where
    the command source is trusted (e.g., LLM-generated code in a known pipeline).

    For higher isolation requirements (untrusted code, CTF challenges, or security
    research), provide isolation at the application layer (containers, VMs).
    """

    def execute(self, command: str) -> ExecutionResult:
        """Execute bash command after safety checks.

        Args:
            command (str): The bash command to execute.

        Returns:
            ExecutionResult: Execution outcome with captured stdout/stderr and
            success flag, or a skipped result if safety checks fail.
        """
        validated = self._validate_command(command)
        if isinstance(validated, ExecutionResult):
            return validated

        argv = validated

        # Execute command with shell=False to prevent shell metacharacter bypass
        try:
            result = subprocess.run(
                argv,
                shell=False,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                cwd=self.working_dir,
            )

            stdout, stdout_truncated = _truncate_output(result.stdout.strip())
            stderr, stderr_truncated = _truncate_output(result.stderr.strip())

            # Append truncation warnings if needed
            if stdout_truncated:
                stdout += "\n[Output truncated - stdout exceeded 10KB]"
            if stderr_truncated:
                stderr += "\n[Output truncated - stderr exceeded 10KB]"

            return ExecutionResult(
                success=result.returncode == 0, stdout=stdout, stderr=stderr
            )
        except subprocess.TimeoutExpired:
            return ExecutionResult(
                success=False,
                stdout=None,
                stderr=None,
                skipped=True,
                skip_message=f"Execution timed out after {self.timeout} seconds",
            )
        except Exception as e:
            return ExecutionResult(
                success=False,
                stdout=None,
                stderr=None,
                skipped=True,
                skip_message=f"Subprocess execution error: {e!s}",
            )


def bash_executor(
    command: str, working_dir: str | None = None, allowed_paths: list[str] | None = None
) -> ExecutionResult:
    """Execute a bash command with denylist safety checks.

    This is the recommended entry point. Commands execute locally with access to
    the host environment (working directory, PATH, git repos, installed tools).

    Safety model: Conservative denylist applied to all commands. The denylist
    refuses sudo, interactive shells, destructive operations (rm -rf, git push
    --force), shell operators (|, >, &&), code execution paths (python -c, bash
    -c), and writes to system paths (/etc, /sys, /proc, etc.).

    Args:
        command: The bash command to execute.
        working_dir: Optional working directory for the command (host path).
        allowed_paths: Optional explicit write allowlist. When provided,
            write-target paths must fall under one of these roots (in addition
            to passing the default dangerous-path checks).

    Returns:
        An ``ExecutionResult`` with stdout, stderr, and success flag. If the
        command was rejected for safety reasons, ``skipped=True`` and
        ``skip_message`` contains the reason.

    Examples:
        Basic execution:
        >>> result = bash_executor("echo hello")
        >>> assert result.success is True
        >>> assert result.stdout == "hello"

        With working directory:
        >>> result = bash_executor("pwd", working_dir="/tmp")
        >>> assert "/tmp" in result.stdout

        With path restrictions:
        >>> result = bash_executor("touch file.txt", allowed_paths=["/tmp"])
        >>> assert result.success is True
    """
    env = _LocalBashEnvironment(allowed_paths=allowed_paths, working_dir=working_dir)
    return env.execute(command)
