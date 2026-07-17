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
    "curl",  # Download/upload files (with -o/-O)
    "wget",  # Download files (with -O)
}


def _check_nested_dangerous_commands(argv: list[str]) -> tuple[bool, str]:
    """Check for dangerous nested commands in wrapper contexts.

    This function validates nested command arguments that only the function-based
    system can handle, because it requires context-aware multi-argument lookahead.
    The pattern framework cannot perform these checks.

    Args:
        argv: Tokenized command arguments.

    Returns:
        A tuple of (is_dangerous, reason_message).
    """
    if not argv:
        return False, ""

    cmd = argv[0].split("/")[-1]  # Get basename of command

    # Flags that consume the next argument (only skip for safe wrappers)
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
        # Skip if argument contains / (it's a path, not a command name)
        if "/" in arg:
            continue
        # Skip if argument starts with - (it's a flag)
        if arg.startswith("-"):
            continue

        # Skip if this argument is the value for a preceding flag (space-separated).
        # IMPORTANT: Only do this if the command is a safe wrapper, because the flags
        # for regular commands (like ssh, git) might precede dangerous nested commands.
        # E.g., in "timeout -t 10 sudo", skip "10" because timeout is a safe wrapper.
        # E.g., in "ssh -t sudo whoami", DON'T skip "sudo" because -t is for ssh, not a wrapper.
        if cmd in SAFE_WRAPPER_COMMANDS and i > 1 and argv[i - 1] in flag_value_flags:
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
            # For shells in safe wrappers, check if they have dangerous flags that bypass
            # the denylist. This includes both code-execution flags (e.g., env bash -c)
            # and interactive flags (e.g., env bash -i).
            if arg_cmd in ("bash", "sh", "zsh", "ksh", "tcsh"):
                # Check for code-execution flags
                shell_code_exec_flags = {"-c", "-e"}
                if any(flag in argv for flag in shell_code_exec_flags):
                    return (
                        True,
                        f"Interpreter code execution ('{arg_cmd} -c' or '{arg_cmd} -e' in nested argument) is not allowed",
                    )
                # Check for interactive flags
                shell_interactive_flags = {"-i", "--interactive", "-l", "--login"}
                if any(flag in argv for flag in shell_interactive_flags):
                    return (
                        True,
                        f"Interactive shell ('{arg_cmd} -i' or '{arg_cmd} -l' in nested argument) is not allowed",
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

    return False, ""


def _is_dangerous_command(argv: list[str]) -> tuple[bool, str]:
    """Check if a command is dangerous based on nested context analysis.

    **DEPRECATED FOR MOST USES**: Use check_all_patterns() for top-level checks.
    This function now only checks nested command contexts that the pattern framework
    cannot handle. It is kept for backward compatibility.

    For new code, prefer:
    1. check_all_patterns() for top-level command validation (records audit trail)
    2. _check_nested_dangerous_commands() for wrapper context validation

    Args:
        argv: Tokenized command arguments.

    Returns:
        A tuple of (is_dangerous, reason_message).
    """
    if not argv:
        return False, ""

    # Only check nested context validation (pattern framework handles top-level)
    return _check_nested_dangerous_commands(argv)


def _extract_write_command(argv: list[str]) -> tuple[str, int]:
    """Extract the actual write command from potentially wrapped argv.

    Wrapper commands (env, nohup, timeout, etc.) may mask write commands.
    This function finds the first non-wrapper command that is a write operation.

    Args:
        argv: Tokenized command arguments.

    Returns:
        A tuple of (write_command_name, index_in_argv). If no write command
        is found, returns ("", -1).

    Examples:
        argv = ["env", "touch", "/etc/passwd"] → ("touch", 1)
        argv = ["timeout", "10", "rm", "/etc/foo"] → ("rm", 2)
        argv = ["curl", "-o", "/etc/passwd", "http://..."] → ("curl", 0)
    """
    if not argv:
        return "", -1

    # Check if argv[0] is a write command (direct case)
    cmd = argv[0].split("/")[-1]
    if cmd in WRITE_COMMANDS:
        return cmd, 0

    # If argv[0] is a wrapper command, look for the nested write command
    if cmd not in SAFE_WRAPPER_COMMANDS:
        return "", -1

    # For wrapper commands, find the first positional argument that is a write command
    # Skip flags and their values
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

    i = 1
    while i < len(argv):
        arg = argv[i]

        # Skip flags
        if arg.startswith("-"):
            # Check if this flag takes a value (space-separated)
            if arg in flag_value_flags and i + 1 < len(argv):
                i += 2  # Skip flag and its value
            else:
                i += 1  # Skip standalone flag
            continue

        # Skip numeric arguments (e.g., timeout duration, signal numbers)
        if arg.isdigit():
            i += 1
            continue

        # Skip variable assignments (e.g., DEBUG=ON, VAR=VALUE)
        if "=" in arg and not arg.startswith("-"):
            i += 1
            continue

        # This is a positional argument, check if it's a write command
        nested_cmd = arg.split("/")[-1]
        if nested_cmd in WRITE_COMMANDS:
            return nested_cmd, i

        # If it's not a write command and not numeric, it might be an argument or another wrapper
        # For now, we assume the first positional non-numeric argument is the command to wrap
        return "", -1

    return "", -1


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
    if not argv:
        return False, ""

    # Check env variable assignments for dangerous paths (e.g., LD_PRELOAD=/etc/passwd)
    # Only check if argv[0] is 'env' to avoid checking all arguments
    if argv and argv[0] in ("env", "/usr/bin/env"):
        for arg in argv[1:]:
            if "=" in arg and not arg.startswith("-"):
                # This is a variable assignment; check the value part
                var_name, var_value = arg.split("=", 1)
                if var_value.startswith(("/", "~")):
                    try:
                        resolved_value = str(
                            Path(var_value).expanduser().resolve(strict=False)
                        )
                        # Check if the path points to a dangerous location
                        for danger_path in DANGEROUS_PATHS:
                            if (
                                resolved_value == danger_path
                                or resolved_value.startswith(danger_path + "/")
                            ):
                                return (
                                    True,
                                    f"Environment variable '{var_name}' points to dangerous path '{var_value}'",
                                )
                    except Exception as e:
                        logger.warning(
                            f"Cannot resolve env variable value '{var_value}': {e}"
                        )

    # Extract the actual write command (handles wrapper commands like env, nohup, timeout)
    write_cmd, write_cmd_idx = _extract_write_command(argv)
    if not write_cmd:
        return False, ""

    cmd = write_cmd
    resolved_allowed_paths = _resolve_allowed_paths(allowed_paths or [])

    # For ln (symlink/hardlink), check both source and target
    # ln can create symlinks pointing outside allowed paths (e.g., ln -s /etc/passwd /allowed/path/link)
    if cmd == "ln":
        source_idx = None
        target_idx = None
        for i, arg in enumerate(argv[write_cmd_idx + 1 :], start=write_cmd_idx + 1):
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

    # For curl/wget, check the output file path specified with -o/-O
    if cmd in ("curl", "wget"):
        output_file = None
        for i, arg in enumerate(argv[write_cmd_idx + 1 :], start=write_cmd_idx + 1):
            if arg in ("-o", "-O", "--output"):
                # Next argument should be the output file
                if i + 1 < len(argv):
                    output_file = argv[i + 1]
                break
            # Also check for combined format like -o/file or -O/file
            elif arg.startswith(("-o", "-O")) and len(arg) > 2:
                output_file = arg[2:]
                break

        if output_file:
            try:
                if output_file.startswith(("/", "~")):
                    resolved_output = str(
                        Path(output_file).expanduser().resolve(strict=False)
                    )
                else:
                    resolved_output = str(
                        Path(output_file).expanduser().resolve(strict=False)
                    )
            except Exception as e:
                logger.warning(f"Cannot resolve {cmd} output file '{output_file}': {e}")
                return (
                    True,
                    f"Cannot validate {cmd} output path '{output_file}': path resolution failed ({type(e).__name__})",
                )
            # Check output against dangerous paths
            for danger_path in DANGEROUS_PATHS:
                if resolved_output == danger_path or resolved_output.startswith(
                    danger_path + "/"
                ):
                    return True, f"Writing to '{resolved_output}' is not allowed"
            if resolved_allowed_paths:
                if not any(
                    _is_path_within(resolved_output, allowed_root)
                    for allowed_root in resolved_allowed_paths
                ):
                    return (
                        True,
                        f"Path '{output_file}' is outside explicitly allowed paths: {', '.join(allowed_paths or [])}",
                    )

    for arg in argv[write_cmd_idx + 1 :]:
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

    if not argv:
        return False, ""

    # Extract the actual write command (handles wrapper commands)
    write_cmd, write_cmd_idx = _extract_write_command(argv)
    if not write_cmd:
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

    for arg in argv[write_cmd_idx + 1 :]:
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

        # Check nested command context (e.g., env sudo, timeout bash -c)
        # This must be checked before patterns because it validates wrapper contexts
        is_dangerous, reason = _check_nested_dangerous_commands(argv)
        if is_dangerous:
            record_bash_violation(
                command=" ".join(argv),
                argv=argv,
                pattern_name="NestedDangerousCommandPattern",
                category="nested_command",
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

        # Check all patterns (top-level validation with audit trail recording)
        is_dangerous, reason = check_all_patterns(
            argv, self.working_dir, self.allowed_paths
        )
        if is_dangerous:
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

        # Validate working_dir exists before subprocess call (fail-closed)
        if self.working_dir:
            try:
                working_dir_path = Path(self.working_dir).expanduser().resolve()
                if not working_dir_path.is_dir():
                    return ExecutionResult(
                        success=False,
                        stdout=None,
                        stderr=None,
                        skipped=True,
                        skip_message=f"Working directory '{self.working_dir}' does not exist or is not a directory",
                    )
            except Exception as e:
                return ExecutionResult(
                    success=False,
                    stdout=None,
                    stderr=None,
                    skipped=True,
                    skip_message=f"Cannot resolve working directory '{self.working_dir}': {e!s}",
                )

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
