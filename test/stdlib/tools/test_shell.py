"""Tests for bash shell execution environments."""

from unittest.mock import patch

import pytest

from mellea.stdlib.tools.shell import (
    StaticBashEnvironment,
    _LocalBashEnvironment,
    bash_executor,
)


class TestStaticBashEnvironment:
    """Tests for static bash command parsing and validation."""

    def test_parse_simple_command(self) -> None:
        """Valid simple command should pass validation."""
        env = StaticBashEnvironment()
        result = env.execute("echo hello")

        assert result.skipped is True
        assert result.success is True
        assert result.analysis_result == ["echo", "hello"]

    def test_parse_command_with_args(self) -> None:
        """Command with quoted arguments should parse correctly."""
        env = StaticBashEnvironment()
        result = env.execute('echo "hello world"')

        assert result.skipped is True
        assert result.success is True
        assert result.analysis_result == ["echo", "hello world"]

    def test_parse_empty_command(self) -> None:
        """Empty command should be rejected."""
        env = StaticBashEnvironment()
        result = env.execute("")

        assert result.skipped is True
        assert result.success is False
        assert result.skip_message is not None
        assert "Empty command" in result.skip_message

    def test_parse_invalid_quoting(self) -> None:
        """Command with invalid quoting should fail to parse."""
        env = StaticBashEnvironment()
        result = env.execute('echo "unclosed quote')

        assert result.skipped is True
        assert result.success is False
        assert result.skip_message is not None
        assert "parse" in result.skip_message.lower()


class TestDangerousCommandDetection:
    """Tests for dangerous command detection."""

    @pytest.mark.parametrize(
        "dangerous_cmd",
        [
            "sudo echo hello",
            "su - root",
            "doas whoami",
            "sudo -i",
            "sudo -s",
            "bash -i",
            "sh -i",
            "zsh -i",
            "passwd",
            "visudo",
            "chsh",
            "chfn",
            "useradd testuser",
            "userdel testuser",
            "usermod -l newname testuser",
        ],
    )
    def test_dangerous_commands_rejected(self, dangerous_cmd: str) -> None:
        """Dangerous commands should be rejected at parse time."""
        env = StaticBashEnvironment()
        result = env.execute(dangerous_cmd)

        assert result.skipped is True
        assert result.success is False
        assert result.skip_message is not None
        assert "not allowed" in result.skip_message.lower()

    def test_safe_shell_commands_allowed(self) -> None:
        """Non-interactive shell commands should be allowed."""
        env = StaticBashEnvironment()

        # bash/sh without -i flag should pass (might be used for scripting)
        result = env.execute("bash script.sh")
        assert result.success is True
        assert result.skipped is True


class TestInterpreterIndirectionBypassAttempts:
    """Tests for interpreter-indirection bypass attempts.

    Interpreter indirection occurs when a program (bash, python, env, timeout, etc.)
    is used to run arbitrary code. These are separate from simple command execution
    and need explicit testing to ensure the safety checks cover them.
    """

    def test_bash_c_string_rejected(self) -> None:
        """bash -c with arbitrary code should be rejected."""
        env = StaticBashEnvironment()
        # bash -c runs a command string, can bypass argv parsing
        result = env.execute("bash -c 'rm -rf /'")

        assert result.skipped is True
        assert result.success is False
        assert result.skip_message is not None
        # Should reject either the -c flag or command substitution
        assert "not allowed" in result.skip_message.lower()

    def test_sh_c_string_rejected(self) -> None:
        """sh -c with arbitrary code should be rejected."""
        env = StaticBashEnvironment()
        result = env.execute("sh -c 'sudo echo'")

        assert result.skipped is True
        assert result.success is False
        assert result.skip_message is not None
        assert "not allowed" in result.skip_message.lower()

    def test_env_bash_c_nested_code_execution_rejected(self) -> None:
        """env bash -c should be rejected (bypasses shell denylist with nested -c).

        Regression test for: env bash -c <payload> was bypassing the denylist because:
        - The code-execution check (lines 176-182) only catches bash as argv[0]
        - Nested bash in env was allowed at line 225-231 (shells allowed in safe wrappers)
        - But the -c flag on the nested bash was never re-checked

        This test ensures that -c/-e flags on nested shells are caught even when
        the top-level wrapper is in SAFE_WRAPPER_COMMANDS.
        """
        env = StaticBashEnvironment()
        result = env.execute("env bash -c 'id'")

        assert result.skipped is True
        assert result.success is False
        assert result.skip_message is not None
        assert "not allowed" in result.skip_message.lower()

    def test_timeout_bash_c_nested_code_execution_rejected(self) -> None:
        """timeout bash -c should be rejected (same bypass as env)."""
        env = StaticBashEnvironment()
        result = env.execute("timeout 5 bash -c 'whoami'")

        assert result.skipped is True
        assert result.success is False
        assert result.skip_message is not None
        assert "not allowed" in result.skip_message.lower()

    def test_env_sh_e_nested_code_execution_rejected(self) -> None:
        """env sh -e should be rejected (similar bypass with -e flag)."""
        env = StaticBashEnvironment()
        result = env.execute("env sh -e 'echo test'")

        assert result.skipped is True
        assert result.success is False
        assert result.skip_message is not None
        assert "not allowed" in result.skip_message.lower()

    def test_env_bash_script_allowed(self) -> None:
        """env bash script.sh should be allowed (legitimate use case)."""
        env = StaticBashEnvironment()
        result = env.execute("env bash script.sh")

        assert result.skipped is True
        assert result.success is True

    def test_timeout_bash_script_allowed(self) -> None:
        """timeout bash script.sh should be allowed (legitimate use case)."""
        env = StaticBashEnvironment()
        result = env.execute("timeout 10 bash script.sh")

        assert result.skipped is True
        assert result.success is True

    def test_env_with_sudo_rejected(self) -> None:
        """env with sudo should be rejected (privilege escalation)."""
        env = StaticBashEnvironment()
        # env is sometimes used to set environment vars for sudo
        result = env.execute("env sudo bash")

        assert result.skipped is True
        assert result.success is False
        assert result.skip_message is not None
        # Should reject sudo
        assert "not allowed" in result.skip_message.lower()

    def test_timeout_with_sudo_rejected(self) -> None:
        """timeout with sudo should be rejected (privilege escalation)."""
        env = StaticBashEnvironment()
        result = env.execute("timeout 10 sudo whoami")

        assert result.skipped is True
        assert result.success is False
        assert result.skip_message is not None
        assert "not allowed" in result.skip_message.lower()

    def test_timeout_with_flag_value_and_sudo_rejected(self) -> None:
        """timeout with --kill-after=value and sudo should be rejected (checks value-taking flags)."""
        env = StaticBashEnvironment()
        # Regression test: ensure sudo is detected despite --kill-after=1 consuming the value
        result = env.execute("timeout --kill-after=1 sudo whoami")

        assert result.skipped is True
        assert result.success is False
        assert result.skip_message is not None
        assert "not allowed" in result.skip_message.lower()

    def test_env_i_with_sudo_rejected(self) -> None:
        """env -i with sudo should be rejected (privilege escalation bypass attempt)."""
        env = StaticBashEnvironment()
        # Regression test for CVE-like: env -i (clear environment) + sudo
        # -i is NOT a value-taking flag; it takes no argument.
        # The skip logic must not incorrectly skip sudo.
        result = env.execute("env -i sudo whoami")

        assert result.skipped is True
        assert result.success is False
        assert result.skip_message is not None
        assert "not allowed" in result.skip_message.lower()

    def test_env_i_with_dangerous_rm_rejected(self) -> None:
        """env -i with rm -rf should be rejected (destructive bypass attempt)."""
        env = StaticBashEnvironment()
        result = env.execute("env -i rm -rf /tmp/test")

        assert result.skipped is True
        assert result.success is False
        assert result.skip_message is not None
        assert "not allowed" in result.skip_message.lower()

    def test_env_with_dangerous_rm_rejected(self) -> None:
        """env with rm -rf should be rejected."""
        env = StaticBashEnvironment()
        result = env.execute("env rm -rf /tmp/test")

        assert result.skipped is True
        assert result.success is False
        assert result.skip_message is not None
        assert "not allowed" in result.skip_message.lower()

    def test_timeout_with_dangerous_rm_rejected(self) -> None:
        """timeout with rm -rf should be rejected."""
        env = StaticBashEnvironment()
        result = env.execute("timeout 10 rm -rf /tmp/test")

        assert result.skipped is True
        assert result.success is False
        assert result.skip_message is not None
        assert "not allowed" in result.skip_message.lower()

    def test_env_with_safe_rm_allowed(self) -> None:
        """env with rm (no -r/-rf) should be allowed."""
        env = StaticBashEnvironment()
        result = env.execute("env rm file.txt")

        assert result.skipped is True
        assert result.success is True

    def test_timeout_with_safe_rm_allowed(self) -> None:
        """timeout with rm (no -r/-rf) should be allowed."""
        env = StaticBashEnvironment()
        result = env.execute("timeout 10 rm file.txt")

        assert result.skipped is True
        assert result.success is True

    def test_env_with_dangerous_command_in_middle_rejected(self) -> None:
        """env with variable assignment followed by dangerous command should be rejected."""
        env = StaticBashEnvironment()
        result = env.execute("env LD_LIBRARY_PATH=/lib sudo whoami")

        assert result.skipped is True
        assert result.success is False
        assert result.skip_message is not None
        assert "not allowed" in result.skip_message.lower()

    def test_python_c_arbitrary_code_rejected(self) -> None:
        """python -c with arbitrary code should be rejected."""
        env = StaticBashEnvironment()
        result = env.execute("python3 -c 'import os; os.system(\"rm -rf /\")'")

        assert result.skipped is True
        assert result.success is False
        assert result.skip_message is not None
        # Should reject the -c flag or command substitution
        assert "not allowed" in result.skip_message.lower()

    def test_python_multiline_code_rejected(self) -> None:
        """python with multiline code (using \\n) should be rejected."""
        env = StaticBashEnvironment()
        # Even with \\n instead of ; to bypass semicolon check
        result = env.execute("python3 -c 'import os\\nos.system(\"bad\")'")

        assert result.skipped is True
        assert result.success is False
        assert result.skip_message is not None
        assert "not allowed" in result.skip_message.lower()

    def test_bash_script_file_allowed(self) -> None:
        """bash with a script file (not -c) should be allowed for scripts."""
        env = StaticBashEnvironment()
        result = env.execute("bash /path/to/script.sh")

        # Should be allowed (script execution is legitimate)
        assert result.skipped is True
        assert result.success is True

    def test_perl_e_code_rejected(self) -> None:
        """perl -e with inline code should be rejected."""
        env = StaticBashEnvironment()
        result = env.execute("perl -e 'system(\"sudo whoami\")'")

        assert result.skipped is True
        assert result.success is False
        assert result.skip_message is not None
        assert "not allowed" in result.skip_message.lower()

    def test_ruby_e_code_rejected(self) -> None:
        """ruby -e with inline code should be rejected."""
        env = StaticBashEnvironment()
        result = env.execute("ruby -e 'system(\"rm -rf /\")'")

        assert result.skipped is True
        assert result.success is False
        assert result.skip_message is not None
        assert "not allowed" in result.skip_message.lower()

    @pytest.mark.parametrize(
        "safe_cmd",
        [
            "echo hello",
            "pwd",
            "ls -la",
            "cat file.txt",
            "grep pattern file.txt",
            "find . -name '*.py'",
        ],
    )
    def test_safe_commands_allowed(self, safe_cmd: str) -> None:
        """Safe commands should pass validation."""
        env = StaticBashEnvironment()
        result = env.execute(safe_cmd)

        assert result.skipped is True
        assert result.success is True
        assert result.analysis_result is not None


class TestDestructivePatternDetection:
    """Tests for detection of destructive operations."""

    @pytest.mark.parametrize(
        "destructive_cmd",
        [
            "rm -rf /",
            "rm -r /home/user",
            "rm -rf .",
            "git push --force origin main",
            "git push -f",
            "git reset --hard HEAD~1",
            "git clean -fd",
        ],
    )
    def test_destructive_operations_rejected(self, destructive_cmd: str) -> None:
        """Destructive operations should be rejected."""
        env = StaticBashEnvironment()
        result = env.execute(destructive_cmd)

        assert result.skipped is True
        assert result.success is False
        assert result.skip_message is not None
        assert "not allowed" in result.skip_message.lower()

    def test_standard_operations_with_flags_allowed(self) -> None:
        """Standard operations with force flags should be allowed.

        These are not inherently destructive:
        - cp -f: copy with force overwrite (standard)
        - mv -f: move with force overwrite (standard)
        - make -f: specify makefile (standard)
        """
        env = StaticBashEnvironment()

        result = env.execute("cp -f largefile /tmp")
        assert result.skipped is True
        assert result.success is True

        result = env.execute("mv -f file /tmp")
        assert result.skipped is True
        assert result.success is True

        result = env.execute("make -f Makefile clean")
        assert result.skipped is True
        assert result.success is True

    def test_safe_git_operations_allowed(self) -> None:
        """Safe git operations without --force should be allowed."""
        env = StaticBashEnvironment()

        result = env.execute("git push origin main")
        assert result.success is True
        assert result.skipped is True

    def test_safe_rm_operations_allowed(self) -> None:
        """rm without -r/-rf flags should be allowed."""
        env = StaticBashEnvironment()

        result = env.execute("rm file.txt")
        assert result.success is True
        assert result.skipped is True


class TestShellOperatorFalsePositives:
    """Tests for legitimate patterns that were previously false positives.

    The shell operator detection originally used substring matching,
    which blocked legitimate patterns like "a&&b" (regex). These tests
    verify that the fix correctly allows such patterns while still
    blocking actual shell operators.
    """

    def test_grep_with_and_in_pattern(self) -> None:
        """grep with && in regex pattern should be allowed."""
        env = StaticBashEnvironment()
        result = env.execute("grep 'a&&b' file.txt")

        assert result.skipped is True
        assert result.success is True

    def test_grep_with_or_in_pattern(self) -> None:
        """grep with || in regex pattern should be allowed."""
        env = StaticBashEnvironment()
        result = env.execute("grep 'a||b' file.txt")

        assert result.skipped is True
        assert result.success is True

    def test_echo_with_redirect_symbol_in_string(self) -> None:
        """echo with >> in string should be allowed."""
        env = StaticBashEnvironment()
        result = env.execute("echo 'a>>b'")

        assert result.skipped is True
        assert result.success is True

    def test_grep_with_heredoc_symbol_in_pattern(self) -> None:
        """grep with << in pattern should be allowed."""
        env = StaticBashEnvironment()
        result = env.execute("grep 'x<<EOF' file.txt")

        assert result.skipped is True
        assert result.success is True

    def test_awk_with_redirect_in_code(self) -> None:
        """awk code with >> in pattern should be allowed."""
        env = StaticBashEnvironment()
        result = env.execute('awk "{print $1>>$2}"')

        assert result.skipped is True
        assert result.success is True

    def test_sed_with_pipe_in_pattern(self) -> None:
        """sed pattern with | should be allowed."""
        env = StaticBashEnvironment()
        result = env.execute("sed 's/a|b/c/'")

        assert result.skipped is True
        assert result.success is True

    def test_actual_shell_redirect_operator_blocked(self) -> None:
        """Actual shell redirect operators with arguments should be blocked."""
        env = StaticBashEnvironment()
        # >&2 is a shell redirect (stderr redirect)
        result = env.execute("echo 'test' >&2")

        assert result.skipped is True
        assert result.success is False
        assert result.skip_message is not None
        # Message from patterns framework may say "redirection" instead of "not allowed"
        assert any(
            phrase in result.skip_message.lower()
            for phrase in ["not allowed", "redirection", "operator"]
        )

    def test_redirect_to_file_blocked(self) -> None:
        """Redirect to file (>filename) should be blocked."""
        env = StaticBashEnvironment()
        result = env.execute("echo 'test' >output.txt")

        assert result.skipped is True
        assert result.success is False
        assert result.skip_message is not None
        # Message from patterns framework may say "redirection" instead of "not allowed"
        assert any(
            phrase in result.skip_message.lower()
            for phrase in ["not allowed", "redirection", "operator"]
        )


class TestShellMetacharacterDetection:
    """Tests for detection of shell metacharacters that bypass argv parsing."""

    @pytest.mark.parametrize(
        "metacharacter_cmd",
        [
            "echo error >&2",  # stderr redirection
            "echo hello > /tmp/file",  # stdout redirection
            "cat file | grep pattern",  # pipe
            "echo a; rm -rf /",  # command chaining
            "echo $(whoami)",  # command substitution
            "echo `date`",  # backtick substitution
            "echo ${HOME}",  # variable expansion with braces
            "ls &",  # background execution
            "find . -name '*.py' && echo done",  # logical AND
            "ls || echo failed",  # logical OR
        ],
    )
    def test_shell_metacharacters_rejected(self, metacharacter_cmd: str) -> None:
        """Shell metacharacters should be rejected to prevent bypass attacks."""
        env = StaticBashEnvironment()
        result = env.execute(metacharacter_cmd)

        assert result.skipped is True
        assert result.success is False
        assert result.skip_message is not None
        # Message can be from patterns framework (more detailed) or manual checks (simpler)
        assert any(
            phrase in result.skip_message.lower()
            for phrase in [
                "not allowed",
                "redirection",
                "operator",
                "substitution",
                "chaining",
                "pipes",
                "background",
                "execution",
            ]
        )


class TestMacOSPrivateVarHandling:
    """Tests for correct handling of macOS /private/var paths.

    On macOS, tempfile.mkdtemp() returns /var/folders/... which resolves to
    /private/var/folders/... . We should allow writes to /private/var/folders/*
    (user temp directories) while blocking /private/var/log, /private/var/www, etc.
    """

    def test_private_var_log_blocked(self) -> None:
        """Writing to /private/var/log should be blocked."""
        env = StaticBashEnvironment()
        result = env.execute("touch /private/var/log/test.log")

        assert result.skipped is True
        assert result.success is False
        assert result.skip_message is not None
        assert "not allowed" in result.skip_message.lower()

    def test_private_var_www_blocked(self) -> None:
        """Writing to /private/var/www should be blocked."""
        env = StaticBashEnvironment()
        result = env.execute("touch /private/var/www/index.html")

        assert result.skipped is True
        assert result.success is False
        assert result.skip_message is not None
        assert "not allowed" in result.skip_message.lower()

    def test_private_var_db_blocked(self) -> None:
        """Writing to /private/var/db should be blocked."""
        env = StaticBashEnvironment()
        result = env.execute("touch /private/var/db/test.db")

        assert result.skipped is True
        assert result.success is False
        assert result.skip_message is not None
        assert "not allowed" in result.skip_message.lower()

    def test_private_var_root_blocked(self) -> None:
        """Writing to /private/var/root should be blocked."""
        env = StaticBashEnvironment()
        result = env.execute("touch /private/var/root/test.txt")

        assert result.skipped is True
        assert result.success is False
        assert result.skip_message is not None
        assert "not allowed" in result.skip_message.lower()

    def test_private_var_folders_allowed(self) -> None:
        """Writing to /private/var/folders/* (macOS temp dirs) should be allowed."""
        env = StaticBashEnvironment()
        # This simulates a resolved path from tempfile on macOS
        result = env.execute("touch /private/var/folders/kl/tmpXXXX/test.txt")

        # Should pass validation (not marked as dangerous path)
        assert result.skipped is True
        assert result.success is True

    def test_macos_temp_directory_resolved_allowed(self) -> None:
        """Resolved macOS temp directory paths should be allowed."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            from pathlib import Path

            # Resolve the temp dir (on macOS this becomes /private/var/folders/...)
            resolved = str(Path(tmpdir).resolve())

            env = StaticBashEnvironment()
            result = env.execute(f"touch {resolved}/test.txt")

            # Should pass (temp dir is safe)
            assert result.skipped is True
            assert result.success is True


class TestSystemPathDetection:
    """Tests for detection of system path access."""

    @pytest.mark.parametrize(
        "system_path_cmd",
        [
            "rm /etc/passwd",
            "touch /etc/config.conf",
            "cp file /sys/module",
            "mkdir /proc/newdir",
        ],
    )
    def test_system_paths_rejected(self, system_path_cmd: str) -> None:
        """Attempts to write to system paths should be rejected."""
        env = StaticBashEnvironment()
        result = env.execute(system_path_cmd)

        assert result.skipped is True
        assert result.success is False
        assert result.skip_message is not None
        assert "not allowed" in result.skip_message.lower()

    @pytest.mark.parametrize(
        "safe_path_cmd",
        [
            "cat /etc/passwd",  # Reading is OK
            "ls /sys",  # Reading is OK
            "touch ~/file.txt",  # Writing to home is OK
            "mkdir /tmp/tmpdir",  # Writing to /tmp is OK
        ],
    )
    def test_safe_paths_allowed(self, safe_path_cmd: str) -> None:
        """Safe path operations should be allowed."""
        env = StaticBashEnvironment()
        result = env.execute(safe_path_cmd)

        assert result.skipped is True
        assert result.success is True

    def test_symlink_to_dangerous_path_rejected(self) -> None:
        """Symlinks pointing to dangerous paths should be rejected."""
        import os
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a symlink in /tmp pointing to /etc
            symlink_path = os.path.join(tmpdir, "link_to_etc")
            try:
                os.symlink("/etc", symlink_path)
            except OSError:
                # Skip if symlink creation fails (e.g., on some filesystems)
                pytest.skip("Cannot create symlinks on this system")

            # Try to write through the symlink
            env = StaticBashEnvironment()
            result = env.execute(f"touch {symlink_path}/config")

            # Should be rejected because symlink resolves to /etc
            assert result.skipped is True
            assert result.success is False
            assert result.skip_message is not None
            assert "not allowed" in result.skip_message.lower()

    def test_chmod_to_system_path_rejected(self) -> None:
        """chmod to system paths should be rejected (was missing from write_commands).

        Regression test for planetf1 comment: chmod, chown, ln, dd weren't in
        the write_commands set, allowing chmod 777 /etc/passwd to bypass.
        """
        env = StaticBashEnvironment()
        result = env.execute("chmod 777 /etc/passwd")

        assert result.skipped is True
        assert result.success is False
        assert result.skip_message is not None
        assert "not allowed" in result.skip_message.lower()

    def test_chown_to_system_path_rejected(self) -> None:
        """chown to system paths should be rejected."""
        env = StaticBashEnvironment()
        result = env.execute("chown root:root /etc/config")

        assert result.skipped is True
        assert result.success is False

    def test_dd_to_dangerous_path_rejected(self) -> None:
        """dd to dangerous paths should be rejected (was missing from write_commands)."""
        env = StaticBashEnvironment()
        result = env.execute("dd if=/dev/urandom of=/boot/vmlinuz")

        assert result.skipped is True
        assert result.success is False
        assert result.skip_message is not None
        assert "not allowed" in result.skip_message.lower()

    def test_ln_symlink_escape_rejected(self) -> None:
        """ln creating symlinks to dangerous paths should be rejected.

        Regression test: ln -sf /etc/passwd /allowed/path/link creates a link
        in an allowed path pointing outside (symlink escape). The link target
        must be validated in allowed_paths context.
        """
        env = StaticBashEnvironment(allowed_paths=["/tmp"])
        # Try to create a symlink in /tmp pointing outside allowed paths
        result = env.execute("ln -sf /etc/passwd /tmp/link_to_etc")

        # Should reject because /etc/passwd is outside allowed paths
        assert result.skipped is True
        assert result.success is False

    def test_ln_symlink_within_allowed_paths_allowed(self) -> None:
        """ln creating symlinks within allowed paths should be allowed."""
        import os
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            source = os.path.join(tmpdir, "source.txt")
            target = os.path.join(tmpdir, "link.txt")
            # Create source file
            open(source, "w").close()

            env = StaticBashEnvironment(allowed_paths=[tmpdir])
            result = env.execute(f"ln -s {source} {target}")

            assert result.skipped is True
            assert result.success is True


class TestWorkingDirRestriction:
    """Tests for working directory restrictions."""

    def test_working_dir_restriction_blocks_outside_writes(self) -> None:
        """Writing outside working_dir should be rejected by working_dir check."""
        env = StaticBashEnvironment(working_dir="/home/user/project")
        # Use a safe path that is not in DANGEROUS_PATHS (so working_dir check fires first)
        result = env.execute("touch /home/other/file.txt")

        assert result.skipped is True
        assert result.success is False
        assert result.skip_message is not None
        # Must be rejected by working_dir check, not dangerous-path check
        assert "outside" in result.skip_message.lower()

    def test_working_dir_allows_inside_writes(self) -> None:
        """Writing inside working_dir should be allowed."""
        env = StaticBashEnvironment(working_dir="/home/user/project")
        result = env.execute("touch /home/user/project/test.txt")

        assert result.skipped is True
        assert result.success is True

    def test_working_dir_allows_tmp_writes(self) -> None:
        """Writing to /tmp should always be allowed."""
        env = StaticBashEnvironment(working_dir="/home/user/project")
        result = env.execute("touch /tmp/tmpfile")

        assert result.skipped is True
        assert result.success is True

    def test_working_dir_relative_path_resolved_within_working_dir(self) -> None:
        """Relative paths should be resolved relative to working_dir, not caller's cwd."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a file relative to working_dir (not caller's cwd)
            env = StaticBashEnvironment(working_dir=tmpdir)
            result = env.execute("touch myfile.txt")

            # Should be allowed: relative path resolves to tmpdir/myfile.txt
            assert result.skipped is True
            assert result.success is True

    def test_working_dir_relative_path_blocks_outside(self) -> None:
        """Relative paths that escape working_dir should be rejected."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            # Try to write outside working_dir using relative path
            env = StaticBashEnvironment(working_dir=tmpdir)
            result = env.execute("touch ../outside.txt")

            # Should be rejected: ../outside.txt escapes working_dir
            assert result.skipped is True
            assert result.success is False
            assert result.skip_message is not None
            assert "outside" in result.skip_message.lower()

    def test_working_dir_unresolvable_fails_closed(self) -> None:
        """Unresolvable working_dir should fail closed (deny writes)."""
        env = StaticBashEnvironment(working_dir="~invalid/nonexistent")
        # Invalid home dir prefix causes RuntimeError in .resolve()
        result = env.execute("touch /tmp/test.txt")

        # Should be rejected: can't resolve working_dir
        assert result.skipped is True
        assert result.success is False
        assert result.skip_message is not None
        assert (
            "not resolvable" in result.skip_message.lower()
            or "cannot validate" in result.skip_message.lower()
        )

    def test_working_dir_unresolvable_blocks_even_etc(self) -> None:
        """Unresolvable working_dir should block attempts to write to /etc."""
        env = StaticBashEnvironment(working_dir="~invalid/path")
        result = env.execute("touch /etc/config")

        # Should be blocked, first by /etc check, but verify it fails
        assert result.skipped is True
        assert result.success is False


class TestPathResolutionFailures:
    """Tests for fail-closed behavior when path resolution fails."""

    def test_unresolvable_argument_path_fails_closed(self) -> None:
        """Unresolvable argument paths should fail closed (deny writes)."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            env = StaticBashEnvironment(working_dir=tmpdir)
            # Invalid home dir prefix in argument causes RuntimeError
            result = env.execute("touch ~invalid/file.txt")

            # Should be rejected: can't resolve argument path
            assert result.skipped is True
            assert result.success is False
            assert result.skip_message is not None
            # Error should mention path resolution failure
            assert (
                "cannot validate" in result.skip_message.lower()
                or "resolution failed" in result.skip_message.lower()
            )


class TestAllowedPaths:
    """Tests for explicit allowed path enforcement."""

    def test_allowed_paths_allows_write_inside_explicit_path(self) -> None:
        """Writing inside an explicit allowed path should be permitted."""
        env = StaticBashEnvironment(allowed_paths=["/home/user/project"])
        result = env.execute("touch /home/user/project/output.txt")

        assert result.skipped is True
        assert result.success is True

    def test_allowed_paths_blocks_write_outside_explicit_path(self) -> None:
        """Writing outside explicit allowed paths should be rejected."""
        env = StaticBashEnvironment(allowed_paths=["/home/user/project"])
        result = env.execute("touch /home/user/other/output.txt")

        assert result.skipped is True
        assert result.success is False
        assert result.skip_message is not None
        assert "outside explicitly allowed paths" in result.skip_message.lower()

    def test_allowed_paths_does_not_override_dangerous_paths(self) -> None:
        """Explicit allowed paths must not permit writes to dangerous system paths."""
        env = StaticBashEnvironment(allowed_paths=["/etc"])
        result = env.execute("touch /etc/config.conf")

        assert result.skipped is True
        assert result.success is False
        assert result.skip_message is not None
        assert "not allowed" in result.skip_message.lower()


class TestLocalBashEnvironment:
    """Tests for local bash environment execution (no isolation)."""

    def test_safe_command_execution(self) -> None:
        """Safe commands should execute successfully."""
        env = _LocalBashEnvironment()
        result = env.execute("echo hello")

        assert result.skipped is False
        assert result.success is True
        assert result.stdout is not None
        assert "hello" in result.stdout

    def test_command_with_failing_exit_code(self) -> None:
        """Commands with non-zero exit should fail."""
        env = _LocalBashEnvironment()
        result = env.execute("false")

        assert result.skipped is False
        assert result.success is False

    def test_shell_metacharacters_rejected(self) -> None:
        """Shell redirections and pipes should be rejected for security."""
        env = _LocalBashEnvironment()
        result = env.execute("echo error >&2")

        assert result.skipped is True
        assert result.success is False
        assert result.skip_message is not None
        # Message from patterns framework may say "redirection" instead of "not allowed"
        assert any(
            phrase in result.skip_message.lower()
            for phrase in ["not allowed", "redirection", "operator"]
        )

    def test_dangerous_command_rejected(self) -> None:
        """Dangerous commands should be rejected even before execution."""
        env = _LocalBashEnvironment()
        result = env.execute("sudo echo test")

        assert result.skipped is True
        assert result.success is False
        assert result.skip_message is not None
        assert "not allowed" in result.skip_message.lower()

    def test_timeout_enforcement(self) -> None:
        """Command exceeding timeout should be interrupted."""
        env = _LocalBashEnvironment(timeout=1)
        result = env.execute("sleep 5")

        assert result.skipped is True
        assert result.success is False
        assert result.skip_message is not None
        assert "timed out" in result.skip_message.lower()

    def test_output_truncation(self) -> None:
        """Very large output should be truncated."""
        from mellea.stdlib.tools.shell import MAX_OUTPUT_SIZE

        env = _LocalBashEnvironment()
        # Generate output larger than MAX_OUTPUT_SIZE (10KB) to trigger truncation
        import os
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a large file to cat (500 lines of 51 bytes each = ~25.5 KB)
            large_file = os.path.join(tmpdir, "large.txt")
            with open(large_file, "w") as f:
                for _ in range(500):
                    f.write("x" * 50 + "\n")

            result = env.execute(f"cat {large_file}")

            assert result.success is True
            # Check that output was truncated
            assert result.stdout is not None
            # Verify truncation marker is present
            assert "Output truncated" in result.stdout
            # Verify output is actually truncated (not full ~25KB content)
            # Truncated output should be MAX_OUTPUT_SIZE + marker message
            # Allow some slack for the exact message format
            assert len(result.stdout) <= MAX_OUTPUT_SIZE + 100

    def test_working_dir_parameter(self) -> None:
        """working_dir should be passed to subprocess."""
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmpdir:
            env = _LocalBashEnvironment(working_dir=tmpdir)
            result = env.execute("pwd")

            assert result.success is True
            assert result.stdout is not None
            assert tmpdir in result.stdout


class TestBashExecutorFunctions:
    """Tests for public bash_executor function."""

    def test_bash_executor_uses_local_by_default(self) -> None:
        """bash_executor should use _LocalBashEnvironment by default (no sandbox)."""
        result = bash_executor("echo test")

        # bash_executor with no sandbox parameter should always succeed (uses local execution)
        assert result.success is True
        assert result.stdout is not None
        assert "test" in result.stdout

    def test_bash_executor_with_working_dir(self) -> None:
        """bash_executor should pass working_dir through to sandbox execution."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            result = bash_executor("pwd", working_dir=tmpdir)

            if result.skip_message is not None and (
                "not installed" in result.skip_message
                or "sandbox execution error" in result.skip_message.lower()
            ):
                assert result.skipped is True
            else:
                assert result.success is True
                assert result.stdout is not None
                assert tmpdir in result.stdout

    def test_bash_executor_with_allowed_paths(self) -> None:
        """bash_executor should accept allowed_paths parameter."""
        # Just verify the parameter is accepted (actual execution may skip on sandbox)
        result = bash_executor(
            "echo test", allowed_paths=["/tmp", "/home/user/project"]
        )

        # Either executes or skips due to sandbox setup
        if result.skip_message is not None and (
            "not installed" in result.skip_message
            or "not a valid" in result.skip_message
        ):
            assert result.skipped is True
        else:
            assert result.success is True

    def test_bash_executor_local_execution(self) -> None:
        """bash_executor should execute locally."""
        result = bash_executor("echo hello")

        assert result.success is True
        assert result.stdout is not None
        assert "hello" in result.stdout

    def test_dangerous_command_rejected(self) -> None:
        """Dangerous commands should be rejected."""
        result = bash_executor("sudo echo test")

        assert result.skipped is True
        assert result.success is False
        assert result.skip_message is not None
        assert "not allowed" in result.skip_message.lower()


class TestCommandParsing:
    """Tests for command parsing and quoting handling."""

    def test_command_with_spaces_in_quotes(self) -> None:
        """Arguments with spaces should be handled correctly."""
        env = StaticBashEnvironment()
        result = env.execute('echo "hello world" "foo bar"')

        assert result.success is True
        assert result.analysis_result == ["echo", "hello world", "foo bar"]

    def test_command_with_escaped_quotes(self) -> None:
        """Escaped quotes should be handled."""
        env = StaticBashEnvironment()
        result = env.execute(r'echo "say \"hello\""')

        assert result.success is True
        assert result.analysis_result is not None
        assert "echo" in result.analysis_result

    def test_command_with_equals_in_args(self) -> None:
        """Arguments with = (like env vars) should parse correctly."""
        env = StaticBashEnvironment()
        result = env.execute("grep FOO=bar file.txt")

        assert result.success is True
        assert result.analysis_result is not None
        assert "FOO=bar" in result.analysis_result


class TestErrorMessages:
    """Tests for clear error messages."""

    def test_sudo_rejection_message(self) -> None:
        """sudo rejection should have clear message."""
        env = StaticBashEnvironment()
        result = env.execute("sudo apt-get install package")

        assert result.skip_message is not None
        assert "not allowed" in result.skip_message.lower()

    def test_dangerous_flag_rejection_message(self) -> None:
        """Dangerous git operation rejection should mention the issue."""
        env = StaticBashEnvironment()
        result = env.execute("git push --force")

        assert result.skip_message is not None
        assert (
            "destructive" in result.skip_message.lower()
            or "--force" in result.skip_message
            or "force" in result.skip_message.lower()
        )

    def test_system_path_rejection_message(self) -> None:
        """System path rejection should mention the path."""
        env = StaticBashEnvironment()
        result = env.execute("rm /etc/passwd")

        assert result.skip_message is not None
        assert "/etc" in result.skip_message or "allowed" in result.skip_message.lower()


@pytest.mark.integration
def test_tool_wrapping() -> None:
    """Test that bash_executor can be wrapped as a MelleaTool."""
    try:
        from mellea.backends.tools import MelleaTool

        tool = MelleaTool.from_callable(bash_executor)

        assert tool.name == "bash_executor"
        # Check that the tool schema is generated correctly
        schema = tool.as_json_tool
        assert "parameters" in schema or "function" in schema  # Schema format may vary
        # The tool should be callable
        assert callable(tool.run)
    except ImportError:
        pytest.skip("MelleaTool not available")
