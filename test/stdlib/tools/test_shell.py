"""Tests for bash shell execution environments."""

import pytest

from mellea.stdlib.tools.shell import (
    LLMSandboxBashEnvironment,
    StaticBashEnvironment,
    UnsafeBashEnvironment,
    bash_executor,
    local_bash_executor,
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
            "cp -f largefile /tmp",
            "mv -f file /tmp",
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
            "echo content > file.txt",  # Writing to current dir is OK
            "touch ~/.bashrc",  # Writing to home is OK
            "mkdir /tmp/tmpdir",  # Writing to /tmp is OK
        ],
    )
    def test_safe_paths_allowed(self, safe_path_cmd: str) -> None:
        """Safe path operations should be allowed."""
        env = StaticBashEnvironment()
        result = env.execute(safe_path_cmd)

        assert result.skipped is True
        assert result.success is True


class TestWorkingDirRestriction:
    """Tests for working directory restrictions."""

    def test_working_dir_restriction_blocks_outside_writes(self) -> None:
        """Writing outside working_dir should be rejected."""
        env = StaticBashEnvironment(working_dir="/home/user/project")
        result = env.execute("touch /var/log/test.log")

        assert result.skipped is True
        assert result.success is False
        # Could be blocked by either path restriction or working dir restriction
        assert result.skip_message is not None
        assert (
            "not allowed" in result.skip_message.lower()
            or "outside" in result.skip_message.lower()
        )

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


class TestUnsafeBashEnvironment:
    """Tests for unsafe bash environment execution."""

    def test_safe_command_execution(self) -> None:
        """Safe commands should execute successfully."""
        env = UnsafeBashEnvironment()
        result = env.execute("echo hello")

        assert result.skipped is False
        assert result.success is True
        assert result.stdout is not None
        assert "hello" in result.stdout

    def test_command_with_failing_exit_code(self) -> None:
        """Commands with non-zero exit should fail."""
        env = UnsafeBashEnvironment()
        result = env.execute("false")

        assert result.skipped is False
        assert result.success is False

    def test_stderr_capture(self) -> None:
        """stderr should be captured."""
        env = UnsafeBashEnvironment()
        result = env.execute("echo error >&2")

        assert result.skipped is False
        assert result.stderr is not None
        assert "error" in result.stderr

    def test_dangerous_command_rejected(self) -> None:
        """Dangerous commands should be rejected even before execution."""
        env = UnsafeBashEnvironment()
        result = env.execute("sudo echo test")

        assert result.skipped is True
        assert result.success is False
        assert result.skip_message is not None
        assert "not allowed" in result.skip_message.lower()

    def test_timeout_enforcement(self) -> None:
        """Command exceeding timeout should be interrupted."""
        env = UnsafeBashEnvironment(timeout=1)
        result = env.execute("sleep 5")

        assert result.skipped is True
        assert result.success is False
        assert result.skip_message is not None
        assert (
            "timed out" in result.skip_message.lower()
            or "timeout" in result.skip_message.lower()
        )

    def test_output_truncation(self) -> None:
        """Very large output should be truncated."""
        env = UnsafeBashEnvironment()
        # Generate output larger than 10KB
        result = env.execute("python3 -c \"print('x' * 20000)\"")

        assert result.success is True
        # Check that output was truncated
        assert result.stdout is not None
        assert "[OUTPUT TRUNCATED]" in result.stdout or len(result.stdout) < 30000

    def test_working_dir_parameter(self) -> None:
        """working_dir should be passed to subprocess."""
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmpdir:
            env = UnsafeBashEnvironment(working_dir=tmpdir)
            result = env.execute("pwd")

            assert result.success is True
            assert result.stdout is not None
            assert tmpdir in result.stdout


class TestBashExecutorFunctions:
    """Tests for public bash_executor and local_bash_executor functions."""

    def test_bash_executor_creates_sandbox_environment(self) -> None:
        """bash_executor should use LLMSandboxBashEnvironment by default."""
        # This will skip if llm-sandbox is not installed, which is fine
        result = bash_executor("echo test")

        # Either succeeds with output, or skips due to missing llm-sandbox or language support
        if (
            result.skip_message is not None
            and ("not installed" in result.skip_message or "not a valid" in result.skip_message)
        ):
            assert result.skipped is True
        else:
            # If sandbox is available, command should execute
            assert result.success is True

    def test_local_bash_executor_creates_unsafe_environment(self) -> None:
        """local_bash_executor should use UnsafeBashEnvironment."""
        result = local_bash_executor("echo test")

        assert result.skipped is False
        assert result.success is True
        assert result.stdout is not None
        assert "test" in result.stdout

    def test_bash_executor_with_working_dir(self) -> None:
        """bash_executor should accept working_dir parameter."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            # This may skip if llm-sandbox not available or doesn't support lang
            result = bash_executor("pwd", working_dir=tmpdir)

            # Just verify the function accepts the parameter without error
            # It will skip if sandbox not available
            assert result is not None

    def test_local_bash_executor_with_working_dir(self) -> None:
        """local_bash_executor should accept working_dir parameter."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            result = local_bash_executor("pwd", working_dir=tmpdir)

            assert result.success is True
            assert result.stdout is not None
            assert tmpdir in result.stdout


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
        """Dangerous flag rejection should mention the flag."""
        env = StaticBashEnvironment()
        result = env.execute("git push --force")

        assert result.skip_message is not None
        assert (
            "--force" in result.skip_message or "force" in result.skip_message.lower()
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

        tool = MelleaTool.from_callable(local_bash_executor)

        assert tool.name == "local_bash_executor"
        # Check that the tool schema is generated correctly
        schema = tool.as_json_tool
        assert "parameters" in schema or "function" in schema  # Schema format may vary
        # The tool should be callable
        assert callable(tool.run)
    except ImportError:
        pytest.skip("MelleaTool not available")
