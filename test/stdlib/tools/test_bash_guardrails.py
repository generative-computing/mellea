"""Tests for bash security guardrails framework.

Verifies:
1. All dangerous commands are properly categorized and documented
2. All security patterns correctly identify violations
3. Guardrails coverage is complete across threat categories
4. New patterns can be added without breaking existing checks
"""

from mellea.stdlib.tools._bash_guardrails import (
    COMMAND_RULES,
    SHELL_OPERATOR_RULES,
    CommandCategory,
    Severity,
    audit_guardrails_coverage,
    get_command_rules_by_category,
    get_high_severity_commands,
)
from mellea.stdlib.tools._bash_patterns import (
    SECURITY_PATTERNS,
    CodeExecutionPattern,
    CommandSubstitutionPattern,
    DangerousCommandPattern,
    DestructiveGitPattern,
    DestructiveRmPattern,
    ShellOperatorPattern,
    check_all_patterns,
    get_pattern_names,
)


class TestCommandRules:
    """Tests for COMMAND_RULES structure and metadata."""

    def test_all_dangerous_commands_have_rules(self) -> None:
        """Every dangerous command should have a defined rule."""
        # From shell.py DANGEROUS_COMMANDS set
        expected_commands = {
            "sudo",
            "su",
            "doas",
            "bash",
            "sh",
            "zsh",
            "ksh",
            "tcsh",
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
        rules_keys = set(COMMAND_RULES.keys())
        assert expected_commands.issubset(rules_keys), (
            f"Missing rules for: {expected_commands - rules_keys}"
        )

    def test_rules_have_required_fields(self) -> None:
        """Each rule should have category, severity, and rationale."""
        for cmd, rule in COMMAND_RULES.items():
            assert rule.category in CommandCategory, f"{cmd}: invalid category"
            assert rule.severity in Severity, f"{cmd}: invalid severity"
            assert isinstance(rule.rationale, str) and len(rule.rationale) > 0, (
                f"{cmd}: rationale must be non-empty string"
            )

    def test_privilege_escalation_are_critical(self) -> None:
        """Privilege escalation commands should be critical severity."""
        priv_esc = get_command_rules_by_category(CommandCategory.PRIVILEGE_ESCALATION)
        for cmd, rule in priv_esc.items():
            assert rule.severity == Severity.CRITICAL, (
                f"{cmd} is privilege escalation but not critical"
            )

    def test_get_command_rules_by_category(self) -> None:
        """get_command_rules_by_category should filter correctly."""
        get_command_rules_by_category(CommandCategory.DESTRUCTIVE)
        for cmd in ["rm", "rmdir"]:
            if cmd in COMMAND_RULES:
                assert COMMAND_RULES[cmd].category == CommandCategory.DESTRUCTIVE

    def test_high_severity_commands_retrieved(self) -> None:
        """get_high_severity_commands should return critical and high."""
        high_severity = get_high_severity_commands()
        for cmd, rule in high_severity.items():
            assert rule.severity in (Severity.CRITICAL, Severity.HIGH), (
                f"{cmd} is in high_severity but not critical/high"
            )

    def test_audit_guardrails_coverage(self) -> None:
        """audit_guardrails_coverage should return all categories."""
        coverage = audit_guardrails_coverage()
        for category in CommandCategory:
            assert category.value in coverage, f"Missing coverage for {category}"
            assert isinstance(coverage[category.value], list)


class TestShellOperatorRules:
    """Tests for SHELL_OPERATOR_RULES structure."""

    def test_all_shell_operators_have_rules(self) -> None:
        """All dangerous shell operators should have rules."""
        expected_operators = {"|", ">", "&&", "||", ";", "&", ">>", ">&", "<<", "|&"}
        rules_keys = set(SHELL_OPERATOR_RULES.keys())
        assert expected_operators.issubset(rules_keys), (
            f"Missing rules for: {expected_operators - rules_keys}"
        )

    def test_operator_rules_have_required_fields(self) -> None:
        """Each operator rule should have category and rationale."""
        for op, rule in SHELL_OPERATOR_RULES.items():
            assert hasattr(rule, "category"), f"{op}: missing category"
            assert isinstance(rule.rationale, str) and len(rule.rationale) > 0


class TestDangerousCommandPattern:
    """Tests for DangerousCommandPattern detection."""

    def test_sudo_rejected(self) -> None:
        """sudo command should be rejected."""
        pattern = DangerousCommandPattern()
        is_dangerous, reason = pattern.check(["sudo", "echo", "test"])
        assert is_dangerous is True
        assert "sudo" in reason.lower()

    def test_interactive_bash_rejected(self) -> None:
        """bash -i should be rejected."""
        pattern = DangerousCommandPattern()
        is_dangerous, _ = pattern.check(["bash", "-i"])
        assert is_dangerous is True

    def test_non_interactive_bash_allowed(self) -> None:
        """bash script.sh should pass (not rejected by pattern)."""
        pattern = DangerousCommandPattern()
        is_dangerous, _ = pattern.check(["bash", "script.sh"])
        assert is_dangerous is False

    def test_passwd_rejected(self) -> None:
        """passwd should be rejected."""
        pattern = DangerousCommandPattern()
        is_dangerous, _ = pattern.check(["passwd"])
        assert is_dangerous is True


class TestShellOperatorPattern:
    """Tests for ShellOperatorPattern detection."""

    def test_pipe_operator_rejected(self) -> None:
        """Pipe operator should be rejected."""
        pattern = ShellOperatorPattern()
        is_dangerous, reason = pattern.check(["cat", "file", "|", "grep", "pattern"])
        assert is_dangerous is True
        assert len(reason) > 0  # Should have a reason

    def test_redirect_operator_rejected(self) -> None:
        """Output redirect should be rejected."""
        pattern = ShellOperatorPattern()
        is_dangerous, _ = pattern.check(["echo", "hello", ">", "file.txt"])
        assert is_dangerous is True

    def test_redirect_prefix_rejected(self) -> None:
        """Redirect as prefix (>&2) should be rejected."""
        pattern = ShellOperatorPattern()
        is_dangerous, _ = pattern.check(["echo", "error", ">&2"])
        assert is_dangerous is True

    def test_and_operator_rejected(self) -> None:
        """AND operator should be rejected."""
        pattern = ShellOperatorPattern()
        is_dangerous, _ = pattern.check(["cmd1", "&&", "cmd2"])
        assert is_dangerous is True

    def test_semicolon_rejected(self) -> None:
        """Semicolon chaining should be rejected."""
        pattern = ShellOperatorPattern()
        is_dangerous, _ = pattern.check(["cmd1", ";", "cmd2"])
        assert is_dangerous is True

    def test_quoted_pipe_in_string_allowed(self) -> None:
        """Pipe inside quoted string should be allowed (after shlex.split)."""
        pattern = ShellOperatorPattern()
        # After shlex.split("grep 'a|b'"), we get ["grep", "a|b"]
        # The pipe is part of the string, not a standalone operator
        is_dangerous, _ = pattern.check(["grep", "a|b"])
        assert is_dangerous is False


class TestCommandSubstitutionPattern:
    """Tests for CommandSubstitutionPattern detection."""

    def test_backtick_substitution_rejected(self) -> None:
        """Backtick command substitution should be rejected."""
        pattern = CommandSubstitutionPattern()
        is_dangerous, _ = pattern.check(["echo", "`date`"])
        assert is_dangerous is True

    def test_dollar_paren_substitution_rejected(self) -> None:
        """$(...) command substitution should be rejected."""
        pattern = CommandSubstitutionPattern()
        is_dangerous, _ = pattern.check(["echo", "$(whoami)"])
        assert is_dangerous is True

    def test_variable_expansion_rejected(self) -> None:
        """Variable expansion ${VAR} should be rejected."""
        pattern = CommandSubstitutionPattern()
        is_dangerous, _ = pattern.check(["echo", "${HOME}"])
        assert is_dangerous is True


class TestCodeExecutionPattern:
    """Tests for CodeExecutionPattern detection."""

    def test_python_c_rejected(self) -> None:
        """python -c should be rejected."""
        pattern = CodeExecutionPattern()
        is_dangerous, _ = pattern.check(["python", "-c", "print('hello')"])
        assert is_dangerous is True

    def test_python_m_rejected(self) -> None:
        """python -m should be rejected."""
        pattern = CodeExecutionPattern()
        is_dangerous, _ = pattern.check(["python", "-m", "http.server"])
        assert is_dangerous is True

    def test_bash_c_rejected(self) -> None:
        """bash -c should be rejected."""
        pattern = CodeExecutionPattern()
        is_dangerous, _ = pattern.check(["bash", "-c", "rm -rf /"])
        assert is_dangerous is True

    def test_perl_e_rejected(self) -> None:
        """perl -e should be rejected."""
        pattern = CodeExecutionPattern()
        is_dangerous, _ = pattern.check(["perl", "-e", "system('rm -rf /')"])
        assert is_dangerous is True

    def test_python_script_allowed(self) -> None:
        """python script.py should be allowed."""
        pattern = CodeExecutionPattern()
        is_dangerous, _ = pattern.check(["python", "script.py"])
        assert is_dangerous is False


class TestDestructiveGitPattern:
    """Tests for DestructiveGitPattern detection."""

    def test_git_push_force_rejected(self) -> None:
        """git push --force should be rejected."""
        pattern = DestructiveGitPattern()
        is_dangerous, _ = pattern.check(["git", "push", "--force", "origin", "main"])
        assert is_dangerous is True

    def test_git_reset_hard_rejected(self) -> None:
        """git reset --hard should be rejected."""
        pattern = DestructiveGitPattern()
        is_dangerous, _ = pattern.check(["git", "reset", "--hard", "HEAD~1"])
        assert is_dangerous is True

    def test_git_clean_f_rejected(self) -> None:
        """git clean -f should be rejected."""
        pattern = DestructiveGitPattern()
        is_dangerous, _ = pattern.check(["git", "clean", "-f"])
        assert is_dangerous is True

    def test_git_log_allowed(self) -> None:
        """git log should be allowed."""
        pattern = DestructiveGitPattern()
        is_dangerous, _ = pattern.check(["git", "log", "--oneline"])
        assert is_dangerous is False


class TestDestructiveRmPattern:
    """Tests for DestructiveRmPattern detection."""

    def test_rm_rf_rejected(self) -> None:
        """rm -rf should be rejected."""
        pattern = DestructiveRmPattern()
        is_dangerous, _ = pattern.check(["rm", "-rf", "/home"])
        assert is_dangerous is True

    def test_rm_r_rejected(self) -> None:
        """rm -r should be rejected."""
        pattern = DestructiveRmPattern()
        is_dangerous, _ = pattern.check(["rm", "-r", "/home"])
        assert is_dangerous is True

    def test_rm_single_file_allowed(self) -> None:
        """rm file.txt should be allowed."""
        pattern = DestructiveRmPattern()
        is_dangerous, _ = pattern.check(["rm", "file.txt"])
        assert is_dangerous is False


class TestPatternRegistry:
    """Tests for SECURITY_PATTERNS registry and composition."""

    def test_all_patterns_registered(self) -> None:
        """All pattern types should be in the registry."""
        pattern_types = {type(p).__name__ for p in SECURITY_PATTERNS}
        expected = {
            "DangerousCommandPattern",
            "ShellOperatorPattern",
            "CommandSubstitutionPattern",
            "CodeExecutionPattern",
            "DestructiveGitPattern",
            "DestructiveRmPattern",
        }
        assert expected.issubset(pattern_types), (
            f"Missing patterns: {expected - pattern_types}"
        )

    def test_check_all_patterns_integration(self) -> None:
        """check_all_patterns should integrate all registered patterns."""
        # Should catch sudo (DangerousCommandPattern)
        is_dangerous, _ = check_all_patterns(["sudo", "echo"])
        assert is_dangerous is True

        # Should catch pipe (ShellOperatorPattern)
        is_dangerous, _ = check_all_patterns(["cat", "file", "|", "grep"])
        assert is_dangerous is True

        # Should catch substitution (CommandSubstitutionPattern)
        is_dangerous, _ = check_all_patterns(["echo", "$(date)"])
        assert is_dangerous is True

        # Should catch code execution (CodeExecutionPattern)
        is_dangerous, _ = check_all_patterns(["python", "-c", "print(1)"])
        assert is_dangerous is True

    def test_get_pattern_names(self) -> None:
        """get_pattern_names should return all pattern class names."""
        names = get_pattern_names()
        assert isinstance(names, list)
        assert len(names) == len(SECURITY_PATTERNS)
        assert all(isinstance(n, str) for n in names)


class TestPatternExtensibility:
    """Tests that new patterns can be added without breaking existing logic."""

    def test_custom_pattern_can_be_created(self) -> None:
        """New pattern subclasses should be creatable."""
        from mellea.stdlib.tools._bash_patterns import BashSecurityPattern

        class CustomPattern(BashSecurityPattern):
            def check(self, argv: list[str]) -> tuple[bool, str]:
                if argv and argv[0] == "dangerous_custom":
                    return True, "Custom dangerous command"
                return False, ""

        pattern = CustomPattern()
        is_dangerous, _ = pattern.check(["dangerous_custom", "arg"])
        assert is_dangerous is True

    def test_safe_command_passes_all_patterns(self) -> None:
        """Safe commands should pass all patterns."""
        safe_commands = [
            ["echo", "hello"],
            ["pwd"],
            ["ls", "-la"],
            ["cat", "file.txt"],
            ["grep", "pattern", "file.txt"],
        ]

        for cmd in safe_commands:
            is_dangerous, reason = check_all_patterns(cmd)
            assert is_dangerous is False, f"Safe command {cmd} failed with: {reason}"
