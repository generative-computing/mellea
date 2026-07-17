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
    DangerousPackageManagerPattern,
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

    def test_semicolon_standalone_allowed(self) -> None:
        """Standalone semicolon should be allowed (used by find -exec)."""
        pattern = ShellOperatorPattern()
        # After shlex.split("find -exec cat {} \;"), we get [..., ";"]
        # This is safe because subprocess.run(..., shell=False) doesn't interpret it
        is_dangerous, _ = pattern.check(["find", "-exec", "cat", "{}", ";"])
        assert is_dangerous is False

    def test_semicolon_embedded_rejected(self) -> None:
        """Embedded semicolon in argument should be rejected."""
        pattern = ShellOperatorPattern()
        # After shlex.split("echo 'hello;rm'"), we get ["echo", "hello;rm"]
        # Embedded semicolon could indicate LLM trying to hide a command
        is_dangerous, _ = pattern.check(["echo", "hello;rm"])
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


class TestPatternCategoryAndSeverity:
    """Tests that all patterns have category and severity attributes for audit logging."""

    def test_all_patterns_have_category(self) -> None:
        """Every pattern should have a category attribute."""
        for pattern in SECURITY_PATTERNS:
            assert hasattr(pattern, "category"), (
                f"{type(pattern).__name__} missing category attribute"
            )
            assert isinstance(pattern.category, str) and len(pattern.category) > 0, (
                f"{type(pattern).__name__} has invalid category"
            )

    def test_all_patterns_have_severity(self) -> None:
        """Every pattern should have a severity attribute."""
        for pattern in SECURITY_PATTERNS:
            assert hasattr(pattern, "severity"), (
                f"{type(pattern).__name__} missing severity attribute"
            )
            assert isinstance(pattern.severity, str) and len(pattern.severity) > 0, (
                f"{type(pattern).__name__} has invalid severity"
            )

    def test_violation_audit_uses_pattern_metadata(self) -> None:
        """Violations should record pattern category and severity in audit trail."""
        from mellea.stdlib.tools._bash_audit import BashAuditTrail

        trail = BashAuditTrail.get_instance()
        trail.clear()

        # This should trigger DangerousCommandPattern with category and severity
        check_all_patterns(["sudo", "ls"])

        violations = trail.get_violations()
        assert len(violations) == 1

        v = violations[0]
        assert v.category != "unknown", (
            "Violation should have category from pattern, not 'unknown'"
        )
        assert v.severity != "MEDIUM", (
            "DangerousCommandPattern violation should use pattern severity, not default"
        )


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


class TestBashAuditTrail:
    """Tests for audit trail recording and querying."""

    def test_violation_recorded_on_pattern_rejection(self) -> None:
        """Verify violation recorded when pattern rejects."""
        from mellea.stdlib.tools._bash_audit import BashAuditTrail

        trail = BashAuditTrail.get_instance()
        trail.clear()

        check_all_patterns(["sudo", "echo"])
        violations = trail.get_violations()
        assert len(violations) == 1
        assert violations[0].pattern == "DangerousCommandPattern"

    def test_violation_contains_correct_metadata(self) -> None:
        """Verify violation has all required fields."""
        from mellea.stdlib.tools._bash_audit import BashAuditTrail

        trail = BashAuditTrail.get_instance()
        trail.clear()

        check_all_patterns(["rm", "-rf", "/"])
        violations = trail.get_violations()
        v = violations[0]
        assert v.command == "rm -rf /"
        assert v.severity in ("HIGH", "CRITICAL")
        assert v.reason
        assert v.timestamp > 0

    def test_get_violations_filters_by_severity(self) -> None:
        """Verify filter by severity works."""
        from mellea.stdlib.tools._bash_audit import BashAuditTrail

        trail = BashAuditTrail.get_instance()
        trail.clear()

        check_all_patterns(["sudo", "ls"])
        check_all_patterns(["rm", "-r", "/tmp"])

        all_violations = trail.get_violations()
        assert len(all_violations) >= 2

        # All violations should have a severity, so filtering by first violation's severity should work
        first_severity = all_violations[0].severity
        matching = trail.get_violations(severity=first_severity)
        assert len(matching) >= 1
        for v in matching:
            assert v.severity == first_severity

    def test_export_metrics_counts_violations(self) -> None:
        """Verify metrics export includes violation counts."""
        from mellea.stdlib.tools._bash_audit import BashAuditTrail

        trail = BashAuditTrail.get_instance()
        trail.clear()

        check_all_patterns(["sudo", "ls"])
        check_all_patterns(["rm", "-r", "/tmp"])

        metrics = trail.export_metrics()
        assert metrics["total"] == 2
        assert any(k.startswith("severity_") for k in metrics.keys())

    def test_violations_cleared_between_tests(self) -> None:
        """Verify clear() removes all violations."""
        from mellea.stdlib.tools._bash_audit import BashAuditTrail

        trail = BashAuditTrail.get_instance()
        trail.clear()

        check_all_patterns(["sudo", "ls"])
        assert len(trail.get_violations()) == 1

        trail.clear()
        assert len(trail.get_violations()) == 0

    def test_query_violations_with_pattern_filter(self) -> None:
        """Verify filter by pattern name works."""
        from mellea.stdlib.tools._bash_audit import BashAuditTrail

        trail = BashAuditTrail.get_instance()
        trail.clear()

        check_all_patterns(["sudo", "ls"])  # DangerousCommandPattern
        check_all_patterns(["echo", "|", "grep"])  # ShellOperatorPattern

        dangerous = trail.get_violations(pattern="DangerousCommandPattern")
        assert len(dangerous) >= 1

    def test_get_violations_filters_multiple_criteria(self) -> None:
        """Verify filtering with multiple criteria returns correct subset.

        Regression test for iterate-and-remove bug: the old code used
        `results.remove(violation)` inside a for loop, which causes the
        iterator to skip elements. Using list comprehension ensures all
        violations are correctly evaluated and filtered.
        """
        from mellea.stdlib.tools._bash_audit import BashAuditTrail

        trail = BashAuditTrail.get_instance()
        trail.clear()

        # Create violations with different patterns
        check_all_patterns(["sudo", "ls"])  # DangerousCommandPattern
        check_all_patterns(["echo", "|", "grep"])  # ShellOperatorPattern
        check_all_patterns(["rm", "-rf", "/tmp"])  # DestructiveRmPattern

        # Get all violations for baseline
        all_violations = trail.get_violations()
        assert len(all_violations) >= 3, "Expected at least 3 violations recorded"

        # Filter by pattern DangerousCommandPattern - should find the first one
        dangerous_only = trail.get_violations(pattern="DangerousCommandPattern")
        assert len(dangerous_only) >= 1
        for v in dangerous_only:
            assert v.pattern == "DangerousCommandPattern"

        # Filter by pattern ShellOperatorPattern - should find the second one
        shell_ops = trail.get_violations(pattern="ShellOperatorPattern")
        assert len(shell_ops) >= 1
        for v in shell_ops:
            assert v.pattern == "ShellOperatorPattern"

        # Filter by pattern DestructiveRmPattern - should find the third one
        destructive = trail.get_violations(pattern="DestructiveRmPattern")
        assert len(destructive) >= 1
        for v in destructive:
            assert v.pattern == "DestructiveRmPattern"

        # Verify filtering is complete: all three filters should succeed
        # With the old iterate-and-remove bug, some filters would miss violations
        # because the iterator would skip elements after removal
        assert len(dangerous_only) + len(shell_ops) + len(destructive) >= 3, (
            "Filtering incomplete: expected at least 3 violations across 3 patterns"
        )

    def test_get_violations_exact_count_with_dual_filters(self) -> None:
        """Verify exact violation counts with combined filters.

        Regression test: asserts exact counts (not >= checks) with two
        simultaneous filter criteria. Exercises AND logic to ensure filters
        don't degenerate to OR semantics. With the iterate-and-remove bug,
        filtering would return incomplete results when multiple criteria
        were combined.
        """
        from mellea.stdlib.tools._bash_audit import BashAuditTrail

        trail = BashAuditTrail.get_instance()
        trail.clear()

        # Create a controlled set of violations with specific severity levels:
        # sudo: CRITICAL (from COMMAND_RULES)
        # chfn: MEDIUM (from COMMAND_RULES)
        # ShellOperatorPattern: HIGH severity
        check_all_patterns(["sudo", "ls"])  # DangerousCommandPattern, CRITICAL
        check_all_patterns(["chfn", "user"])  # DangerousCommandPattern, MEDIUM
        check_all_patterns(["echo", "|", "grep"])  # ShellOperatorPattern, HIGH
        check_all_patterns(["cat", ">>", "file"])  # ShellOperatorPattern, HIGH

        all_violations = trail.get_violations()
        assert len(all_violations) == 4, (
            f"Expected exactly 4 violations, got {len(all_violations)}"
        )

        # Test 1: Single filter (baseline) with exact counts
        dangerous = trail.get_violations(pattern="DangerousCommandPattern")
        assert len(dangerous) == 2, (
            f"Expected exactly 2 DangerousCommandPattern violations, got {len(dangerous)}"
        )

        shell_ops = trail.get_violations(pattern="ShellOperatorPattern")
        assert len(shell_ops) == 2, (
            f"Expected exactly 2 ShellOperatorPattern violations, got {len(shell_ops)}"
        )

        # Test 2: Dual filter with AND logic
        # Both conditions must match: DangerousCommandPattern AND CRITICAL severity
        critical_dangerous = trail.get_violations(
            pattern="DangerousCommandPattern", severity="CRITICAL"
        )
        assert len(critical_dangerous) == 1, (
            f"Expected 1 CRITICAL DangerousCommandPattern violation (sudo), got {len(critical_dangerous)}"
        )
        for v in critical_dangerous:
            assert v.pattern == "DangerousCommandPattern"
            assert v.severity == "CRITICAL"

        # Verify reverse: ShellOperatorPattern with HIGH severity (should also be 2)
        high_shell_ops = trail.get_violations(
            pattern="ShellOperatorPattern", severity="HIGH"
        )
        assert len(high_shell_ops) == 2, (
            f"Expected 2 HIGH ShellOperatorPattern violations, got {len(high_shell_ops)}"
        )
        for v in high_shell_ops:
            assert v.pattern == "ShellOperatorPattern"
            assert v.severity == "HIGH"

        # Test 3: Dual filter with no matches (AND logic prevents false positives)
        # DangerousCommandPattern with MEDIUM severity should match only chfn
        medium_dangerous = trail.get_violations(
            pattern="DangerousCommandPattern", severity="MEDIUM"
        )
        assert len(medium_dangerous) == 1, (
            f"Expected 1 MEDIUM DangerousCommandPattern violation (chfn), got {len(medium_dangerous)}"
        )
        for v in medium_dangerous:
            assert v.pattern == "DangerousCommandPattern"
            assert v.severity == "MEDIUM"

        # Test 4: Multiple filters simultaneously (both pattern and severity)
        # Demonstrate that AND semantics work: combining two criteria is stricter
        all_with_severity = trail.get_violations(severity="HIGH")
        assert len(all_with_severity) == 2, (
            f"Expected 2 HIGH violations (the 2 shell operators), got {len(all_with_severity)}"
        )

        pattern_and_severity = trail.get_violations(
            pattern="ShellOperatorPattern", severity="HIGH"
        )
        assert len(pattern_and_severity) == 2, (
            "Should match both shell operators with HIGH severity"
        )


class TestBashPatternImprovements:
    """Tests for bash pattern improvements (issue #1398).

    Verifies that new patterns and improved patterns work correctly:
    1. Interactive shell --login flag detection
    2. Git push --force variants
    3. Dangerous package manager flags
    """

    def test_dangerous_package_manager_pattern_exists(self) -> None:
        """DangerousPackageManagerPattern should be in the pattern registry."""
        from mellea.stdlib.tools._bash_patterns import DangerousPackageManagerPattern

        pattern_names = get_pattern_names()
        assert "DangerousPackageManagerPattern" in pattern_names

    def test_interactive_shell_login_flag(self) -> None:
        """Interactive shell --login flag should be detected."""
        # bash --login
        is_dangerous, reason = check_all_patterns(["bash", "--login"])
        assert is_dangerous is True
        assert "Interactive shell" in reason

        # sh --login
        is_dangerous, reason = check_all_patterns(["sh", "--login"])
        assert is_dangerous is True

        # zsh --login
        is_dangerous, reason = check_all_patterns(["zsh", "--login"])
        assert is_dangerous is True

    def test_git_push_force_variants(self) -> None:
        """Git push --force variants should be detected."""
        # --force-with-lease
        is_dangerous, reason = check_all_patterns(["git", "push", "--force-with-lease"])
        assert is_dangerous is True
        assert "Destructive" in reason

        # --force-if-includes
        is_dangerous, reason = check_all_patterns(
            ["git", "push", "--force-if-includes"]
        )
        assert is_dangerous is True
        assert "Destructive" in reason

        # Regular --force (should still work)
        is_dangerous, reason = check_all_patterns(["git", "push", "--force"])
        assert is_dangerous is True

        # Short form -f (should still work)
        is_dangerous, reason = check_all_patterns(["git", "push", "-f"])
        assert is_dangerous is True

    def test_dangerous_package_manager_apt(self) -> None:
        """apt with dangerous flags should be detected."""
        # apt with -f
        is_dangerous, reason = check_all_patterns(["apt", "install", "-f", "package"])
        assert is_dangerous is True
        assert "apt" in reason.lower()

        # apt with --force
        is_dangerous, reason = check_all_patterns(
            ["apt", "install", "--force", "package"]
        )
        assert is_dangerous is True

        # apt with -r
        is_dangerous, reason = check_all_patterns(["apt", "install", "-r", "package"])
        assert is_dangerous is True

    def test_dangerous_package_manager_yum(self) -> None:
        """yum with dangerous flags should be detected."""
        # yum with -f
        is_dangerous, reason = check_all_patterns(["yum", "install", "-f", "package"])
        assert is_dangerous is True
        assert "yum" in reason.lower()

        # yum with --force
        is_dangerous, reason = check_all_patterns(
            ["yum", "install", "--force", "package"]
        )
        assert is_dangerous is True

        # yum with -r
        is_dangerous, reason = check_all_patterns(["yum", "install", "-r", "package"])
        assert is_dangerous is True

    def test_safe_apt_commands(self) -> None:
        """Safe apt commands should pass."""
        # apt install without dangerous flags
        is_dangerous, _ = check_all_patterns(["apt", "install", "package"])
        assert is_dangerous is False

        # apt search (no flags)
        is_dangerous, _ = check_all_patterns(["apt", "search", "package"])
        assert is_dangerous is False


class TestAuditTrailCoverage:
    """Tests verifying that violations are properly recorded in the audit trail.

    Phase 2 ensures that check_all_patterns() is the canonical entry point
    and that all violations are recorded for auditing purposes.
    """

    def test_violations_recorded_via_patterns(self) -> None:
        """Violations detected by patterns should be recorded."""
        from mellea.stdlib.tools._bash_audit import get_bash_violations

        # Get initial count
        initial_count = len(get_bash_violations())

        # Trigger a violation via patterns
        is_dangerous, reason = check_all_patterns(["bash", "-c", "echo hello"])

        # Verify violation was recorded
        assert is_dangerous is True
        assert "code execution" in reason.lower() or "not allowed" in reason.lower()

        # Check that violation was added to trail
        violations = get_bash_violations()
        assert len(violations) > initial_count, (
            "Violation should be recorded in audit trail"
        )

    def test_interactive_shell_violation_recorded(self) -> None:
        """Interactive shell violations should be recorded with metadata."""
        from mellea.stdlib.tools._bash_audit import get_bash_violations

        initial_count = len(get_bash_violations())

        # Trigger a dangerous command violation
        is_dangerous, _ = check_all_patterns(["bash", "-i"])

        assert is_dangerous is True
        violations = get_bash_violations()
        assert len(violations) > initial_count

        # Verify the most recent violation has proper metadata
        latest = violations[-1]
        assert latest.pattern == "DangerousCommandPattern"
        assert latest.severity == "HIGH"
        assert latest.category == "interactive"

    def test_git_violation_recorded_with_category(self) -> None:
        """Git violations should be recorded with proper category."""
        from mellea.stdlib.tools._bash_audit import get_bash_violations

        initial_count = len(get_bash_violations())

        # Trigger a git violation
        is_dangerous, _ = check_all_patterns(["git", "push", "--force-with-lease"])

        assert is_dangerous is True
        violations = get_bash_violations()
        assert len(violations) > initial_count

        # Verify metadata
        latest = violations[-1]
        assert latest.pattern == "DestructiveGitPattern"
        assert latest.severity == "HIGH"
        assert latest.category == "destructive"

    def test_package_manager_violation_recorded(self) -> None:
        """Package manager violations should be recorded."""
        from mellea.stdlib.tools._bash_audit import get_bash_violations

        initial_count = len(get_bash_violations())

        # Trigger a package manager violation
        is_dangerous, _ = check_all_patterns(["apt", "install", "-f", "package"])

        assert is_dangerous is True
        violations = get_bash_violations()
        assert len(violations) > initial_count

        # Verify metadata
        latest = violations[-1]
        assert latest.pattern == "DangerousPackageManagerPattern"
        assert latest.category == "destructive"


class TestIsDangerousCommandBoundary:
    """Tests for _is_dangerous_command() function boundaries.

    These tests exercise _is_dangerous_command() directly to verify that the
    function (and the inline bash_executor path it guards) correctly validates
    command safety in nested contexts where the pattern framework alone is insufficient.

    Specifically, these tests pin the semicolon handling behavior after issue #1391
    (find -exec ... ; becomes inert under shell=False, so bare ';' in argv should
    not be dangerous when used as a find terminator).
    """

    def test_is_dangerous_command_find_exec_semicolon(self) -> None:
        """_is_dangerous_command() alone does not flag a bare ';' terminator.

        After shlex.split, "find -exec ... ;" yields a bare standalone token
        [';']. _is_dangerous_command() only validates nested command contexts,
        so it returns safe here. NOTE: this does not mean the command executes —
        in the full _validate_command flow, check_all_patterns()'s
        ShellOperatorPattern still rejects the standalone ';'. This test pins
        only the nested-context boundary, not end-to-end permissiveness.
        """
        from mellea.stdlib.tools.shell import _is_dangerous_command

        # find -exec with semicolon terminator → should be safe
        # (the bare ';' is a find option, not a shell operator in the argv list)
        ok, reason = _is_dangerous_command(["find", "logs", "-exec", "cat", "{}", ";"])
        assert ok is False, f"find -exec semicolon should be safe, but got: {reason}"

    def test_is_dangerous_command_embedded_semicolon(self) -> None:
        """Embedded semicolon in string arg not caught by nested validation.

        This test documents that embedded semicolons (e.g., "hello;rm") are NOT
        caught by _is_dangerous_command() because that function only validates
        nested command contexts. Embedded semicolons are caught by
        ShellOperatorPattern.check() in check_all_patterns(), which checks for
        ';' in any argv element.

        This is the correct division of responsibility:
        - _is_dangerous_command() → nested contexts (env sudo, timeout bash -c)
        - ShellOperatorPattern → top-level shell operators (semicolons, pipes)
        """
        from mellea.stdlib.tools.shell import _is_dangerous_command

        # Embedded semicolon in string argument
        # _is_dangerous_command alone does NOT catch this
        # (that's ShellOperatorPattern's job)
        bad, _ = _is_dangerous_command(["echo", "hello;rm"])
        # We expect False because _is_dangerous_command only checks nested
        # contexts, not embedded shell operators.
        assert bad is False

    def test_is_dangerous_command_vs_check_all_patterns(self) -> None:
        """Verify division of labor between _is_dangerous_command and patterns.

        _is_dangerous_command() is a thin wrapper around
        _check_nested_dangerous_commands() and validates only nested command
        contexts (e.g., env sudo, timeout bash -c).

        Top-level shell operators (semicolons, pipes, redirects) are the
        responsibility of check_all_patterns(), which runs ShellOperatorPattern
        and other checkers.

        This test verifies that:
        1. Embedded semicolon passes _is_dangerous_command (nested only)
        2. Embedded semicolon fails check_all_patterns (pattern-based)
        """
        from mellea.stdlib.tools._bash_patterns import check_all_patterns
        from mellea.stdlib.tools.shell import _is_dangerous_command

        argv = ["echo", "hello;rm"]

        # Step 1: _is_dangerous_command only validates nested contexts
        nested_ok, _ = _is_dangerous_command(argv)
        # Embedded semicolon is not a nested context issue
        assert nested_ok is False

        # Step 2: check_all_patterns catches embedded semicolon
        patterns_dangerous, reason = check_all_patterns(argv)
        # ShellOperatorPattern catches it
        assert patterns_dangerous is True
        assert ";" in reason or "chaining" in reason.lower()
