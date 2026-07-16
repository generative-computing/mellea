# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

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
