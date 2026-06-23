"""Structured bash security guardrails framework.

Provides systematic organization of bash command safety rules with categories,
severity levels, and rationale. This enables:
- Clear documentation of what is blocked and why
- Auditing of coverage across threat categories
- Principled decision-making about future allowlist expansions
- Test-driven verification of guardrail completeness
"""

from dataclasses import dataclass
from enum import Enum


class CommandCategory(Enum):
    """Categorization of dangerous commands by threat profile."""

    PRIVILEGE_ESCALATION = "privilege_escalation"
    INTERACTIVE = "interactive"
    DESTRUCTIVE = "destructive"
    ENVIRONMENT_CHANGING = "environment_changing"
    FILE_PERMISSIONS = "file_permissions"


class Severity(Enum):
    """Risk severity of a command if misused."""

    CRITICAL = "critical"  # Allows full system compromise
    HIGH = "high"  # Causes significant damage or data loss
    MEDIUM = "medium"  # Limited damage but noteworthy risk
    LOW = "low"  # Mostly informational; rarely legitimately needed


@dataclass
class CommandRule:
    """Rule defining why a command is dangerous and how it's handled.

    Attributes:
        category: Threat category this command belongs to.
        severity: Risk severity if the command is executed unexpectedly.
        rationale: Explanation of why this command is blocked.
        safe_with: List of conditions under which the command might be safe
            (informational only; not enforced yet).
    """

    category: CommandCategory
    severity: Severity
    rationale: str
    safe_with: list[str] | None = None


# Canonical mapping of dangerous commands to their security rules.
# This is the authoritative reference for why each command is blocked.
COMMAND_RULES: dict[str, CommandRule] = {
    # Privilege escalation: always dangerous
    "sudo": CommandRule(
        category=CommandCategory.PRIVILEGE_ESCALATION,
        severity=Severity.CRITICAL,
        rationale="Elevation to root requires human interaction or stored credentials. Cannot be automated safely in an untrusted pipeline.",
    ),
    "su": CommandRule(
        category=CommandCategory.PRIVILEGE_ESCALATION,
        severity=Severity.CRITICAL,
        rationale="User switching (su) requires password input or stored credentials. Cannot be automated safely.",
    ),
    "doas": CommandRule(
        category=CommandCategory.PRIVILEGE_ESCALATION,
        severity=Severity.CRITICAL,
        rationale="Alternative privilege escalation (OpenBSD/BSD). Same risk as sudo.",
    ),
    # Interactive shells: would block LLM workflow
    "bash": CommandRule(
        category=CommandCategory.INTERACTIVE,
        severity=Severity.HIGH,
        rationale="Interactive bash shells (-i flag) block the LLM workflow. Non-interactive usage (e.g., bash script.sh) is allowed.",
        safe_with=["non_interactive_mode"],
    ),
    "sh": CommandRule(
        category=CommandCategory.INTERACTIVE,
        severity=Severity.HIGH,
        rationale="Interactive sh shells block the LLM workflow. Non-interactive usage is allowed.",
        safe_with=["non_interactive_mode"],
    ),
    "zsh": CommandRule(
        category=CommandCategory.INTERACTIVE,
        severity=Severity.HIGH,
        rationale="Interactive zsh shells block the LLM workflow.",
    ),
    "ksh": CommandRule(
        category=CommandCategory.INTERACTIVE,
        severity=Severity.HIGH,
        rationale="Interactive ksh shells block the LLM workflow.",
    ),
    "tcsh": CommandRule(
        category=CommandCategory.INTERACTIVE,
        severity=Severity.HIGH,
        rationale="Interactive tcsh shells block the LLM workflow.",
    ),
    # User/group/password management: permanent system changes
    "passwd": CommandRule(
        category=CommandCategory.ENVIRONMENT_CHANGING,
        severity=Severity.CRITICAL,
        rationale="Password changes require user interaction and permanently alter system state.",
    ),
    "visudo": CommandRule(
        category=CommandCategory.ENVIRONMENT_CHANGING,
        severity=Severity.CRITICAL,
        rationale="Sudo configuration changes require human validation and affect system security model.",
    ),
    "chsh": CommandRule(
        category=CommandCategory.ENVIRONMENT_CHANGING,
        severity=Severity.HIGH,
        rationale="Change shell permanently alters user environment; rarely needed in automation.",
    ),
    "chfn": CommandRule(
        category=CommandCategory.ENVIRONMENT_CHANGING,
        severity=Severity.MEDIUM,
        rationale="Change GECOS (user info) has low direct risk but indicates attempt to alter user identity.",
    ),
    "useradd": CommandRule(
        category=CommandCategory.ENVIRONMENT_CHANGING,
        severity=Severity.HIGH,
        rationale="User creation permanently alters system and requires elevated privileges.",
    ),
    "userdel": CommandRule(
        category=CommandCategory.ENVIRONMENT_CHANGING,
        severity=Severity.HIGH,
        rationale="User deletion permanently alters system and affects file ownership.",
    ),
    "usermod": CommandRule(
        category=CommandCategory.ENVIRONMENT_CHANGING,
        severity=Severity.HIGH,
        rationale="User modification (groups, shells, etc.) permanently alters system state.",
    ),
    "groupadd": CommandRule(
        category=CommandCategory.ENVIRONMENT_CHANGING,
        severity=Severity.HIGH,
        rationale="Group creation permanently alters system.",
    ),
    "groupdel": CommandRule(
        category=CommandCategory.ENVIRONMENT_CHANGING,
        severity=Severity.HIGH,
        rationale="Group deletion permanently alters system.",
    ),
    "groupmod": CommandRule(
        category=CommandCategory.ENVIRONMENT_CHANGING,
        severity=Severity.HIGH,
        rationale="Group modification permanently alters system.",
    ),
}


class ShellOperatorCategory(Enum):
    """Category of shell operators that bypass argv parsing."""

    REDIRECTION = "redirection"  # >, >>, <, >&, etc.
    PIPE = "pipe"  # |, |&
    CHAINING = "chaining"  # ;, &&, ||
    BACKGROUND = "background"  # &
    SUBSTITUTION = "substitution"  # $(...), `...`, ${...}


@dataclass
class ShellOperatorRule:
    """Rule for detecting and blocking shell operators.

    Attributes:
        operator: The operator token (e.g., ">>", "&&").
        category: Category of operator (redirection, pipe, etc.).
        rationale: Why this operator is blocked.
        blocked_if: Description of when it's blocked (e.g., "always", "as standalone token").
    """

    operator: str
    category: ShellOperatorCategory
    rationale: str
    blocked_if: str = "always"


# Canonical shell operators that are always dangerous.
SHELL_OPERATOR_RULES: dict[str, ShellOperatorRule] = {
    # Redirection operators
    ">": ShellOperatorRule(
        operator=">",
        category=ShellOperatorCategory.REDIRECTION,
        rationale="Output redirection allows writing arbitrary output to any file. Use subprocess or file operations instead.",
        blocked_if="standalone or as prefix (e.g., >file, >&2)",
    ),
    ">>": ShellOperatorRule(
        operator=">>",
        category=ShellOperatorCategory.REDIRECTION,
        rationale="Append redirection allows modifying arbitrary files.",
        blocked_if="standalone or as prefix",
    ),
    "<": ShellOperatorRule(
        operator="<",
        category=ShellOperatorCategory.REDIRECTION,
        rationale="Input redirection bypasses file access controls.",
        blocked_if="standalone or as prefix",
    ),
    ">&": ShellOperatorRule(
        operator=">&",
        category=ShellOperatorCategory.REDIRECTION,
        rationale="Stream redirection (stderr/stdout redirect) bypasses output controls.",
        blocked_if="as prefix (e.g., >&2)",
    ),
    "<<": ShellOperatorRule(
        operator="<<",
        category=ShellOperatorCategory.REDIRECTION,
        rationale="Heredoc redirection embeds multi-line input, reducing transparency.",
        blocked_if="standalone or as prefix",
    ),
    # Pipe operators
    "|": ShellOperatorRule(
        operator="|",
        category=ShellOperatorCategory.PIPE,
        rationale="Pipes chain commands without explicit control flow, enabling complex attacks.",
        blocked_if="standalone",
    ),
    "|&": ShellOperatorRule(
        operator="|&",
        category=ShellOperatorCategory.PIPE,
        rationale="Coproc pipes (bash 4.0+) enable bidirectional command interaction.",
        blocked_if="standalone",
    ),
    # Chaining operators
    ";": ShellOperatorRule(
        operator=";",
        category=ShellOperatorCategory.CHAINING,
        rationale="Semicolon chains commands regardless of success. Use Python control flow instead.",
        blocked_if="substring (dangerous even in quoted contexts in some shells)",
    ),
    "&&": ShellOperatorRule(
        operator="&&",
        category=ShellOperatorCategory.CHAINING,
        rationale="AND operator chains commands conditionally. Use Python if/else instead.",
        blocked_if="standalone",
    ),
    "||": ShellOperatorRule(
        operator="||",
        category=ShellOperatorCategory.CHAINING,
        rationale="OR operator chains commands on failure. Use Python try/except instead.",
        blocked_if="standalone",
    ),
    # Background operator
    "&": ShellOperatorRule(
        operator="&",
        category=ShellOperatorCategory.BACKGROUND,
        rationale="Background execution reduces visibility and control over command lifetime.",
        blocked_if="standalone",
    ),
}


def get_command_rules_by_category(category: CommandCategory) -> dict[str, CommandRule]:
    """Get all commands in a specific category.

    Args:
        category: The category to filter by.

    Returns:
        Dictionary of command -> rule for all commands in the category.
    """
    return {
        cmd: rule for cmd, rule in COMMAND_RULES.items() if rule.category == category
    }


def get_high_severity_commands() -> dict[str, CommandRule]:
    """Get all commands with high or critical severity.

    Returns:
        Dictionary of command -> rule for high/critical severity commands.
    """
    return {
        cmd: rule
        for cmd, rule in COMMAND_RULES.items()
        if rule.severity in (Severity.CRITICAL, Severity.HIGH)
    }


def audit_guardrails_coverage() -> dict[str, list[str]]:
    """Audit the coverage of guardrails across threat categories.

    Returns:
        Dictionary mapping category names to lists of commands in that category.
    """
    coverage: dict[str, list[str]] = {}
    for category in CommandCategory:
        commands = list(get_command_rules_by_category(category).keys())
        coverage[category.value] = commands
    return coverage
