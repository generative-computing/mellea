"""Prompt linting utilities for detecting mega-prompt anti-patterns. Enable via MELLEA_WARN_ANTIPATTERNS=1."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from enum import Enum

from mellea.helpers.fancy_logger import FancyLogger


class AntiPatternType(Enum):
    """Types of anti-patterns that can be detected."""

    LONG_INSTRUCTION = "long_instruction"
    AGENT_STYLE_INSTRUCTION = "agent_style_instruction"
    MULTI_SECTION_INSTRUCTION = "multi_section_instruction"
    LONG_DOCSTRING = "long_docstring"
    AGENT_STYLE_DOCSTRING = "agent_style_docstring"


@dataclass
class AntiPatternWarning:
    """A warning about a detected anti-pattern."""

    pattern_type: AntiPatternType
    message: str
    suggestion: str


# Thresholds for detection
_INSTRUCTION_CHAR_THRESHOLD = 500
_INSTRUCTION_WORD_THRESHOLD = 100
_DOCSTRING_CHAR_THRESHOLD = 300
_DOCSTRING_WORD_THRESHOLD = 60

# Agent-style keywords that suggest this should be m.chat() instead
_AGENT_KEYWORDS = [
    r"\byou are\b",
    r"\byour role\b",
    r"\bstep \d+\b",
    r"\bfirst,?\s",
    r"\bthen,?\s",
    r"\bfinally,?\s",
    r"\byou will\b",
    r"\byou should\b",
    r"\byou must\b",
    r"\bact as\b",
    r"\bbehave as\b",
    r"\bimagine you\b",
    r"\bpretend you\b",
    r"\brole:\s*\w+",
    r"\btask:\s*\w+",
    r"\bcontext:\s*\w+",
    r"\binstructions:\s*\w+",
    r"^#+ ",  # Markdown headers
]

# Compiled regex for efficiency
_AGENT_PATTERN = re.compile("|".join(_AGENT_KEYWORDS), re.IGNORECASE | re.MULTILINE)

# Pattern to detect multi-section prompts
_SECTION_PATTERN = re.compile(r"\n\n\n+|\n#{1,3} |\n\*{2,}|\n-{3,}", re.MULTILINE)


def _is_warnings_enabled() -> bool:
    """Check if anti-pattern warnings are enabled via environment variable."""
    return os.environ.get("MELLEA_WARN_ANTIPATTERNS", "").lower() in (
        "1",
        "true",
        "yes",
    )


def _count_words(text: str) -> int:
    """Count the number of words in a text."""
    return len(text.split())


def _has_agent_keywords(text: str) -> bool:
    """Check if text contains agent-style keywords."""
    return bool(_AGENT_PATTERN.search(text))


def _has_multiple_sections(text: str) -> int:
    """Count the number of section breaks in text."""
    return len(_SECTION_PATTERN.findall(text))


def check_instruction_description(description: str) -> list[AntiPatternWarning]:
    """Check an instruction description for anti-patterns."""
    warnings: list[AntiPatternWarning] = []

    if description is None:
        return warnings

    char_count = len(description)
    word_count = _count_words(description)

    # Check for excessive length
    if (
        char_count > _INSTRUCTION_CHAR_THRESHOLD
        or word_count > _INSTRUCTION_WORD_THRESHOLD
    ):
        warnings.append(
            AntiPatternWarning(
                pattern_type=AntiPatternType.LONG_INSTRUCTION,
                message=f"Instruction description is very long ({word_count} words, {char_count} chars).",
                suggestion="Consider using m.chat() for complex prompts or decomposing into smaller instructions.",
            )
        )

    # Check for agent-style keywords
    if _has_agent_keywords(description):
        warnings.append(
            AntiPatternWarning(
                pattern_type=AntiPatternType.AGENT_STYLE_INSTRUCTION,
                message="Instruction description contains agent-style language patterns.",
                suggestion="Use m.chat() for agent-style prompts or refactor into composable instructions.",
            )
        )

    # Check for multi-section structure
    section_count = _has_multiple_sections(description)
    if section_count >= 2:
        warnings.append(
            AntiPatternWarning(
                pattern_type=AntiPatternType.MULTI_SECTION_INSTRUCTION,
                message=f"Instruction description has multiple sections ({section_count + 1} detected).",
                suggestion="Decompose into separate instructions or use m.chat() for complex prompts.",
            )
        )

    return warnings


def check_genslot_docstring(docstring: str | None) -> list[AntiPatternWarning]:
    """Check a generative slot docstring for anti-patterns."""
    warnings: list[AntiPatternWarning] = []

    if docstring is None:
        return warnings

    char_count = len(docstring)
    word_count = _count_words(docstring)

    # Check for excessive length
    if char_count > _DOCSTRING_CHAR_THRESHOLD or word_count > _DOCSTRING_WORD_THRESHOLD:
        warnings.append(
            AntiPatternWarning(
                pattern_type=AntiPatternType.LONG_DOCSTRING,
                message=f"Generative slot docstring is very long ({word_count} words, {char_count} chars).",
                suggestion="Docstrings should be concise function descriptions. Use m.instruct() or m.chat() for complex prompts.",
            )
        )

    # Check for agent-style keywords
    if _has_agent_keywords(docstring):
        warnings.append(
            AntiPatternWarning(
                pattern_type=AntiPatternType.AGENT_STYLE_DOCSTRING,
                message="Generative slot docstring contains agent-style language patterns.",
                suggestion="Docstrings should describe what the function does, not how the LLM should behave. Consider using m.chat() instead.",
            )
        )

    return warnings


def warn_instruction_antipatterns(description: str | None) -> None:
    """Emit warnings if anti-patterns detected. Respects MELLEA_WARN_ANTIPATTERNS env var."""
    if not _is_warnings_enabled():
        return

    if description is None:
        return

    desc_str = str(description)
    warnings = check_instruction_description(desc_str)

    for warning in warnings:
        FancyLogger.get_logger().warning(
            f"[Mellea Anti-Pattern] {warning.message} {warning.suggestion}"
        )


def warn_genslot_antipatterns(docstring: str | None) -> None:
    """Emit warnings if anti-patterns detected. Respects MELLEA_WARN_ANTIPATTERNS env var."""
    if not _is_warnings_enabled():
        return

    if docstring is None:
        return

    warnings = check_genslot_docstring(docstring)

    for warning in warnings:
        FancyLogger.get_logger().warning(
            f"[Mellea Anti-Pattern] {warning.message} {warning.suggestion}"
        )
