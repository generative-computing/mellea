"""Tests for prompt linting utilities."""

import os
from unittest.mock import patch

from mellea.helpers.prompt_linting import (
    _DOCSTRING_WORD_THRESHOLD,
    _INSTRUCTION_WORD_THRESHOLD,
    AntiPatternType,
    AntiPatternWarning,
    _count_words,
    _has_agent_keywords,
    _has_multiple_sections,
    _is_warnings_enabled,
    check_genslot_docstring,
    check_instruction_description,
    warn_genslot_antipatterns,
    warn_instruction_antipatterns,
)


def test_count_words():
    assert _count_words("") == 0
    assert _count_words("hello") == 1
    assert _count_words("hello world") == 2
    assert _count_words("one two three four five") == 5


def test_has_agent_keywords_positive():
    assert _has_agent_keywords("You are a helpful assistant")
    assert _has_agent_keywords("Your role is to help users")
    assert _has_agent_keywords("Step 1: Do this")
    assert _has_agent_keywords("First, analyze the input")
    assert _has_agent_keywords("Then, process the data")
    assert _has_agent_keywords("Finally, return the result")
    assert _has_agent_keywords("You will perform this task")
    assert _has_agent_keywords("You should be helpful")
    assert _has_agent_keywords("You must follow these rules")
    assert _has_agent_keywords("Act as a coding assistant")
    assert _has_agent_keywords("Behave as an expert")
    assert _has_agent_keywords("Imagine you are a robot")
    assert _has_agent_keywords("Pretend you are an AI")
    assert _has_agent_keywords("Role: Assistant")
    assert _has_agent_keywords("Task: Summarize")
    assert _has_agent_keywords("# Header with markdown")
    assert _has_agent_keywords("Context: Some context here")
    assert _has_agent_keywords("Instructions: Follow these")


def test_has_agent_keywords_negative():
    assert not _has_agent_keywords("Summarize the document")
    assert not _has_agent_keywords("Extract the main points")
    assert not _has_agent_keywords("Translate to French")
    assert not _has_agent_keywords("Generate a short story")
    assert not _has_agent_keywords("Write a haiku about nature")


def test_has_multiple_sections():
    assert _has_multiple_sections("Just a simple prompt") == 0
    assert _has_multiple_sections("Section 1\n\n\nSection 2") >= 1
    assert _has_multiple_sections("Intro\n# Section 1\nContent") >= 1
    assert _has_multiple_sections("Part 1\n---\nPart 2") >= 1


def test_is_warnings_enabled():
    with patch.dict(os.environ, {}, clear=True):
        os.environ.pop("MELLEA_WARN_ANTIPATTERNS", None)
        assert not _is_warnings_enabled()

    with patch.dict(os.environ, {"MELLEA_WARN_ANTIPATTERNS": "1"}):
        assert _is_warnings_enabled()

    with patch.dict(os.environ, {"MELLEA_WARN_ANTIPATTERNS": "true"}):
        assert _is_warnings_enabled()

    with patch.dict(os.environ, {"MELLEA_WARN_ANTIPATTERNS": "yes"}):
        assert _is_warnings_enabled()

    with patch.dict(os.environ, {"MELLEA_WARN_ANTIPATTERNS": "0"}):
        assert not _is_warnings_enabled()

    with patch.dict(os.environ, {"MELLEA_WARN_ANTIPATTERNS": "false"}):
        assert not _is_warnings_enabled()


def test_check_instruction_description_none():
    warnings = check_instruction_description(None)
    assert warnings == []


def test_check_instruction_description_short_clean():
    warnings = check_instruction_description("Summarize the document")
    assert warnings == []


def test_check_instruction_description_long():
    long_text = "word " * (_INSTRUCTION_WORD_THRESHOLD + 10)
    warnings = check_instruction_description(long_text)
    assert len(warnings) >= 1
    assert any(w.pattern_type == AntiPatternType.LONG_INSTRUCTION for w in warnings)


def test_check_instruction_description_agent_style():
    agent_prompt = "You are a helpful AI assistant. Your role is to help users."
    warnings = check_instruction_description(agent_prompt)
    assert len(warnings) >= 1
    assert any(
        w.pattern_type == AntiPatternType.AGENT_STYLE_INSTRUCTION for w in warnings
    )


def test_check_instruction_description_multi_section():
    multi_section = "Introduction\n\n\nSection 1: Overview\n\n\nSection 2: Details"
    warnings = check_instruction_description(multi_section)
    assert len(warnings) >= 1
    assert any(
        w.pattern_type == AntiPatternType.MULTI_SECTION_INSTRUCTION for w in warnings
    )


def test_check_instruction_description_combined_antipatterns():
    combined = """You are an expert data analyst.

Your role is to analyze data carefully.

# Step 1: Data Collection
First, collect all relevant data from the sources provided.

# Step 2: Analysis
Then, analyze the data using statistical methods.
"""
    warnings = check_instruction_description(combined)
    pattern_types = {w.pattern_type for w in warnings}
    assert AntiPatternType.AGENT_STYLE_INSTRUCTION in pattern_types


def test_check_genslot_docstring_none():
    warnings = check_genslot_docstring(None)
    assert warnings == []


def test_check_genslot_docstring_short_clean():
    warnings = check_genslot_docstring("Extract sentiment from text.")
    assert warnings == []


def test_check_genslot_docstring_long():
    long_text = "word " * (_DOCSTRING_WORD_THRESHOLD + 10)
    warnings = check_genslot_docstring(long_text)
    assert len(warnings) >= 1
    assert any(w.pattern_type == AntiPatternType.LONG_DOCSTRING for w in warnings)


def test_check_genslot_docstring_agent_style():
    agent_docstring = "You are a classifier. Your role is to classify text."
    warnings = check_genslot_docstring(agent_docstring)
    assert len(warnings) >= 1
    assert any(
        w.pattern_type == AntiPatternType.AGENT_STYLE_DOCSTRING for w in warnings
    )


def test_warn_instruction_antipatterns_disabled_by_default():
    with patch.dict(os.environ, {}, clear=True):
        os.environ.pop("MELLEA_WARN_ANTIPATTERNS", None)
        with patch("mellea.helpers.prompt_linting.FancyLogger") as mock_logger:
            warn_instruction_antipatterns("You are a helpful assistant.")
            mock_logger.get_logger.assert_not_called()


def test_warn_instruction_antipatterns_enabled():
    with patch.dict(os.environ, {"MELLEA_WARN_ANTIPATTERNS": "1"}):
        with patch("mellea.helpers.prompt_linting.FancyLogger") as mock_logger:
            mock_warning = mock_logger.get_logger.return_value.warning
            warn_instruction_antipatterns("You are a helpful assistant.")
            mock_warning.assert_called()


def test_warn_genslot_antipatterns_disabled_by_default():
    with patch.dict(os.environ, {}, clear=True):
        os.environ.pop("MELLEA_WARN_ANTIPATTERNS", None)
        with patch("mellea.helpers.prompt_linting.FancyLogger") as mock_logger:
            warn_genslot_antipatterns("You are a classifier assistant.")
            mock_logger.get_logger.assert_not_called()


def test_warn_genslot_antipatterns_enabled():
    with patch.dict(os.environ, {"MELLEA_WARN_ANTIPATTERNS": "1"}):
        with patch("mellea.helpers.prompt_linting.FancyLogger") as mock_logger:
            mock_warning = mock_logger.get_logger.return_value.warning
            warn_genslot_antipatterns("You are a classifier assistant.")
            mock_warning.assert_called()


def test_warn_instruction_antipatterns_none_input():
    with patch.dict(os.environ, {"MELLEA_WARN_ANTIPATTERNS": "1"}):
        warn_instruction_antipatterns(None)


def test_warn_genslot_antipatterns_none_input():
    with patch.dict(os.environ, {"MELLEA_WARN_ANTIPATTERNS": "1"}):
        warn_genslot_antipatterns(None)


def test_instruction_integration():
    from mellea.stdlib.instruction import Instruction

    with patch.dict(os.environ, {"MELLEA_WARN_ANTIPATTERNS": "1"}):
        with patch("mellea.helpers.prompt_linting.FancyLogger") as mock_logger:
            mock_warning = mock_logger.get_logger.return_value.warning
            _ = Instruction(
                description="You are a helpful assistant. Your role is to help users."
            )
            mock_warning.assert_called()


def test_generative_slot_integration():
    from mellea.stdlib.genslot import SyncGenerativeSlot

    with patch.dict(os.environ, {"MELLEA_WARN_ANTIPATTERNS": "1"}):
        with patch("mellea.helpers.prompt_linting.FancyLogger") as mock_logger:
            mock_warning = mock_logger.get_logger.return_value.warning

            @SyncGenerativeSlot
            def classify(text: str) -> str:
                """You are a classifier. Your role is to classify text."""

            mock_warning.assert_called()


def test_antipattern_warning_creation():
    warning = AntiPatternWarning(
        pattern_type=AntiPatternType.LONG_INSTRUCTION,
        message="Test message",
        suggestion="Test suggestion",
    )
    assert warning.pattern_type == AntiPatternType.LONG_INSTRUCTION
    assert warning.message == "Test message"
    assert warning.suggestion == "Test suggestion"


def test_antipattern_type_enum():
    assert AntiPatternType.LONG_INSTRUCTION
    assert AntiPatternType.AGENT_STYLE_INSTRUCTION
    assert AntiPatternType.MULTI_SECTION_INSTRUCTION
    assert AntiPatternType.LONG_DOCSTRING
    assert AntiPatternType.AGENT_STYLE_DOCSTRING
