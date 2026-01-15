"""Unit tests for SOFAISamplingStrategy."""

from unittest.mock import MagicMock

import pytest

from mellea.backends import Backend
from mellea.core import Requirement, ValidationResult
from mellea.stdlib.components import Instruction, Message, ModelOutputThunk
from mellea.stdlib.context import ChatContext
from mellea.stdlib.sampling import SOFAISamplingStrategy


class TestSOFAIInit:
    """Test SOFAISamplingStrategy initialization."""

    def test_init_valid_backends(self):
        """Test initialization with valid backends."""
        s1 = MagicMock(spec=Backend)
        s2 = MagicMock(spec=Backend)

        strategy = SOFAISamplingStrategy(s1_solver_backend=s1, s2_solver_backend=s2)

        assert strategy.s1_solver_backend is s1
        assert strategy.s2_solver_backend is s2
        assert strategy.loop_budget == 3  # default
        assert strategy.s2_solver_mode == "fresh_start"  # default

    def test_init_with_judge_backend(self):
        """Test initialization with judge backend."""
        s1 = MagicMock(spec=Backend)
        s2 = MagicMock(spec=Backend)
        judge = MagicMock(spec=Backend)

        strategy = SOFAISamplingStrategy(
            s1_solver_backend=s1,
            s2_solver_backend=s2,
            judge_backend=judge,
            feedback_strategy="all_errors",
        )

        assert strategy.judge_backend is judge
        assert strategy.feedback_strategy == "all_errors"

    def test_init_invalid_s1_backend_raises(self):
        """Test that non-Backend s1_solver raises TypeError."""
        s2 = MagicMock(spec=Backend)

        with pytest.raises(TypeError):
            SOFAISamplingStrategy(
                s1_solver_backend="not a backend",  # type: ignore
                s2_solver_backend=s2,
            )

    def test_init_invalid_s2_backend_raises(self):
        """Test that non-Backend s2_solver raises TypeError."""
        s1 = MagicMock(spec=Backend)

        with pytest.raises(TypeError):
            SOFAISamplingStrategy(
                s1_solver_backend=s1,
                s2_solver_backend="not a backend",  # type: ignore
            )

    def test_init_invalid_judge_backend_raises(self):
        """Test that non-Backend judge raises TypeError."""
        s1 = MagicMock(spec=Backend)
        s2 = MagicMock(spec=Backend)

        with pytest.raises(TypeError):
            SOFAISamplingStrategy(
                s1_solver_backend=s1,
                s2_solver_backend=s2,
                judge_backend="not a backend",  # type: ignore
            )

    def test_init_invalid_loop_budget_raises(self):
        """Test that loop_budget <= 0 raises ValueError."""
        s1 = MagicMock(spec=Backend)
        s2 = MagicMock(spec=Backend)

        with pytest.raises(ValueError):
            SOFAISamplingStrategy(
                s1_solver_backend=s1, s2_solver_backend=s2, loop_budget=0
            )

    def test_init_all_s2_modes(self):
        """Test all three s2_solver_mode options."""
        s1 = MagicMock(spec=Backend)
        s2 = MagicMock(spec=Backend)

        for mode in ["fresh_start", "continue_chat", "best_attempt"]:
            strategy = SOFAISamplingStrategy(
                s1_solver_backend=s1,
                s2_solver_backend=s2,
                s2_solver_mode=mode,  # type: ignore
            )
            assert strategy.s2_solver_mode == mode


class TestSOFAIRepair:
    """Test SOFAISamplingStrategy repair method."""

    def test_repair_creates_message_with_feedback(self):
        """Test that repair creates a Message with feedback from ValidationResult."""
        old_ctx = MagicMock(spec=ChatContext)
        new_ctx = MagicMock(spec=ChatContext)

        past_actions = [MagicMock(spec=Instruction)]
        past_results = [MagicMock(spec=ModelOutputThunk)]

        # Create validation results with reasons
        req1 = Requirement(description="Be formal")
        val1 = ValidationResult(False, reason="Output was too casual")
        req2 = Requirement(description="Use greeting")
        val2 = ValidationResult(True)  # Passed

        past_val = [[(req1, val1), (req2, val2)]]

        next_action, returned_ctx = SOFAISamplingStrategy.repair(
            old_ctx, new_ctx, past_actions, past_results, past_val
        )

        assert isinstance(next_action, Message)
        assert next_action.role == "user"
        assert "Output was too casual" in next_action.content
        assert "Be formal" not in next_action.content  # Uses reason, not description
        assert returned_ctx is new_ctx

    def test_repair_uses_description_when_no_reason(self):
        """Test that repair uses description when ValidationResult has no reason."""
        old_ctx = MagicMock(spec=ChatContext)
        new_ctx = MagicMock(spec=ChatContext)

        past_actions = [MagicMock(spec=Instruction)]
        past_results = [MagicMock(spec=ModelOutputThunk)]

        req1 = Requirement(description="Be formal")
        val1 = ValidationResult(False)  # No reason

        past_val = [[(req1, val1)]]

        next_action, _ = SOFAISamplingStrategy.repair(
            old_ctx, new_ctx, past_actions, past_results, past_val
        )

        assert isinstance(next_action, Message)
        assert "Be formal" in next_action.content  # Falls back to description


class TestSOFAISelectFromFailure:
    """Test SOFAISamplingStrategy select_from_failure method."""

    def test_select_from_failure_returns_last(self):
        """Test that select_from_failure returns last index (-1)."""
        actions = [MagicMock(), MagicMock(), MagicMock()]
        results = [MagicMock(), MagicMock(), MagicMock()]
        validations = [[], [], []]

        index = SOFAISamplingStrategy.select_from_failure(actions, results, validations)
        assert index == -1


class TestSOFAISelectBestAttempt:
    """Test SOFAISamplingStrategy _select_best_attempt method."""

    def test_select_best_attempt_single(self):
        """Test with single attempt."""
        req1 = Requirement(description="R1")
        val1 = ValidationResult(True)

        sampled_val = [[(req1, val1)]]
        index = SOFAISamplingStrategy._select_best_attempt(sampled_val)
        assert index == 0

    def test_select_best_attempt_prefers_more_passes(self):
        """Test that attempt with more passes is selected."""
        req1 = Requirement(description="R1")
        req2 = Requirement(description="R2")

        # First attempt: 1 pass
        val1_1 = ValidationResult(True)
        val1_2 = ValidationResult(False)

        # Second attempt: 2 passes
        val2_1 = ValidationResult(True)
        val2_2 = ValidationResult(True)

        sampled_val = [
            [(req1, val1_1), (req2, val1_2)],
            [(req1, val2_1), (req2, val2_2)],
        ]

        index = SOFAISamplingStrategy._select_best_attempt(sampled_val)
        assert index == 1

    def test_select_best_attempt_prefers_later_on_tie(self):
        """Test that later attempt is selected on tie."""
        req1 = Requirement(description="R1")

        # Both attempts: 1 pass
        val1 = ValidationResult(True)
        val2 = ValidationResult(True)

        sampled_val = [[(req1, val1)], [(req1, val2)]]

        index = SOFAISamplingStrategy._select_best_attempt(sampled_val)
        assert index == 1  # Prefers later


class TestSOFAIParseJudgment:
    """Test SOFAISamplingStrategy _parse_judgment method."""

    def test_parse_yes(self):
        """Test parsing 'Yes' responses."""
        assert SOFAISamplingStrategy._parse_judgment("Yes") is True
        assert SOFAISamplingStrategy._parse_judgment("yes") is True
        assert SOFAISamplingStrategy._parse_judgment("YES") is True
        assert (
            SOFAISamplingStrategy._parse_judgment("Yes, the requirement is satisfied.")
            is True
        )

    def test_parse_no(self):
        """Test parsing 'No' responses."""
        assert SOFAISamplingStrategy._parse_judgment("No") is False
        assert SOFAISamplingStrategy._parse_judgment("no") is False
        assert SOFAISamplingStrategy._parse_judgment("NO") is False
        assert SOFAISamplingStrategy._parse_judgment("No, there are errors.") is False

    def test_parse_yes_in_first_line(self):
        """Test that 'yes' in first line returns True."""
        assert SOFAISamplingStrategy._parse_judgment("The answer is yes.") is True

    def test_parse_no_first_line_default(self):
        """Test that ambiguous response defaults to False."""
        assert SOFAISamplingStrategy._parse_judgment("Maybe") is False
        assert SOFAISamplingStrategy._parse_judgment("I'm not sure") is False


class TestSOFAIExtractFeedback:
    """Test SOFAISamplingStrategy _extract_feedback method."""

    def test_extract_from_tags(self):
        """Test extracting feedback from <feedback> tags."""
        response = "No\n<feedback>The error is X.</feedback>"
        assert SOFAISamplingStrategy._extract_feedback(response) == "The error is X."

    def test_extract_multiline(self):
        """Test extracting multiline feedback."""
        response = """No
<feedback>
Error 1: X
Error 2: Y
</feedback>"""
        feedback = SOFAISamplingStrategy._extract_feedback(response)
        assert "Error 1: X" in feedback
        assert "Error 2: Y" in feedback

    def test_fallback_without_tags(self):
        """Test fallback to full response when no tags."""
        response = "No, the output is wrong."
        assert (
            SOFAISamplingStrategy._extract_feedback(response)
            == "No, the output is wrong."
        )


@pytest.mark.qualitative
class TestSOFAIIntegration:
    """Integration tests for SOFAISamplingStrategy.

    These tests require actual LLM backends and are marked as qualitative.
    """

    def test_sofai_with_ollama(self, gh_run):
        """Test SOFAI with Ollama backends."""
        from mellea import start_session
        from mellea.backends.ollama import OllamaModelBackend

        # Use smaller models for testing
        s1 = OllamaModelBackend(model_id="llama3.2:1b")
        s2 = OllamaModelBackend(model_id="llama3.2:1b")

        strategy = SOFAISamplingStrategy(
            s1_solver_backend=s1, s2_solver_backend=s2, loop_budget=2
        )

        m = start_session("ollama", model_id="llama3.2:1b", ctx=ChatContext())

        result = m.instruct(
            "Write a haiku about coding.",
            requirements=["The haiku should have 3 lines"],
            strategy=strategy,
            return_sampling_results=True,
        )

        assert result is not None
        assert hasattr(result, "success")
        assert len(result.sample_generations) >= 1


if __name__ == "__main__":
    pytest.main(["-v", __file__])
