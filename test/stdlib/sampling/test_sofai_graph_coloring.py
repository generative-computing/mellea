"""Comprehensive tests for SOFAISamplingStrategy using Graph Coloring domain.

This test suite validates the SOFAI sampling strategy through a graph coloring
constraint satisfaction problem, testing all modes, flows, and edge cases.
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mellea.backends import Backend
from mellea.core import GenerateLog, Requirement, ValidationResult
from mellea.stdlib.components import Instruction, Message, ModelOutputThunk
from mellea.stdlib.context import ChatContext
from mellea.stdlib.sampling import SOFAISamplingStrategy, SamplingResult


# =============================================================================
# Graph Coloring Test Domain
# =============================================================================

# Simple graph: A - B - C (linear)
SIMPLE_GRAPH = {"A": ["B"], "B": ["A", "C"], "C": ["B"]}
SIMPLE_COLORS = ["Red", "Blue"]

# Triangle graph: A - B - C - A (cycle)
TRIANGLE_GRAPH = {"A": ["B", "C"], "B": ["A", "C"], "C": ["A", "B"]}
TRIANGLE_COLORS = ["Red", "Blue", "Green"]


def parse_coloring(output_str: str) -> dict | None:
    """Parse LLM output as JSON, handling markdown code blocks."""
    try:
        output_str = output_str.strip()
        if output_str.startswith("```json"):
            output_str = output_str[7:].split("```")[0].strip()
        elif output_str.startswith("```"):
            output_str = output_str[3:].split("```")[0].strip()
        parsed = json.loads(output_str)
        if not isinstance(parsed, dict):
            return None
        return parsed
    except (json.JSONDecodeError, Exception):
        return None


def create_graph_coloring_validator(graph: dict, colors: list):
    """Create a validator function for graph coloring."""

    def check_graph_coloring(ctx) -> ValidationResult:
        output = ctx.last_output()
        if output is None:
            return ValidationResult(False, reason="No output found.")

        coloring = parse_coloring(str(output.value))
        if coloring is None:
            return ValidationResult(False, reason="Could not parse as valid JSON.")

        errors = []

        # Check all nodes are colored
        colored_nodes = set(coloring.keys())
        graph_nodes = set(graph.keys())
        missing = graph_nodes - colored_nodes
        extra = colored_nodes - graph_nodes

        if missing:
            errors.append(f"Missing nodes: {', '.join(sorted(missing))}")
        if extra:
            errors.append(f"Unexpected nodes: {', '.join(sorted(extra))}")

        # Check valid colors
        invalid_colors = [c for c in coloring.values() if c not in colors]
        if invalid_colors:
            errors.append(f"Invalid colors: {set(invalid_colors)}")

        # Check adjacency constraints
        if not errors:
            for node, neighbors in graph.items():
                if node not in coloring:
                    continue
                node_color = coloring[node]
                for neighbor in neighbors:
                    if neighbor in coloring and coloring[neighbor] == node_color:
                        errors.append(
                            f"Adjacent nodes {node} and {neighbor} have same color '{node_color}'"
                        )

        if errors:
            return ValidationResult(False, reason=" | ".join(errors))
        return ValidationResult(True, reason="Valid graph coloring!")

    return check_graph_coloring


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def mock_s1_backend():
    """Create a mock S1 solver backend."""
    backend = MagicMock(spec=Backend)
    backend.model_id = "s1-fast-model"
    return backend


@pytest.fixture
def mock_s2_backend():
    """Create a mock S2 solver backend."""
    backend = MagicMock(spec=Backend)
    backend.model_id = "s2-slow-model"
    return backend


@pytest.fixture
def mock_judge_backend():
    """Create a mock judge backend."""
    backend = MagicMock(spec=Backend)
    backend.model_id = "judge-model"
    return backend


@pytest.fixture
def chat_context():
    """Create a ChatContext for testing."""
    return ChatContext()


@pytest.fixture
def graph_coloring_instruction():
    """Create graph coloring instruction."""
    return Instruction(
        description=(
            "Color the nodes of the graph (A, B, C) using Red or Blue. "
            "Adjacent nodes must have different colors. "
            'Respond with JSON like {"A": "Red", "B": "Blue", "C": "Red"}.'
        )
    )


@pytest.fixture
def graph_coloring_requirement():
    """Create graph coloring requirement with validator."""
    return Requirement(
        description="Valid graph coloring with no adjacent same-color nodes",
        validation_fn=create_graph_coloring_validator(SIMPLE_GRAPH, SIMPLE_COLORS),
    )


def create_mock_mot(value: str) -> ModelOutputThunk:
    """Create a mock ModelOutputThunk with given value."""
    mot = MagicMock(spec=ModelOutputThunk)
    mot.value = value
    mot.parsed_repr = value
    mot._generate_log = MagicMock(spec=GenerateLog)
    mot._generate_log.is_final_result = False
    # Required for Message._parse() to work correctly
    mot.tool_calls = None
    mot._meta = {}

    async def mock_avalue():
        return value

    mot.avalue = mock_avalue
    return mot


def create_mock_context_with_output(output_value: str) -> MagicMock:
    """Create a mock context that returns a specific output."""
    ctx = MagicMock(spec=ChatContext)
    mot = create_mock_mot(output_value)
    ctx.last_output = MagicMock(return_value=mot)
    return ctx


# =============================================================================
# Test Initialization with Graph Coloring Context
# =============================================================================


class TestSOFAIInitGraphColoring:
    """Test SOFAI initialization in graph coloring context."""

    def test_init_for_graph_coloring(self, mock_s1_backend, mock_s2_backend):
        """Test basic initialization for graph coloring task."""
        strategy = SOFAISamplingStrategy(
            s1_solver_backend=mock_s1_backend,
            s2_solver_backend=mock_s2_backend,
            loop_budget=3,
        )

        assert strategy.s1_solver_backend is mock_s1_backend
        assert strategy.s2_solver_backend is mock_s2_backend
        assert strategy.loop_budget == 3

    @pytest.mark.parametrize("mode", ["fresh_start", "continue_chat", "best_attempt"])
    def test_all_s2_modes_init(self, mock_s1_backend, mock_s2_backend, mode):
        """Test initialization with all S2 modes."""
        strategy = SOFAISamplingStrategy(
            s1_solver_backend=mock_s1_backend,
            s2_solver_backend=mock_s2_backend,
            s2_solver_mode=mode,
        )
        assert strategy.s2_solver_mode == mode

    def test_with_judge_backend(
        self, mock_s1_backend, mock_s2_backend, mock_judge_backend
    ):
        """Test initialization with judge backend for validation."""
        strategy = SOFAISamplingStrategy(
            s1_solver_backend=mock_s1_backend,
            s2_solver_backend=mock_s2_backend,
            judge_backend=mock_judge_backend,
            feedback_strategy="all_errors",
        )

        assert strategy.judge_backend is mock_judge_backend
        assert strategy.feedback_strategy == "all_errors"


# =============================================================================
# Test Graph Coloring Validator
# =============================================================================


class TestGraphColoringValidator:
    """Test the graph coloring validator function."""

    def test_valid_coloring(self):
        """Test that valid coloring passes."""
        validator = create_graph_coloring_validator(SIMPLE_GRAPH, SIMPLE_COLORS)
        ctx = create_mock_context_with_output('{"A": "Red", "B": "Blue", "C": "Red"}')

        result = validator(ctx)
        assert result.as_bool() is True

    def test_invalid_adjacent_same_color(self):
        """Test that adjacent nodes with same color fails."""
        validator = create_graph_coloring_validator(SIMPLE_GRAPH, SIMPLE_COLORS)
        # A and B are adjacent, both Red
        ctx = create_mock_context_with_output('{"A": "Red", "B": "Red", "C": "Blue"}')

        result = validator(ctx)
        assert result.as_bool() is False
        assert "same color" in result.reason.lower()

    def test_missing_node(self):
        """Test that missing nodes fails."""
        validator = create_graph_coloring_validator(SIMPLE_GRAPH, SIMPLE_COLORS)
        ctx = create_mock_context_with_output('{"A": "Red", "B": "Blue"}')  # Missing C

        result = validator(ctx)
        assert result.as_bool() is False
        assert "Missing nodes" in result.reason

    def test_invalid_color(self):
        """Test that invalid colors fail."""
        validator = create_graph_coloring_validator(SIMPLE_GRAPH, SIMPLE_COLORS)
        ctx = create_mock_context_with_output(
            '{"A": "Red", "B": "Green", "C": "Blue"}'
        )  # Green not allowed

        result = validator(ctx)
        assert result.as_bool() is False
        assert "Invalid colors" in result.reason

    def test_invalid_json(self):
        """Test that invalid JSON fails."""
        validator = create_graph_coloring_validator(SIMPLE_GRAPH, SIMPLE_COLORS)
        ctx = create_mock_context_with_output("not json")

        result = validator(ctx)
        assert result.as_bool() is False
        assert "JSON" in result.reason

    def test_json_in_code_block(self):
        """Test that JSON in markdown code block is parsed."""
        validator = create_graph_coloring_validator(SIMPLE_GRAPH, SIMPLE_COLORS)
        ctx = create_mock_context_with_output(
            '```json\n{"A": "Red", "B": "Blue", "C": "Red"}\n```'
        )

        result = validator(ctx)
        assert result.as_bool() is True


# =============================================================================
# Test Repair Method with Graph Coloring
# =============================================================================


class TestSOFAIRepairGraphColoring:
    """Test repair method with graph coloring validation feedback."""

    def test_repair_includes_coloring_error(self):
        """Test that repair message includes specific coloring error."""
        old_ctx = MagicMock(spec=ChatContext)
        new_ctx = MagicMock(spec=ChatContext)

        past_actions = [MagicMock(spec=Instruction)]
        past_results = [MagicMock(spec=ModelOutputThunk)]

        req = Requirement(description="Valid coloring")
        val = ValidationResult(
            False, reason="Adjacent nodes A and B have same color 'Red'"
        )

        past_val = [[(req, val)]]

        next_action, _ = SOFAISamplingStrategy.repair(
            old_ctx, new_ctx, past_actions, past_results, past_val
        )

        assert isinstance(next_action, Message)
        assert "Adjacent nodes A and B" in next_action.content
        assert "same color" in next_action.content

    def test_repair_with_multiple_errors(self):
        """Test repair with multiple validation errors."""
        old_ctx = MagicMock(spec=ChatContext)
        new_ctx = MagicMock(spec=ChatContext)

        past_actions = [MagicMock(spec=Instruction)]
        past_results = [MagicMock(spec=ModelOutputThunk)]

        req1 = Requirement(description="All nodes colored")
        val1 = ValidationResult(False, reason="Missing nodes: C")
        req2 = Requirement(description="Valid colors only")
        val2 = ValidationResult(False, reason="Invalid colors: {'Green'}")

        past_val = [[(req1, val1), (req2, val2)]]

        next_action, _ = SOFAISamplingStrategy.repair(
            old_ctx, new_ctx, past_actions, past_results, past_val
        )

        assert "Missing nodes: C" in next_action.content
        assert "Invalid colors" in next_action.content


# =============================================================================
# Test Select Best Attempt with Graph Coloring
# =============================================================================


class TestSOFAISelectBestAttemptGraphColoring:
    """Test _select_best_attempt with graph coloring scenarios."""

    def test_select_attempt_with_fewer_errors(self):
        """Test selecting attempt with fewer constraint violations."""
        req1 = Requirement(description="Valid adjacency")
        req2 = Requirement(description="All nodes colored")
        req3 = Requirement(description="Valid colors")

        # Attempt 1: 1 pass (only colors valid)
        attempt1 = [
            (req1, ValidationResult(False)),
            (req2, ValidationResult(False)),
            (req3, ValidationResult(True)),
        ]

        # Attempt 2: 2 passes (colors valid, all nodes colored)
        attempt2 = [
            (req1, ValidationResult(False)),
            (req2, ValidationResult(True)),
            (req3, ValidationResult(True)),
        ]

        sampled_val = [attempt1, attempt2]
        index = SOFAISamplingStrategy._select_best_attempt(sampled_val)

        assert index == 1  # Attempt 2 has more passes


# =============================================================================
# Test Prepare S2 Context with Graph Coloring
# =============================================================================


class TestSOFAIPrepareS2ContextGraphColoring:
    """Test _prepare_s2_context with graph coloring scenarios."""

    def test_best_attempt_includes_coloring_feedback(
        self, mock_s1_backend, mock_s2_backend
    ):
        """Test that best_attempt mode includes coloring feedback."""
        strategy = SOFAISamplingStrategy(
            s1_solver_backend=mock_s1_backend,
            s2_solver_backend=mock_s2_backend,
            s2_solver_mode="best_attempt",
        )

        original_action = Instruction(description="Color the graph")
        original_context = ChatContext()
        last_result_ctx = ChatContext()
        last_action = Message(role="user", content="Fix the coloring")

        # S1 produced an invalid coloring
        mot = create_mock_mot('{"A": "Red", "B": "Red", "C": "Blue"}')
        sampled_results = [mot]
        sampled_scores = [
            [
                (
                    Requirement(description="Valid"),
                    ValidationResult(False, reason="A and B both Red but are adjacent"),
                )
            ]
        ]

        s2_action, s2_context = strategy._prepare_s2_context(
            s2_mode="best_attempt",
            original_action=original_action,
            original_context=original_context,
            last_result_ctx=last_result_ctx,
            last_action=last_action,
            sampled_results=sampled_results,
            sampled_scores=sampled_scores,
            loop_count=1,
        )

        assert isinstance(s2_action, Message)
        # Should include the original task
        assert "Color the graph" in s2_action.content
        # Should include the failed attempt
        assert '{"A": "Red", "B": "Red", "C": "Blue"}' in s2_action.content
        # Should include the error feedback
        assert "A and B both Red" in s2_action.content or "issues" in s2_action.content


# =============================================================================
# Test Full Sample Flow with Graph Coloring
# =============================================================================


class TestSOFAISampleFlowGraphColoring:
    """Test the complete sample() flow using graph coloring."""

    @pytest.mark.asyncio
    async def test_s1_produces_valid_coloring_immediately(
        self, mock_s1_backend, mock_s2_backend, graph_coloring_instruction
    ):
        """Test S1 succeeds on first try with valid coloring."""
        strategy = SOFAISamplingStrategy(
            s1_solver_backend=mock_s1_backend,
            s2_solver_backend=mock_s2_backend,
            loop_budget=3,
        )

        # S1 produces valid coloring
        valid_coloring = '{"A": "Red", "B": "Blue", "C": "Red"}'
        mot = create_mock_mot(valid_coloring)

        async def mock_generate(*args, **kwargs):
            return mot, ChatContext()

        mock_s1_backend.generate_from_context = AsyncMock(side_effect=mock_generate)

        graph_coloring_instruction.parse = MagicMock(return_value=valid_coloring)
        context = ChatContext()

        # Use the actual validator
        req = Requirement(
            description="Valid coloring",
            validation_fn=create_graph_coloring_validator(SIMPLE_GRAPH, SIMPLE_COLORS),
        )

        with patch("mellea.stdlib.sampling.sofai.mfuncs") as mock_mfuncs:
            mock_mfuncs.avalidate = AsyncMock(
                return_value=[ValidationResult(True, reason="Valid!")]
            )

            result = await strategy.sample(
                action=graph_coloring_instruction,
                context=context,
                backend=mock_s1_backend,
                requirements=[req],
            )

        assert result.success is True
        assert len(result.sample_generations) == 1
        mock_s2_backend.generate_from_context.assert_not_called()

    @pytest.mark.asyncio
    async def test_s1_fails_s2_produces_valid_coloring(
        self, mock_s1_backend, mock_s2_backend, graph_coloring_instruction
    ):
        """Test S1 fails, S2 produces valid coloring."""
        strategy = SOFAISamplingStrategy(
            s1_solver_backend=mock_s1_backend,
            s2_solver_backend=mock_s2_backend,
            s2_solver_mode="fresh_start",
            loop_budget=1,
        )

        # S1 produces invalid coloring (adjacent same color)
        invalid_coloring = '{"A": "Red", "B": "Red", "C": "Blue"}'
        s1_mot = create_mock_mot(invalid_coloring)

        # S2 produces valid coloring
        valid_coloring = '{"A": "Red", "B": "Blue", "C": "Red"}'
        s2_mot = create_mock_mot(valid_coloring)

        async def s1_generate(*args, **kwargs):
            return s1_mot, ChatContext()

        async def s2_generate(*args, **kwargs):
            return s2_mot, ChatContext()

        mock_s1_backend.generate_from_context = AsyncMock(side_effect=s1_generate)
        mock_s2_backend.generate_from_context = AsyncMock(side_effect=s2_generate)

        graph_coloring_instruction.parse = MagicMock(side_effect=lambda x: x.value)
        context = ChatContext()

        req = Requirement(description="Valid coloring")

        with patch("mellea.stdlib.sampling.sofai.mfuncs") as mock_mfuncs:
            mock_mfuncs.avalidate = AsyncMock(
                side_effect=[
                    [ValidationResult(False, reason="A and B same color")],  # S1 fails
                    [ValidationResult(True)],  # S2 succeeds
                ]
            )

            result = await strategy.sample(
                action=graph_coloring_instruction,
                context=context,
                backend=mock_s1_backend,
                requirements=[req],
            )

        assert result.success is True
        assert len(result.sample_generations) == 2
        mock_s2_backend.generate_from_context.assert_called_once()

    @pytest.mark.asyncio
    async def test_s1_improves_across_iterations(
        self, mock_s1_backend, mock_s2_backend, graph_coloring_instruction
    ):
        """Test S1 improves across multiple iterations before success."""
        strategy = SOFAISamplingStrategy(
            s1_solver_backend=mock_s1_backend,
            s2_solver_backend=mock_s2_backend,
            loop_budget=3,
        )

        # Sequence of improving colorings
        colorings = [
            '{"A": "Red"}',  # Missing B, C
            '{"A": "Red", "B": "Red", "C": "Blue"}',  # A,B conflict
            '{"A": "Red", "B": "Blue", "C": "Red"}',  # Valid!
        ]
        call_count = [0]

        async def mock_generate(*args, **kwargs):
            idx = min(call_count[0], len(colorings) - 1)
            mot = create_mock_mot(colorings[idx])
            call_count[0] += 1
            return mot, ChatContext()

        mock_s1_backend.generate_from_context = AsyncMock(side_effect=mock_generate)

        graph_coloring_instruction.parse = MagicMock(side_effect=lambda x: x.value)
        context = ChatContext()

        req = Requirement(description="Valid coloring")

        validation_results = [
            [ValidationResult(False, reason="Missing nodes: B, C")],
            [ValidationResult(False, reason="A and B same color")],
            [ValidationResult(True)],
        ]
        val_idx = [0]

        async def mock_validate(*args, **kwargs):
            result = validation_results[val_idx[0]]
            val_idx[0] += 1
            return result

        with patch("mellea.stdlib.sampling.sofai.mfuncs") as mock_mfuncs:
            mock_mfuncs.avalidate = AsyncMock(side_effect=mock_validate)

            result = await strategy.sample(
                action=graph_coloring_instruction,
                context=context,
                backend=mock_s1_backend,
                requirements=[req],
            )

        assert result.success is True
        assert len(result.sample_generations) == 3
        mock_s2_backend.generate_from_context.assert_not_called()

    @pytest.mark.asyncio
    async def test_both_solvers_fail_on_impossible_graph(
        self, mock_s1_backend, mock_s2_backend
    ):
        """Test both solvers fail on impossible coloring (2 colors for triangle)."""
        strategy = SOFAISamplingStrategy(
            s1_solver_backend=mock_s1_backend,
            s2_solver_backend=mock_s2_backend,
            loop_budget=2,
        )

        # Triangle graph needs 3 colors, but we only allow 2
        instruction = Instruction(
            description="Color triangle A-B-C with only Red and Blue"
        )

        mot = create_mock_mot('{"A": "Red", "B": "Blue", "C": "Red"}')

        async def mock_generate(*args, **kwargs):
            return mot, ChatContext()

        mock_s1_backend.generate_from_context = AsyncMock(side_effect=mock_generate)
        mock_s2_backend.generate_from_context = AsyncMock(side_effect=mock_generate)

        instruction.parse = MagicMock(return_value=mot.value)
        context = ChatContext()

        # This will always fail because triangle needs 3 colors
        req = Requirement(
            description="Valid coloring",
            validation_fn=create_graph_coloring_validator(
                TRIANGLE_GRAPH, ["Red", "Blue"]
            ),  # Only 2 colors!
        )

        with patch("mellea.stdlib.sampling.sofai.mfuncs") as mock_mfuncs:
            # All validations fail
            mock_mfuncs.avalidate = AsyncMock(
                return_value=[
                    ValidationResult(False, reason="C adjacent to both A and B")
                ]
            )

            result = await strategy.sample(
                action=instruction,
                context=context,
                backend=mock_s1_backend,
                requirements=[req],
            )

        assert result.success is False
        # Should have loop_budget S1 attempts + 1 S2 attempt
        assert len(result.sample_generations) == 3


# =============================================================================
# Test All S2 Modes with Graph Coloring
# =============================================================================


class TestSOFAIS2ModesGraphColoring:
    """Test all S2 solver modes with graph coloring domain."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("mode", ["fresh_start", "continue_chat", "best_attempt"])
    async def test_s2_mode_fallback(self, mock_s1_backend, mock_s2_backend, mode):
        """Test S2 fallback works correctly in each mode."""
        strategy = SOFAISamplingStrategy(
            s1_solver_backend=mock_s1_backend,
            s2_solver_backend=mock_s2_backend,
            s2_solver_mode=mode,
            loop_budget=1,
        )

        instruction = Instruction(description="Color the graph")
        s1_mot = create_mock_mot('{"A": "Red", "B": "Red", "C": "Blue"}')
        s2_mot = create_mock_mot('{"A": "Red", "B": "Blue", "C": "Red"}')

        async def s1_generate(*args, **kwargs):
            return s1_mot, ChatContext()

        async def s2_generate(*args, **kwargs):
            return s2_mot, ChatContext()

        mock_s1_backend.generate_from_context = AsyncMock(side_effect=s1_generate)
        mock_s2_backend.generate_from_context = AsyncMock(side_effect=s2_generate)

        instruction.parse = MagicMock(side_effect=lambda x: x.value)
        context = ChatContext()

        req = Requirement(description="Valid coloring")

        with patch("mellea.stdlib.sampling.sofai.mfuncs") as mock_mfuncs:
            mock_mfuncs.avalidate = AsyncMock(
                side_effect=[[ValidationResult(False)], [ValidationResult(True)]]
            )

            result = await strategy.sample(
                action=instruction,
                context=context,
                backend=mock_s1_backend,
                requirements=[req],
            )

        assert result.success is True
        assert len(result.sample_generations) == 2
        mock_s2_backend.generate_from_context.assert_called_once()


# =============================================================================
# Integration Tests (Qualitative)
# =============================================================================


@pytest.mark.qualitative
class TestSOFAIGraphColoringIntegration:
    """Integration tests with actual LLM backends.

    These tests are marked qualitative and skipped in CI.
    """

    def test_graph_coloring_fresh_start(self):
        """Test real graph coloring with fresh_start mode."""
        from mellea import start_session
        from mellea.backends.ollama import OllamaModelBackend
        from mellea.stdlib.requirements import req

        s1 = OllamaModelBackend(model_id="llama3.2:1b")
        s2 = OllamaModelBackend(model_id="llama3.2:1b")

        strategy = SOFAISamplingStrategy(
            s1_solver_backend=s1,
            s2_solver_backend=s2,
            s2_solver_mode="fresh_start",
            loop_budget=2,
        )

        validator = create_graph_coloring_validator(SIMPLE_GRAPH, SIMPLE_COLORS)

        m = start_session("ollama", model_id="llama3.2:1b", ctx=ChatContext())

        result = m.instruct(
            "Color graph nodes A, B, C with Red or Blue. "
            "A connects to B, B connects to A and C, C connects to B. "
            "Adjacent nodes must have different colors. "
            'Reply with JSON only: {"A": "color", "B": "color", "C": "color"}',
            requirements=[
                req(description="Valid graph coloring", validation_fn=validator)
            ],
            strategy=strategy,
            return_sampling_results=True,
        )

        assert result is not None
        assert hasattr(result, "success")
        assert len(result.sample_generations) >= 1

    def test_graph_coloring_continue_chat(self):
        """Test real graph coloring with continue_chat mode."""
        from mellea import start_session
        from mellea.backends.ollama import OllamaModelBackend
        from mellea.stdlib.requirements import req

        s1 = OllamaModelBackend(model_id="llama3.2:1b")
        s2 = OllamaModelBackend(model_id="llama3.2:1b")

        strategy = SOFAISamplingStrategy(
            s1_solver_backend=s1,
            s2_solver_backend=s2,
            s2_solver_mode="continue_chat",
            loop_budget=2,
        )

        validator = create_graph_coloring_validator(SIMPLE_GRAPH, SIMPLE_COLORS)

        m = start_session("ollama", model_id="llama3.2:1b", ctx=ChatContext())

        result = m.instruct(
            "Color graph A-B-C with Red/Blue. Adjacent nodes need different colors. "
            'Output JSON: {"A": "...", "B": "...", "C": "..."}',
            requirements=[
                req(description="Valid graph coloring", validation_fn=validator)
            ],
            strategy=strategy,
            return_sampling_results=True,
        )

        assert result is not None

    def test_graph_coloring_best_attempt(self):
        """Test real graph coloring with best_attempt mode."""
        from mellea import start_session
        from mellea.backends.ollama import OllamaModelBackend
        from mellea.stdlib.requirements import req

        s1 = OllamaModelBackend(model_id="llama3.2:1b")
        s2 = OllamaModelBackend(model_id="llama3.2:1b")

        strategy = SOFAISamplingStrategy(
            s1_solver_backend=s1,
            s2_solver_backend=s2,
            s2_solver_mode="best_attempt",
            loop_budget=2,
        )

        validator = create_graph_coloring_validator(SIMPLE_GRAPH, SIMPLE_COLORS)

        m = start_session("ollama", model_id="llama3.2:1b", ctx=ChatContext())

        result = m.instruct(
            "Solve graph coloring: nodes A,B,C, edges A-B and B-C. "
            "Use Red or Blue only. Adjacent = different color. "
            "JSON output required.",
            requirements=[
                req(description="Valid graph coloring", validation_fn=validator)
            ],
            strategy=strategy,
            return_sampling_results=True,
        )

        assert result is not None


if __name__ == "__main__":
    pytest.main(["-v", __file__])
