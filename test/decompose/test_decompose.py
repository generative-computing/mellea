"""Tests for cli/decompose/decompose.py functions.

This module tests the reorder_subtasks and verify_user_variables functions
which handle dependency ordering and validation of decomposition results.
"""

import pytest

from cli.decompose.decompose import reorder_subtasks, verify_user_variables
from cli.decompose.pipeline import DecompPipelineResult, DecompSubtasksResult

# ============================================================================
# Tests for reorder_subtasks
# ============================================================================


class TestReorderSubtasksHappyPath:
    """Happy path tests for reorder_subtasks function."""

    def test_no_dependencies(self):
        """Test subtasks with no dependencies remain in original order."""
        subtasks: list[DecompSubtasksResult] = [
            {
                "subtask": "Task A",
                "tag": "TASK_A",
                "constraints": [],
                "prompt_template": "Do A",
                "input_vars_required": [],
                "depends_on": [],
            },
            {
                "subtask": "Task B",
                "tag": "TASK_B",
                "constraints": [],
                "prompt_template": "Do B",
                "input_vars_required": [],
                "depends_on": [],
            },
            {
                "subtask": "Task C",
                "tag": "TASK_C",
                "constraints": [],
                "prompt_template": "Do C",
                "input_vars_required": [],
                "depends_on": [],
            },
        ]

        result = reorder_subtasks(subtasks)

        # Should maintain alphabetical order (topological sort is stable)
        assert len(result) == 3
        assert result[0]["tag"] == "TASK_A"
        assert result[1]["tag"] == "TASK_B"
        assert result[2]["tag"] == "TASK_C"

    def test_simple_linear_dependency(self):
        """Test simple linear dependency chain: C -> B -> A."""
        subtasks: list[DecompSubtasksResult] = [
            {
                "subtask": "Task C",
                "tag": "TASK_C",
                "constraints": [],
                "prompt_template": "Do C",
                "input_vars_required": [],
                "depends_on": ["TASK_B"],
            },
            {
                "subtask": "Task B",
                "tag": "TASK_B",
                "constraints": [],
                "prompt_template": "Do B",
                "input_vars_required": [],
                "depends_on": ["TASK_A"],
            },
            {
                "subtask": "Task A",
                "tag": "TASK_A",
                "constraints": [],
                "prompt_template": "Do A",
                "input_vars_required": [],
                "depends_on": [],
            },
        ]

        result = reorder_subtasks(subtasks)

        # Should reorder to A, B, C
        assert len(result) == 3
        assert result[0]["tag"] == "TASK_A"
        assert result[1]["tag"] == "TASK_B"
        assert result[2]["tag"] == "TASK_C"

    def test_diamond_dependency(self):
        """Test diamond dependency: D depends on B and C, both depend on A."""
        subtasks: list[DecompSubtasksResult] = [
            {
                "subtask": "Task D",
                "tag": "TASK_D",
                "constraints": [],
                "prompt_template": "Do D",
                "input_vars_required": [],
                "depends_on": ["TASK_B", "TASK_C"],
            },
            {
                "subtask": "Task C",
                "tag": "TASK_C",
                "constraints": [],
                "prompt_template": "Do C",
                "input_vars_required": [],
                "depends_on": ["TASK_A"],
            },
            {
                "subtask": "Task B",
                "tag": "TASK_B",
                "constraints": [],
                "prompt_template": "Do B",
                "input_vars_required": [],
                "depends_on": ["TASK_A"],
            },
            {
                "subtask": "Task A",
                "tag": "TASK_A",
                "constraints": [],
                "prompt_template": "Do A",
                "input_vars_required": [],
                "depends_on": [],
            },
        ]

        result = reorder_subtasks(subtasks)

        # A must be first, D must be last, B and C can be in either order
        assert len(result) == 4
        assert result[0]["tag"] == "TASK_A"
        assert result[3]["tag"] == "TASK_D"
        assert {result[1]["tag"], result[2]["tag"]} == {"TASK_B", "TASK_C"}

    def test_case_insensitive_dependencies(self):
        """Test that dependencies are case-insensitive."""
        subtasks: list[DecompSubtasksResult] = [
            {
                "subtask": "Task B",
                "tag": "task_b",
                "constraints": [],
                "prompt_template": "Do B",
                "input_vars_required": [],
                "depends_on": ["TASK_A"],  # Uppercase reference
            },
            {
                "subtask": "Task A",
                "tag": "TASK_A",
                "constraints": [],
                "prompt_template": "Do A",
                "input_vars_required": [],
                "depends_on": [],
            },
        ]

        result = reorder_subtasks(subtasks)

        assert len(result) == 2
        assert result[0]["tag"] == "TASK_A"
        assert result[1]["tag"] == "task_b"

    def test_multiple_independent_chains(self):
        """Test multiple independent dependency chains."""
        subtasks: list[DecompSubtasksResult] = [
            # Chain 1: B -> A
            {
                "subtask": "Task B",
                "tag": "TASK_B",
                "constraints": [],
                "prompt_template": "Do B",
                "input_vars_required": [],
                "depends_on": ["TASK_A"],
            },
            {
                "subtask": "Task A",
                "tag": "TASK_A",
                "constraints": [],
                "prompt_template": "Do A",
                "input_vars_required": [],
                "depends_on": [],
            },
            # Chain 2: D -> C
            {
                "subtask": "Task D",
                "tag": "TASK_D",
                "constraints": [],
                "prompt_template": "Do D",
                "input_vars_required": [],
                "depends_on": ["TASK_C"],
            },
            {
                "subtask": "Task C",
                "tag": "TASK_C",
                "constraints": [],
                "prompt_template": "Do C",
                "input_vars_required": [],
                "depends_on": [],
            },
        ]

        result = reorder_subtasks(subtasks)

        # A before B, C before D
        assert len(result) == 4
        a_idx = next(i for i, t in enumerate(result) if t["tag"] == "TASK_A")
        b_idx = next(i for i, t in enumerate(result) if t["tag"] == "TASK_B")
        c_idx = next(i for i, t in enumerate(result) if t["tag"] == "TASK_C")
        d_idx = next(i for i, t in enumerate(result) if t["tag"] == "TASK_D")
        assert a_idx < b_idx
        assert c_idx < d_idx

    def test_nonexistent_dependency_ignored(self):
        """Test that dependencies referencing non-existent tasks are ignored."""
        subtasks: list[DecompSubtasksResult] = [
            {
                "subtask": "Task B",
                "tag": "TASK_B",
                "constraints": [],
                "prompt_template": "Do B",
                "input_vars_required": [],
                "depends_on": [
                    "TASK_A",
                    "NONEXISTENT",
                ],  # NONEXISTENT should be ignored
            },
            {
                "subtask": "Task A",
                "tag": "TASK_A",
                "constraints": [],
                "prompt_template": "Do A",
                "input_vars_required": [],
                "depends_on": [],
            },
        ]

        result = reorder_subtasks(subtasks)

        # Should still work, ignoring the nonexistent dependency
        assert len(result) == 2
        assert result[0]["tag"] == "TASK_A"
        assert result[1]["tag"] == "TASK_B"


class TestReorderSubtasksUnhappyPath:
    """Negative tests for reorder_subtasks function."""

    def test_circular_dependency_two_nodes(self):
        """Test circular dependency between two nodes."""
        subtasks: list[DecompSubtasksResult] = [
            {
                "subtask": "Task A",
                "tag": "TASK_A",
                "constraints": [],
                "prompt_template": "Do A",
                "input_vars_required": [],
                "depends_on": ["TASK_B"],
            },
            {
                "subtask": "Task B",
                "tag": "TASK_B",
                "constraints": [],
                "prompt_template": "Do B",
                "input_vars_required": [],
                "depends_on": ["TASK_A"],
            },
        ]

        with pytest.raises(ValueError, match="Circular dependency detected"):
            reorder_subtasks(subtasks)

    def test_circular_dependency_three_nodes(self):
        """Test circular dependency in a chain of three nodes."""
        subtasks: list[DecompSubtasksResult] = [
            {
                "subtask": "Task A",
                "tag": "TASK_A",
                "constraints": [],
                "prompt_template": "Do A",
                "input_vars_required": [],
                "depends_on": ["TASK_C"],
            },
            {
                "subtask": "Task B",
                "tag": "TASK_B",
                "constraints": [],
                "prompt_template": "Do B",
                "input_vars_required": [],
                "depends_on": ["TASK_A"],
            },
            {
                "subtask": "Task C",
                "tag": "TASK_C",
                "constraints": [],
                "prompt_template": "Do C",
                "input_vars_required": [],
                "depends_on": ["TASK_B"],
            },
        ]

        with pytest.raises(ValueError, match="Circular dependency detected"):
            reorder_subtasks(subtasks)

    def test_self_dependency(self):
        """Test task depending on itself."""
        subtasks: list[DecompSubtasksResult] = [
            {
                "subtask": "Task A",
                "tag": "TASK_A",
                "constraints": [],
                "prompt_template": "Do A",
                "input_vars_required": [],
                "depends_on": ["TASK_A"],
            }
        ]

        with pytest.raises(ValueError, match="Circular dependency detected"):
            reorder_subtasks(subtasks)

    def test_empty_subtasks_list(self):
        """Test with empty subtasks list."""
        subtasks: list[DecompSubtasksResult] = []

        result = reorder_subtasks(subtasks)

        assert result == []


# ============================================================================
# Tests for verify_user_variables
# ============================================================================


class TestVerifyUserVariablesHappyPath:
    """Happy path tests for verify_user_variables function."""

    def test_no_input_vars_no_dependencies(self):
        """Test with no input variables and no dependencies."""
        decomp_data: DecompPipelineResult = {
            "original_task_prompt": "Test task",
            "subtask_list": ["Task A"],
            "identified_constraints": [],
            "subtasks": [
                {
                    "subtask": "Task A",
                    "tag": "TASK_A",
                    "constraints": [],
                    "prompt_template": "Do A",
                    "input_vars_required": [],
                    "depends_on": [],
                }
            ],
        }

        result = verify_user_variables(decomp_data, None)

        assert result == decomp_data
        assert len(result["subtasks"]) == 1

    def test_valid_input_vars(self):
        """Test with valid input variables."""
        decomp_data: DecompPipelineResult = {
            "original_task_prompt": "Test task",
            "subtask_list": ["Task A"],
            "identified_constraints": [],
            "subtasks": [
                {
                    "subtask": "Task A",
                    "tag": "TASK_A",
                    "constraints": [],
                    "prompt_template": "Do A with {{ USER_INPUT }}",
                    "input_vars_required": ["USER_INPUT"],
                    "depends_on": [],
                }
            ],
        }

        result = verify_user_variables(decomp_data, ["USER_INPUT"])

        assert result == decomp_data

    def test_case_insensitive_input_vars(self):
        """Test that input variable matching is case-insensitive."""
        decomp_data: DecompPipelineResult = {
            "original_task_prompt": "Test task",
            "subtask_list": ["Task A"],
            "identified_constraints": [],
            "subtasks": [
                {
                    "subtask": "Task A",
                    "tag": "TASK_A",
                    "constraints": [],
                    "prompt_template": "Do A",
                    "input_vars_required": ["user_input"],  # lowercase
                    "depends_on": [],
                }
            ],
        }

        # Should work with uppercase input
        result = verify_user_variables(decomp_data, ["USER_INPUT"])

        assert result == decomp_data

    def test_valid_dependencies_in_order(self):
        """Test with valid dependencies already in correct order."""
        decomp_data: DecompPipelineResult = {
            "original_task_prompt": "Test task",
            "subtask_list": ["Task A", "Task B"],
            "identified_constraints": [],
            "subtasks": [
                {
                    "subtask": "Task A",
                    "tag": "TASK_A",
                    "constraints": [],
                    "prompt_template": "Do A",
                    "input_vars_required": [],
                    "depends_on": [],
                },
                {
                    "subtask": "Task B",
                    "tag": "TASK_B",
                    "constraints": [],
                    "prompt_template": "Do B",
                    "input_vars_required": [],
                    "depends_on": ["TASK_A"],
                },
            ],
        }

        result = verify_user_variables(decomp_data, None)

        # Should not reorder since already correct
        assert result["subtasks"][0]["tag"] == "TASK_A"
        assert result["subtasks"][1]["tag"] == "TASK_B"

    def test_dependencies_out_of_order_triggers_reorder(self):
        """Test that out-of-order dependencies trigger automatic reordering."""
        decomp_data: DecompPipelineResult = {
            "original_task_prompt": "Test task",
            "subtask_list": ["Task B", "Task A"],
            "identified_constraints": [],
            "subtasks": [
                {
                    "subtask": "Task B",
                    "tag": "TASK_B",
                    "constraints": [],
                    "prompt_template": "Do B",
                    "input_vars_required": [],
                    "depends_on": ["TASK_A"],
                },
                {
                    "subtask": "Task A",
                    "tag": "TASK_A",
                    "constraints": [],
                    "prompt_template": "Do A",
                    "input_vars_required": [],
                    "depends_on": [],
                },
            ],
        }

        result = verify_user_variables(decomp_data, None)

        # Should reorder to A, B
        assert result["subtasks"][0]["tag"] == "TASK_A"
        assert result["subtasks"][1]["tag"] == "TASK_B"

    def test_complex_reordering(self):
        """Test complex dependency reordering."""
        decomp_data: DecompPipelineResult = {
            "original_task_prompt": "Test task",
            "subtask_list": ["Task D", "Task C", "Task B", "Task A"],
            "identified_constraints": [],
            "subtasks": [
                {
                    "subtask": "Task D",
                    "tag": "TASK_D",
                    "constraints": [],
                    "prompt_template": "Do D",
                    "input_vars_required": [],
                    "depends_on": ["TASK_B", "TASK_C"],
                },
                {
                    "subtask": "Task C",
                    "tag": "TASK_C",
                    "constraints": [],
                    "prompt_template": "Do C",
                    "input_vars_required": [],
                    "depends_on": ["TASK_A"],
                },
                {
                    "subtask": "Task B",
                    "tag": "TASK_B",
                    "constraints": [],
                    "prompt_template": "Do B",
                    "input_vars_required": [],
                    "depends_on": ["TASK_A"],
                },
                {
                    "subtask": "Task A",
                    "tag": "TASK_A",
                    "constraints": [],
                    "prompt_template": "Do A",
                    "input_vars_required": [],
                    "depends_on": [],
                },
            ],
        }

        result = verify_user_variables(decomp_data, None)

        # A must be first, D must be last
        assert result["subtasks"][0]["tag"] == "TASK_A"
        assert result["subtasks"][3]["tag"] == "TASK_D"


class TestVerifyUserVariablesUnHappyPath:
    """Negative tests for verify_user_variables function."""

    def test_missing_required_input_var(self):
        """Test error when required input variable is not provided."""
        decomp_data: DecompPipelineResult = {
            "original_task_prompt": "Test task",
            "subtask_list": ["Task A"],
            "identified_constraints": [],
            "subtasks": [
                {
                    "subtask": "Task A",
                    "tag": "TASK_A",
                    "constraints": [],
                    "prompt_template": "Do A",
                    "input_vars_required": ["MISSING_VAR"],
                    "depends_on": [],
                }
            ],
        }

        with pytest.raises(
            ValueError,
            match='Subtask "task_a" requires input variable "MISSING_VAR" which was not provided',
        ):
            verify_user_variables(decomp_data, None)

    def test_missing_required_input_var_with_some_provided(self):
        """Test error when one of multiple required variables is missing."""
        decomp_data: DecompPipelineResult = {
            "original_task_prompt": "Test task",
            "subtask_list": ["Task A"],
            "identified_constraints": [],
            "subtasks": [
                {
                    "subtask": "Task A",
                    "tag": "TASK_A",
                    "constraints": [],
                    "prompt_template": "Do A",
                    "input_vars_required": ["VAR1", "VAR2"],
                    "depends_on": [],
                }
            ],
        }

        with pytest.raises(
            ValueError,
            match='Subtask "task_a" requires input variable "VAR2" which was not provided',
        ):
            verify_user_variables(decomp_data, ["VAR1"])

    def test_dependency_on_nonexistent_subtask(self):
        """Test error when subtask depends on non-existent subtask."""
        decomp_data: DecompPipelineResult = {
            "original_task_prompt": "Test task",
            "subtask_list": ["Task A"],
            "identified_constraints": [],
            "subtasks": [
                {
                    "subtask": "Task A",
                    "tag": "TASK_A",
                    "constraints": [],
                    "prompt_template": "Do A",
                    "input_vars_required": [],
                    "depends_on": ["NONEXISTENT_TASK"],
                }
            ],
        }

        with pytest.raises(
            ValueError,
            match='Subtask "task_a" depends on variable "NONEXISTENT_TASK" which does not exist',
        ):
            verify_user_variables(decomp_data, None)

    def test_circular_dependency_detected(self):
        """Test that circular dependencies are caught during reordering."""
        decomp_data: DecompPipelineResult = {
            "original_task_prompt": "Test task",
            "subtask_list": ["Task A", "Task B"],
            "identified_constraints": [],
            "subtasks": [
                {
                    "subtask": "Task B",
                    "tag": "TASK_B",
                    "constraints": [],
                    "prompt_template": "Do B",
                    "input_vars_required": [],
                    "depends_on": ["TASK_A"],
                },
                {
                    "subtask": "Task A",
                    "tag": "TASK_A",
                    "constraints": [],
                    "prompt_template": "Do A",
                    "input_vars_required": [],
                    "depends_on": ["TASK_B"],
                },
            ],
        }

        with pytest.raises(ValueError, match="Circular dependency detected"):
            verify_user_variables(decomp_data, None)

    def test_empty_input_var_list_treated_as_none(self):
        """Test that empty input_var list is treated same as None."""
        decomp_data: DecompPipelineResult = {
            "original_task_prompt": "Test task",
            "subtask_list": ["Task A"],
            "identified_constraints": [],
            "subtasks": [
                {
                    "subtask": "Task A",
                    "tag": "TASK_A",
                    "constraints": [],
                    "prompt_template": "Do A",
                    "input_vars_required": ["REQUIRED_VAR"],
                    "depends_on": [],
                }
            ],
        }

        # Both should raise the same error
        with pytest.raises(ValueError, match="requires input variable"):
            verify_user_variables(decomp_data, [])

        with pytest.raises(ValueError, match="requires input variable"):
            verify_user_variables(decomp_data, None)
