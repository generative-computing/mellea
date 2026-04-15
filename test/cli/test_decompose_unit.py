"""Unit tests for decompose pure-logic helpers — no backend, no file I/O required.

Covers reorder_subtasks, verify_user_variables, validate_filename.
"""

import pytest

from cli.decompose.decompose import reorder_subtasks, verify_user_variables
from cli.decompose.utils import validate_filename

# --- reorder_subtasks ---


def _subtask(tag: str, subtask: str, depends_on: list[str] | None = None):
    """Minimal subtask dict for testing."""
    d = {
        "tag": tag,
        "subtask": subtask,
        "constraints": [],
        "prompt_template": "",
        "general_instructions": "",
        "input_vars_required": [],
        "depends_on": depends_on or [],
    }
    return d


def test_reorder_no_dependencies():
    subtasks = [_subtask("a", "1. Task A"), _subtask("b", "2. Task B")]
    result = reorder_subtasks(subtasks)
    # No dependencies — all tasks present, order unconstrained
    assert {s["tag"] for s in result} == {"a", "b"}


def test_reorder_respects_dependency():
    subtasks = [
        _subtask("b", "1. Task B", depends_on=["a"]),
        _subtask("a", "2. Task A"),
    ]
    result = reorder_subtasks(subtasks)
    tags = [s["tag"] for s in result]
    assert tags.index("a") < tags.index("b")


def test_reorder_chain_dependency():
    subtasks = [
        _subtask("c", "1. C", depends_on=["b"]),
        _subtask("b", "2. B", depends_on=["a"]),
        _subtask("a", "3. A"),
    ]
    result = reorder_subtasks(subtasks)
    tags = [s["tag"] for s in result]
    assert tags == ["a", "b", "c"]


def test_reorder_circular_raises():
    subtasks = [
        _subtask("a", "1. A", depends_on=["b"]),
        _subtask("b", "2. B", depends_on=["a"]),
    ]
    with pytest.raises(ValueError, match="Circular dependency"):
        reorder_subtasks(subtasks)


def test_reorder_renumbers_subtasks():
    subtasks = [
        _subtask("b", "2. Task B", depends_on=["a"]),
        _subtask("a", "1. Task A"),
    ]
    result = reorder_subtasks(subtasks)
    # After reordering, numbering should be updated
    assert result[0]["subtask"].startswith("1. ")
    assert result[1]["subtask"].startswith("2. ")


def test_reorder_invalid_dependency_ignored():
    subtasks = [_subtask("a", "1. A", depends_on=["nonexistent"])]
    result = reorder_subtasks(subtasks)
    assert len(result) == 1
    assert result[0]["tag"] == "a"


def test_reorder_case_insensitive_dependency():
    # Tags and depends_on are lowercased before lookup — mixed case must resolve correctly
    subtasks = [
        _subtask("b", "1. Task B", depends_on=["A"]),
        _subtask("a", "2. Task A"),
    ]
    result = reorder_subtasks(subtasks)
    tags = [s["tag"] for s in result]
    assert tags.index("a") < tags.index("b")


# --- verify_user_variables ---


def _decomp_data(subtasks):
    return {
        "original_task_prompt": "",
        "subtask_list": [],
        "identified_constraints": [],
        "subtasks": subtasks,
    }


def test_verify_valid_input_vars():
    data = _decomp_data([_subtask("a", "A", depends_on=[])])
    data["subtasks"][0]["input_vars_required"] = ["doc"]
    result = verify_user_variables(data, input_var=["doc"])
    assert result is data


def test_verify_missing_input_var_raises():
    data = _decomp_data([_subtask("a", "A")])
    data["subtasks"][0]["input_vars_required"] = ["doc"]
    with pytest.raises(ValueError, match="requires input variable"):
        verify_user_variables(data, input_var=[])


def test_verify_missing_dependency_raises():
    data = _decomp_data([_subtask("a", "A", depends_on=["nonexistent"])])
    with pytest.raises(ValueError, match="does not exist"):
        verify_user_variables(data, input_var=[])


def test_verify_reorders_when_needed():
    data = _decomp_data(
        [_subtask("b", "1. B", depends_on=["a"]), _subtask("a", "2. A")]
    )
    result = verify_user_variables(data, input_var=None)
    tags = [s["tag"] for s in result["subtasks"]]
    assert tags.index("a") < tags.index("b")


def test_verify_no_reorder_when_already_sorted():
    data = _decomp_data(
        [_subtask("a", "1. A"), _subtask("b", "2. B", depends_on=["a"])]
    )
    result = verify_user_variables(data, input_var=None)
    tags = [s["tag"] for s in result["subtasks"]]
    assert tags == ["a", "b"]


def test_verify_none_input_var_treated_as_empty():
    data = _decomp_data([_subtask("a", "A")])
    result = verify_user_variables(data, input_var=None)
    assert result is data


# --- validate_filename ---


def test_valid_filename():
    assert validate_filename("my_output_file") is True


def test_valid_filename_with_extension():
    assert validate_filename("results.json") is True


def test_valid_filename_with_hyphen():
    assert validate_filename("my-output") is True


def test_valid_filename_with_spaces():
    assert validate_filename("my output file") is True


def test_invalid_filename_slash():
    assert validate_filename("path/to/file") is False


def test_invalid_filename_empty():
    assert validate_filename("") is False


def test_invalid_filename_single_char():
    # Pattern requires at least 2 chars (first char + rest)
    assert validate_filename("a") is False


def test_invalid_filename_starts_with_hyphen():
    assert validate_filename("-badname") is False


def test_valid_filename_starts_with_dot():
    assert validate_filename(".hidden_file") is True


def test_invalid_filename_too_long():
    assert validate_filename("a" * 251) is False


def test_valid_filename_max_length():
    assert validate_filename("a" * 250) is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
