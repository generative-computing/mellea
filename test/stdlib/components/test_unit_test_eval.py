"""Unit tests for mellea.stdlib.components.unit_test_eval."""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from mellea.stdlib.components.unit_test_eval import (
    Example,
    Message,
    TestBasedEval,
    TestData,
)

# --- Pydantic model validation ---


def test_example_defaults():
    ex = Example(input=[Message(role="user", content="hi")])
    assert ex.targets == []
    assert ex.input_id == ""


def test_test_data_empty_examples_rejected():
    with pytest.raises(ValidationError, match="examples list cannot be empty"):
        TestData(source="test", name="test", instructions="test", examples=[], id="t1")


# --- TestBasedEval init & defaults ---


def test_init_defaults():
    eval_obj = TestBasedEval(source="s", name="n", instructions="i", inputs=["q"])
    assert eval_obj.targets == []
    assert eval_obj.input_ids == []
    assert eval_obj.test_id is None


def test_init_explicit():
    eval_obj = TestBasedEval(
        source="s",
        name="n",
        instructions="i",
        inputs=["q1", "q2"],
        targets=[["a1"], ["a2"]],
        test_id="tid",
        input_ids=["id1", "id2"],
    )
    assert eval_obj.targets == [["a1"], ["a2"]]
    assert eval_obj.input_ids == ["id1", "id2"]
    assert eval_obj.test_id == "tid"


# --- format_for_llm ---


def test_format_for_llm_no_context():
    eval_obj = TestBasedEval(source="s", name="n", instructions="i", inputs=["q"])
    rep = eval_obj.format_for_llm()
    assert rep.args == {}


def test_format_for_llm_with_context():
    eval_obj = TestBasedEval(source="s", name="n", instructions="i", inputs=["q"])
    eval_obj.set_judge_context("input", "pred", ["target"])
    rep = eval_obj.format_for_llm()
    assert rep.args["input"] == "input"
    assert rep.args["prediction"] == "pred"
    assert rep.args["target"] == "target"
    assert rep.args["guidelines"] == "i"


# --- set_judge_context ---


def test_set_judge_context_no_targets():
    eval_obj = TestBasedEval(source="s", name="n", instructions="i", inputs=["q"])
    eval_obj.set_judge_context("in", "pred", [])
    assert eval_obj._judge_context["target"] == "N/A"


def test_set_judge_context_single_target():
    eval_obj = TestBasedEval(source="s", name="n", instructions="i", inputs=["q"])
    eval_obj.set_judge_context("in", "pred", ["only"])
    assert eval_obj._judge_context["target"] == "only"


def test_set_judge_context_multiple_targets():
    eval_obj = TestBasedEval(source="s", name="n", instructions="i", inputs=["q"])
    eval_obj.set_judge_context("in", "pred", ["a", "b", "c"])
    expected = "1. a\n2. b\n3. c"
    assert eval_obj._judge_context["target"] == expected


# --- from_json_file ---


def _minimal_test_data(
    *,
    source: str = "src",
    name: str = "test",
    instructions: str = "eval",
    test_id: str = "t1",
    examples: list | None = None,
) -> dict:
    if examples is None:
        examples = [
            {
                "input": [{"role": "user", "content": "q1"}],
                "targets": [{"role": "assistant", "content": "a1"}],
                "input_id": "ex1",
            }
        ]
    return {
        "source": source,
        "name": name,
        "instructions": instructions,
        "id": test_id,
        "examples": examples,
    }


def _write_test_json(tmp_path: Path, data: dict | list) -> str:
    """Write test data to a JSON file and return the path string."""
    p = tmp_path / "data.json"
    p.write_text(json.dumps(data))
    return str(p)


def test_from_json_file_single_object(tmp_path):
    path = _write_test_json(tmp_path, _minimal_test_data())
    evals = TestBasedEval.from_json_file(path)
    assert len(evals) == 1
    assert evals[0].source == "src"
    assert evals[0].inputs == ["q1"]
    assert evals[0].targets == [["a1"]]
    assert evals[0].input_ids == ["ex1"]


def test_from_json_file_array(tmp_path):
    data = [_minimal_test_data(test_id="t1"), _minimal_test_data(test_id="t2")]
    path = _write_test_json(tmp_path, data)
    evals = TestBasedEval.from_json_file(path)
    assert len(evals) == 2


def test_from_json_file_invalid_data(tmp_path):
    path = _write_test_json(tmp_path, {"bad": "data"})
    with pytest.raises(ValueError, match="Invalid test data"):
        TestBasedEval.from_json_file(path)


def test_from_json_file_multiple_user_messages(tmp_path):
    """Only the last user message is used as input."""
    data = _minimal_test_data(
        examples=[
            {
                "input": [
                    {"role": "user", "content": "first"},
                    {"role": "assistant", "content": "mid"},
                    {"role": "user", "content": "second"},
                ],
                "targets": [],
                "input_id": "",
            }
        ]
    )
    path = _write_test_json(tmp_path, data)
    evals = TestBasedEval.from_json_file(path)
    assert evals[0].inputs == ["second"]


def test_from_json_file_no_user_messages(tmp_path):
    """Example with no user messages produces no input entry."""
    data = _minimal_test_data(
        examples=[
            {
                "input": [{"role": "system", "content": "sys"}],
                "targets": [],
                "input_id": "",
            }
        ]
    )
    path = _write_test_json(tmp_path, data)
    evals = TestBasedEval.from_json_file(path)
    assert evals[0].inputs == []


def test_from_json_file_filters_non_assistant_targets(tmp_path):
    """Only assistant-role messages are extracted as targets."""
    data = _minimal_test_data(
        examples=[
            {
                "input": [{"role": "user", "content": "q"}],
                "targets": [
                    {"role": "assistant", "content": "good"},
                    {"role": "user", "content": "ignored"},
                    {"role": "assistant", "content": "also good"},
                ],
                "input_id": "",
            }
        ]
    )
    path = _write_test_json(tmp_path, data)
    evals = TestBasedEval.from_json_file(path)
    assert evals[0].targets == [["good", "also good"]]


if __name__ == "__main__":
    pytest.main([__file__])
