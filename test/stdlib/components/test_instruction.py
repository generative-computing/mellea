"""Unit tests for the Instruction component — init, jinja rendering, copy/repair, parts, format."""

import pytest

from mellea.core import CBlock, ModelOutputThunk, Requirement, TemplateRepresentation
from mellea.stdlib.components.instruction import Instruction

# --- basic init ---


def test_init_minimal():
    ins = Instruction(description="summarise the text")
    assert ins._description is not None
    assert str(ins._description) == "summarise the text"
    assert ins._requirements == []
    assert ins._icl_examples == []
    assert ins._grounding_context == {}
    assert ins._repair_string is None


def test_init_no_args():
    ins = Instruction()
    assert ins._description is None
    assert ins._requirements == []


def test_init_converts_string_description_to_cblock():
    ins = Instruction(description="hello")
    assert isinstance(ins._description, CBlock)


def test_init_accepts_cblock_description():
    cb = CBlock("already a block")
    ins = Instruction(description=cb)
    assert ins._description is cb


def test_init_string_requirements_converted():
    ins = Instruction(requirements=["must be concise", "must be accurate"])
    assert len(ins._requirements) == 2
    for r in ins._requirements:
        assert isinstance(r, Requirement)


def test_init_requirement_objects_preserved():
    r = Requirement(description="no profanity")
    ins = Instruction(requirements=[r])
    assert ins._requirements[0].description == "no profanity"


def test_init_grounding_context_strings_blockified():
    ins = Instruction(grounding_context={"doc1": "some content"})
    assert isinstance(ins._grounding_context["doc1"], CBlock)


def test_init_prefix_converted():
    ins = Instruction(prefix="Answer:")
    assert isinstance(ins._prefix, CBlock)


def test_init_output_prefix_raises():
    """output_prefix is currently unsupported; should raise AssertionError."""
    with pytest.raises(
        AssertionError, match="output_prefix is not currently supported"
    ):
        Instruction(user_variables={"x": "y"}, output_prefix="Result:")


# --- apply_user_dict_from_jinja ---


def test_jinja_simple_substitution():
    result = Instruction.apply_user_dict_from_jinja(
        {"name": "world"}, "Hello {{ name }}!"
    )
    assert result == "Hello world!"


def test_jinja_multiple_variables():
    result = Instruction.apply_user_dict_from_jinja(
        {"a": "foo", "b": "bar"}, "{{ a }} and {{ b }}"
    )
    assert result == "foo and bar"


def test_jinja_missing_variable_renders_empty():
    result = Instruction.apply_user_dict_from_jinja({}, "Hello {{ name }}!")
    assert result == "Hello !"


def test_jinja_no_variables():
    result = Instruction.apply_user_dict_from_jinja({}, "plain string")
    assert result == "plain string"


# --- user_variables applied to fields ---


def test_user_variables_applied_to_description():
    ins = Instruction(
        description="Task: {{ task }}", user_variables={"task": "translate"}
    )
    assert str(ins._description) == "Task: translate"


def test_user_variables_applied_to_prefix():
    ins = Instruction(
        prefix="{{ prefix_word }}:", user_variables={"prefix_word": "Answer"}
    )
    assert str(ins._prefix) == "Answer:"


def test_user_variables_applied_to_requirements():
    ins = Instruction(
        requirements=["must be in {{ lang }}"], user_variables={"lang": "French"}
    )
    assert ins._requirements[0].description == "must be in French"


def test_user_variables_applied_to_icl_examples():
    ins = Instruction(icl_examples=["Example: {{ ex }}"], user_variables={"ex": "blue"})
    assert str(ins._icl_examples[0]) == "Example: blue"


def test_user_variables_applied_to_grounding_context():
    ins = Instruction(
        grounding_context={"doc": "See {{ ref }}"}, user_variables={"ref": "section 3"}
    )
    assert str(ins._grounding_context["doc"]) == "See section 3"


def test_user_variables_description_must_be_string():
    with pytest.raises(AssertionError, match="description must be a string"):
        Instruction(description=CBlock("not a string"), user_variables={"x": "y"})


def test_user_variables_requirement_object_description_rendered():
    r = Requirement(description="must be in {{ lang }}")
    ins = Instruction(requirements=[r], user_variables={"lang": "Spanish"})
    assert ins._requirements[0].description == "must be in Spanish"


# --- parts() ---


def test_parts_includes_description():
    ins = Instruction(description="do something")
    parts = ins.parts()
    assert ins._description in parts


def test_parts_includes_requirements():
    r = Requirement(description="be concise")
    ins = Instruction(description="task", requirements=[r])
    assert r in ins.parts()


def test_parts_includes_grounding_context_values():
    ins = Instruction(grounding_context={"doc": "content"})
    parts = ins.parts()
    assert ins._grounding_context["doc"] in parts


def test_parts_empty_instruction():
    ins = Instruction()
    # No description, no requirements, no grounding context
    assert ins.parts() == []


def test_parts_includes_icl_examples():
    ins = Instruction(icl_examples=["example 1"])
    parts = ins.parts()
    assert len(parts) == 1


# --- format_for_llm ---


def test_format_for_llm_returns_template_representation():
    ins = Instruction(description="do something")
    result = ins.format_for_llm()
    assert isinstance(result, TemplateRepresentation)


def test_format_for_llm_args_structure():
    ins = Instruction(description="task", requirements=["req 1"], icl_examples=["ex 1"])
    result = ins.format_for_llm()
    assert "description" in result.args
    assert "requirements" in result.args
    assert "icl_examples" in result.args
    assert "grounding_context" in result.args
    assert "repair" in result.args


def test_format_for_llm_check_only_req_excluded():
    r = Requirement(description="internal check", check_only=True)
    ins = Instruction(requirements=[r])
    result = ins.format_for_llm()
    assert r.description not in result.args["requirements"]


def test_format_for_llm_repair_is_none_by_default():
    ins = Instruction(description="task")
    result = ins.format_for_llm()
    assert result.args["repair"] is None


# --- copy_and_repair ---


def test_copy_and_repair_sets_repair_string():
    ins = Instruction(description="task", requirements=["be brief"])
    repaired = ins.copy_and_repair("requirement 'be brief' not met")
    assert repaired._repair_string == "requirement 'be brief' not met"


def test_copy_and_repair_does_not_mutate_original():
    ins = Instruction(description="task")
    _ = ins.copy_and_repair("failed")
    assert ins._repair_string is None


def test_copy_and_repair_deep_copy():
    ins = Instruction(description="task", requirements=["be brief"])
    repaired = ins.copy_and_repair("reason")
    # Mutating the copy's requirements should not affect the original
    repaired._requirements.append(Requirement(description="new"))
    assert len(ins._requirements) == 1


def test_copy_and_repair_format_includes_repair():
    ins = Instruction(description="task")
    repaired = ins.copy_and_repair("please fix this")
    result = repaired.format_for_llm()
    assert result.args["repair"] == "please fix this"


# --- _parse ---


def test_parse_returns_value():
    ins = Instruction(description="x")
    mot = ModelOutputThunk(value="answer")
    assert ins._parse(mot) == "answer"


def test_parse_none_returns_empty_string():
    ins = Instruction(description="x")
    mot = ModelOutputThunk(value=None)
    assert ins._parse(mot) == ""


# --- requirements property ---


def test_requirements_property():
    ins = Instruction(requirements=["be brief", "be accurate"])
    reqs = ins.requirements
    assert len(reqs) == 2
    assert all(isinstance(r, Requirement) for r in reqs)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
