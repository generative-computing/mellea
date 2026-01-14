import importlib
import os
import sys
import tempfile
from typing import List

import pytest

from mellea.formatters import TemplateFormatter
from mellea.backends.model_ids import ModelIdentifier, IBM_GRANITE_3_2_8B
from mellea.core import CBlock, Component, ModelOutputThunk, TemplateRepresentation
from mellea.stdlib.components import Message, Instruction
from mellea.stdlib.components import MObject


@pytest.fixture(scope="module")
def tf() -> TemplateFormatter:
    return TemplateFormatter(
        model_id="default",  # Don't specify a model family so template variability doesn't impact results.
        use_template_cache=False,  # We want to exercise all lookup paths without defaulting to previously used instruction templates.
    )


@pytest.fixture()
def instr() -> Instruction:
    return Instruction(
        description="Write an essay about LLMs.",
        requirements=["Be sure to mention IBM's Granite models."],
    )


def test_cblock_print(tf: TemplateFormatter):
    assert tf.print(CBlock(value="cblock value")) == "cblock value", (
        "printed value did not match cblock value"
    )


def test_component_print(tf: TemplateFormatter, instr: Instruction):
    output = tf.print(instr)

    assert isinstance(instr._description, CBlock), (
        "after printing, an instruction's description is not longer a CBlock"
    )
    assert instr._description.value is not None, (
        "after printing, an instruction's description value is None"
    )
    assert instr._description.value in output, (
        "printed instruction did not contain description"
    )

    assert instr._requirements[0].description is not None, (
        "specified requirement should not have description == None"
    )
    assert instr._requirements[0].description in output, (
        "printed instruction did not contain requirement description"
    )


def test_to_chat_messages(tf: TemplateFormatter):
    sys_msg = Message(role="system", content="system message")
    response = ModelOutputThunk(value="response")
    msgs = tf.to_chat_messages(
        [sys_msg, CBlock(value="cblock 1"), CBlock(value="cblock 2"), response]
    )
    assert all(isinstance(msg, Message) for msg in msgs), (
        "to_chat_messages had a non-message item returned"
    )


def test_custom_template_string(tf: TemplateFormatter):
    class _TemplInstruction(Instruction):
        def format_for_llm(self) -> TemplateRepresentation:
            instr_args = super().format_for_llm().args
            return TemplateRepresentation(
                obj=self, args=instr_args, template="""{{description}}"""
            )

    c = _TemplInstruction("description text", ["req1"])
    out = tf.print(c)
    assert out == "description text"
    assert "req1" not in out, (
        "custom template field failed, requirements shouldn't be included"
    )


def test_string_repre(tf: TemplateFormatter):
    str_repr = "string repr of _StringRepr"

    class _StringRepr(MObject):
        def format_for_llm(self) -> str:
            return str_repr

    output = tf.print(_StringRepr())
    assert output == str_repr, (
        "print output should match string output for format_for_llm"
    )


def test_user_path(instr: Instruction):
    """Ensures that paths with no templates don't prevent default template lookups.

    Also creates a temporary dir to use as a user-specified dir and ensures template lookup
    logic is correct."""
    tf = TemplateFormatter(
        "granite3.3", template_path="/fake/path", use_template_cache=False
    )

    repr = instr.format_for_llm()
    assert type(repr) is TemplateRepresentation, (
        "instruction's llm repr should be a TemplateRepresentation"
    )

    tmpl = tf._load_template(repr)
    assert tmpl.name != "", (
        "a fake template path impacted template lookup; formatter should've defaulted to `mellea`"
    )

    # Point to a user-specified directory with templates.
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
        tf._template_path = os.path.join(td, "templates")

        default_template_dir = os.path.join(td, "templates", "prompts", "default")
        template_path = os.path.join(default_template_dir, "Instruction.jinja2")

        os.makedirs(default_template_dir, exist_ok=True)
        with open(template_path, "w") as temp_f:
            temp_f.write("user provided template.")
        ud_tmpl = tf.print(instr)
        assert "user provided template." in ud_tmpl

        granite_template_dir = os.path.join(td, "templates", "prompts", "granite")
        granite_template_path = os.path.join(granite_template_dir, "Instruction.jinja2")
        os.makedirs(granite_template_dir, exist_ok=True)
        with open(granite_template_path, "w") as temp_f:
            temp_f.write("user provided granite template.")
        ud_tmpl = tf.print(instr)
        assert "user provided granite template." in ud_tmpl

        granite33_template_dir = os.path.join(
            td, "templates", "prompts", "granite", "granite3.3"
        )
        granite33_template_path = os.path.join(
            granite33_template_dir, "Instruction.jinja2"
        )
        os.makedirs(granite33_template_dir, exist_ok=True)
        with open(granite33_template_path, "w") as temp_f:
            temp_f.write("user provided granite33 template.")
        ud_tmpl = tf.print(instr)
        assert "user provided granite33 template." in ud_tmpl


def test_no_module(tf: TemplateFormatter):
    c = Instruction("description")
    c.__module__ = "fake-module"
    tmpl = tf._load_template(c.format_for_llm())
    assert tmpl.name == "prompts/default/Instruction.jinja2", (
        "did not load Mellea default package"
    )


def test_no_template(tf: TemplateFormatter):
    class _NoTemplate(Component[str]):
        def parts(self) -> List[Component | CBlock]:
            return []

        def format_for_llm(self) -> TemplateRepresentation:
            return TemplateRepresentation(self, {})

        def _parse(self, computed: ModelOutputThunk) -> str:
            return ""

    with pytest.raises(Exception):
        tf._load_template(_NoTemplate().format_for_llm())


def test_load_with_model_id(instr: Instruction):
    tf = TemplateFormatter(IBM_GRANITE_3_2_8B)
    tmpl = tf._load_template(instr.format_for_llm())
    assert tmpl.name is not None
    assert "granite" in tmpl.name, (
        "there should always be a granite specific instruction template"
    )


def test_fake_model_id(instr: Instruction):
    tf = TemplateFormatter("fake-model")
    tmpl = tf._load_template(instr.format_for_llm())
    assert tmpl.name is not None
    assert "default" in tmpl.name, (
        "there should always be a default instruction template"
    )


def test_custom_model_id():
    model_id = ModelIdentifier(mlx_name="new-model-here")
    tf = TemplateFormatter(model_id=model_id)
    assert tf._get_model_id() == "new-model-here", (
        "getting the model id should always give a string if one exists"
    )


def test_empty_model_id(instr: Instruction):
    model_id = ModelIdentifier()
    tf = TemplateFormatter(model_id=model_id)
    assert tf._get_model_id() == ""

    tmpl = tf._load_template(instr.format_for_llm())
    assert tmpl.name is not None
    assert "default" in tmpl.name, (
        "there should always be a default instruction template"
    )


def test_template_caching(instr: Instruction):
    """Caching shouldn't be interacted with this way by users.

    Only toggling these internal variables to test code paths."""
    tf = TemplateFormatter("default", use_template_cache=True)
    assert tf._template_cache is not None

    default_tmpl = tf._load_template(instr.format_for_llm())
    res = tf._template_cache.cache.get(instr.__class__.__name__)
    assert res is default_tmpl

    # This test relies on granite having a specific instruction template.
    tf.model_id = "granite"
    tmpl = tf._load_template(instr.format_for_llm())
    assert tmpl is default_tmpl

    tf._use_template_cache = False
    tmpl = tf._load_template(instr.format_for_llm())
    assert tmpl is not default_tmpl, (
        "template formatter appeared to use cache or grab the wrong template when caching was disabled"
    )


def test_custom_component_external_package(tf: TemplateFormatter):
    """Creates a fake package with a custom component and loads the package.
    Ensures template loading works for custom components defined in other packages."""

    new_component_content = """
from mellea.core import Component, TemplateRepresentation, ModelOutputThunk
class NewComponent(Component[str]):
    def parts(self):
        raise NotImplementedError(
            "Disallowing use of `parts` until we figure out exactly what it's supposed to be for"
        )

    def format_for_llm(self) -> dict:
        return TemplateRepresentation(
            self,
            {"text": "template arg version of new component"}
        )

    def _parse(self, computed: ModelOutputThunk) -> str:
        return ""
"""

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
        init_path = os.path.join(td, "__init__.py")
        component_path = os.path.join(td, "new_component.py")
        template_dir = os.path.join(td, "templates", "prompts", "default")
        template_path = os.path.join(template_dir, "NewComponent.jinja2")

        with open(init_path, "w") as init:
            init.write("")

        with open(component_path, "w") as comp_f:
            comp_f.write(new_component_content)

        os.makedirs(template_dir, exist_ok=True)
        with open(template_path, "w") as temp_f:
            temp_f.write("{{text}}")

        sys.path.append(td)
        temp_module = importlib.import_module("new_component")

        NewComponent = getattr(temp_module, "NewComponent")
        nc = NewComponent()
        assert tf.print(nc) == "template arg version of new component", (
            "did not get expected template for a 3rd party package NewComponent"
        )


def test_custom_template_order(tf: TemplateFormatter, instr: Instruction):
    repr = instr.format_for_llm()
    repr.template_order = ["Query"]

    tmpl = tf._load_template(repr)
    assert tmpl.name == "prompts/default/Query.jinja2", (
        "changing template order did not change the template retrieved"
    )

    repr.template_order = ["FakeQuery"]
    with pytest.raises(ValueError):
        tf._load_template(repr)


def test_to_chat_messages_not_parsed_repr(tf: TemplateFormatter):
    action = ModelOutputThunk(value="test value")

    parsed_repr = Message(role="assistant", content="different content")

    messages = tf.to_chat_messages([action])
    assert messages[0].content == "test value"

    action.parsed_repr = parsed_repr
    messages = tf.to_chat_messages([action])
    assert messages[0] is parsed_repr
    assert messages[0].content == "different content"


if __name__ == "__main__":
    pytest.main([__file__])
