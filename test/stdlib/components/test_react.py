"""Unit tests for mellea.stdlib.components.react."""

import pytest

from mellea.backends.tools import MelleaTool
from mellea.core import CBlock
from mellea.core.base import ModelOutputThunk, TemplateRepresentation
from mellea.stdlib.components.react import (
    MELLEA_FINALIZER_TOOL,
    ReactInitiator,
    ReactThought,
)

# --- ReactInitiator ---


def test_initiator_goal_wrapped_as_cblock():
    ri = ReactInitiator(goal="solve X", tools=None)
    assert isinstance(ri.goal, CBlock)
    assert ri.goal.value == "solve X"


def test_initiator_parts():
    ri = ReactInitiator(goal="g", tools=None)
    parts = ri.parts()
    assert len(parts) == 1
    assert parts[0] is ri.goal


def test_initiator_format_injects_finalizer():
    ri = ReactInitiator(goal="g", tools=None)
    rep = ri.format_for_llm()
    assert isinstance(rep, TemplateRepresentation)
    assert rep.tools is not None
    assert MELLEA_FINALIZER_TOOL in rep.tools
    assert rep.args["finalizer_tool_name"] == MELLEA_FINALIZER_TOOL


def test_initiator_format_with_user_tools():
    tool = MelleaTool.from_callable(lambda q: q, "search")
    ri = ReactInitiator(goal="g", tools=[tool])
    rep = ri.format_for_llm()
    assert rep.tools is not None
    assert "search" in rep.tools
    assert MELLEA_FINALIZER_TOOL in rep.tools


def test_initiator_format_overrides_user_finalizer():
    """User tool named 'final_answer' is overridden with a warning."""
    user_tool = MelleaTool.from_callable(lambda a: a, MELLEA_FINALIZER_TOOL)
    ri = ReactInitiator(goal="g", tools=[user_tool])
    rep = ri.format_for_llm()
    assert rep.tools is not None
    assert MELLEA_FINALIZER_TOOL in rep.tools
    # The finalizer in the representation should be the internally-created one, not the user's
    assert rep.tools[MELLEA_FINALIZER_TOOL] is not user_tool


# --- ReactThought ---


def test_thought_format():
    rt = ReactThought()
    rep = rt.format_for_llm()
    assert isinstance(rep, TemplateRepresentation)
    assert rep.args == {}
    assert rep.template_order == ["*", "ReactThought"]


# --- _parse ---


def test_initiator_parse_value():
    ri = ReactInitiator(goal="g", tools=None)
    mot = ModelOutputThunk(value="answer")
    assert ri._parse(mot) == "answer"


def test_initiator_parse_none():
    ri = ReactInitiator(goal="g", tools=None)
    mot = ModelOutputThunk(value=None)
    assert ri._parse(mot) == ""


def test_thought_parse_value():
    rt = ReactThought()
    mot = ModelOutputThunk(value="thinking")
    assert rt._parse(mot) == "thinking"


def test_thought_parse_none():
    rt = ReactThought()
    mot = ModelOutputThunk(value=None)
    assert rt._parse(mot) == ""


if __name__ == "__main__":
    pytest.main([__file__])
