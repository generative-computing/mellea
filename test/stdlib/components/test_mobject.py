"""Unit tests for Query, Transform, and MObject — no docling, no backend required."""

import pytest

from mellea.core import ModelOutputThunk, TemplateRepresentation
from mellea.stdlib.components.mobject import MObject, Query, Transform

# --- helpers ---


class _SimpleComponent(MObject):
    """Minimal MObject subclass for testing."""

    def __init__(self, content: str = "hello") -> None:
        super().__init__()
        self._content = content

    def content_as_string(self) -> str:
        return self._content

    def format_for_llm(self) -> str:
        return self._content

    def parts(self):
        return []

    def _parse(self, computed):
        return computed.value or ""


# --- Query ---


def test_query_parts_returns_wrapped_object():
    obj = _SimpleComponent("doc text")
    q = Query(obj, "what is this?")
    parts = q.parts()
    assert len(parts) == 1
    assert parts[0] is obj


def test_query_format_for_llm_returns_template_repr():
    obj = _SimpleComponent("text")
    q = Query(obj, "summarise")
    result = q.format_for_llm()
    assert isinstance(result, TemplateRepresentation)


def test_query_format_for_llm_query_field():
    obj = _SimpleComponent("text")
    q = Query(obj, "what colour?")
    result = q.format_for_llm()
    assert result.args["query"] == "what colour?"


def test_query_format_for_llm_content_is_wrapped_object():
    obj = _SimpleComponent("text")
    q = Query(obj, "q")
    result = q.format_for_llm()
    assert result.args["content"] is obj


def test_query_parse_returns_value():
    obj = _SimpleComponent()
    q = Query(obj, "q")
    mot = ModelOutputThunk(value="answer")
    assert q._parse(mot) == "answer"


def test_query_parse_none_returns_empty():
    obj = _SimpleComponent()
    q = Query(obj, "q")
    mot = ModelOutputThunk(value=None)
    assert q._parse(mot) == ""


# --- Transform ---


def test_transform_parts_returns_wrapped_object():
    obj = _SimpleComponent("doc text")
    t = Transform(obj, "translate to French")
    parts = t.parts()
    assert len(parts) == 1
    assert parts[0] is obj


def test_transform_format_for_llm_returns_template_repr():
    obj = _SimpleComponent("text")
    t = Transform(obj, "rewrite formally")
    result = t.format_for_llm()
    assert isinstance(result, TemplateRepresentation)


def test_transform_format_for_llm_transformation_field():
    obj = _SimpleComponent("text")
    t = Transform(obj, "make it shorter")
    result = t.format_for_llm()
    assert result.args["transformation"] == "make it shorter"


def test_transform_format_for_llm_content_is_wrapped_object():
    obj = _SimpleComponent("text")
    t = Transform(obj, "x")
    result = t.format_for_llm()
    assert result.args["content"] is obj


def test_transform_parse_returns_value():
    obj = _SimpleComponent()
    t = Transform(obj, "x")
    mot = ModelOutputThunk(value="result")
    assert t._parse(mot) == "result"


# --- MObject ---


def test_mobject_parts_empty():
    obj = _SimpleComponent()
    assert obj.parts() == []


def test_mobject_get_query_object():
    obj = _SimpleComponent("text")
    q = obj.get_query_object("what is this?")
    assert isinstance(q, Query)
    assert q._query == "what is this?"
    assert q._obj is obj


def test_mobject_get_transform_object():
    obj = _SimpleComponent("text")
    t = obj.get_transform_object("shorten it")
    assert isinstance(t, Transform)
    assert t._transformation == "shorten it"
    assert t._obj is obj


def test_mobject_content_as_string():
    obj = _SimpleComponent("my content")
    assert obj.content_as_string() == "my content"


def test_mobject_format_for_llm_returns_template_repr():
    obj = _SimpleComponent("text")
    result = obj.format_for_llm()
    # Uses the overridden format_for_llm returning str
    assert result == "text"


def test_mobject_custom_query_type():
    class _CustomQuery(Query):
        pass

    obj = MObject(query_type=_CustomQuery)
    q = obj.get_query_object("q")
    assert isinstance(q, _CustomQuery)


def test_mobject_custom_transform_type():
    class _CustomTransform(Transform):
        pass

    obj = MObject(transform_type=_CustomTransform)
    t = obj.get_transform_object("t")
    assert isinstance(t, _CustomTransform)


def test_mobj_base_format_for_llm():
    """Test MObject.format_for_llm (not the overridden version) via base class directly."""

    class _MObjectWithTools(MObject):
        def my_tool(self) -> str:
            """A custom tool."""
            return "result"

        def content_as_string(self) -> str:
            return "content"

        def parts(self):
            return []

        def format_for_llm(self):
            return MObject.format_for_llm(self)

        def _parse(self, computed):
            return ""

    obj = _MObjectWithTools()
    result = obj.format_for_llm()
    assert isinstance(result, TemplateRepresentation)
    assert result.args["content"] == "content"


def test_mobj_parse_returns_value():
    class _M(MObject):
        def content_as_string(self):
            return ""

        def parts(self):
            return []

        def _parse(self, computed):
            return MObject._parse(self, computed)

    obj = _M()
    mot = ModelOutputThunk(value="result")
    assert obj._parse(mot) == "result"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
