import pytest

from mellea.core import TemplateRepresentation
from mellea.stdlib.components.docs import TableTransform
from mellea.stdlib.components import MObject, Query, Transform

custom_mobject_description = "custom mobject description"


@pytest.fixture
def mobj() -> MObject:
    class _CustomMObject(MObject):
        def __init__(
            self,
            description: str,
            *,
            query_type: type = Query,
            transform_type: type = Transform,
        ) -> None:
            super().__init__(query_type=query_type, transform_type=transform_type)
            self._description = description

        def fake_tool_function(self) -> str:
            return "wow a tool function"

        def format_for_llm(self) -> str:
            return self._description

    return _CustomMObject(custom_mobject_description)


def test_get_transform_object(mobj: MObject):
    tr_text = "transform text"
    transform = mobj.get_transform_object(tr_text)
    assert isinstance(transform, Transform)

    tr = transform.format_for_llm()
    assert isinstance(tr, TemplateRepresentation)
    assert tr.args.get("transformation", "") == tr_text

    transform_tr_content = tr.args.get("content", None)
    assert transform_tr_content is not None
    assert isinstance(transform_tr_content, MObject)

    original_obj_tr = transform_tr_content.format_for_llm()
    assert original_obj_tr == custom_mobject_description


def test_get_transform_object_custom():
    tr_text = "transform text"
    mobj = MObject(transform_type=TableTransform)
    transform = mobj.get_transform_object(tr_text)
    assert isinstance(transform, TableTransform)

    with pytest.raises(AssertionError):
        tr = transform.format_for_llm()


if __name__ == "__main__":
    pytest.main([__file__])
