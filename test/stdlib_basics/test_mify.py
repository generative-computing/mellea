import pytest

from mellea.backends.formatter import TemplateFormatter
from mellea.stdlib.base import Component, TemplateRepresentation
from mellea.stdlib.mobject import Query, MObjectProtocol, MObject
from mellea.stdlib.mify import mify, MifiedProtocol


def test_protocol_adherence():
    mobj = MObject()
    assert isinstance(mobj, MObjectProtocol), (
        "mobject doesn't conform to mobject protocol"
    )
    assert isinstance(mobj, Component), "mobject doesn't conform to component protocol"

    @mify
    class _Customer:
        def __init__(self, name: str) -> None:
            self.name = name

    mified_class = _Customer("Jake")
    assert isinstance(mified_class, MObjectProtocol), (
        "mified class doesn't conform to mobject protocol"
    )
    assert isinstance(mified_class, Component), (
        "mified class doesn't conform to component protocol"
    )

    class _Customer2:
        def __init__(self, name: str) -> None:
            self.name = name

    c = _Customer2("jake")
    mify(c)
    assert isinstance(c, MObjectProtocol), (
        "mified object doesn't conform to mobject protocol"
    )
    assert isinstance(c, Component), (
        "mified object doesn't conform to component protocol"
    )


def test_mify_class():
    @mify()
    class _Customer:
        def __init__(self, name: str) -> None:
            self.name = name

    c = _Customer("Jake")
    assert isinstance(c, MifiedProtocol)

    @mify
    class _Customer2:
        def __init__(self, name) -> None:
            self.name = name

    c = _Customer2("Jake")
    assert isinstance(c, MifiedProtocol)

    mify(_Customer)


def test_mify_class_with_existing_objects():
    class _Customer:
        def __init__(self, name: str) -> None:
            self.name = name

    c = _Customer("jake")
    c2 = _Customer("jacob")

    mify(_Customer)
    assert isinstance(c, MifiedProtocol)

    # Make sure class/instance fields work as expected.
    c._fields_include = {"field"}
    assert c._fields_include != c2._fields_include  # type: ignore


def test_mify_object():
    class _Customer:
        def __init__(self, name: str) -> None:
            self.name = name

    c = _Customer("Jake")
    c_mified = mify(obj=c)
    assert c == c_mified
    assert isinstance(c, MifiedProtocol)

    # This call will fail if the mify methods aren't actually bound to the object.
    # Leave it here so that it fails if something changes that.
    c.format_for_llm()


def test_mify_object_and_class():
    class _Customer:
        def __init__(self, name: str) -> None:
            self.name = name

    c = _Customer("Jake")
    fields_exclude = {"random_field"}
    mify(c, fields_exclude=fields_exclude)
    mify(_Customer)

    assert isinstance(c, MifiedProtocol)
    assert c._fields_exclude == fields_exclude


def test_mify_mobject():
    mobj = MObject()
    mify(mobj)
    assert hasattr(mobj, "_query_type")
    assert not hasattr(mobj, "_fields_include"), "mobject was incorrectly mified"


class CustomerQuery(Query):
    def format_for_llm(self) -> TemplateRepresentation | str:
        q = super().format_for_llm()
        if isinstance(q, TemplateRepresentation):
            q.template_order = ["CustomerQuery", "Query"]
        return q


@mify(
    query_type=CustomerQuery,
    fields_include={"name"},
    funcs_include={"get_details"},
    template_order=["Custom"],
    stringify_func=lambda x: x.get_details(),  # type: ignore
)
class Customer:
    def __init__(self, name: str, age: int):
        self.name = name
        self.age = age

    def get_details(self) -> str:
        return f"{self.name}, {self.age}"

    def extraneous_func(self):
        """this function does nothing."""
        return


@pytest.fixture(scope="function")
def mified_customer() -> Customer:
    return Customer("Jacob", 26)


def test_mify_class_parameterized(mified_customer: Customer):
    c = mified_customer
    assert isinstance(c, MifiedProtocol)
    assert c._query_type == CustomerQuery
    assert c._fields_include == {"name"}
    assert c._template_order == ["Custom"]

    assert isinstance(c.get_query_object("test"), CustomerQuery)
    assert c.content_as_string() == f"{c.name}, {c.age}"
    c._stringify_func = None
    assert c.content_as_string() is not None


def test_mify_class_parameterized_funcs(mified_customer: Customer):
    c = mified_customer
    assert isinstance(c, MifiedProtocol)
    assert c._funcs_include == {"get_details"}

    members = c._get_all_members()
    func = members.get("get_details", None)
    assert func is not None

    c._funcs_exclude = {"get_details"}
    assert len(c._get_all_members()) == 0

    c._funcs_exclude = None
    c._funcs_include = None
    members = c._get_all_members()
    assert len(members) == 1
    func = members.get("extraneous_func", None)
    assert func is not None

    c._funcs_include = set()
    assert len(c._get_all_members()) == 0


def test_mify_class_parameterized_fields(mified_customer: Customer):
    c = mified_customer
    assert isinstance(c, MifiedProtocol)
    assert c._fields_include == {"name"}

    fields = c._get_all_fields()
    field = fields.get("name", None)
    assert field == "Jacob"

    c._fields_exclude = {"name"}
    assert len(c._get_all_fields()) == 0

    c._fields_include = None
    c._fields_exclude = None
    assert len(c._get_all_fields()) > 0

    c._fields_exclude = {"name"}
    fields = c._get_all_fields()
    assert fields.get("age", None) is not None

    c._fields_exclude = None
    c._fields_include = set()
    assert len(c._get_all_fields()) == 0


def test_mify_class_parameterized_format_for_llm(mified_customer: Customer):
    c = mified_customer
    assert isinstance(c, MifiedProtocol)

    c._fields_include = None
    c._fields_exclude = None
    c._funcs_exclude = None
    c._funcs_include = None
    tr = c.format_for_llm()
    assert tr.args.get("content", None) == f"{c.name}, {c.age}"
    assert tr.template_order == ["Custom"]
    assert tr.obj is c
    assert tr.fields is not None
    assert len(tr.fields) == 0

    assert tr.tools is not None
    assert tr.tools.get("extraneous_func", None) is not None
    assert len(tr.tools) == 1

    c._fields_include = {"name"}
    tr = c.format_for_llm()
    assert len(tr.args) == 1
    assert tr.args.get("name", None) is not None

    c._template = "{{test}}"
    tr = c.format_for_llm()
    assert tr.template is not None
    assert tr.template == "{{test}}"

    c._template_order = "test"
    tr = c.format_for_llm()
    assert tr.template_order == ["test"]


def test_mify_class_parameterized_formatter(mified_customer: Customer):
    c = mified_customer
    assert isinstance(c, MifiedProtocol)

    c._fields_include = None
    c._fields_exclude = None
    c._template_order = None  # Default to MObject's ordering.

    c._template = "{{content}}"
    tf = TemplateFormatter("default")
    transform = c.get_transform_object("transform")
    transform_string = tf.print(transform)
    assert f"{c.name}, {c.age}" in transform_string

    c._template = None
    transform_string = tf.print(transform)
    assert f"{c.name}, {c.age}" in transform_string


if __name__ == "__main__":
    pytest.main([__file__])
