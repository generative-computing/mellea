"""``MObject``, ``Query``, ``Transform``, and ``MObjectProtocol`` for query/transform workflows.

Defines the ``MObjectProtocol`` protocol for objects that can be queried and
transformed by an LLM, and the concrete ``MObject`` base class that implements it.
Also provides the ``Query`` and ``Transform`` ``Component`` subtypes, which wrap an
object with a natural-language question or mutation instruction respectively. These
primitives underpin ``@mify`` and can be composed directly to build document Q&A
or structured extraction pipelines.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Protocol, runtime_checkable

from ...backends.tools import MelleaTool
from ...core import CBlock, Component, ModelOutputThunk, TemplateRepresentation
from ...security import SecLevel


class Query(Component[str]):
    """A ``Component`` that pairs an ``MObject`` with a natural-language question.

    Wraps the object and its query string into a ``TemplateRepresentation`` so the
    formatter can render both together in a prompt, optionally forwarding the
    object's tools and fields to the template.

    Args:
        obj (Component): The object to be queried.
        query (str): The natural-language question to ask about the object.
    """

    def __init__(self, obj: Component, query: str) -> None:
        """Initialize Query with the object to query and a natural-language question string."""
        self._obj = obj
        self._query = query
        self._sec_level: SecLevel | None = None

    @property
    def sec_level(self) -> SecLevel | None:
        """Get the security level for this Component.

        Returns:
            SecLevel if present, None otherwise
        """
        return self._sec_level

    def parts(self) -> list[Component | CBlock]:
        """Return the constituent parts of this query component.

        Returns:
            list[Component | CBlock]: A list containing the wrapped object.
        """
        return [self._obj]

    def format_for_llm(self) -> TemplateRepresentation | str:
        """Format this query for the language model.

        Returns:
            TemplateRepresentation | str: A ``TemplateRepresentation`` containing
            the query string, the wrapped object, and any tools or fields from the
            object's own representation.
        """
        object_repr = self._obj.format_for_llm()
        return TemplateRepresentation(
            args={
                "query": self._query,
                "content": self._obj,  # Put the object here so the object template can be applied first.
            },
            obj=self,
            tools=(
                object_repr.tools
                if isinstance(object_repr, TemplateRepresentation)
                else None
            ),
            fields=(
                object_repr.fields
                if isinstance(object_repr, TemplateRepresentation)
                else None
            ),
            template_order=["Query"],
        )

    def _parse(self, computed: ModelOutputThunk) -> str:
        """Parse the model output. Returns string value for now."""
        return computed.value if computed.value is not None else ""


class Transform(Component[str]):
    """A ``Component`` that pairs an ``MObject`` with a natural-language mutation instruction.

    Wraps the object and its transformation description into a
    ``TemplateRepresentation`` so the formatter can render both together in a prompt,
    optionally forwarding the object's tools and fields to the template.

    Args:
        obj (Component): The object to be transformed.
        transformation (str): The natural-language description of the transformation.
    """

    def __init__(self, obj: Component, transformation: str) -> None:
        """Initialize Transform with the object to transform and a natural-language description."""
        self._obj = obj
        self._transformation = transformation
        self._sec_level: SecLevel | None = None

    @property
    def sec_level(self) -> SecLevel | None:
        """Get the security level for this Component.

        Returns:
            SecLevel if present, None otherwise
        """
        return self._sec_level

    def parts(self) -> list[Component | CBlock]:
        """Return the constituent parts of this transform component.

        Returns:
            list[Component | CBlock]: A list containing the wrapped object.
        """
        return [self._obj]

    def format_for_llm(self) -> TemplateRepresentation | str:
        """Format this transform for the language model.

        Returns:
            TemplateRepresentation | str: A ``TemplateRepresentation`` containing
            the transformation description, the wrapped object, and any tools or
            fields from the object's own representation.
        """
        object_repr = self._obj.format_for_llm()
        return TemplateRepresentation(
            args={
                "transformation": self._transformation,
                "content": self._obj,  # Put the object here so the object template can be applied first.
            },
            obj=self,
            tools=(
                object_repr.tools
                if isinstance(object_repr, TemplateRepresentation)
                else None
            ),
            fields=(
                object_repr.fields
                if isinstance(object_repr, TemplateRepresentation)
                else None
            ),
            template_order=["Transform"],
        )

    def _parse(self, computed: ModelOutputThunk) -> str:
        """Parse the model output. Returns string value for now."""
        return computed.value if computed.value is not None else ""


@runtime_checkable
class MObjectProtocol(Protocol):
    """Protocol to describe the necessary functionality of a MObject. Implementers should prefer inheriting from MObject than MObjectProtocol."""

    def parts(self) -> list[Component | CBlock]:
        """Return a list of parts for this MObject.

        Returns:
            list[Component | CBlock]: The constituent sub-components.
        """
        ...

    def get_query_object(self, query: str) -> Query:
        """Return the instantiated query object.

        Args:
            query (str): The query string.

        Returns:
            Query: A ``Query`` component wrapping this object and the given
            query string.
        """
        ...

    def get_transform_object(self, transformation: str) -> Transform:
        """Return the instantiated transform object.

        Args:
            transformation (str): The transformation description string.

        Returns:
            Transform: A ``Transform`` component wrapping this object and the
            given transformation description.
        """
        ...

    def content_as_string(self) -> str:
        """Return the content of this MObject as a plain string.

        The default value is just ``str(self)``.
        Subclasses should override this method.

        Returns:
            str: String representation of this object's content.
        """
        ...

    def _get_all_members(self) -> dict[str, Callable]:
        """Return all methods from this MObject that are not inherited from the superclass.

        Undocumented methods and methods with ``[no-index]`` in their docstring
        are ignored.
        """
        ...

    def format_for_llm(self) -> TemplateRepresentation | str:
        """Return the template representation used by the formatter.

        The default ``TemplateRepresentation`` uses automatic parsing for tools
        and fields. Content is retrieved from ``content_as_string()``.

        Returns:
            TemplateRepresentation | str: The formatted representation for the
            language model.
        """
        ...

    def _parse(self, computed: ModelOutputThunk) -> str:
        """Parse the model output."""
        ...


class MObject(Component[str]):
    """An extension of ``Component`` for adding query and transform operations.

    Args:
        query_type (type): The ``Query`` subclass to use when constructing query
            components. Defaults to ``Query``.
        transform_type (type): The ``Transform`` subclass to use when constructing
            transform components. Defaults to ``Transform``.
    """

    def __init__(
        self, *, query_type: type = Query, transform_type: type = Transform
    ) -> None:
        """Initialize MObject with a query type and transform type for building query/transform components."""
        self._query_type = query_type
        self._transform_type = transform_type
        self._sec_level: SecLevel | None = None

    @property
    def sec_level(self) -> SecLevel | None:
        """Get the security level for this Component.

        Returns:
            SecLevel if present, None otherwise
        """
        return self._sec_level

    def parts(self) -> list[Component | CBlock]:
        """MObject has no parts because of how format_for_llm is defined.

        Returns:
            list[Component | CBlock]: Always an empty list.
        """
        return []

    def get_query_object(self, query: str) -> Query:
        """Return the instantiated query object.

        Args:
            query (str): The query string.

        Returns:
            Query: A ``Query`` component wrapping this object and the given
            query string.
        """
        return self._query_type(self, query)

    def get_transform_object(self, transformation: str) -> Transform:
        """Return the instantiated transform object.

        Args:
            transformation (str): The transformation description string.

        Returns:
            Transform: A ``Transform`` component wrapping this object and the
            given transformation description.
        """
        return self._transform_type(self, transformation)

    def content_as_string(self) -> str:
        """Return the content of this MObject as a plain string.

        The default value is just ``str(self)``.
        Subclasses should override this method.

        Returns:
            str: String representation of this object's content.
        """
        return str(self)

    def _get_all_members(self) -> dict[str, Callable]:
        """Return all methods from this MObject except methods of the superclass.

        Undocumented methods and methods with ``[no-index]`` in their docstring
        are ignored.
        """
        all_members: dict[str, Callable] = dict(
            inspect.getmembers(self, predicate=inspect.ismethod)
        )
        unique_members = {}

        # Get members of superclass
        superclass_members = dict(inspect.getmembers(MObject)).keys()

        # Filter out members that are also in superclasses
        for name, member in all_members.items():
            if name not in superclass_members and (
                hasattr(member, "__doc__")
                and member.__doc__ is not None
                and "[no-index]" not in member.__doc__.strip()
            ):
                unique_members[name] = member
        return unique_members

    def format_for_llm(self) -> TemplateRepresentation | str:
        """Return the template representation used by the formatter.

        The default ``TemplateRepresentation`` uses automatic parsing for tools
        and fields. Content is retrieved from ``content_as_string()``.

        Returns:
            TemplateRepresentation | str: The formatted representation for the
            language model.
        """
        tools = {
            k: MelleaTool.from_callable(c) for k, c in self._get_all_members().items()
        }
        return TemplateRepresentation(
            args={"content": self.content_as_string()},
            obj=self,
            tools=tools,  # type: ignore[arg-type]
            fields=[],
            template_order=["*", "MObject"],
        )

    def _parse(self, computed: ModelOutputThunk) -> str:
        """Parse the model output. Returns string value for now."""
        return computed.value if computed.value is not None else ""
