"""MObject."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Protocol, runtime_checkable

from ...core import CBlock, Component, ModelOutputThunk, TemplateRepresentation
from ...security import SecLevel


class Query(Component[str]):
    """A Query component."""

    def __init__(self, obj: Component, query: str) -> None:
        """Initializes a new instance of Query with the provided object and query.

        Args:
            obj : The object to be queried.
            query:  The query string used for querying the object.
        """
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
        """Get the parts of the query."""
        return [self._obj]

    def format_for_llm(self) -> TemplateRepresentation | str:
        """Format the query for llm."""
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
    """A Transform component."""

    def __init__(self, obj: Component, transformation: str) -> None:
        """Initializes a new instance of Transform with the provided object and transformation description.

        Args:
            obj : The object to be queried.
            transformation:  The string used for transforming the object.
        """
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
        """Get the parts of the transform."""
        return [self._obj]

    def format_for_llm(self) -> TemplateRepresentation | str:
        """Format the transform for llm."""
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
        """Returns a list of parts for MObject."""
        ...

    def get_query_object(self, query: str) -> Query:
        """Returns the instantiated query object.

        Args:
            query : The query string.
        """
        ...

    def get_transform_object(self, transformation: str) -> Transform:
        """Returns the instantiated transform object.

        Args:
            transformation: the transform string
        """
        ...

    def content_as_string(self) -> str:
        """Returns the content of MObject as a string.

        The default value is just `str(self)`.
        Subclasses should override this method.
        """
        ...

    def _get_all_members(self) -> dict[str, Callable]:
        """Returns a list of all methods from the MObject except methods of the super class.

        Undocumented and methods with [no-index] in doc string are ignored.
        """
        ...

    def format_for_llm(self) -> TemplateRepresentation | str:
        """The template representation used by the formatter.

        The default `TemplateRepresentation` uses an automatic
        parsing for tools and fields. The content is retrieved
        from `content_as_string()`.
        """
        ...

    def _parse(self, computed: ModelOutputThunk) -> str:
        """Parse the model output."""
        ...


class MObject(Component[str]):
    """An extension of `Component` for adding query and transform operations."""

    def __init__(
        self, *, query_type: type = Query, transform_type: type = Transform
    ) -> None:
        """Initializes a new instance of MObject with a specified query type and transformation type.

        Args:
            query_type : The type of query to be used, defaults to Query if not provided.
            transform_type : The type of transform to be used, defaults to Transform if not provided.
        """
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
        """MObject has no parts because of how format_for_llm is defined."""
        return []

    def get_query_object(self, query: str) -> Query:
        """Returns the instantiated query object.

        Args:
            query : The query string.
        """
        return self._query_type(self, query)

    def get_transform_object(self, transformation: str) -> Transform:
        """Returns the instantiated transform object.

        Args:
            transformation: the transform string
        """
        return self._transform_type(self, transformation)

    def content_as_string(self) -> str:
        """Returns the content of MObject as a string.

        The default value is just `str(self)`.
        Subclasses should override this method.
        """
        return str(self)

    def _get_all_members(self) -> dict[str, Callable]:
        """Returns a list of all methods from the MObject except methods of the super class.

        Undocumented and methods with [no-index] in doc string are ignored.
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
        """The template representation used by the formatter.

        The default `TemplateRepresentation` uses an automatic
        parsing for tools and fields. The content is retrieved
        from `content_as_string()`.
        """
        return TemplateRepresentation(
            args={"content": self.content_as_string()},
            obj=self,
            tools=self._get_all_members(),
            fields=[],
            template_order=["*", "MObject"],
        )

    def _parse(self, computed: ModelOutputThunk) -> str:
        """Parse the model output. Returns string value for now."""
        return computed.value if computed.value is not None else ""
