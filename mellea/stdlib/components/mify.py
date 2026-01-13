"""Mify classes and objects."""

import inspect
import types
from collections.abc import Callable
from typing import Any, Protocol, TypeVar, overload, runtime_checkable

from ...core import (
    CBlock,
    Component,
    ComponentParseError,
    ModelOutputThunk,
    TemplateRepresentation,
)
from .mobject import MObjectProtocol, Query, Transform


@runtime_checkable
class MifiedProtocol(MObjectProtocol, Protocol):
    """Adds additional functionality to the MObjectProtocol and modifies MObject functions so that mified objects can be more easily interacted with and modified.

    See the mify decorator for more information.
    """

    _query_type: type = Query
    _transform_type: type = Transform
    _fields_include: set[str] | None = None
    _fields_exclude: set[str] | None = None
    _funcs_include: set[str] | None = None
    _funcs_exclude: set[str] | None = None
    _template: str | None = None
    _template_order: str | list[str] | None = None
    _parsing_func: Callable[[str], object] | None = None
    _stringify_func: Callable[[object], str] | None = None

    def parts(self) -> list[Component | CBlock]:
        """TODO: we need to rewrite this component to use format_for_llm and initializer correctly.

        For now an empty list is the correct behavior.

        [no-index]
        """
        return []

    def get_query_object(self, query: str) -> Query:
        """Returns the instantiated query object.

        [no-index]

        Args:
            query : The query string.
        """
        return self._query_type(self, query)

    def get_transform_object(self, transformation: str) -> Transform:
        """Returns the instantiated transform object.

        [no-index]

        Args:
            transformation: the transform string
        """
        return self._transform_type(self, transformation)

    def content_as_string(self) -> str:
        """Returns the content of the Mified object as a string.

        [no-index]

        Will use the passed in stringify function if provided.
        """
        if self._stringify_func:
            return self._stringify_func()  # type: ignore
        return str(self)

    def _get_all_members(self) -> dict[str, Callable]:
        """Returns a dict of methods from this object that are not shared with the `object` base class.

        Undocumented methods and methods with [no-index] in doc string are ignored.

        It will also take into consideration its funcs_include and funcs_exclude fields.
        Functions that were specifically included will ignore the undocumented and [no-index] requirements.
        See mify decorator for more info.
        """
        # It doesn't matter if the funcs to exclude is an empty set, so
        # we handle it being none here to simplify the code below.
        funcs_exclude = set()
        if self._funcs_exclude:
            funcs_exclude = self._funcs_exclude

        # This includes members defined by any superclasses, but not the object superclass.
        all_members = _get_non_duplicate_members(self, object)

        # It does matter if include is an empty set. Handle it's cases here.
        if self._funcs_include is not None:
            # Include is empty. Early return nothing.
            if len(self._funcs_include) == 0:
                return {}

            # All the fields that should be included were also specifically excluded.
            # Early return nothing.
            if funcs_exclude.issuperset(self._funcs_include):
                return {}

            narrowed = {}
            for name, func in all_members.items():
                if name in self._funcs_include:
                    narrowed[name] = func
            return narrowed

        # Deal with the exclude list only if there's no include list.
        # If the exclude list is empty, this will only filter out items
        # that are undocumented or have [no-index].
        narrowed = {}
        for name, func in all_members.items():
            if name not in funcs_exclude and (
                hasattr(func, "__doc__")
                and func.__doc__ is not None
                and "[no-index]" not in func.__doc__.strip()
            ):
                narrowed[name] = func
        return narrowed

    def _get_all_fields(self) -> dict[str, Any]:
        """Returns a dict of fields that are not shared with the `object` superclass.

        This will return dunder fields as well. As a result, it's advised to always set
        fields_include if using this.

        [no-index]

        It will also take into consideration its fields_include and fields_exclude fields.
        See mify decorator for more info.
        """
        # It doesn't matter if the fields to exclude is an empty set, so
        # we handle it being none here to simplify the code below.
        fields_exclude = set()
        if self._fields_exclude:
            fields_exclude = self._fields_exclude

        # This includes fields defined by any superclasses, as long as it's not Protocol.
        all_fields = _get_non_duplicate_fields(self, Protocol)

        # It does matter if include is an empty set. Handle it's cases here.
        if self._fields_include is not None:
            # Include is empty. Early return nothing.
            if len(self._fields_include) == 0:
                return {}

            # All the fields that should be included were also specifically excluded.
            # Early return nothing.
            if fields_exclude.issuperset(self._fields_include):
                return {}

            narrowed = {}
            for name, field in all_fields.items():
                if name in self._fields_include:
                    narrowed[name] = field
            return narrowed

        # Deal with the exclude list only if there's no include list.
        # If the exclude list is empty, this will pass on all fields.
        narrowed = {}
        for name, field in all_fields.items():
            if name not in fields_exclude:
                narrowed[name] = field
        return narrowed

    def format_for_llm(self) -> TemplateRepresentation:
        """The representation of an object given to the backend.

        [no-index]

        Sets the TemplateRepresentation fields based on the object and the values
        specified during mify.

        See mify decorator for more details.
        """
        args = {"content": self.content_as_string()}

        # Filter out the fields we don't care about.
        if self._fields_include or self._fields_exclude:
            args = self._get_all_fields()

        template_order = ["*", "MObject"]
        if self._template_order:
            if isinstance(self._template_order, str):
                template_order = [self._template_order]
            else:
                template_order = self._template_order

        return TemplateRepresentation(
            args=args,  # type: ignore
            obj=self,
            tools=self._get_all_members(),
            fields=[],
            template=self._template,
            template_order=template_order,
        )

    def _parse(self, computed: ModelOutputThunk) -> str:
        """Parse the model output. Returns string value for now.

        [no-index]
        """
        return computed.value if computed.value is not None else ""

    def parse(self, computed: ModelOutputThunk) -> str:
        """Parse the model output. Returns string value for now.

        [no-index]
        """
        try:
            return self._parse(computed)
        except Exception as e:
            raise ComponentParseError(f"component parsing failed: {e}")


T = TypeVar("T")


@overload
def mify(
    *,
    query_type: type = Query,
    transform_type: type = Transform,
    fields_include: set[str] | None = None,
    fields_exclude: set[str] | None = None,
    funcs_include: set[str] | None = None,
    funcs_exclude: set[str] | None = None,
    template: str | None = None,
    template_order: str | list[str] | None = None,
    parsing_func: Callable[[str], T] | None = None,
    stringify_func: Callable[[T], str] | None = None,
) -> Callable[..., T]: ...  # Overloads for @mify()


@overload
def mify(
    obj: T,
    *,
    query_type: type = Query,
    transform_type: type = Transform,
    fields_include: set[str] | None = None,
    fields_exclude: set[str] | None = None,
    funcs_include: set[str] | None = None,
    funcs_exclude: set[str] | None = None,
    template: str | None = None,
    template_order: str | list[str] | None = None,
    parsing_func: Callable[[str], T] | None = None,
    stringify_func: Callable[[T], str] | None = None,
) -> T: ...  # Overloads for @mify and mify(obj|cls)


def mify(*args, **kwargs):  # noqa: D417
    """M-ify an object or class.

    Allows the object (or instances of the class) to be used in m sessions and with m functions.

    For the args below, only specify an _include or an _exclude of for fields and funcs. If both are specified,
    include takes precedence. If you specify the same item to be included and excluded, nothing will be included.

    If fields_include or fields_exclude are set:
    - the stringify_func will not be used to represent this object to the model
    - you must specify a template field or a template in the template_order field that handles a dict with those fields as keys
    - it's advised to use fields_include due to the many dunder fields and inherited fields an object/class might have

    Mify sets attributes on the object/class. If the object isn't already an mified/mobject, it will overwrite
    the attributes and methods of the object/class necessary for it to be mified.

    Args:
        obj: either a class or an instance of the class
        fields_include: fields of the object to include in its representation to models
        fields_exclude: fields of the object to exclude from its representation to models
        funcs_include: functions of the object to include in its representation to models
        funcs_exclude: functions of the object to exclude from its representation to models
        query_type: a specific query component type to use when querying a model
        transform_type: a specific transform component type to use when transforming with a model
        template: a string representation of a jinja template; takes precedence over template_order
        template_order: a template ordering to use when searching for applicable templates
        parsing_func: not yet implemented
        stringify_func: used to create a string representation of the object

    Returns:
        An object if an object was passed in or a decorator (callable) to mify classes.

        If an object is returned, that object will be the same object that was passed in.
        For example,
        ```
        obj = mify(obj)
        obj.format_for_llm()
        ```
        and
        ```
        mify(obj)
        obj.format_for_llm()
        ```
        are equivalent.

        Most IDEs will not correctly show the type hints for the newly added functions
        for either an mify object or instances of an mified class. For IDE support, write
        ```
        assert isinstance(obj, MifiedProtocol)
        ```
    """
    # Grab and remove obj if it exists in kwargs. Otherwise, it's the only arg.
    obj = kwargs.pop("obj", None)
    if len(args) == 1:
        obj = args[0]

    return _mify(obj=obj, **kwargs)


def _mify(
    *,
    obj: T  # type: ignore
    | None = None,  # Necessary if the decorator is called without args or directly on the class.
    query_type: type = Query,
    transform_type: type = Transform,
    fields_include: set[str] | None = None,
    fields_exclude: set[str] | None = None,
    funcs_include: set[str] | None = None,
    funcs_exclude: set[str] | None = None,
    template: str | None = None,
    template_order: str | list[str] | None = None,
    parsing_func: Callable[[str], object] | None = None,
    stringify_func: Callable[[object], str] | None = None,
):
    """Returns either a decorator or the mified object."""

    def mification(obj: T) -> T:
        """The function that actually performs the m-ification."""
        # Don't need to call this on a object/class that is already mified or already a mobject.
        if isinstance(obj, MifiedProtocol) or isinstance(obj, MObjectProtocol):
            return obj  # type: ignore

        # Add necessary functions from the MifiedProtocol to this object/class.
        is_class = inspect.isclass(obj)
        current_members = dict(inspect.getmembers(obj))
        for name, func in _get_non_duplicate_members(MifiedProtocol, Protocol).items():
            if name not in current_members.keys():
                if is_class:
                    setattr(obj, name, func)
                else:
                    # For objects, have to specifically bind methods.
                    setattr(obj, name, types.MethodType(func, obj))

        # Set the defaults for the object/class.
        setattr(obj, "_query_type", query_type)
        setattr(obj, "_transform_type", transform_type)
        setattr(obj, "_fields_include", fields_include)
        setattr(obj, "_fields_exclude", fields_exclude)
        setattr(obj, "_funcs_include", funcs_include)
        setattr(obj, "_funcs_exclude", funcs_exclude)
        setattr(obj, "_template", template)
        setattr(obj, "_template_order", template_order)
        setattr(obj, "_parsing_func", parsing_func)
        setattr(obj, "_stringify_func", stringify_func)

        # Necessary if we want to support changing these defaults during instantiation.
        # if inspect.isclass(obj):
        #     # If this is a class, wrap the init function to add new fields at a class level on new instances.
        #     original_init = obj.__init__

        #     @functools.wraps(original_init)
        #     def mify_init(self, *args, **kwargs):
        #         original_init(self, *args, **kwargs)

        #         # Set the necessary MifiedProtocol fields.
        #         self._query_type = query_type
        #         self._transform_type = transform_type
        #         self._fields_include = fields_include
        #         self._fields_exclude = fields_exclude
        #         self._funcs_include = funcs_include
        #         self._funcs_exclude = funcs_exclude
        #         self._template = template
        #         self._template_order = template_order
        #         self._parsing_func = parsing_func
        #         self._stringify_func = stringify_func

        #     obj.__init__ = mify_init
        return obj

    if obj is not None:
        # Call the decorator now. Necessary if @mify is called without args
        # or directly on the class.
        return mification(obj)
    return mification


def _get_non_duplicate_members(
    obj: object, check_duplicates: object
) -> dict[str, Callable]:
    """Returns all methods/functions unique to the object."""
    members = dict(
        inspect.getmembers(
            obj,
            # Checks for ismethod or isfunction because of the methods added from the MifiedProtocol.
            predicate=lambda x: (inspect.ismethod(x) or inspect.isfunction(x))
            and x.__name__ not in dict(inspect.getmembers(check_duplicates)).keys(),
        )
    )
    return members


def _get_non_duplicate_fields(
    object: object, check_duplicates: object
) -> dict[str, Callable]:
    """Returns all fields unique to the object."""
    # Get the fields.
    members = dict(
        inspect.getmembers(
            object,
            predicate=lambda x: not inspect.isfunction(x) and not inspect.ismethod(x),
        )
    )

    narrowed = {}
    for k, v in members.items():
        if k not in dict(
            inspect.getmembers(
                check_duplicates,
                predicate=lambda x: not inspect.isfunction(x)
                and not inspect.ismethod(x),
            )
        ):
            narrowed[k] = v

    return narrowed
