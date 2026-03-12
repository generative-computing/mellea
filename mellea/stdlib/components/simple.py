"""``SimpleComponent``: a lightweight named-span component.

``SimpleComponent`` accepts arbitrary keyword arguments (strings, ``CBlock``s, or
``Component``s) and renders them as a JSON object keyed by the argument names. It is
the go-to component type for ad-hoc prompts that do not require a dedicated
``Component`` subclass or a Jinja2 template.
"""

from ...core import CBlock, Component, ModelOutputThunk


class SimpleComponent(Component[str]):
    """A Component that is make up of named spans."""

    def __init__(self, **kwargs):
        """Initialized a simple component of the constructor's kwargs."""
        for key in kwargs.keys():
            if type(kwargs[key]) is str:
                kwargs[key] = CBlock(value=kwargs[key])
        self._kwargs_type_check(kwargs)
        self._kwargs = kwargs

    def parts(self):
        """Returns the values of the kwargs."""
        return list(self._kwargs.values())

    def _kwargs_type_check(self, kwargs):
        for key in kwargs.keys():
            value = kwargs[key]
            assert issubclass(type(value), Component) or issubclass(
                type(value), CBlock
            ), f"Expected span but found {type(value)} of value: {value}"
            assert type(key) is str
        return True

    @staticmethod
    def make_simple_string(kwargs):
        """Render keyword arguments as ``<|key|>value</|key|>`` tagged strings.

        Args:
            kwargs (dict): Mapping of span names to their ``CBlock`` or
                ``Component`` values.

        Returns:
            str: Newline-joined tagged representation of all keyword arguments.
        """
        return "\n".join(
            [f"<|{key}|>{value}</|{key}|>" for (key, value) in kwargs.items()]
        )

    @staticmethod
    def make_json_string(kwargs):
        """Serialize keyword arguments to a JSON string.

        Each value is converted to its string representation: ``CBlock`` and
        ``ModelOutputThunk`` values use their ``.value`` attribute, while
        ``Component`` values use ``format_for_llm()``.

        Args:
            kwargs (dict): Mapping of span names to ``CBlock``, ``Component``,
                or ``ModelOutputThunk`` values.

        Returns:
            str: JSON-encoded representation of the keyword arguments.
        """
        str_args = dict()
        for key in kwargs.keys():
            match kwargs[key]:
                case ModelOutputThunk() | CBlock():
                    str_args[key] = kwargs[key].value
                case Component():
                    str_args[key] = kwargs[key].format_for_llm()
        import json

        return json.dumps(str_args)

    def format_for_llm(self):
        """Format this component as a JSON string representation for the language model.

        Delegates to ``make_json_string`` using the stored keyword arguments.

        Returns:
            str: JSON-encoded string of all named spans in this component.
        """
        return SimpleComponent.make_json_string(self._kwargs)

    def _parse(self, computed: ModelOutputThunk) -> str:
        """Parse the model output. Returns string value for now."""
        return computed.value if computed.value is not None else ""
