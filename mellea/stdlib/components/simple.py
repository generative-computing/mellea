"""SimpleComponent."""

from ...core import CBlock, Component, ModelOutputThunk


class SimpleComponent(Component[str]):
    """A Component that is make up of named spans."""

    def __init__(self, **kwargs):
        """Initialized a simple component of the constructor's kwargs."""
        for key in kwargs.keys():
            if type(kwargs[key]) is str:
                kwargs[key] = CBlock(value=kwargs[key])
        # Filter out all None-valued arguments by default as a convienance wrapper.
        kwargs = dict([(key, value) for (key,value) in kwargs.items() if value is not None])
        self._kwargs_type_check(kwargs)
        self._kwargs = kwargs

    def _parts(self, value):
        """Returns the values of the kwargs."""
        match value:
            case ModelOutputThunk():
                return [value]
            case CBlock():
                return [value]
            case Component():
                return [value]
            case list():
                all = []
                for lpart in value:
                    all.extend(self._part(lpart))
                return all
            case dict():
                raise ValueError("TODO")
    
    def parts(self):
        all_parts = []
        for value in self._kwargs.values():
            all_parts.extend(self._parts(value))
        assert all([type(part) in [CBlock, Component, ModelOutputThunk] for part in all_parts])
        return all_parts

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
        """Uses <|key|>value</|key|> to represent a simple component."""
        return "\n".join(
            [f"<|{key}|>{value}</|{key}|>" for (key, value) in kwargs.items()]
        )

    @staticmethod
    def make_json_string(kwargs):
        """Uses json."""
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
        """Uses a string rep."""
        return SimpleComponent.make_json_string(self._kwargs)

    def _parse(self, computed: ModelOutputThunk) -> str:
        """Parse the model output. Returns string value for now."""
        return computed.value if computed.value is not None else ""
