"""LLM tool definitions, parsing, and validation for mellea backends.

Provides the ``MelleaTool`` class (and the ``@tool`` decorator shorthand) for
wrapping Python callables as OpenAI-compatible tool schemas, with factory methods
for LangChain and smolagents interoperability. Also includes helpers for converting
tool lists to JSON, extracting tool call requests from raw LLM output strings, and
validating/coercing tool arguments against the tool's JSON schema using Pydantic.
"""

import copy
import inspect
import json
import re
from collections import defaultdict
from collections.abc import Callable, Generator, Iterable, Mapping, Sequence
from typing import Any, Literal, overload

from pydantic import BaseModel, ConfigDict, Field

from mellea.core.utils import MelleaLogger

from ..core import CBlock, Component, TemplateRepresentation
from ..core.base import AbstractMelleaTool
from .model_options import ModelOption


class MelleaTool(AbstractMelleaTool):
    """Tool class to represent a callable tool with an OpenAI-compatible JSON schema.

    Wraps a Python callable alongside its JSON schema representation so it can be
    registered with backends that support tool calling (OpenAI, Ollama, HuggingFace, etc.).

    Args:
        name (str): The tool name used for identification and lookup.
        tool_call (Callable): The underlying Python callable to invoke when the tool is run.
        as_json_tool (dict[str, Any]): The OpenAI-compatible JSON schema dict describing
            the tool's parameters.

    """

    # Tool is what we pass as a model option / as input
    # Our ModelToolCall is the class that has a reference to the tool and actually calls with arguments

    name: str
    _as_json_tool: dict[str, Any]
    _call_tool: Callable[..., Any]

    def __init__(
        self, name: str, tool_call: Callable, as_json_tool: dict[str, Any]
    ) -> None:
        """Initialize the tool with a name, tool call and as_json_tool dict."""
        self.name = name
        self._as_json_tool = as_json_tool
        self._call_tool = tool_call

    def run(self, *args, **kwargs) -> Any:
        """Run the tool with the given arguments.

        Args:
            args: Positional arguments forwarded to the underlying callable.
            kwargs: Keyword arguments forwarded to the underlying callable.

        Returns:
            Any: The return value of the underlying callable.
        """
        return self._call_tool(*args, **kwargs)

    @property
    def as_json_tool(self) -> dict[str, Any]:
        """Return the tool converted to a OpenAI compatible JSON object."""
        return self._as_json_tool.copy()

    @classmethod
    def from_langchain(cls, tool: Any) -> "MelleaTool":
        """Create a MelleaTool from a LangChain tool object.

        Args:
            tool (Any): A ``langchain_core.tools.BaseTool`` instance to wrap.

        Returns:
            MelleaTool: A Mellea tool wrapping the LangChain tool.

        Raises:
            ImportError: If ``langchain-core`` is not installed.
            ValueError: If ``tool`` is not a ``BaseTool`` instance.
        """
        try:
            from langchain_core.tools import BaseTool  # type: ignore[import-not-found]
            from langchain_core.utils.function_calling import (  # type: ignore[import-not-found]
                convert_to_openai_tool,
            )

            if isinstance(tool, BaseTool):
                tool_name = tool.name
                as_json = convert_to_openai_tool(tool)

                def parameter_remapper(*args, **kwargs):
                    """Langchain tools expect their first argument to be 'tool_input'."""
                    if args:
                        # This shouldn't happen. Our ModelToolCall.call_func actually passes in everything as kwargs.
                        MelleaLogger.get_logger().warning(
                            f"ignoring unexpected args while calling langchain tool ({tool_name}): ({args})"
                        )

                    return tool.run(tool_input={**kwargs})

                tool_call = parameter_remapper
                return MelleaTool(tool_name, tool_call, as_json)
            else:
                raise ValueError(
                    f"tool parameter must be a langchain tool type; got: {type(tool)}"
                )

        except ImportError as e:
            raise ImportError(
                f"It appears you are attempting to utilize a langchain tool '{type(tool)}'. "
                "Please install mellea with tools support: pip install 'mellea[tools]'"
            ) from e

    @classmethod
    def from_smolagents(cls, tool: Any) -> "MelleaTool":
        """Create a Tool from a HuggingFace smolagents tool object.

        Args:
            tool: A smolagents.Tool instance

        Returns:
            MelleaTool: A Mellea tool wrapping the smolagents tool

        Raises:
            ImportError: If smolagents is not installed
            ValueError: If tool is not a smolagents Tool instance

        Example:
            >>> from smolagents import PythonInterpreterTool
            >>> tool = PythonInterpreterTool()
            >>> mellea_tool = MelleaTool.from_smolagents(tool)
        """
        try:
            from smolagents import (  # type: ignore[import-not-found]
                Tool as SmolagentsTool,
            )
            from smolagents.models import (  # type: ignore[import-not-found]
                get_tool_json_schema,
            )

            if not isinstance(tool, SmolagentsTool):
                raise ValueError(
                    f"tool parameter must be a smolagents Tool type; got: {type(tool)}"
                )

            tool_name = tool.name

            # Use smolagents' built-in conversion to OpenAI format
            as_json = get_tool_json_schema(tool)

            # Wrap the tool's forward method
            def tool_call(*args, **kwargs):
                """Wrapper for smolagents tool forward method."""
                if args:
                    # This shouldn't happen. Our ModelToolCall.call_func passes everything as kwargs.
                    MelleaLogger.get_logger().warning(
                        f"ignoring unexpected args while calling smolagents tool ({tool_name}): ({args})"
                    )
                return tool.forward(**kwargs)

            return MelleaTool(tool_name, tool_call, as_json)

        except ImportError as e:
            raise ImportError(
                f"It appears you are attempting to utilize a smolagents tool '{type(tool)}'. "
                "Please install mellea with tools support: pip install 'mellea[tools]'"
            ) from e

    @classmethod
    def from_callable(cls, func: Callable, name: str | None = None) -> "MelleaTool":
        """Create a MelleaTool from a plain Python callable.

        Introspects the callable's signature and docstring to build an
        OpenAI-compatible JSON schema automatically.

        Args:
            func (Callable): The Python callable to wrap as a tool.
            name (str | None): Optional name override; defaults to ``func.__name__``.

        Returns:
            MelleaTool: A Mellea tool wrapping the callable.
        """
        # Use the function name if the name is '' or None.
        tool_name = name or func.__name__
        as_json = convert_function_to_ollama_tool(func, tool_name).model_dump(
            exclude_none=True
        )
        tool_call = func
        return MelleaTool(tool_name, tool_call, as_json)


@overload
def tool(func: Callable, *, name: str | None = None) -> MelleaTool: ...


@overload
def tool(*, name: str | None = None) -> Callable[[Callable], MelleaTool]: ...


def tool(
    func: Callable | None = None, name: str | None = None
) -> MelleaTool | Callable[[Callable], MelleaTool]:
    """Decorator to mark a function as a Mellea tool.

    This decorator wraps a function to make it usable as a tool without
    requiring explicit MelleaTool.from_callable() calls. The decorated
    function returns a MelleaTool instance that must be called via .run().

    Args:
        func: The function to decorate (when used without arguments)
        name: Optional custom name for the tool (defaults to function name)

    Returns:
        A MelleaTool instance. Use .run() to invoke the tool.
        The returned object passes isinstance(result, MelleaTool) checks.

    Examples:
        Basic usage:
        >>> @tool
        ... def get_weather(location: str, days: int = 1) -> dict:
        ...     '''Get weather forecast.
        ...
        ...     Args:
        ...         location: City name
        ...         days: Number of days to forecast
        ...     '''
        ...     return {"location": location, "forecast": "sunny"}
        >>>
        >>> # The decorated function IS a MelleaTool
        >>> isinstance(get_weather, MelleaTool)  # True
        >>>
        >>> # Can be used directly in tools list (no extraction needed)
        >>> tools = [get_weather]
        >>>
        >>> # Must use .run() to invoke the tool
        >>> result = get_weather.run(location="Boston")

        With custom name (as decorator):
        >>> @tool(name="weather_api")
        ... def get_weather(location: str) -> dict:
        ...     return {"location": location}
        >>>
        >>> result = get_weather.run(location="New York")

        With custom name (as function):
        >>> def new_tool(): ...
        >>> differently_named_tool = tool(new_tool, name="different_name")
    """

    def decorator(f: Callable) -> MelleaTool:
        # Simply return the base MelleaTool instance
        return MelleaTool.from_callable(f, name=name)

    # Handle both @tool and @tool() syntax
    if func is None:
        # Called with arguments: @tool(name="custom")
        return decorator
    else:
        # Called without arguments: @tool
        return decorator(func)


def add_tools_from_model_options(
    tools_dict: dict[str, AbstractMelleaTool], model_options: dict[str, Any]
):
    """If model_options has tools, add those tools to the tools_dict.

    Accepts MelleaTool instances or @tool decorated functions.

    Args:
        tools_dict: Mutable mapping of tool name to tool instance; modified in-place.
        model_options: Model options dict that may contain a ``ModelOption.TOOLS``
            entry (either a list of ``MelleaTool`` or a ``dict[str, MelleaTool]``).
    """
    model_opts_tools = model_options.get(ModelOption.TOOLS, None)
    if model_opts_tools is None:
        return

    # Mappings are iterable.
    assert isinstance(model_opts_tools, Iterable), (
        "ModelOption.TOOLS must be a list of Tool or dict[str, Tool]"
    )

    if isinstance(model_opts_tools, Mapping):
        # Handle the dict case.
        for tool_name, tool_instance in model_opts_tools.items():
            assert isinstance(tool_name, str), (
                f"If ModelOption.TOOLS is a dict, it must be a dict of [str, Tool]; found {type(tool_name)} as the key instead"
            )
            assert isinstance(tool_instance, AbstractMelleaTool), (
                f"If ModelOption.TOOLS is a dict, it must be a dict of [str, Tool]; found {type(tool_instance)} as the value instead"
            )
            tools_dict[tool_name] = tool_instance
    else:
        # Handle any other iterable / list here.
        for tool_instance in model_opts_tools:
            assert isinstance(tool_instance, AbstractMelleaTool), (
                f"If ModelOption.TOOLS is a list, it must be a list of Tool; found {type(tool_instance)}"
            )
            # MelleaTool (and subclasses like CallableMelleaTool) have a name attribute
            assert isinstance(tool_instance, MelleaTool), (
                f"Tool must be a MelleaTool instance with a name attribute; found {type(tool_instance)}"
            )
            tools_dict[tool_instance.name] = tool_instance


def add_tools_from_context_actions(
    tools_dict: dict[str, AbstractMelleaTool],
    ctx_actions: list[Component | CBlock] | None,
):
    """If any of the actions in ctx_actions have tools in their template_representation, add those to the tools_dict.

    Args:
        tools_dict: Mutable mapping of tool name to tool instance; modified in-place.
        ctx_actions: List of ``Component`` or ``CBlock`` objects whose template
            representations may declare tools, or ``None`` to skip.
    """
    if ctx_actions is None:
        return

    for action in ctx_actions:
        if not isinstance(action, Component):
            continue  # Only components have template representations.

        tr = action.format_for_llm()
        if not isinstance(tr, TemplateRepresentation) or tr.tools is None:
            continue

        for tool_name, func in tr.tools.items():
            tools_dict[tool_name] = func


def convert_tools_to_json(tools: dict[str, AbstractMelleaTool]) -> list[dict]:
    """Convert tools to json dict representation.

    Args:
        tools: Mapping of tool name to ``AbstractMelleaTool`` instance.

    Returns:
        List of OpenAI-compatible JSON tool schema dicts, one per tool.

    Notes:
    - Huggingface transformers library lets you pass in an array of functions but doesn't like methods.
    - WatsonxAI uses `from langchain_ibm.chat_models import convert_to_openai_tool` in their demos, but it gives the same values.
    - OpenAI uses the same format / schema.
    """
    return [t.as_json_tool for t in tools.values()]


def json_extraction(text: str) -> Generator[dict, None, None]:
    """Yield the next valid JSON object found in a given string.

    Args:
        text: Input string potentially containing one or more JSON objects.

    Yields:
        Each valid JSON object found in ``text``, in order of appearance.
    """
    index = 0
    decoder = json.JSONDecoder()

    # Keep trying to find valid json by jumping to the next
    # opening curly bracket. Will ignore non-json text.
    index = text.find("{", index)
    while index != -1:
        try:
            j, index = decoder.raw_decode(text, index)
            yield j
        except GeneratorExit:
            return  # allow for early exits from the generator.
        except Exception:
            index += 1

        index = text.find("{", index)


def find_func(d: object) -> tuple[str | None, Mapping | None]:
    """Find the first function in a json-like dictionary.

    Most llms output tool requests in the form ``...{"name": string, "arguments": {}}...``

    Args:
        d: A JSON-like Python object (typically a ``dict``) to search for a function
            call record.

    Returns:
        A ``(name, args)`` tuple where ``name`` is the tool name string and ``args``
        is the arguments mapping, or ``(None, None)`` if no function call was found.
    """
    if not isinstance(d, dict):
        return None, None

    name = d.get("name", None)
    args = None

    args_names = ["arguments", "args", "parameters"]
    for an in args_names:
        args = d.get(an, None)
        if isinstance(args, Mapping):
            break
        else:
            args = None

    if name is not None and args is not None:
        # args is usually output as `{}` if none are required.
        return name, args

    for v in d.values():
        return find_func(v)
    return None, None


def parse_tools(llm_response: str) -> list[tuple[str, Mapping]]:
    """A simple parser that will scan a string for tools and attempt to extract them; only works for json based outputs.

    Args:
        llm_response: Raw string output from a language model.

    Returns:
        List of ``(tool_name, arguments)`` tuples for each tool call found.
    """
    processed = " ".join(llm_response.split())

    tools = []
    for possible_tool in json_extraction(processed):
        tool_name, tool_arguments = find_func(possible_tool)
        if tool_name is not None and tool_arguments is not None:
            tools.append((tool_name, tool_arguments))
    return tools


def validate_tool_arguments(
    tool: AbstractMelleaTool,
    args: Mapping[str, Any],
    *,
    coerce_types: bool = True,
    strict: bool = False,
) -> dict[str, Any]:
    """Validate and optionally coerce tool arguments against tool's JSON schema.

    This function validates tool call arguments extracted from LLM responses against
    the tool's JSON schema from as_json_tool. It can automatically coerce common type
    mismatches (e.g., string "30" to int 30) and provides detailed error messages.

    Args:
        tool: The MelleaTool instance to validate against
        args: Raw arguments from model (post-JSON parsing)
        coerce_types: If True, attempt type coercion for common cases (default: True)
        strict: If True, raise ValidationError on failures; if False, log warnings
                and return original args (default: False)

    Returns:
        Validated and optionally coerced arguments dict

    Raises:
        ValidationError: If strict=True and validation fails

    Examples:
        >>> def get_weather(location: str, days: int = 1) -> dict:
        ...     return {"location": location, "days": days}
        >>> tool = MelleaTool.from_callable(get_weather)

        >>> # LLM returns days as string
        >>> args = {"location": "Boston", "days": "3"}
        >>> validated = validate_tool_arguments(tool, args)
        >>> validated
        {'location': 'Boston', 'days': 3}

        >>> # Strict mode raises on validation errors
        >>> bad_args = {"location": "Boston", "days": "not_a_number"}
        >>> validate_tool_arguments(tool, bad_args, strict=True)
        Traceback (most recent call last):
        ...
        pydantic.ValidationError: ...
    """
    from pydantic import ValidationError, create_model

    from ..core import MelleaLogger

    # Extract JSON schema from tool
    tool_schema = tool.as_json_tool.get("function", {})
    tool_name = tool_schema.get("name", "unknown_tool")
    parameters = tool_schema.get("parameters", {})
    properties = parameters.get("properties", {})
    required_fields = parameters.get("required", [])

    # Map JSON schema types to Python types
    JSON_TYPE_TO_PYTHON = {
        "string": str,
        "integer": int,
        "number": float,
        "boolean": bool,
        "array": list,
        "object": dict,
    }

    # Helper function to build Pydantic model from nested JSON schema
    def _build_pydantic_type_from_schema(schema: dict[str, Any]) -> Any:
        """Recursively build Pydantic type from JSON schema.

        Note: This assumes all $ref references have been resolved by
        convert_function_to_ollama_tool before validation. If a schema
        still contains $ref entries, this will return Any as a fallback.
        """
        json_type = schema.get("type", "string")

        # Handle nested objects with properties
        if json_type == "object" and "properties" in schema:
            nested_properties = schema.get("properties", {})
            nested_required = schema.get("required", [])
            nested_fields: dict[str, Any] = {}

            for nested_name, nested_schema in nested_properties.items():
                nested_type = _build_pydantic_type_from_schema(nested_schema)
                if nested_name in nested_required:
                    nested_fields[nested_name] = (nested_type, ...)
                else:
                    nested_fields[nested_name] = (nested_type, None)

            # Create a nested Pydantic model with deterministic name
            # Use sorted field names for consistent naming across runs
            model_name = f"Nested_{'_'.join(sorted(nested_fields.keys()))}"
            return create_model(
                model_name, __config__=ConfigDict(extra="allow"), **nested_fields
            )

        # Handle arrays
        if json_type == "array":
            item_schema = schema.get("items", {})
            item_type = _build_pydantic_type_from_schema(item_schema)
            return list[item_type]  # type: ignore

        # Handle comma-separated types (Union types)
        if isinstance(json_type, str) and "," in json_type:
            type_list = [t.strip() for t in json_type.split(",")]
            python_types = [JSON_TYPE_TO_PYTHON.get(t, Any) for t in type_list]
            seen = set()
            unique_types = []
            for t in python_types:
                if t not in seen:
                    seen.add(t)
                    unique_types.append(t)

            if len(unique_types) == 1:
                return unique_types[0]
            else:
                from functools import reduce
                from operator import or_

                return reduce(or_, unique_types)

        # Handle anyOf (union types in JSON schema)
        if "anyOf" in schema:
            types_list = []
            for sub_schema in schema["anyOf"]:
                sub_type = _build_pydantic_type_from_schema(sub_schema)
                types_list.append(sub_type)

            if len(types_list) == 1:
                return types_list[0]
            else:
                from functools import reduce
                from operator import or_

                return reduce(or_, types_list)

        # Simple type mapping
        return JSON_TYPE_TO_PYTHON.get(json_type, Any)

    # Build Pydantic model from JSON schema
    field_definitions: dict[str, Any] = {}

    for param_name, param_schema in properties.items():
        param_type = _build_pydantic_type_from_schema(param_schema)

        # Determine if parameter is required
        if param_name in required_fields:
            # Required parameter
            field_definitions[param_name] = (param_type, ...)
        else:
            # Optional parameter (default to None)
            field_definitions[param_name] = (param_type, None)

    # Configure model for type coercion if requested
    if coerce_types:
        model_config = ConfigDict(
            str_strip_whitespace=True,
            strict=False,  # Allow type coercion
            extra="forbid" if strict else "allow",  # Handle extra fields
            # Enable coercion modes for common LLM output issues
            coerce_numbers_to_str=True,  # Allow int/float -> str
        )
    else:
        model_config = ConfigDict(
            strict=True,  # No coercion
            extra="forbid" if strict else "allow",
        )

    # Create dynamic Pydantic model for validation
    ValidatorModel = create_model(
        f"{tool_name}_Validator", __config__=model_config, **field_definitions
    )

    try:
        # Validate using Pydantic
        validated_model = ValidatorModel(**args)
        validated_args = validated_model.model_dump()

        # In lenient mode with extra="allow", Pydantic includes extra fields
        # but we need to preserve them from the original args
        if not strict:
            # Add back any extra fields that weren't in the model
            for key, value in args.items():
                if key not in field_definitions:
                    validated_args[key] = value

        # Log successful validation with coercion details
        coerced_fields = []
        for key, original_value in args.items():
            validated_value = validated_args.get(key)
            if type(original_value) is not type(validated_value):
                coerced_fields.append(
                    f"{key}: {type(original_value).__name__} → {type(validated_value).__name__}"
                )

        if coerced_fields and coerce_types:
            MelleaLogger.get_logger().debug(
                f"Tool '{tool_name}' arguments coerced: {', '.join(coerced_fields)}"
            )

        return validated_args

    except ValidationError as e:
        # Format error message
        error_details = []
        for error in e.errors():
            field = ".".join(str(loc) for loc in error["loc"])
            msg = error["msg"]
            error_details.append(f"  - {field}: {msg}")

        error_msg = f"Tool argument validation failed for '{tool_name}':\n" + "\n".join(
            error_details
        )

        if strict:
            # Re-raise with enhanced message
            MelleaLogger.get_logger().error(error_msg)
            raise
        else:
            # Log warning and return original args
            MelleaLogger.get_logger().warning(
                error_msg + "\nReturning original arguments without validation."
            )
            return dict(args)

    except Exception as e:
        # Catch any other errors during validation
        error_msg = f"Unexpected error validating tool '{tool_name}' arguments: {e}"

        if strict:
            MelleaLogger.get_logger().error(error_msg)
            raise
        else:
            MelleaLogger.get_logger().warning(
                error_msg + "\nReturning original arguments without validation."
            )
            return dict(args)


# Below functions and classes extracted from Ollama Python SDK (v0.6.1)
# so that all backends don't need it installed.
# https://github.com/ollama/ollama-python/blob/60e7b2f9ce710eeb57ef2986c46ea612ae7516af/ollama/_types.py#L19-L101
class SubscriptableBaseModel(BaseModel):
    """Pydantic ``BaseModel`` subclass that also supports subscript (``[]``) access.

    Imported from the Ollama Python client. Allows model fields to be accessed
    via ``model["field"]`` in addition to ``model.field``, which is required for
    compatibility with Ollama's internal response parsing.
    """

    def __getitem__(self, key: str) -> Any:
        """Getitem.

        >>> msg = Message(role='user')
        >>> msg['role']
        'user'
        >>> msg = Message(role='user')
        >>> msg['nonexistent']
        Traceback (most recent call last):
        KeyError: 'nonexistent'
        """
        if key in self:
            return getattr(self, key)

        raise KeyError(key)

    def __setitem__(self, key: str, value: Any) -> None:
        """Setitem.

        >>> msg = Message(role='user')
        >>> msg['role'] = 'assistant'
        >>> msg['role']
        'assistant'
        >>> tool_call = Message.ToolCall(function=Message.ToolCall.Function(name='foo', arguments={}))
        >>> msg = Message(role='user', content='hello')
        >>> msg['tool_calls'] = [tool_call]
        >>> msg['tool_calls'][0]['function']['name']
        'foo'
        """
        setattr(self, key, value)

    def __contains__(self, key: str) -> bool:
        """Contains.

        >>> msg = Message(role='user')
        >>> 'nonexistent' in msg
        False
        >>> 'role' in msg
        True
        >>> 'content' in msg
        False
        >>> msg.content = 'hello!'
        >>> 'content' in msg
        True
        >>> msg = Message(role='user', content='hello!')
        >>> 'content' in msg
        True
        >>> 'tool_calls' in msg
        False
        >>> msg['tool_calls'] = []
        >>> 'tool_calls' in msg
        True
        >>> msg['tool_calls'] = [Message.ToolCall(function=Message.ToolCall.Function(name='foo', arguments={}))]
        >>> 'tool_calls' in msg
        True
        >>> msg['tool_calls'] = None
        >>> 'tool_calls' in msg
        True
        >>> tool = OllamaTool()
        >>> 'type' in tool
        True
        """
        if key in self.model_fields_set:
            return True

        if value := self.__class__.model_fields.get(key):
            return value.default is not None

        return False

    def get(self, key: str, default: Any = None) -> Any:
        """Return the value of a field by name, or a default if the field does not exist.

        Args:
            key (str): The field name to look up on the model.
            default (Any): Value to return when ``key`` is not a field on the model.
                Defaults to ``None``.

        Returns:
            Any: The field value if the attribute exists, otherwise ``default``.

        >>> msg = Message(role='user')
        >>> msg.get('role')
        'user'
        >>> msg = Message(role='user')
        >>> msg.get('nonexistent')
        >>> msg = Message(role='user')
        >>> msg.get('nonexistent', 'default')
        'default'
        >>> msg = Message(role='user', tool_calls=[ Message.ToolCall(function=Message.ToolCall.Function(name='foo', arguments={}))])
        >>> msg.get('tool_calls')[0]['function']['name']
        'foo'
        """
        return getattr(self, key) if hasattr(self, key) else default


# https://github.com/ollama/ollama-python/blob/60e7b2f9ce710eeb57ef2986c46ea612ae7516af/ollama/_types.py#L337-L363
class OllamaTool(SubscriptableBaseModel):
    """Pydantic model for an Ollama-compatible tool schema, imported from the Ollama Python SDK.

    Represents the JSON structure that Ollama (and OpenAI-compatible endpoints) expect
    when a tool is passed to the chat API. Mellea builds these objects internally via
    ``convert_function_to_ollama_tool`` and never exposes them to end users directly.

    Attributes:
        type (str | None): Tool type; always ``"function"`` for function-calling tools.
        function (Function | None): Nested object containing the function name,
            description, and parameters schema.
    """

    type: str | None = "function"

    class Function(SubscriptableBaseModel):
        """Pydantic model for the ``function`` field of an Ollama tool schema, imported from the Ollama Python SDK.

        Attributes:
            name (str | None): The name of the function being described.
            description (str | None): Human-readable description of what the function does.
            parameters (Parameters | None): Schema describing the function's parameters.
        """

        name: str | None = None
        description: str | None = None

        class Parameters(SubscriptableBaseModel):
            """Pydantic model for the ``parameters`` field of an Ollama function schema, imported from the Ollama Python SDK.

            Attributes:
                type (Literal["object"] | None): Always ``"object"`` for function parameters.
                defs (Any | None): JSON Schema ``$defs`` for referenced sub-schemas.
                items (Any | None): Array item schema, if applicable.
                required (Sequence[str] | None): List of required parameter names.
                properties (Mapping[str, Property] | None): Parameter property definitions.
            """

            model_config = ConfigDict(populate_by_name=True)
            type: Literal["object"] | None = "object"
            defs: Any | None = Field(None, alias="$defs")
            items: Any | None = None
            required: Sequence[str] | None = None

            class Property(SubscriptableBaseModel):
                """Pydantic model for a single parameter property in an Ollama tool schema, imported from the Ollama Python SDK.

                Attributes:
                    type (str | Sequence[str] | None): JSON Schema type string or list of type strings.
                    items (Any | None): Schema for array element types, if applicable.
                    description (str | None): Human-readable description of this parameter.
                    enum (Sequence[Any] | None): Allowed values for this parameter, if constrained.
                    properties (Mapping[str, Any] | None): Nested properties for object types.
                    required (Sequence[str] | None): Required fields for nested objects.
                    title (str | None): Title for the property schema.
                """

                model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

                type: str | Sequence[str] | None = None
                items: Any | None = None
                description: str | None = None
                enum: Sequence[Any] | None = None
                properties: Mapping[str, Any] | None = None
                required: Sequence[str] | None = None
                title: str | None = None

            properties: Mapping[str, Property] | None = None

        parameters: Parameters | None = None

    function: Function | None = None


# https://github.com/ollama/ollama-python/blob/main/ollama/_utils.py#L13-L53
def _parse_docstring(doc_string: str | None) -> dict[str, str]:
    """Imported from Ollama."""
    parsed_docstring: defaultdict[str, str] = defaultdict(str)
    if not doc_string:
        return parsed_docstring

    key = str(hash(doc_string))
    for line in doc_string.splitlines():
        lowered_line = line.lower().strip()
        if lowered_line.startswith("args:"):
            key = "args"
        elif lowered_line.startswith(("returns:", "yields:", "raises:")):
            key = "_"

        else:
            # maybe change to a list and join later
            parsed_docstring[key] += f"{line.strip()}\n"

    last_key = None
    for line in parsed_docstring["args"].splitlines():
        line = line.strip()
        if ":" in line:
            # Split the line on either:
            # 1. A parenthetical expression like (integer) - captured in group 1
            # 2. A colon :
            # Followed by optional whitespace. Only split on first occurrence.
            parts = re.split(r"(?:\(([^)]*)\)|:)\s*", line, maxsplit=1)

            arg_name = parts[0].strip()
            last_key = arg_name

            # Get the description - will be in parts[1] if parenthetical or parts[-1] if after colon
            arg_description = parts[-1].strip()
            if len(parts) > 2 and parts[1]:  # Has parenthetical content
                arg_description = parts[-1].split(":", 1)[-1].strip()

            parsed_docstring[last_key] = arg_description

        elif last_key and line:
            parsed_docstring[last_key] += " " + line

    return parsed_docstring


def _resolve_ref(ref_path: str, defs: dict) -> dict:
    """Resolve a $ref path like '#/$defs/Email' to the actual schema."""
    if ref_path.startswith("#/$defs/"):
        def_name = ref_path.split("/")[-1]
        return defs.get(def_name, {})
    elif ref_path.startswith("#/definitions/"):
        def_name = ref_path.split("/")[-1]
        return defs.get(def_name, {})
    return {}


def _is_complex_anyof(v: dict) -> bool:
    """Check if anyOf contains complex types (refs or nested objects)."""
    any_of_schemas = v.get("anyOf", [])
    for sub_schema in any_of_schemas:
        # Skip null types - they just indicate optionality
        if sub_schema.get("type") == "null":
            continue
        # Check for references or nested properties
        if "$ref" in sub_schema or "properties" in sub_schema:
            return True
    return False


# https://github.com/ollama/ollama-python/blob/60e7b2f9ce710eeb57ef2986c46ea612ae7516af/ollama/_utils.py#L56-L90
def convert_function_to_ollama_tool(
    func: Callable, name: str | None = None
) -> OllamaTool:
    """Convert a Python callable to an Ollama-compatible tool schema.

    Imported from Ollama, with enhancements to support Pydantic BaseModel parameters.

    Args:
        func: The Python callable to convert.
        name: Optional override for the tool name; defaults to ``func.__name__``.

    Returns:
        An ``OllamaTool`` instance representing the function as an OpenAI-compatible
        tool schema.
    """
    doc_string_hash = str(hash(inspect.getdoc(func)))
    parsed_docstring = _parse_docstring(inspect.getdoc(func))
    schema = type(
        func.__name__,
        (BaseModel,),
        {
            "__annotations__": {
                k: v.annotation if v.annotation != inspect._empty else str
                for k, v in inspect.signature(func).parameters.items()
            },
            "__signature__": inspect.signature(func),
            "__doc__": parsed_docstring[doc_string_hash],
        },
    ).model_json_schema()  # type: ignore

    defs = schema.get("$defs", schema.get("definitions", {}))

    for k, v in schema.get("properties", {}).items():
        # Check if this property has a $ref (reference to a definition)
        if "$ref" in v:
            # Resolve the reference and inline it
            ref_schema = _resolve_ref(v["$ref"], defs)
            if ref_schema:
                # Inline the referenced schema (deep copy to avoid mutations)
                inlined = copy.deepcopy(ref_schema)
                # Add description from docstring if available
                if parsed_docstring.get(k):
                    inlined["description"] = parsed_docstring[k]
                schema["properties"][k] = inlined
            else:
                # Fallback if we can't resolve
                schema["properties"][k] = {
                    "description": parsed_docstring.get(k, ""),
                    "type": "object",
                }
        # Check if this property is a nested object (has 'properties' or complex types)
        # Narrow anyOf check to only complex unions, not Optional[primitive]
        elif (
            "properties" in v or "allOf" in v or ("anyOf" in v and _is_complex_anyof(v))
        ):
            # This is a complex/nested type - preserve the full schema
            # Only add description if we have one
            if parsed_docstring.get(k):
                v["description"] = parsed_docstring[k]
            schema["properties"][k] = v
        else:
            # Simple type - use the original flattening logic
            # This now handles Optional[primitive] types correctly
            if "anyOf" in v:
                types = {t.get("type", "string") for t in v.get("anyOf")}
            else:
                types = {v.get("type", "string")}

            if "null" in types:
                if k in schema.get("required", []):
                    schema["required"].remove(k)
                types.discard("null")

            schema["properties"][k] = {
                "description": parsed_docstring.get(k, ""),
                "type": ", ".join(types),
            }

    tool = OllamaTool(
        type="function",
        function=OllamaTool.Function(
            name=name or func.__name__,
            description=schema.get("description", ""),
            parameters=OllamaTool.Function.Parameters(**schema),
        ),
    )

    return OllamaTool.model_validate(tool)
