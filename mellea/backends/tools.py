"""Utilities for dealing with LLM tools."""

import inspect
import json
import re
from collections import defaultdict
from collections.abc import Callable, Generator, Iterable, Mapping, Sequence
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from mellea.core.utils import FancyLogger

from ..core import CBlock, Component, TemplateRepresentation
from ..core.base import AbstractMelleaTool
from .model_options import ModelOption


class MelleaTool(AbstractMelleaTool):
    """Tool class to represent a tool."""

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
        """Run the tool with the given arguments."""
        return self._call_tool(*args, **kwargs)

    @property
    def as_json_tool(self) -> dict[str, Any]:
        """Return the tool converted to a OpenAI compatible JSON object."""
        return self._as_json_tool.copy()

    @classmethod
    def from_langchain(cls, tool: Any):
        """Create a Tool from a langchain tool object."""
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
                    if args is not None or len(args) != 0:
                        # This shouldn't happen. Our ModelToolCall.call_func actually passes in everything as kwargs.
                        FancyLogger.get_logger().warning(
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
                f"It appears you are attempting to utilize a langchain tool '{type(tool)}'"
                "Please install langchain core: pip install 'langchain_core'."
            ) from e

    @classmethod
    def from_callable(cls, func: Callable, name: str | None = None):
        """Create a Tool from a callable."""
        # Use the function name if the name is '' or None.
        tool_name = name or func.__name__
        as_json = convert_function_to_ollama_tool(func, tool_name).model_dump(
            exclude_none=True
        )
        tool_call = func
        return MelleaTool(tool_name, tool_call, as_json)


def add_tools_from_model_options(
    tools_dict: dict[str, AbstractMelleaTool], model_options: dict[str, Any]
):
    """If model_options has tools, add those tools to the tools_dict."""
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
            assert isinstance(tool_instance, MelleaTool), (
                f"If ModelOption.TOOLS is a dict, it must be a dict of [str, Tool]; found {type(tool_instance)} as the value instead"
            )
            tools_dict[tool_name] = tool_instance
    else:
        # Handle any other iterable / list here.
        for tool_instance in model_opts_tools:
            assert isinstance(tool_instance, MelleaTool), (
                f"If ModelOption.TOOLS is a list, it must be a list of Tool; found {type(tool_instance)}"
            )
            tools_dict[tool_instance.name] = tool_instance


def add_tools_from_context_actions(
    tools_dict: dict[str, AbstractMelleaTool],
    ctx_actions: list[Component | CBlock] | None,
):
    """If any of the actions in ctx_actions have tools in their template_representation, add those to the tools_dict."""
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

    Notes:
    - Huggingface transformers library lets you pass in an array of functions but doesn't like methods.
    - WatsonxAI uses `from langchain_ibm.chat_models import convert_to_openai_tool` in their demos, but it gives the same values.
    - OpenAI uses the same format / schema.
    """
    return [t.as_json_tool for t in tools.values()]


def json_extraction(text: str) -> Generator[dict, None, None]:
    """Yields the next valid json object in a given string."""
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


def find_func(d) -> tuple[str | None, Mapping | None]:
    """Find the first function in a json-like dictionary.

    Most llms output tool requests in the form `...{"name": string, "arguments": {}}...`
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
    """A simple parser that will scan a string for tools and attempt to extract them; only works for json based outputs."""
    processed = " ".join(llm_response.split())

    tools = []
    for possible_tool in json_extraction(processed):
        tool_name, tool_arguments = find_func(possible_tool)
        if tool_name is not None and tool_arguments is not None:
            tools.append((tool_name, tool_arguments))
    return tools


def validate_tool_arguments(
    func: Callable,
    args: Mapping[str, Any],
    *,
    coerce_types: bool = True,
    strict: bool = False,
) -> dict[str, Any]:
    """Validate and optionally coerce tool arguments against function signature.

    This function validates tool call arguments extracted from LLM responses against
    the expected function signature. It can automatically coerce common type mismatches
    (e.g., string "30" to int 30) and provides detailed error messages.

    Args:
        func: The tool function to validate against
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

        >>> # LLM returns days as string
        >>> args = {"location": "Boston", "days": "3"}
        >>> validated = validate_tool_arguments(get_weather, args)
        >>> validated
        {'location': 'Boston', 'days': 3}

        >>> # Strict mode raises on validation errors
        >>> bad_args = {"location": "Boston", "days": "not_a_number"}
        >>> validate_tool_arguments(get_weather, bad_args, strict=True)
        Traceback (most recent call last):
        ...
        pydantic.ValidationError: ...
    """
    from pydantic import ValidationError, create_model

    from ..core import FancyLogger

    # Get function signature
    sig = inspect.signature(func)

    # Build Pydantic model from function signature
    # This reuses the logic from convert_function_to_tool
    field_definitions: dict[str, Any] = {}

    for param_name, param in sig.parameters.items():
        # Skip *args and **kwargs
        if param.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue

        # Get type annotation
        param_type = param.annotation
        if param_type == inspect.Parameter.empty:
            # No type hint, default to Any
            param_type = Any

        # Handle default values
        if param.default == inspect.Parameter.empty:
            # Required parameter
            field_definitions[param_name] = (param_type, ...)
        else:
            # Optional parameter with default
            field_definitions[param_name] = (param_type, param.default)

    # Create dynamic Pydantic model for validation
    ValidatorModel = create_model(f"{func.__name__}_Validator", **field_definitions)

    # Configure model for type coercion if requested
    if coerce_types:
        # Pydantic v2 uses model_config
        ValidatorModel.model_config = ConfigDict(
            str_strip_whitespace=True  # Strip whitespace from strings
            # Pydantic automatically coerces compatible types
        )

    try:
        # Validate using Pydantic
        validated_model = ValidatorModel(**args)
        validated_args = validated_model.model_dump()

        # Log successful validation with coercion details
        coerced_fields = []
        for key, original_value in args.items():
            validated_value = validated_args.get(key)
            if type(original_value) is not type(validated_value):
                coerced_fields.append(
                    f"{key}: {type(original_value).__name__} → {type(validated_value).__name__}"
                )

        if coerced_fields and coerce_types:
            FancyLogger.get_logger().debug(
                f"Tool '{func.__name__}' arguments coerced: {', '.join(coerced_fields)}"
            )

        return validated_args

    except ValidationError as e:
        # Format error message
        error_details = []
        for error in e.errors():
            field = ".".join(str(loc) for loc in error["loc"])
            msg = error["msg"]
            error_details.append(f"  - {field}: {msg}")

        error_msg = (
            f"Tool argument validation failed for '{func.__name__}':\n"
            + "\n".join(error_details)
        )

        if strict:
            # Re-raise with enhanced message
            FancyLogger.get_logger().error(error_msg)
            raise
        else:
            # Log warning and return original args
            FancyLogger.get_logger().warning(
                error_msg + "\nReturning original arguments without validation."
            )
            return dict(args)

    except Exception as e:
        # Catch any other errors during validation
        error_msg = f"Unexpected error validating tool '{func.__name__}' arguments: {e}"

        if strict:
            FancyLogger.get_logger().error(error_msg)
            raise
        else:
            FancyLogger.get_logger().warning(
                error_msg + "\nReturning original arguments without validation."
            )
            return dict(args)


# Below functions and classes extracted from Ollama Python SDK (v0.6.1)
# so that all backends don't need it installed.
# https://github.com/ollama/ollama-python/blob/60e7b2f9ce710eeb57ef2986c46ea612ae7516af/ollama/_types.py#L19-L101
class SubscriptableBaseModel(BaseModel):
    """Class imported from Ollama."""

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
        """Get.

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
    """Class imported from Ollama."""

    type: str | None = "function"

    class Function(SubscriptableBaseModel):
        """Class imported from Ollama."""

        name: str | None = None
        description: str | None = None

        class Parameters(SubscriptableBaseModel):
            """Class imported from Ollama."""

            model_config = ConfigDict(populate_by_name=True)
            type: Literal["object"] | None = "object"
            defs: Any | None = Field(None, alias="$defs")
            items: Any | None = None
            required: Sequence[str] | None = None

            class Property(SubscriptableBaseModel):
                """Class imported from Ollama."""

                model_config = ConfigDict(arbitrary_types_allowed=True)

                type: str | Sequence[str] | None = None
                items: Any | None = None
                description: str | None = None
                enum: Sequence[Any] | None = None

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


# https://github.com/ollama/ollama-python/blob/60e7b2f9ce710eeb57ef2986c46ea612ae7516af/ollama/_utils.py#L56-L90
def convert_function_to_ollama_tool(
    func: Callable, name: str | None = None
) -> OllamaTool:
    """Imported from Ollama."""
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

    for k, v in schema.get("properties", {}).items():
        # If type is missing, the default is string
        types = (
            {t.get("type", "string") for t in v.get("anyOf")}
            if "anyOf" in v
            else {v.get("type", "string")}
        )
        if "null" in types:
            schema["required"].remove(k)
            types.discard("null")

        schema["properties"][k] = {
            "description": parsed_docstring[k],
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
