"""Utilities for dealing with tools."""

import json
from collections.abc import Callable, Generator, Iterable, Mapping
from typing import Any

from ollama._utils import convert_function_to_tool

from mellea.backends.types import ModelOption
from mellea.stdlib.base import Component, TemplateRepresentation


def add_tools_from_model_options(
    tools_dict: dict[str, Callable], model_options: dict[str, Any]
):
    """If model_options has tools, it will add those tools to the tools_dict."""
    model_opts_tools = model_options.get(ModelOption.TOOLS, None)

    if model_opts_tools is None:
        return

    assert isinstance(model_opts_tools, Iterable), (
        "ModelOption.TOOLS must be a list of Callables"
    )
    for func in model_opts_tools:
        assert callable(func), (
            f"ModelOption.TOOLS must be a list of Callables, found {type(func)}"
        )
        tools_dict[func.__name__] = func


def get_tools_from_action(action: Any) -> dict[str, Callable]:
    """If an object is a Component with a TemplateRepresentation, grabs it's tools field.

    Returns:
        dict: mapping function names to callables
    """
    if isinstance(action, Component):
        tr = action.format_for_llm()
        if isinstance(tr, TemplateRepresentation):
            if tr.tools:
                return tr.tools

    return {}


def convert_tools_to_json(tools: dict[str, Callable]) -> list[dict]:
    """Convert tools to json dict representation.

    Notes:
    - Huggingface transformers library lets you pass in an array of functions but doesn't like methods.
    - WatsonxAI uses `from langchain_ibm.chat_models import convert_to_openai_tool` in their demos, but it gives the same values.
    - OpenAI uses the same format / schema.
    """
    converted: list[dict[str, Any]] = []
    for tool in tools.values():
        try:
            converted.append(
                convert_function_to_tool(tool).model_dump(exclude_none=True)
            )
        except Exception:
            pass

    return converted


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


# NOTE: these extraction tools only work for json based outputs.
def parse_tools(llm_response: str) -> list[tuple[str, Mapping]]:
    """A simple parser that will scan a string for tools and attempt to extract them."""
    processed = " ".join(llm_response.split())

    tools = []
    for possible_tool in json_extraction(processed):
        tool_name, tool_arguments = find_func(possible_tool)
        if tool_name is not None and tool_arguments is not None:
            tools.append((tool_name, tool_arguments))

    return tools
