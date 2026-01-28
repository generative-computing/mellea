
from collections.abc import Callable
from typing import Any
from langchain_community.tools import DuckDuckGoSearchRun, DuckDuckGoSearchResults

from mellea.backends.tools import convert_function_to_tool

search = DuckDuckGoSearchRun()

# print(search.invoke("Obama's first name?"))

search = DuckDuckGoSearchResults()

# print(search.invoke("Obama"))

search = DuckDuckGoSearchResults(output_format="list")

# print(search.invoke("Obama"))
# print(search.tool_call_schema)


from langchain_openai import ChatOpenAI
# llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# tools = [search]
# llm_with_tools = llm.bind_tools(tools)
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_core.tools import BaseTool
import langchain_core.tools as lg_tools


# Tool is what we pass as a model option / as input
# Our ModelToolCall is the class that has a reference to the tool and actually calls with arguments
class Tool():

    name: str
    _as_json_tool: dict[str, Any]
    _call_tool: Callable[..., Any]

    def __init__(self, name: str, tool_call: Callable, as_json_tool: dict[str, Any]) -> None:
        self.name = name
        self._as_json_tool = as_json_tool
        self._call_tool = tool_call

    def run(self, *args, **kwargs) -> Any:
        return self._call_tool(*args, **kwargs)

    @property
    def as_json_tool(self):
        return self._as_json_tool

    @classmethod
    def from_langchain(cls, tool: Any):
        try:
            from langchain_core.tools import BaseTool

            if isinstance(tool, BaseTool):
                tool_name = tool.name
                as_json = convert_to_openai_tool(tool)
                tool_call = tool.run
                return Tool(tool_name, tool_call, as_json)
            else:
                raise ValueError(f"tool parameter must be a langchain tool type; got: {type(tool)}")

        except ImportError as e:
            raise ImportError(
                f"It appears you are attempting to utilize a langchain tool '{type(tool)}'"
                "Please install langchain core: pip install 'langchain_core'."
            ) from e

    @classmethod
    def from_callable(cls, func: Callable, name: str | None = None):
        # Use the function name if the name is '' or None.
        tool_name = name or func.__name__
        as_json = convert_function_to_tool(func, tool_name).model_dump(exclude_none=True)
        tool_call = func
        return Tool(tool_name, tool_call, as_json)

# TODO: JAL. Move this to a test, and check for the name override.
def testing(inpt: int) -> int:
    """Description."""
    return inpt
t = Tool.from_callable(testing, "file")
print(type(search))
print(search.__class__)
print(search.__class__.__bases__)
exit()

# TODO: JAL. Make a test for langchain.
search_tool = Tool.from_langchain(search)

    # @classmethod
    # def from_smolagent():
    #     ...

# TODO: JAL. Instead of doing special handling here and in model tool call class, we should just create our own Tool class that standardizes this process. Ie we can go from hf / langchain / non-callable tool to our tool when instantiated. This would require users to do special behavior when adding tool calls to model options / etc... unless we add special handling for the Callable version.
def convert_tool_to_json(tool: Callable | Any):
        # TODO: JAL. Do regular conversion on callables here...

        # TODO: JAL. Add tests for these magic strings.
        if "langchain" in str(type(tool)):
            try:
                from langchain_core.tools import BaseTool

                if isinstance(tool, BaseTool):
                    ...
                    # print(convert_to_openai_tool(tool))
                    # TODO: JAL. Need to call differently?

            except ImportError as e:
                raise ImportError(
                    f"It appears you are attempting to utilize a langchain tool '{type(tool)}'"
                    "Please install langchain core: pip install 'langchain_core'."
                ) from e
        
        if "smolagent...":
            ...
            # TODO: JAL. Actually implement this.
            # format_smolagent_tool


convert_tool_to_json(search)

# print(convert_to_openai_tool(search))

def test(inpt: int) -> bool:
    """description"""
    ...


# print(convert_to_openai_tool(test))
# print(convert_function_to_tool(test).model_dump(exclude_none=True))
# {'type': 'function', 'function': {'name': 'test', 'description': 'description', 'parameters': {'properties': {'inpt': {'type': 'integer'}}, 'required': ['inpt'], 'type': 'object'}}}
# TODO: JAL. our version has an extra description for the parameter even though it's empty here
# {'type': 'function', 'function': {'name': 'test', 'description': 'description', 'parameters': {'type': 'object', 'required': ['inpt'], 'properties': {'inpt': {'type': 'integer', 'description': ''}}}}}


from smolagents import Tool

class HFModelDownloadsTool(Tool):
    name = "model_download_counter"
    description = """
    This is a tool that returns the most downloaded model of a given task on the Hugging Face Hub.
    It returns the name of the checkpoint."""
    inputs = {
        "task": {
            "type": "string",
            "description": "the task category (such as text-classification, depth-estimation, etc)",
        }
    }
    output_type = "string"

    def forward(self, task: str):
        from huggingface_hub import list_models

        model = next(iter(list_models(filter=task, sort="downloads", direction=-1)))
        return model.id

model_downloads_tool = HFModelDownloadsTool()

from smolagents import load_tool, CodeAgent

image_generation_tool = Tool.from_space(
    "black-forest-labs/FLUX.1-schnell",
    name="image_generator",
    description="Generate an image from a prompt"
)

# print(image_generation_tool.to_tool_calling_prompt())

# {'type': 'function', 'function': {'name': 'test', 'description': 'description', 'parameters': {'properties': {'inpt': {'type': 'integer'}}, 'required': ['inpt'], 'type': 'object'}}}

def format_smolagent_tool(tool: Tool):
    # TODO: JAL. If we want to support huggingface tools, we need to make sure this matches our existing tool calling format and works with our models.
    json_output = {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.inputs,
        }
    }
    print(json_output)

format_smolagent_tool(image_generation_tool)