
from PIL import Image as PILImage

from mellea import start_session
from mellea.core.backend import Backend, BaseModelSubclass
from mellea.core.base import Component, Context, ImageBlock, ModelOutputThunk
from mellea.stdlib.components.docs.document import Document

from mellea.stdlib import functional as mfuncs
from mellea.stdlib.sampling.base import RejectionSamplingStrategy

# Agent Patterns
"""
ReAct
Query -> Thought -> Tool -> Output -> Answer
           \<----------------/
"""


# ---------
import datetime
import inspect
import json
from collections.abc import Callable
from typing import Literal

import pydantic
from jinja2 import Template

import mellea
import mellea.stdlib.components.chat
from mellea.core import FancyLogger
from mellea.stdlib.context import ChatContext

FancyLogger.get_logger().setLevel("ERROR")

react_system_template: Template = Template(
    """Answer the user's question as best you can.

Today is {{- today }} and you can use the following tool names with associated descriptions:
{% for tool in tools %} * {{- tool.get_name() }}: {{- tool.get_description()}}{% endfor %}"""
)


class ReactTool(pydantic.BaseModel):
    """This is a helper model for React tools.

    Args:
        fn: The tool.
        name: The name of the tool. The default value is the function's name.
        description: A description of the tool for the ReACT system prompt. The default value is the first line of the function's docstring.
    """

    fn: Callable
    name: str | None
    description: str | None

    def get_name(self):
        if self.name is None:
            return self.fn.__name__
        else:
            return self.name

    def get_description(self):
        if self.description is None:
            return self.fn.__doc__.splitlines[0]
        else:
            return self.description

    def args_schema(self):
        sig = inspect.signature(self.fn)
        fields = dict()
        for param_name, param in sig.parameters.items():
            fields[param_name] = str
        return pydantic.create_model(
            f"{self.fn.__name__.capitalize()}ToolSchema", **fields
        )


class ReactToolbox(pydantic.BaseModel):
    """A convienance wrapper around ReactTool."""

    tools: list[ReactTool]

    def tool_names(self):
        return [tool.get_name() for tool in self.tools]

    def tools_dict(self):
        """Formats the tools for passing into backends' tools= parameter."""
        return {tool.get_name(): tool.fn for tool in self.tools}

    def get_tool_from_name(self, name: str) -> ReactTool | None:
        for tool in self.tools:
            if tool.get_name() == name:
                return tool
        return None

    def call_tool(self, tool: ReactTool, kwargs_json: str):
        fn = tool.fn
        kwargs = json.loads(kwargs_json)
        return fn(**kwargs)

    def tool_name_schema(self):
        names = self.tool_names()
        fields = dict()
        fields["tool"] = Literal[*names]
        return pydantic.create_model("ToolSelectionSchema", **fields)

    def get_tool_from_schema(self, content: str):
        schema = self.tool_name_schema()
        validated = schema.model_validate_json(content)
        return self.get_tool_from_name(validated.tool)


class IsDoneModel(pydantic.BaseModel):
    is_done: bool

# ------
class ReactInitiator(Component[str]):
    def __init__(*args, **kwargs):
        ...

class ReactThought(Component[str]):
    def __init__(*args, **kwargs):
        ...

class ReactAction(Component[str]):
    def __init__(*args, **kwargs):
        ...

# Mfunc version.
async def react(
    goal: str,
    context: Context, # TODO: Should we be able to enter a react loop during a conversation?
    backend: Backend,
    *,
    images: list[ImageBlock] | list[PILImage.Image] | None = None,
    docs: list[Document] | None = None,
    format: type[BaseModelSubclass] | None = None,
    model_options: dict | None = None,
    tools: list[Callable] | None,
) -> tuple[ModelOutputThunk, Context]:
    # TODO: Add two budgets -> tool sampling and the outer loop

    assert isinstance(context, ChatContext)
    test_ctx_lin = context.view_for_generation()
    assert test_ctx_lin is not None and len(test_ctx_lin) == 0, (
        "ReACT expects a fresh context."
    )

    context = context.add(ReactInitiator(goal, tools, role="system", format=format))
    # ReactInitiator will add `def final(format)` as a tool call.

    done = False
    turn_num = 0
    while not done:
        turn_num += 1
        FancyLogger.get_logger().info(f"## ReACT TURN NUMBER {turn_num}")

        thought, context = await mfuncs.aact(ReactThought(...), context=context, backend=backend)

        samples = await mfuncs.aact(
            action=ReactAction(...),
            context=context,
            backend=backend,
            requirements=[], # Valid Tool Call Requirement
            strategy=RejectionSamplingStrategy(),
            return_sampling_results=True
        )

        if not samples.success:
            raise ValueError("could not generate correct action")

        action = samples.result
        context = samples.result_ctx

        assert action.tool_calls is not None, "tool calls should be populated, something went wrong with the requirement validation"

        is_final = False
        tool_responses = mfuncs._call_tools(action, backend=backend)
        for tool_res in tool_responses:
            context = context.add(tool_res)
            if tool_res.name == "final_tool": # TODO: JAL. Make this nicer / write a function that has a constant.
                is_final = True

        if is_final:
            assert len(tool_responses) == 1, "multiple tools were called with 'final'"
            done = True
            action._underlying_value = str(tool_responses[0].arguments) # TODO: clean this up so that it's either the actual answer or the expected format
            return action, context
            # TODO: have to convert to model output thunk...

    raise Exception("did not complete")


def react(
    m: mellea.MelleaSession,
    goal: str,
    state_description: str | None,
    react_toolbox: ReactToolbox,
):
    assert m.ctx.is_chat_context, "ReACT requires a chat context."
    test_ctx_lin = m.ctx.view_for_generation()
    assert test_ctx_lin is not None and len(test_ctx_lin) == 0, (
        "ReACT expects a fresh context."
    )

    # Construct the system prompt for ReACT.
    _sys_prompt = react_system_template.render(
        {"today": datetime.date.today(), "tools": react_toolbox.tools}
    )

    # Add the system prompt and the goal to the chat history.
    m.ctx = m.ctx.add(
        mellea.stdlib.components.chat.Message(role="system", content=_sys_prompt)
    ).add(mellea.stdlib.components.chat.Message(role="user", content=f"{goal}"))

    # The main ReACT loop as a dynamic program:
    # (  ?(not done) ;
    #    (thought request ; thought response) ;
    #    (action request ; action response) ;
    #    (action args request ; action args response) ;
    #    observation from the tool call ;
    #    (is done request ; is done response) ;
    #    { ?(model indicated done) ; emit_final_answer ; done := true }
    # )*
    done = False
    turn_num = 0
    while not done:
        turn_num += 1
        print(f"## ReACT TURN NUMBER {turn_num}")

        print("### Thought")
        thought = m.chat(
            "What should you do next? Respond with a description of the next piece of information you need or the next action you need to take."
        )
        print(thought.content)

        # TODO: Can unify choosing action and inputs.
        print("### Action")
        act = m.chat(
            "Choose your next action. Respond with a nothing other than a tool name.",
            # model_options={mellea.backends.types.ModelOption.TOOLS: react_toolbox.tools_dict()},
            format=react_toolbox.tool_name_schema(),
        )
        selected_tool: ReactTool = react_toolbox.get_tool_from_schema(act.content)
        print(selected_tool.get_name())

        print("### Arguments for action")
        act_args = m.chat(
            "Choose arguments for the tool. Respond using JSON and include only the tool arguments in your response.",
            format=selected_tool.args_schema(),
        )
        print(f"```json\n{json.dumps(json.loads(act_args.content), indent=2)}\n```")

        print("### Observation")
        tool_output = react_toolbox.call_tool(selected_tool, act_args.content)
        m.ctx = m.ctx.add(
            mellea.stdlib.components.chat.Message(role="tool", content=tool_output)
        )
        print(tool_output)

        # TODO: Rework this check.
        print("### Done Check")
        is_done = IsDoneModel.model_validate_json(
            m.chat(
                f"Do you know the answer to the user's original query ({goal})? If so, respond with Yes. If you need to take more actions, then respond No.",
                format=IsDoneModel,
            ).content
        ).is_done
        if is_done:
            print("Done. Will summarize and return output now.")
            done = True
            return m.chat(
                f"Please provide your final answer to the original query ({goal})."
            ).content
        else:
            print("Not done.")
            done = False


if __name__ == "__main__":
    m = mellea.start_session(ctx=ChatContext())

    def zip_lookup_tool_fn(city: str):
        """Returns the ZIP code for the `city`."""
        return "03285"

    zip_lookup_tool = ReactTool(
        name="Zip Code Lookup",
        fn=zip_lookup_tool_fn,
        description="Returns the ZIP code given a town name.",
    )

    def weather_lookup_fn(zip_code: str):
        """Looks up the weather for a town given a five-digit `zip_code`."""
        return "The weather in Thornton, NH is sunny with a high of 78 and a low of 52. Scattered showers are possible in the afternoon."

    weather_lookup_tool = ReactTool(
        name="Get the weather",
        fn=weather_lookup_fn,
        description="Returns the weather given a ZIP code.",
    )

    result = react(
        m,
        goal="What is today's high temperature in Thornton, NH?",
        state_description=None,
        react_toolbox=ReactToolbox(tools=[zip_lookup_tool, weather_lookup_tool]),
    )

    print("## Final Answer")
    print(result)

m = start_session()
