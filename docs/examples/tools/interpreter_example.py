# pytest: ollama, llm

from mellea import MelleaSession, start_session
from mellea.backends import ModelOption
from mellea.backends.tools import MelleaTool
from mellea.stdlib.requirements import tool_arg_validator, uses_tool
from mellea.stdlib.tools import code_interpreter, local_code_interpreter


def example_1(m: MelleaSession):
    # First, let's see how the code interpreter function works without an LLM in the loop:
    result = code_interpreter("print(1+1)")
    print(result)


# Now let's ask the LLM to make a plot.


def example_2(m: MelleaSession):
    plot_output = m.instruct(
        description="Make a plot of y=x^2",
        model_options={
            ModelOption.TOOLS: [MelleaTool.from_callable(local_code_interpreter)]
        },
    )
    print(plot_output)


# Notice that the model did not actually generate a plot. Let's force tool use:


def example_3(m: MelleaSession):
    plot_output = m.instruct(
        description="Use the code interpreter tool to make a plot of y=x^2.",
        requirements=[uses_tool(local_code_interpreter)],
        model_options={
            ModelOption.TOOLS: [MelleaTool.from_callable(local_code_interpreter)]
        },
        tool_calls=True,
    )

    if plot_output.tool_calls is None:
        raise ValueError("Expected tool_calls but got None")

    code = plot_output.tool_calls["local_code_interpreter"].args["code"]
    print(f"Going to execute the following code:\n```python\n{code}\n```")

    # Call the tool.
    exec_result = plot_output.tool_calls["local_code_interpreter"].call_func()

    print(exec_result)


# Notice that the model did make a plot, but it just "showed" the plot.
# We would actually like this to be written out to a file.


def example_4(m: MelleaSession):
    plot_output = m.instruct(
        description="Use the code interpreter tool to make a plot of y=x^2.",
        requirements=[
            uses_tool(local_code_interpreter),
            tool_arg_validator(
                "The plot should be written to /tmp/output.png",
                tool_name=local_code_interpreter,
                arg_name="code",
                validation_fn=lambda code_snippet: "/tmp/output.png" in code_snippet
                and "plt.show()" not in code_snippet,
            ),
        ],
        model_options={
            ModelOption.TOOLS: [MelleaTool.from_callable(local_code_interpreter)]
        },
        tool_calls=True,
    )

    if plot_output.tool_calls is None:
        raise ValueError("Expected tool_calls but got None")

    code = plot_output.tool_calls["local_code_interpreter"].args["code"]
    print(f"Going to execute the following code:\n```python\n{code}\n```")

    # Call the tool.
    exec_result = plot_output.tool_calls["local_code_interpreter"].call_func()

    print(exec_result)


# m = start_session(backend_name="ollama", model_id=OPENAI_GPT_OSS_20B)
m = start_session()
example_4(m)
