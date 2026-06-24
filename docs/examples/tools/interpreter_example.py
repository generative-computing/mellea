# pytest: ollama, e2e

from mellea import MelleaSession, start_session
from mellea.backends import ModelOption
from mellea.stdlib.requirements import tool_arg_validator, uses_tool
from mellea.stdlib.tools import python_tool

# tier="docker_unsafe" matches the isolation level of the deprecated code_interpreter()
docker_tool = python_tool(tier="docker_unsafe")
# tier="local_unsafe" for LLM-driven examples where the model writes the code
tool = python_tool(tier="local_unsafe", name="python")


def example_1(m: MelleaSession):
    # First, let's see how the code interpreter function works without an LLM in the loop:
    result = docker_tool.run(code="print(1+1)")
    print(result.stdout)


# Now let's ask the LLM to make a plot.


def example_2(m: MelleaSession):
    plot_output = m.instruct(
        description="Make a plot of y=x^2", model_options={ModelOption.TOOLS: [tool]}
    )
    print(plot_output)


# Notice that the model did not actually generate a plot. Let's force tool use:


def example_3(m: MelleaSession):
    plot_output = m.instruct(
        description="Use the code interpreter tool to make a plot of y=x^2.",
        requirements=[uses_tool("python")],
        model_options={ModelOption.TOOLS: [tool]},
        tool_calls=True,
    )

    if plot_output.tool_calls is None:
        raise ValueError("Expected tool_calls but got None")

    code = plot_output.tool_calls["python"].args["code"]
    print(f"Going to execute the following code:\n```python\n{code}\n```")

    # Call the tool.
    exec_result = plot_output.tool_calls["python"].call_func()

    print(exec_result)


# Notice that the model did make a plot, but it just "showed" the plot.
# We would actually like this to be written out to a file.


def example_4(m: MelleaSession):
    plot_output = m.instruct(
        description="Use the code interpreter tool to make a plot of y=x^2.",
        requirements=[
            uses_tool("python"),
            tool_arg_validator(
                "The plot should be written to /tmp/output.png",
                tool_name="python",
                arg_name="code",
                validation_fn=lambda code_snippet: (
                    "/tmp/output.png" in code_snippet
                    and "plt.show()" not in code_snippet
                ),
            ),
        ],
        model_options={ModelOption.TOOLS: [tool]},
        tool_calls=True,
    )

    if plot_output.tool_calls is None:
        raise ValueError("Expected tool_calls but got None")

    code = plot_output.tool_calls["python"].args["code"]
    print(f"Going to execute the following code:\n```python\n{code}\n```")

    # Call the tool.
    exec_result = plot_output.tool_calls["python"].call_func()

    print(exec_result)


# m = start_session(backend_name="ollama", model_id=OPENAI_GPT_OSS_20B)
m = start_session()
example_4(m)
