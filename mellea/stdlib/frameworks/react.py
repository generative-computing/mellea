"""ReACT Agentic Pattern."""

# from PIL import Image as PILImage
from mellea.backends.model_options import ModelOption
from mellea.core.backend import Backend, BaseModelSubclass
from mellea.core.base import AbstractMelleaTool, ModelOutputThunk
from mellea.core.utils import FancyLogger
from mellea.stdlib import functional as mfuncs

# from mellea.stdlib.components.docs.document import Document
from mellea.stdlib.components.chat import ToolMessage
from mellea.stdlib.components.react import (
    MELLEA_FINALIZER_TOOL,
    ReactInitiator,
    ReactThought,
)
from mellea.stdlib.context import ChatContext

# TODO: JAL. Change output type.
async def react(
    goal: str,
    context: ChatContext,
    backend: Backend,
    *,
    # TODO: These should be added when Components generically support them.
    # images: list[ImageBlock] | list[PILImage.Image] | None = None,
    # docs: list[Document] | None = None,
    format: type[BaseModelSubclass] | None = None,
    model_options: dict | None = None,
    tools: list[AbstractMelleaTool] | None,
    loop_budget: int = 10,
) -> tuple[ModelOutputThunk, ChatContext]:
    """Asynchronous ReACT pattern (Think -> Act -> Observe -> Repeat Until Done); attempts to accomplish the provided goal given the provided tools.

    Args:
        goal: the goal to be accomplished or the question to answer
        context: the context being used; a type of ChatContext
        backend: the backend used to generate the response.
        format: if set, the BaseModel to use for constrained decoding.
        model_options: additional model options, which will upsert into the model/backend's defaults.
        tools: the list of tools to use
        loop_budget: the number of steps allowed; use -1 for unlimited

    Returns:
        A (ModelOutputThunk, Context) if `return_sampling_results` is `False`, else returns a `SamplingResult`.

    Raises:
        RuntimeError: if the loop ends before a final answer is found

    """
    assert isinstance(context, ChatContext), (
        f"ReACT must use a type of chat context, got: {type(context)}"
    )
    # We are currently allowing react to be slotted into an existing context. Uncomment the below lines
    # and change the docstring if we want to change that.
    # test_ctx_lin = context.view_for_generation()
    # assert test_ctx_lin is not None and len(test_ctx_lin) == 0, (
    #     "ReACT expects a fresh context."
    # )

    tools = tools or []
    mo_tools: list[AbstractMelleaTool] | None = (
        model_options.get(ModelOption.TOOLS, None)
        if model_options is not None
        else None
    )
    if mo_tools is not None:
        tools.extend(mo_tools)

    context = context.add(ReactInitiator(goal, tools, format=format))

    turn_num = 0
    while (turn_num < loop_budget) or (loop_budget == -1):
        turn_num += 1
        FancyLogger.get_logger().info(f"## ReACT TURN NUMBER {turn_num}")

        step, next_context = await mfuncs.aact(
            action=ReactThought(),
            context=context,
            backend=backend,
            requirements=[],
            strategy=None,
            model_options=model_options,
            tool_calls=True,
        )

        # Have to assert this due to type hints.
        assert isinstance(next_context, ChatContext)
        context = next_context

        is_final = False
        tool_responses: list[ToolMessage] = []
        if step.tool_calls is not None:
            # Code below assumes the tool is called here.
            tool_responses = mfuncs._call_tools(step, backend=backend)
            for tool_res in tool_responses:
                context = context.add(tool_res)
                if tool_res.name == MELLEA_FINALIZER_TOOL:
                    is_final = True

        if is_final:
            assert len(tool_responses) == 1, "multiple tools were called with 'final'"
            step._underlying_value = str(tool_responses[0].content)
            return step, context

    raise RuntimeError(f"could not complete react loop in {loop_budget} iterations")
