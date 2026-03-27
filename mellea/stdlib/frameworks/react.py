"""ReACT (Reason + Act) agentic pattern implementation.

Provides the ``react()`` async function, which drives a tool-use loop: the model
reasons about a goal, selects a tool, receives the result as an observation, and
repeats until it calls ``final_answer`` or the ``loop_budget`` is exhausted. Accepts
any list of ``AbstractMelleaTool`` instances and a ``ChatContext`` for multi-turn
history tracking. Raises ``RuntimeError`` if the loop ends without a final answer.
"""

# from PIL import Image as PILImage
from mellea.backends.model_options import ModelOption
from mellea.core.backend import Backend, BaseModelSubclass
from mellea.core.base import AbstractMelleaTool, ComputedModelOutputThunk
from mellea.core.utils import MelleaLogger
from mellea.stdlib import functional as mfuncs

# from mellea.stdlib.components.docs.document import Document
from mellea.stdlib.components.chat import ToolMessage
from mellea.stdlib.components.react import (
    MELLEA_FINALIZER_TOOL,
    ReactInitiator,
    ReactThought,
)
from mellea.stdlib.context import ChatContext


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
) -> tuple[ComputedModelOutputThunk[str], ChatContext]:
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

    context = context.add(ReactInitiator(goal, tools))

    turn_num = 0
    while (turn_num < loop_budget) or (loop_budget == -1):
        turn_num += 1
        MelleaLogger.get_logger().info(f"## ReACT TURN NUMBER {turn_num}")

        step, next_context = await mfuncs.aact(
            action=ReactThought(),
            context=context,
            backend=backend,
            requirements=[],
            strategy=None,
            model_options=model_options,
            tool_calls=True,
            await_result=True,
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

        # Check if we should return: either finalizer was called or model gave direct answer
        should_return = is_final or (step.tool_calls is None and step.value is not None)

        if should_return:
            if is_final:
                assert len(tool_responses) == 1, (
                    "multiple tools were called with 'final'"
                )
                if format is None:
                    # The tool has already been called above.
                    step._underlying_value = str(tool_responses[0].content)

            # Apply format if requested (works for both finalizer and direct answer cases)
            if format is not None:
                step, next_context = await mfuncs.aact(
                    action=ReactThought(),
                    context=context,
                    backend=backend,
                    requirements=[],
                    strategy=None,
                    model_options=model_options,
                    format=format,
                    await_result=True,
                )
                assert isinstance(next_context, ChatContext)
                context = next_context

            return step, context

    raise RuntimeError(f"could not complete react loop in {loop_budget} iterations")
