"""ReACT Agentic Pattern."""

from collections.abc import Callable

# from PIL import Image as PILImage
from mellea.core.backend import Backend, BaseModelSubclass
from mellea.core.base import Context, ImageBlock, ModelOutputThunk
from mellea.core.utils import FancyLogger
from mellea.stdlib import functional as mfuncs

# from mellea.stdlib.components.docs.document import Document
from mellea.stdlib.components.react import ReactAction, ReactInitiator, ReactThought
from mellea.stdlib.context import ChatContext
from mellea.stdlib.sampling.base import RejectionSamplingStrategy


async def react(
    goal: str,
    context: ChatContext, # TODO: Should we be able to enter a react loop during a conversation?
    backend: Backend,
    *,
    # images: list[ImageBlock] | list[PILImage.Image] | None = None, # TODO: JAL. These should be added when Components generically support them.
    # docs: list[Document] | None = None,
    format: type[BaseModelSubclass] | None = None,
    model_options: dict | None = None,
    tools: list[Callable] | None, # TODO: JAL. This should be a list of tools.
    loop_budget: int = 10
) -> tuple[ModelOutputThunk, ChatContext]:
    """Asynchronous ReACT pattern (Think -> Act -> Observe -> Repeat Until Done); attempts to accomplish the provided goal given the provided tools.

    Args:
        goal: the goal to be accomplished or the question to answer
        context: the context being used; must be an empty ChatContext
        backend: the backend used to generate the response.
        format: if set, the BaseModel to use for constrained decoding.
        model_options: additional model options, which will upsert into the model/backend's defaults.
        tools: the list of tools to use
        loop_budget: the number of times to attempt to repeat the Think -> Act -> Observe loop

    Returns:
        A (ModelOutputThunk, Context) if `return_sampling_results` is `False`, else returns a `SamplingResult`.
    """
    assert isinstance(context, ChatContext), f"ReACT must use a type of chat context, got: {type(context)}"
    test_ctx_lin = context.view_for_generation()
    assert test_ctx_lin is not None and len(test_ctx_lin) == 0, (
        "ReACT expects a fresh context."
    )

    context = context.add(ReactInitiator(goal, tools, format=format))

    turn_num = 0
    while turn_num < loop_budget:
        turn_num += 1
        FancyLogger.get_logger().info(f"## ReACT TURN NUMBER {turn_num}")

        # TODO: JAL. Issue: It calls the tools here.
        thought, context = await mfuncs.aact(
            action=ReactThought(),
            context=context,
            backend=backend,
            model_options=model_options,
            tool_calls=True,
            strategy=None,
            requirements=[] # TODO: JAL. Do a no tool call requirement here...
        ) # type: ignore

        samples = await mfuncs.aact(
            action=ReactAction(),
            context=context,
            backend=backend,
            requirements=[], # TODO: JAL. Valid Tool Call Check
            strategy=RejectionSamplingStrategy(),
            return_sampling_results=True,
            model_options=model_options,
            tool_calls=True
        )

        if not samples.success:
            # TODO: JAL. Make this nicer.
            raise ValueError("could not generate correct action")

        action = samples.result
        assert action.tool_calls is not None, "tool calls should be populated, something went wrong with the requirement validation"

        context = samples.result_ctx  # type: ignore
        assert isinstance(context, ChatContext), "sampling strategy returned a non-ChatContext in ReACT"

        is_final = False
        tool_responses = mfuncs._call_tools(action, backend=backend)
        for tool_res in tool_responses:
            context = context.add(tool_res)
            if tool_res.name == "final_tool": # TODO: JAL. Make this nicer / write a function that has a constant.
                is_final = True

        if is_final:
            assert len(tool_responses) == 1, "multiple tools were called with 'final'"
            action._underlying_value = str(tool_responses[0].arguments) # TODO: clean this up so that it's either the actual answer or the expected format
            return action, context
            # TODO: JAL. have to convert to model output thunk...

    # TODO: JAL. Make this nicer.
    raise Exception("did not complete")
