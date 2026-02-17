"""Functions for Mellea operations like Instruct, Chat, etc..."""

from __future__ import annotations

import asyncio
from collections.abc import Coroutine
from typing import Any, Literal, overload

from PIL import Image as PILImage

from ..backends import FormatterBackend
from ..core import (
    Backend,
    BaseModelSubclass,
    CBlock,
    Component,
    Context,
    FancyLogger,
    GenerateLog,
    ImageBlock,
    ModelOutputThunk,
    Requirement,
    S,
    SamplingResult,
    SamplingStrategy,
    ValidationResult,
)
from ..helpers import _run_async_in_thread
from ..plugins.manager import has_plugins, invoke_hook
from ..plugins.types import MelleaHookType
from ..telemetry import set_span_attribute, trace_application
from .components import Instruction, Message, MObjectProtocol, ToolMessage, mify
from .context import SimpleContext
from .sampling import RejectionSamplingStrategy


@overload
def act(
    action: Component[S],
    context: Context,
    backend: Backend,
    *,
    requirements: list[Requirement] | None = None,
    strategy: SamplingStrategy | None = RejectionSamplingStrategy(loop_budget=2),
    return_sampling_results: Literal[False] = False,
    format: type[BaseModelSubclass] | None = None,
    model_options: dict | None = None,
    tool_calls: bool = False,
) -> tuple[ModelOutputThunk[S], Context]: ...


@overload
def act(
    action: Component[S],
    context: Context,
    backend: Backend,
    *,
    requirements: list[Requirement] | None = None,
    strategy: SamplingStrategy | None = RejectionSamplingStrategy(loop_budget=2),
    return_sampling_results: Literal[True],
    format: type[BaseModelSubclass] | None = None,
    model_options: dict | None = None,
    tool_calls: bool = False,
) -> SamplingResult[S]: ...


def act(
    action: Component[S],
    context: Context,
    backend: Backend,
    *,
    requirements: list[Requirement] | None = None,
    strategy: SamplingStrategy | None = RejectionSamplingStrategy(loop_budget=2),
    return_sampling_results: bool = False,
    format: type[BaseModelSubclass] | None = None,
    model_options: dict | None = None,
    tool_calls: bool = False,
) -> tuple[ModelOutputThunk[S], Context] | SamplingResult[S]:
    """Runs a generic action, and adds both the action and the result to the context.

    Args:
        action: the Component from which to generate.
        context: the context being used as a history from which to generate the response.
        backend: the backend used to generate the response.
        requirements: used as additional requirements when a sampling strategy is provided.
        strategy: a SamplingStrategy that describes the strategy for validating and repairing/retrying for the instruct-validate-repair pattern. None means that no particular sampling strategy is used.
        return_sampling_results: attach the (successful and failed) sampling attempts to the results.
        format: if set, the BaseModel to use for constrained decoding.
        model_options: additional model options, which will upsert into the model/backend's defaults.
        tool_calls: if true, tool calling is enabled.

    Returns:
        A (ModelOutputThunk, Context) if `return_sampling_results` is `False`, else returns a `SamplingResult`.
    """
    out = _run_async_in_thread(
        aact(
            action,
            context,
            backend,
            requirements=requirements,
            strategy=strategy,
            return_sampling_results=return_sampling_results,
            format=format,
            model_options=model_options,
            tool_calls=tool_calls,
            silence_context_type_warning=True,  # We can safely silence this here since it's in a sync function.
        )  # type: ignore[call-overload]
        # Mypy doesn't like the bool for return_sampling_results.
    )

    return out


@overload
def instruct(
    description: str,
    context: Context,
    backend: Backend,
    *,
    images: list[ImageBlock] | list[PILImage.Image] | None = None,
    requirements: list[Requirement | str] | None = None,
    icl_examples: list[str | CBlock] | None = None,
    grounding_context: dict[str, str | CBlock | Component] | None = None,
    user_variables: dict[str, str] | None = None,
    prefix: str | CBlock | None = None,
    output_prefix: str | CBlock | None = None,
    strategy: SamplingStrategy | None = RejectionSamplingStrategy(loop_budget=2),
    return_sampling_results: Literal[False] = False,
    format: type[BaseModelSubclass] | None = None,
    model_options: dict | None = None,
    tool_calls: bool = False,
) -> tuple[ModelOutputThunk[str], Context]: ...


@overload
def instruct(
    description: str,
    context: Context,
    backend: Backend,
    *,
    images: list[ImageBlock] | list[PILImage.Image] | None = None,
    requirements: list[Requirement | str] | None = None,
    icl_examples: list[str | CBlock] | None = None,
    grounding_context: dict[str, str | CBlock | Component] | None = None,
    user_variables: dict[str, str] | None = None,
    prefix: str | CBlock | None = None,
    output_prefix: str | CBlock | None = None,
    strategy: SamplingStrategy | None = RejectionSamplingStrategy(loop_budget=2),
    return_sampling_results: Literal[True],
    format: type[BaseModelSubclass] | None = None,
    model_options: dict | None = None,
    tool_calls: bool = False,
) -> SamplingResult[str]: ...


def instruct(
    description: str,
    context: Context,
    backend: Backend,
    *,
    images: list[ImageBlock] | list[PILImage.Image] | None = None,
    requirements: list[Requirement | str] | None = None,
    icl_examples: list[str | CBlock] | None = None,
    grounding_context: dict[str, str | CBlock | Component] | None = None,
    user_variables: dict[str, str] | None = None,
    prefix: str | CBlock | None = None,
    output_prefix: str | CBlock | None = None,
    strategy: SamplingStrategy | None = RejectionSamplingStrategy(loop_budget=2),
    return_sampling_results: bool = False,
    format: type[BaseModelSubclass] | None = None,
    model_options: dict | None = None,
    tool_calls: bool = False,
) -> tuple[ModelOutputThunk[str], Context] | SamplingResult[str]:
    """Generates from an instruction.

    Args:
        description: The description of the instruction.
        context: the context being used as a history from which to generate the response.
        backend: the backend used to generate the response.
        requirements: A list of requirements that the instruction can be validated against.
        icl_examples: A list of in-context-learning examples that the instruction can be validated against.
        grounding_context: A list of grounding contexts that the instruction can use. They can bind as variables using a (key: str, value: str | ContentBlock) tuple.
        user_variables: A dict of user-defined variables used to fill in Jinja placeholders in other parameters. This requires that all other provided parameters are provided as strings.
        prefix: A prefix string or ContentBlock to use when generating the instruction.
        output_prefix: A string or ContentBlock that defines a prefix for the output generation. Usually you do not need this.
        strategy: A SamplingStrategy that describes the strategy for validating and repairing/retrying for the instruct-validate-repair pattern. None means that no particular sampling strategy is used.
        return_sampling_results: attach the (successful and failed) sampling attempts to the results.
        format: If set, the BaseModel to use for constrained decoding.
        model_options: Additional model options, which will upsert into the model/backend's defaults.
        tool_calls: If true, tool calling is enabled.
        images: A list of images to be used in the instruction or None if none.

    Returns:
        A (ModelOutputThunk, Context) if `return_sampling_results` is `False`, else returns a `SamplingResult`.
    """
    requirements = [] if requirements is None else requirements
    icl_examples = [] if icl_examples is None else icl_examples
    grounding_context = dict() if grounding_context is None else grounding_context

    images = _parse_and_clean_image_args(images)

    # All instruction options are forwarded to create a new Instruction object.
    i = Instruction(
        description=description,
        requirements=requirements,
        icl_examples=icl_examples,
        grounding_context=grounding_context,
        user_variables=user_variables,
        prefix=prefix,
        output_prefix=output_prefix,
        images=images,
    )

    return act(
        i,
        context=context,
        backend=backend,
        requirements=i.requirements,
        strategy=strategy,
        return_sampling_results=return_sampling_results,
        format=format,
        model_options=model_options,
        tool_calls=tool_calls,
    )  # type: ignore[call-overload]


def chat(
    content: str,
    context: Context,
    backend: Backend,
    *,
    role: Message.Role = "user",
    images: list[ImageBlock] | list[PILImage.Image] | None = None,
    user_variables: dict[str, str] | None = None,
    format: type[BaseModelSubclass] | None = None,
    model_options: dict | None = None,
    tool_calls: bool = False,
) -> tuple[Message, Context]:
    """Sends a simple chat message and returns the response. Adds both messages to the Context."""
    if user_variables is not None:
        content_resolved = Instruction.apply_user_dict_from_jinja(
            user_variables, content
        )
    else:
        content_resolved = content
    images = _parse_and_clean_image_args(images)
    user_message = Message(role=role, content=content_resolved, images=images)

    result, new_ctx = act(
        user_message,
        context=context,
        backend=backend,
        strategy=None,  # Explicitly pass `None` since this can't pass requirements.
        format=format,
        model_options=model_options,
        tool_calls=tool_calls,
    )
    parsed_assistant_message = result.parsed_repr
    assert isinstance(parsed_assistant_message, Message)

    return parsed_assistant_message, new_ctx


def validate(
    reqs: Requirement | list[Requirement],
    context: Context,
    backend: Backend,
    *,
    output: CBlock | None = None,
    format: type[BaseModelSubclass] | None = None,
    model_options: dict | None = None,
    generate_logs: list[GenerateLog]
    | None = None,  # TODO: Can we get rid of gen logs here and in act?
    input: CBlock | None = None,
) -> list[ValidationResult]:
    """Validates a set of requirements over the output (if provided) or the current context (if the output is not provided)."""
    # Run everything in the specific event loop for this session.

    out = _run_async_in_thread(
        avalidate(
            reqs=reqs,
            context=context,
            backend=backend,
            output=output,
            format=format,
            model_options=model_options,
            generate_logs=generate_logs,
            input=input,
        )
    )

    # Wait for and return the result.
    return out


def query(
    obj: Any,
    query: str,
    context: Context,
    backend: Backend,
    *,
    format: type[BaseModelSubclass] | None = None,
    model_options: dict | None = None,
    tool_calls: bool = False,
) -> tuple[ModelOutputThunk, Context]:
    """Query method for retrieving information from an object.

    Args:
        obj : The object to be queried. It should be an instance of MObject or can be converted to one if necessary.
        query:  The string representing the query to be executed against the object.
        context: the context being used as a history from which to generate the response.
        backend: the backend used to generate the response.
        format:  format for output parsing.
        model_options: Model options to pass to the backend.
        tool_calls: If true, the model may make tool calls. Defaults to False.

    Returns:
        ModelOutputThunk: The result of the query as processed by the backend.
    """
    if not isinstance(obj, MObjectProtocol):
        obj = mify(obj)

    assert isinstance(obj, MObjectProtocol)
    q = obj.get_query_object(query)

    answer = act(
        q,
        context=context,
        backend=backend,
        strategy=None,  # Explicitly pass `None` since this can't pass requirements.
        format=format,
        model_options=model_options,
        tool_calls=tool_calls,
    )
    return answer


def transform(
    obj: Any,
    transformation: str,
    context: Context,
    backend: Backend,
    *,
    format: type[BaseModelSubclass] | None = None,
    model_options: dict | None = None,
) -> tuple[ModelOutputThunk | Any, Context]:
    """Transform method for creating a new object with the transformation applied.

    Args:
        obj: The object to be queried. It should be an instance of MObject or can be converted to one if necessary.
        transformation:  The string representing the query to be executed against the object.
        context: the context being used as a history from which to generate the response.
        backend: the backend used to generate the response.
        format: format for output parsing; usually not needed with transform.
        model_options: Model options to pass to the backend.

    Returns:
        (ModelOutputThunk | Any, Context): The result of the transformation as processed by the backend. If no tools were called,
        the return type will be always be (ModelOutputThunk, Context). If a tool was called, the return type will be the return type
        of the function called, usually the type of the object passed in.
    """
    if not isinstance(obj, MObjectProtocol):
        obj = mify(obj)

    assert isinstance(obj, MObjectProtocol)
    t = obj.get_transform_object(transformation)

    # Check that your model / backend supports tool calling.
    # This might throw an error when tools are provided but can't be handled by one or the other.
    transformed, new_ctx = act(
        t,
        context=context,
        backend=backend,
        strategy=None,  # Explicitly pass `None` since this can't pass requirements.
        format=format,
        model_options=model_options,
        tool_calls=True,
    )

    tools = _call_tools(transformed, backend)

    # Transform only supports calling one tool call since it cannot currently synthesize multiple outputs.
    # Attempt to choose the best one to call.
    chosen_tool: ToolMessage | None = None
    if len(tools) == 1:
        # Only one function was called. Choose that one.
        chosen_tool = tools[0]

    elif len(tools) > 1:
        for output in tools:
            if type(output._tool_output) is type(obj):
                chosen_tool = output
                break

        if chosen_tool is None:
            chosen_tool = tools[0]

        FancyLogger.get_logger().warning(
            f"multiple tool calls returned in transform of {obj} with description '{transformation}'; picked `{chosen_tool.name}`"
            # type: ignore
        )

    if chosen_tool:
        # Tell the user the function they should've called if no generated values were added.
        if len(chosen_tool._tool.args.keys()) == 0:
            FancyLogger.get_logger().warning(
                f"the transform of {obj} with transformation description '{transformation}' resulted in a tool call with no generated arguments; consider calling the function `{chosen_tool._tool.name}` directly"
            )

        new_ctx.add(chosen_tool)
        FancyLogger.get_logger().info(
            "added a tool message from transform to the context"
        )
        return chosen_tool._tool_output, new_ctx

    return transformed, new_ctx


@overload
async def aact(
    action: Component[S],
    context: Context,
    backend: Backend,
    *,
    requirements: list[Requirement] | None = None,
    strategy: SamplingStrategy | None = RejectionSamplingStrategy(loop_budget=2),
    return_sampling_results: Literal[False] = False,
    format: type[BaseModelSubclass] | None = None,
    model_options: dict | None = None,
    tool_calls: bool = False,
    silence_context_type_warning: bool = False,
) -> tuple[ModelOutputThunk[S], Context]: ...


@overload
async def aact(
    action: Component[S],
    context: Context,
    backend: Backend,
    *,
    requirements: list[Requirement] | None = None,
    strategy: SamplingStrategy | None = RejectionSamplingStrategy(loop_budget=2),
    return_sampling_results: Literal[True],
    format: type[BaseModelSubclass] | None = None,
    model_options: dict | None = None,
    tool_calls: bool = False,
    silence_context_type_warning: bool = False,
) -> SamplingResult[S]: ...


async def aact(
    action: Component[S],
    context: Context,
    backend: Backend,
    *,
    requirements: list[Requirement] | None = None,
    strategy: SamplingStrategy | None = RejectionSamplingStrategy(loop_budget=2),
    return_sampling_results: bool = False,
    format: type[BaseModelSubclass] | None = None,
    model_options: dict | None = None,
    tool_calls: bool = False,
    silence_context_type_warning: bool = False,
) -> tuple[ModelOutputThunk[S], Context] | SamplingResult:
    """Asynchronous version of .act; runs a generic action, and adds both the action and the result to the context.

    Args:
        action: the Component from which to generate.
        context: the context being used as a history from which to generate the response.
        backend: the backend used to generate the response.
        requirements: used as additional requirements when a sampling strategy is provided
        strategy: a SamplingStrategy that describes the strategy for validating and repairing/retrying for the instruct-validate-repair pattern. None means that no particular sampling strategy is used.
        return_sampling_results: attach the (successful and failed) sampling attempts to the results.
        format: if set, the BaseModel to use for constrained decoding.
        model_options: additional model options, which will upsert into the model/backend's defaults.
        tool_calls: if true, tool calling is enabled.
        silence_context_type_warning: if called directly from an asynchronous function, will log a warning if not using a SimpleContext

    Returns:
        A (ModelOutputThunk, Context) if `return_sampling_results` is `False`, else returns a `SamplingResult`.
    """
    import time
    import traceback

    with trace_application(
        "aact",
        action_type=action.__class__.__name__,
        has_requirements=requirements is not None and len(requirements) > 0,
        has_strategy=strategy is not None,
        strategy_type=strategy.__class__.__name__ if strategy else None,
        has_format=format is not None,
        tool_calls=tool_calls,
    ) as span:
        if not silence_context_type_warning and not isinstance(context, SimpleContext):
            FancyLogger().get_logger().warning(
                "Not using a SimpleContext with asynchronous requests could cause unexpected results due to stale contexts. Ensure you await between requests."
                "\nSee the async section of the tutorial: https://github.com/generative-computing/mellea/blob/main/docs/tutorial.md#chapter-12-asynchronicity"
            )

        # --- component_pre_execute hook ---
        if has_plugins():
            from ..plugins.hooks.component import ComponentPreExecutePayload

            pre_exec_payload = ComponentPreExecutePayload(
                component_type=type(action).__name__,
                action=action,
                context=context,
                requirements=requirements or [],
                model_options=model_options or {},
                format=format,
                strategy=strategy,
                tool_calls_enabled=tool_calls,
            )
            _, pre_exec_payload = await invoke_hook(
                MelleaHookType.COMPONENT_PRE_EXECUTE,
                pre_exec_payload,
                backend=backend,
                context=context,
            )
            action = pre_exec_payload.action
            context = pre_exec_payload.context
            requirements = pre_exec_payload.requirements or requirements
            model_options = pre_exec_payload.model_options or model_options
            format = pre_exec_payload.format
            strategy = pre_exec_payload.strategy
            tool_calls = pre_exec_payload.tool_calls_enabled

        t0 = time.monotonic()

        try:
            sampling_result: SamplingResult | None = None
            generate_logs: list[GenerateLog] = []

            if return_sampling_results:
                assert strategy is not None, (
                    "Must provide a SamplingStrategy when return_sampling_results==True"
                )

            if strategy is None:
                # Only use the strategy if one is provided. Add a warning if requirements were passed in though.
                if requirements is not None and len(requirements) > 0:
                    FancyLogger.get_logger().warning(
                        "Calling the function with NO strategy BUT requirements. No requirement is being checked!"
                    )

                result, new_ctx = await backend.generate_from_context_with_hooks(
                    action,
                    ctx=context,
                    format=format,
                    model_options=model_options,
                    tool_calls=tool_calls,
                )
                await result.avalue()

                # ._generate_log should never be None after generation.
                assert result._generate_log is not None
                result._generate_log.is_final_result = True
                generate_logs.append(result._generate_log)

            else:
                # Always sample if a strategy is provided, even if no requirements were provided.
                # Some sampling strategies don't use requirements or set them when instantiated.

                sampling_result = await strategy.sample(
                    action,
                    context=context,
                    backend=backend,
                    requirements=requirements,
                    validation_ctx=None,
                    format=format,
                    model_options=model_options,
                    tool_calls=tool_calls,
                )

                assert sampling_result.sample_generations is not None
                for result in sampling_result.sample_generations:
                    assert (
                        result._generate_log is not None
                    )  # Cannot be None after generation.
                    generate_logs.append(result._generate_log)

                new_ctx = sampling_result.result_ctx
                result = sampling_result.result
                assert sampling_result.result._generate_log is not None
                assert sampling_result.result._generate_log.is_final_result, (
                    "generate logs from the final result returned by the sampling strategy must be marked as final"
                )

            # Add span attributes for the result
            set_span_attribute(span, "num_generate_logs", len(generate_logs))
            if sampling_result:
                set_span_attribute(
                    span, "sampling_success", bool(sampling_result.result)
                )

            # Log the model response (truncated for large responses)
            try:
                response_value = (
                    str(result.value)
                    if hasattr(result, "value") and result.value
                    else str(result)
                )
                # Truncate to 500 chars to avoid overwhelming trace storage
                if len(response_value) > 500:
                    response_value = response_value[:500] + "..."
                set_span_attribute(span, "response", response_value)
                set_span_attribute(
                    span,
                    "response_length",
                    len(str(result.value) if hasattr(result, "value") else str(result)),
                )
            except Exception:
                # If we can't get the response, don't fail the trace
                pass

            # --- component_post_success hook ---
            if has_plugins():
                from ..plugins.hooks.component import ComponentPostSuccessPayload

                success_payload = ComponentPostSuccessPayload(
                    component_type=type(action).__name__,
                    action=action,
                    result=result,
                    context_before=context,
                    context_after=new_ctx,
                    generate_log=generate_logs[-1] if generate_logs else None,
                    sampling_results=sampling_result.sample_generations
                    if sampling_result
                    else None,
                    latency_ms=int((time.monotonic() - t0) * 1000),
                )
                _, success_payload = await invoke_hook(
                    MelleaHookType.COMPONENT_POST_SUCCESS,
                    success_payload,
                    backend=backend,
                    context=new_ctx,
                )
                if (
                    success_payload.result is not None
                    and success_payload.result is not result
                ):
                    result = success_payload.result

            if return_sampling_results:
                assert (
                    sampling_result is not None
                )  # Needed for the type checker but should never happen.
                return sampling_result
            else:
                return result, new_ctx

        except Exception as exc:
            # --- component_post_error hook ---
            if has_plugins():
                from ..plugins.hooks.component import ComponentPostErrorPayload

                error_payload = ComponentPostErrorPayload(
                    component_type=type(action).__name__,
                    action=action,
                    error=exc,
                    error_type=type(exc).__name__,
                    stack_trace=traceback.format_exc(),
                    context=context,
                    model_options=model_options or {},
                )
                await invoke_hook(
                    MelleaHookType.COMPONENT_POST_ERROR,
                    error_payload,
                    backend=backend,
                    context=context,
                    violations_as_exceptions=False,
                )
            raise


@overload
async def ainstruct(
    description: str,
    context: Context,
    backend: Backend,
    *,
    images: list[ImageBlock] | list[PILImage.Image] | None = None,
    requirements: list[Requirement | str] | None = None,
    icl_examples: list[str | CBlock] | None = None,
    grounding_context: dict[str, str | CBlock | Component] | None = None,
    user_variables: dict[str, str] | None = None,
    prefix: str | CBlock | None = None,
    output_prefix: str | CBlock | None = None,
    strategy: SamplingStrategy | None = RejectionSamplingStrategy(loop_budget=2),
    return_sampling_results: Literal[False] = False,
    format: type[BaseModelSubclass] | None = None,
    model_options: dict | None = None,
    tool_calls: bool = False,
) -> tuple[ModelOutputThunk[str], Context]: ...


@overload
async def ainstruct(
    description: str,
    context: Context,
    backend: Backend,
    *,
    images: list[ImageBlock] | list[PILImage.Image] | None = None,
    requirements: list[Requirement | str] | None = None,
    icl_examples: list[str | CBlock] | None = None,
    grounding_context: dict[str, str | CBlock | Component] | None = None,
    user_variables: dict[str, str] | None = None,
    prefix: str | CBlock | None = None,
    output_prefix: str | CBlock | None = None,
    strategy: SamplingStrategy | None = RejectionSamplingStrategy(loop_budget=2),
    return_sampling_results: Literal[True],
    format: type[BaseModelSubclass] | None = None,
    model_options: dict | None = None,
    tool_calls: bool = False,
) -> SamplingResult[S]: ...


async def ainstruct(
    description: str,
    context: Context,
    backend: Backend,
    *,
    images: list[ImageBlock] | list[PILImage.Image] | None = None,
    requirements: list[Requirement | str] | None = None,
    icl_examples: list[str | CBlock] | None = None,
    grounding_context: dict[str, str | CBlock | Component] | None = None,
    user_variables: dict[str, str] | None = None,
    prefix: str | CBlock | None = None,
    output_prefix: str | CBlock | None = None,
    strategy: SamplingStrategy | None = RejectionSamplingStrategy(loop_budget=2),
    return_sampling_results: bool = False,
    format: type[BaseModelSubclass] | None = None,
    model_options: dict | None = None,
    tool_calls: bool = False,
) -> tuple[ModelOutputThunk[str], Context] | SamplingResult:
    """Generates from an instruction.

    Args:
        description: The description of the instruction.
        context: the context being used as a history from which to generate the response.
        backend: the backend used to generate the response.
        requirements: A list of requirements that the instruction can be validated against.
        icl_examples: A list of in-context-learning examples that the instruction can be validated against.
        grounding_context: A list of grounding contexts that the instruction can use. They can bind as variables using a (key: str, value: str | ContentBlock) tuple.
        user_variables: A dict of user-defined variables used to fill in Jinja placeholders in other parameters. This requires that all other provided parameters are provided as strings.
        prefix: A prefix string or ContentBlock to use when generating the instruction.
        output_prefix: A string or ContentBlock that defines a prefix for the output generation. Usually you do not need this.
        strategy: A SamplingStrategy that describes the strategy for validating and repairing/retrying for the instruct-validate-repair pattern. None means that no particular sampling strategy is used.
        return_sampling_results: attach the (successful and failed) sampling attempts to the results.
        format: If set, the BaseModel to use for constrained decoding.
        model_options: Additional model options, which will upsert into the model/backend's defaults.
        tool_calls: If true, tool calling is enabled.
        images: A list of images to be used in the instruction or None if none.

    Returns:
        A (ModelOutputThunk, Context) if `return_sampling_results` is `False`, else returns a `SamplingResult`.
    """
    requirements = [] if requirements is None else requirements
    icl_examples = [] if icl_examples is None else icl_examples
    grounding_context = dict() if grounding_context is None else grounding_context

    images = _parse_and_clean_image_args(images)

    # --- component_pre_create hook ---
    if has_plugins():
        from ..plugins.hooks.component import ComponentPreCreatePayload

        pre_payload = ComponentPreCreatePayload(
            component_type="Instruction",
            description=description,
            images=images,
            requirements=requirements,
            icl_examples=icl_examples,
            grounding_context=grounding_context,
            user_variables=user_variables,
            prefix=prefix,
        )
        _, pre_payload = await invoke_hook(
            MelleaHookType.COMPONENT_PRE_CREATE,
            pre_payload,
            backend=backend,
            context=context,
        )
        description = pre_payload.description
        images = pre_payload.images
        requirements = pre_payload.requirements
        icl_examples = pre_payload.icl_examples
        grounding_context = pre_payload.grounding_context
        user_variables = pre_payload.user_variables
        prefix = pre_payload.prefix

    # All instruction options are forwarded to create a new Instruction object.
    i = Instruction(
        description=description,
        requirements=requirements,
        icl_examples=icl_examples,
        grounding_context=grounding_context,
        user_variables=user_variables,
        prefix=prefix,
        output_prefix=output_prefix,
        images=images,
    )

    # --- component_post_create hook ---
    if has_plugins():
        from ..plugins.hooks.component import ComponentPostCreatePayload

        post_payload = ComponentPostCreatePayload(
            component_type="Instruction", component=i
        )
        _, post_payload = await invoke_hook(
            MelleaHookType.COMPONENT_POST_CREATE,
            post_payload,
            backend=backend,
            context=context,
        )
        if post_payload.component is not None and post_payload.component is not i:
            i = post_payload.component

    return await aact(
        i,
        context=context,
        backend=backend,
        requirements=i.requirements,
        strategy=strategy,
        return_sampling_results=return_sampling_results,
        format=format,
        model_options=model_options,
        tool_calls=tool_calls,
    )  # type: ignore[call-overload]


async def achat(
    content: str,
    context: Context,
    backend: Backend,
    *,
    role: Message.Role = "user",
    images: list[ImageBlock] | list[PILImage.Image] | None = None,
    user_variables: dict[str, str] | None = None,
    format: type[BaseModelSubclass] | None = None,
    model_options: dict | None = None,
    tool_calls: bool = False,
) -> tuple[Message, Context]:
    """Sends a simple chat message and returns the response. Adds both messages to the Context."""
    if user_variables is not None:
        content_resolved = Instruction.apply_user_dict_from_jinja(
            user_variables, content
        )
    else:
        content_resolved = content
    images = _parse_and_clean_image_args(images)

    # --- component_pre_create hook ---
    if has_plugins():
        from ..plugins.hooks.component import ComponentPreCreatePayload

        pre_payload = ComponentPreCreatePayload(
            component_type="Message", description=content_resolved, images=images
        )
        _, pre_payload = await invoke_hook(
            MelleaHookType.COMPONENT_PRE_CREATE,
            pre_payload,
            backend=backend,
            context=context,
        )
        content_resolved = pre_payload.description
        images = pre_payload.images

    user_message = Message(role=role, content=content_resolved, images=images)

    # --- component_post_create hook ---
    if has_plugins():
        from ..plugins.hooks.component import ComponentPostCreatePayload

        post_payload = ComponentPostCreatePayload(
            component_type="Message", component=user_message
        )
        _, post_payload = await invoke_hook(
            MelleaHookType.COMPONENT_POST_CREATE,
            post_payload,
            backend=backend,
            context=context,
        )
        if (
            post_payload.component is not None
            and post_payload.component is not user_message
        ):
            user_message = post_payload.component

    result, new_ctx = await aact(
        user_message,
        context=context,
        backend=backend,
        strategy=None,  # Explicitly pass `None` since this can't pass requirements.
        format=format,
        model_options=model_options,
        tool_calls=tool_calls,
    )
    parsed_assistant_message = result.parsed_repr
    assert isinstance(parsed_assistant_message, Message)

    return parsed_assistant_message, new_ctx


async def avalidate(
    reqs: Requirement | list[Requirement],
    context: Context,
    backend: Backend,
    *,
    output: CBlock | None = None,
    format: type[BaseModelSubclass] | None = None,
    model_options: dict | None = None,
    generate_logs: list[GenerateLog] | None = None,
    input: CBlock | None = None,
) -> list[ValidationResult]:
    """Asynchronous version of .validate; validates a set of requirements over the output (if provided) or the current context (if the output is not provided)."""
    # Turn a solitary requirement in to a list of requirements, and then reqify if needed.
    reqs = [reqs] if not isinstance(reqs, list) else reqs
    reqs = [Requirement(req) if type(req) is str else req for req in reqs]

    if output is None:
        validation_target_ctx = context
    else:
        validation_target_ctx = SimpleContext()

        # Add the input/output to the validation context
        if input is not None:
            validation_target_ctx = validation_target_ctx.add(input)
        validation_target_ctx = validation_target_ctx.add(output)

    # --- validation_pre_check hook ---
    if has_plugins():
        from ..plugins.hooks.validation import ValidationPreCheckPayload

        pre_payload = ValidationPreCheckPayload(
            requirements=reqs,
            target=output,
            context=validation_target_ctx,
            model_options=model_options or {},
        )
        _, pre_payload = await invoke_hook(
            MelleaHookType.VALIDATION_PRE_CHECK,
            pre_payload,
            backend=backend,
            context=context,
        )
        reqs = pre_payload.requirements
        model_options = pre_payload.model_options or model_options

    rvs: list[ValidationResult] = []
    coroutines: list[Coroutine[Any, Any, ValidationResult]] = []

    for requirement in reqs:
        val_result_co = requirement.validate(
            backend, validation_target_ctx, format=format, model_options=model_options
        )
        coroutines.append(val_result_co)

    for val_result in await asyncio.gather(*coroutines):
        rvs.append(val_result)

        # If the validator utilized a backend to generate a result, attach the corresponding
        # info to the generate_logs list.
        if generate_logs is not None:
            if val_result.thunk is not None:
                thunk = val_result.thunk
                assert (
                    thunk._generate_log is not None
                )  # Cannot be None after generation.
                generate_logs.append(thunk._generate_log)
            else:
                # We have to append None here so that the logs line-up.
                # TODO: A better solution should be found for this edge case.
                #       This is the only scenario where ValidationResults are supposed to line
                #       up with GenerateLogs.
                generate_logs.append(None)  # type: ignore

    # --- validation_post_check hook ---
    if has_plugins():
        from ..plugins.hooks.validation import ValidationPostCheckPayload

        post_payload = ValidationPostCheckPayload(
            requirements=reqs,
            results=rvs,
            all_passed=all(bool(r) for r in rvs),
            passed_count=sum(1 for r in rvs if bool(r)),
            failed_count=sum(1 for r in rvs if not bool(r)),
        )
        _, post_payload = await invoke_hook(
            MelleaHookType.VALIDATION_POST_CHECK,
            post_payload,
            backend=backend,
            context=context,
        )
        rvs = post_payload.results

    return rvs


async def aquery(
    obj: Any,
    query: str,
    context: Context,
    backend: Backend,
    *,
    format: type[BaseModelSubclass] | None = None,
    model_options: dict | None = None,
    tool_calls: bool = False,
) -> tuple[ModelOutputThunk, Context]:
    """Query method for retrieving information from an object.

    Args:
        obj : The object to be queried. It should be an instance of MObject or can be converted to one if necessary.
        query:  The string representing the query to be executed against the object.
        context: the context being used as a history from which to generate the response.
        backend: the backend used to generate the response.
        format:  format for output parsing.
        model_options: Model options to pass to the backend.
        tool_calls: If true, the model may make tool calls. Defaults to False.

    Returns:
        ModelOutputThunk: The result of the query as processed by the backend.
    """
    if not isinstance(obj, MObjectProtocol):
        obj = mify(obj)

    assert isinstance(obj, MObjectProtocol)
    q = obj.get_query_object(query)

    answer = await aact(
        q,
        context=context,
        backend=backend,
        strategy=None,  # Explicitly pass `None` since this can't pass requirements.
        format=format,
        model_options=model_options,
        tool_calls=tool_calls,
    )
    return answer


async def atransform(
    obj: Any,
    transformation: str,
    context: Context,
    backend: Backend,
    *,
    format: type[BaseModelSubclass] | None = None,
    model_options: dict | None = None,
) -> tuple[ModelOutputThunk | Any, Context]:
    """Transform method for creating a new object with the transformation applied.

    Args:
        obj: The object to be queried. It should be an instance of MObject or can be converted to one if necessary.
        transformation:  The string representing the query to be executed against the object.
        context: the context being used as a history from which to generate the response.
        backend: the backend used to generate the response.
        format: format for output parsing; usually not needed with transform.
        model_options: Model options to pass to the backend.

    Returns:
        ModelOutputThunk|Any: The result of the transformation as processed by the backend. If no tools were called,
        the return type will be always be ModelOutputThunk. If a tool was called, the return type will be the return type
        of the function called, usually the type of the object passed in.
    """
    if not isinstance(obj, MObjectProtocol):
        obj = mify(obj)

    assert isinstance(obj, MObjectProtocol)
    t = obj.get_transform_object(transformation)

    # Check that your model / backend supports tool calling.
    # This might throw an error when tools are provided but can't be handled by one or the other.
    transformed, new_ctx = await aact(
        t,
        context=context,
        backend=backend,
        strategy=None,  # Explicitly pass `None` since this can't pass requirements.
        format=format,
        model_options=model_options,
        tool_calls=True,
    )

    tools = _call_tools(transformed, backend)

    # Transform only supports calling one tool call since it cannot currently synthesize multiple outputs.
    # Attempt to choose the best one to call.
    chosen_tool: ToolMessage | None = None
    if len(tools) == 1:
        # Only one function was called. Choose that one.
        chosen_tool = tools[0]

    elif len(tools) > 1:
        for output in tools:
            if type(output._tool_output) is type(obj):
                chosen_tool = output
                break

        if chosen_tool is None:
            chosen_tool = tools[0]

        FancyLogger.get_logger().warning(
            f"multiple tool calls returned in transform of {obj} with description '{transformation}'; picked `{chosen_tool.name}`"
            # type: ignore
        )

    if chosen_tool:
        # Tell the user the function they should've called if no generated values were added.
        if len(chosen_tool._tool.args.keys()) == 0:
            FancyLogger.get_logger().warning(
                f"the transform of {obj} with transformation description '{transformation}' resulted in a tool call with no generated arguments; consider calling the function `{chosen_tool._tool.name}` directly"
            )

        new_ctx.add(chosen_tool)
        FancyLogger.get_logger().info(
            "added a tool message from transform to the context"
        )
        return chosen_tool._tool_output, new_ctx

    return transformed, new_ctx


def _parse_and_clean_image_args(
    images_: list[ImageBlock] | list[PILImage.Image] | None = None,
) -> list[ImageBlock] | None:
    images: list[ImageBlock] | None = None
    if images_ is not None:
        assert isinstance(images_, list), "Images should be a list or None."

        if len(images_) > 0:
            if isinstance(images_[0], PILImage.Image):
                images = [
                    ImageBlock.from_pil_image(i)
                    for i in images_
                    if isinstance(i, PILImage.Image)
                ]
            else:
                images = images_  # type: ignore
            assert isinstance(images, list)
            assert all(isinstance(i, ImageBlock) for i in images), (
                "All images should be ImageBlocks now."
            )
        else:
            images = None
    return images


def _call_tools(result: ModelOutputThunk, backend: Backend) -> list[ToolMessage]:
    """Call all the tools requested in a result's tool calls object.

    Returns:
        list[ToolMessage]: A list of tool messages that can be empty.
    """
    # There might be multiple tool calls returned.
    outputs: list[ToolMessage] = []
    tool_calls = result.tool_calls
    if tool_calls:
        # Call the tools and decide what to do.
        for name, tool in tool_calls.items():
            try:
                output = tool.call_func()
            except Exception as e:
                output = e

            # Default to the output. Attempt to properly print it. If that doesn't result in a str,
            # stringify it.
            content = str(output)
            if isinstance(backend, FormatterBackend):
                content = backend.formatter.print(output)  # type: ignore
                content = str(content)

            outputs.append(
                ToolMessage(
                    role="tool",
                    content=content,
                    tool_output=output,
                    name=name,
                    args=tool.args,
                    tool=tool,
                )
            )
    return outputs
