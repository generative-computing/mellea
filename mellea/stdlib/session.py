"""Mellea Sessions."""

from __future__ import annotations

import asyncio
import contextvars
import threading
from collections.abc import Coroutine
from copy import deepcopy
from typing import Any, Literal, overload

from PIL import Image as PILImage

from mellea.backends import Backend, BaseModelSubclass
from mellea.backends.formatter import FormatterBackend
from mellea.backends.model_ids import (
    IBM_GRANITE_3_2_8B,
    IBM_GRANITE_3_3_8B,
    ModelIdentifier,
)
from mellea.backends.ollama import OllamaModelBackend
from mellea.backends.openai import OpenAIBackend
from mellea.helpers.fancy_logger import FancyLogger
from mellea.stdlib.base import (
    CBlock,
    Component,
    Context,
    ContextTurn,
    GenerateLog,
    ImageBlock,
    LinearContext,
    ModelOutputThunk,
    SimpleContext,
)
from mellea.stdlib.chat import Message, ToolMessage
from mellea.stdlib.instruction import Instruction
from mellea.stdlib.mify import mify
from mellea.stdlib.mobject import MObjectProtocol
from mellea.stdlib.requirement import Requirement, ValidationResult, check, req
from mellea.stdlib.sampling import SamplingResult, SamplingStrategy

# Global context variable for the context session
_context_session: contextvars.ContextVar[MelleaSession | None] = contextvars.ContextVar(
    "context_session", default=None
)


def get_session() -> MelleaSession:
    """Get the current session from context.

    Raises:
        RuntimeError: If no session is currently active.
    """
    session = _context_session.get()
    if session is None:
        raise RuntimeError(
            "No active session found. Use 'with start_session(...):' to create one."
        )
    return session


def backend_name_to_class(name: str) -> Any:
    """Resolves backend names to Backend classes."""
    if name == "ollama":
        return OllamaModelBackend
    elif name == "hf" or name == "huggingface":
        from mellea.backends.huggingface import LocalHFBackend

        return LocalHFBackend
    elif name == "openai":
        return OpenAIBackend
    elif name == "watsonx":
        from mellea.backends.watsonx import WatsonxAIBackend

        return WatsonxAIBackend
    elif name == "litellm":
        from mellea.backends.litellm import LiteLLMBackend

        return LiteLLMBackend
    else:
        return None


def start_session(
    backend_name: Literal["ollama", "hf", "openai", "watsonx", "litellm"] = "ollama",
    model_id: str | ModelIdentifier = IBM_GRANITE_3_3_8B,
    ctx: Context | None = SimpleContext(),
    *,
    model_options: dict | None = None,
    **backend_kwargs,
) -> MelleaSession:
    """Start a new Mellea session. Can be used as a context manager or called directly.

    This function creates and configures a new Mellea session with the specified backend
    and model. When used as a context manager (with `with` statement), it automatically
    sets the session as the current active session for use with convenience functions
    like `instruct()`, `chat()`, `query()`, and `transform()`. When called directly,
    it returns a session object that can be used directly.

    Args:
        backend_name: The backend to use. Options are:
            - "ollama": Use Ollama backend for local models
            - "hf" or "huggingface": Use HuggingFace transformers backend
            - "openai": Use OpenAI API backend
            - "watsonx": Use IBM WatsonX backend
        model_id: Model identifier or name. Can be a `ModelIdentifier` from
            mellea.backends.model_ids or a string model name.
        ctx: Context manager for conversation history. Defaults to SimpleContext().
            Use LinearContext() for chat-style conversations.
        model_options: Additional model configuration options that will be passed
            to the backend (e.g., temperature, max_tokens, etc.).
        **backend_kwargs: Additional keyword arguments passed to the backend constructor.

    Returns:
        MelleaSession: A session object that can be used as a context manager
        or called directly with session methods.

    Usage:
        # As a context manager (sets global session):
        with start_session("ollama", "granite3.3:8b") as session:
            result = instruct("Generate a story")  # Uses current session
            # session is also available directly
            other_result = session.chat("Hello")

        # Direct usage (no global session set):
        session = start_session("ollama", "granite3.3:8b")
        result = session.instruct("Generate a story")
        # Remember to call session.cleanup() when done
        session.cleanup()

    Examples:
        # Basic usage with default settings
        with start_session() as session:
            response = instruct("Explain quantum computing")

        # Using OpenAI with custom model options
        with start_session("openai", "gpt-4", model_options={"temperature": 0.7}):
            response = chat("Write a poem")

        # Using HuggingFace with LinearContext for conversations
        from mellea.stdlib.base import LinearContext
        with start_session("hf", "microsoft/DialoGPT-medium", ctx=LinearContext()):
            chat("Hello!")
            chat("How are you?")  # Remembers previous message

        # Direct usage without context manager
        session = start_session()
        response = session.instruct("Explain quantum computing")
        session.cleanup()
    """
    backend_class = backend_name_to_class(backend_name)
    if backend_class is None:
        raise Exception(
            f"Backend name {backend_name} unknown. Please see the docstring for `mellea.stdlib.session.start_session` for a list of options."
        )
    assert backend_class is not None
    backend = backend_class(model_id, model_options=model_options, **backend_kwargs)
    return MelleaSession(backend, ctx)


class MelleaSession:
    """Mellea sessions are a THIN wrapper around `m` convenience functions with NO special semantics.

    Using a Mellea session is not required, but it does represent the "happy path" of Mellea programming. Some nice things about ussing a `MelleaSession`:
    1. In most cases you want to keep a Context together with the Backend from which it came.
    2. You can directly run an instruction or a send a chat, instead of first creating the `Instruction` or `Chat` object and then later calling backend.generate on the object.
    3. The context is "threaded-through" for you, which allows you to issue a sequence of commands instead of first calling backend.generate on something and then appending it to your context.

    These are all relatively simple code hygiene and state management benefits, but they add up over time.
    If you are doing complicating programming (e.g., non-trivial inference scaling) then you might be better off forgoing `MelleaSession`s and managing your Context and Backend directly.

    Note: we put the `instruct`, `validate`, and other convenience functions here instead of in `Context` or `Backend` to avoid import resolution issues.
    """

    def __init__(self, backend: Backend, ctx: Context | None = None):
        """Initializes a new Mellea session with the provided backend and context.

        Args:
            backend (Backend): This is always required.
            ctx (Context): The way in which the model's context will be managed. By default, each interaction with the model is a stand-alone interaction, so we use SimpleContext as the default.
            model_options (Optional[dict]): model options, which will upsert into the model/backend's defaults.
        """
        self.backend = backend
        self.ctx = ctx if ctx is not None else SimpleContext()
        self._backend_stack: list[tuple[Backend, dict | None]] = []
        self._session_logger = FancyLogger.get_logger()
        self._context_token = None

        # Necessary for async. `m.*` functions should always run in this event loop.
        self._event_loop = asyncio.new_event_loop()
        threading.Thread(target=self._event_loop.run_forever, daemon=True).start()

    def __enter__(self):
        """Enter context manager and set this session as the current global session."""
        self._context_token = _context_session.set(self)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager and cleanup session."""
        self.cleanup()
        if self._context_token is not None:
            _context_session.reset(self._context_token)
            self._context_token = None

    def _push_model_state(self, new_backend: Backend, new_model_opts: dict):
        """The backend and model options used within a `Context` can be temporarily changed. This method changes the model's backend and model_opts, while saving the current settings in the `self._backend_stack`.

        Question: should this logic be moved into context? I really want to keep `Session` as simple as possible... see true motivation in the docstring for the class.
        """
        self._backend_stack.append((self.backend, self.model_options))
        self.backend = new_backend
        self.opts = new_model_opts

    def _pop_model_state(self) -> bool:
        """Pops the model state.

        The backend and model options used within a `Context` can be temporarily changed by pushing and popping from the model state.
        This function restores the model's previous backend and model_opts from the `self._backend_stack`.

        Question: should this logic be moved into context? I really want to keep `Session` as simple as possible... see true motivation in the docstring for the class.
        """
        try:
            b, b_model_opts = self._backend_stack.pop()
            self.backend = b
            self.model_options = b_model_opts
            return True
        except Exception:
            return False

    def reset(self):
        """Reset the context state."""
        self.ctx.reset()

    def cleanup(self) -> None:
        """Clean up session resources."""
        self.reset()
        self._backend_stack.clear()
        if hasattr(self.backend, "close"):
            self.backend.close()  # type: ignore

    def summarize(self) -> ModelOutputThunk:
        """Summarizes the current context."""
        raise NotImplementedError()

    @staticmethod
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

    @overload
    def act(
        self,
        action: Component,
        *,
        strategy: SamplingStrategy | None = None,
        return_sampling_results: Literal[False] = False,
        format: type[BaseModelSubclass] | None = None,
        model_options: dict | None = None,
        tool_calls: bool = False,
        stream: bool = False,
    ) -> ModelOutputThunk: ...

    @overload
    def act(
        self,
        action: Component,
        *,
        strategy: SamplingStrategy | None = None,
        return_sampling_results: Literal[True],
        format: type[BaseModelSubclass] | None = None,
        model_options: dict | None = None,
        tool_calls: bool = False,
        stream: bool = False,
    ) -> SamplingResult: ...

    def act(
        self,
        action: Component,
        *,
        requirements: list[Requirement] | None = None,
        strategy: SamplingStrategy | None = None,
        return_sampling_results: bool = False,
        format: type[BaseModelSubclass] | None = None,
        model_options: dict | None = None,
        tool_calls: bool = False,
        stream: bool = False,
    ) -> ModelOutputThunk | SamplingResult:
        """Runs a generic action, and adds both the action and the result to the context.

        Args:
            action: the Component from which to generate.
            requirements: used as additional requirements when a sampling strategy is provided
            strategy: a SamplingStrategy that describes the strategy for validating and repairing/retrying for the instruct-validate-repair pattern. None means that no particular sampling strategy is used.
            return_sampling_results: attach the (successful and failed) sampling attempts to the results.
            format: if set, the BaseModel to use for constrained decoding.
            model_options: additional model options, which will upsert into the model/backend's defaults.
            tool_calls: if true, tool calling is enabled.

        Returns:
            A ModelOutputThunk if `return_sampling_results` is `False`, else returns a `SamplingResult`.
        """

        # Run everything in the specific event loop for this session.
        out = asyncio.run_coroutine_threadsafe(
            self._act(
                action,
                requirements=requirements,
                strategy=strategy,
                return_sampling_results=return_sampling_results,
                format=format,
                model_options=model_options,
                tool_calls=tool_calls,
                stream=stream,
            ),
            self._event_loop,
        )

        return out.result()

    async def _act(
        self,
        action: Component,
        *,
        requirements: list[Requirement] | None = None,
        strategy: SamplingStrategy | None = None,
        return_sampling_results: bool = False,
        format: type[BaseModelSubclass] | None = None,
        model_options: dict | None = None,
        tool_calls: bool = False,
        stream: bool = False,
    ) -> ModelOutputThunk | SamplingResult:
        """Runs a generic action, and adds both the action and the result to the context.

        Args:
            action: the Component from which to generate.
            requirements: used as additional requirements when a sampling strategy is provided
            strategy: a SamplingStrategy that describes the strategy for validating and repairing/retrying for the instruct-validate-repair pattern. None means that no particular sampling strategy is used.
            return_sampling_results: attach the (successful and failed) sampling attempts to the results.
            format: if set, the BaseModel to use for constrained decoding.
            model_options: additional model options, which will upsert into the model/backend's defaults.
            tool_calls: if true, tool calling is enabled.

        Returns:
            A ModelOutputThunk if `return_sampling_results` is `False`, else returns a `SamplingResult`.
        """
        sampling_result: SamplingResult | None = None
        generate_logs: list[GenerateLog] = []

        if return_sampling_results:
            assert strategy is not None, (
                "Must provide a SamplingStrategy when return_sampling_results==True"
            )

        if strategy is None:
            result = self.backend.generate_from_context(
                action,
                ctx=self.ctx,
                format=format,
                model_options=model_options,
                tool_calls=tool_calls,
                stream=stream,
            )
            await result.avalue()

            # ._generate_log should never be None after generation.
            assert result._generate_log is not None
            result._generate_log.is_final_result = True
            generate_logs.append(result._generate_log)

        else:
            # Default validation strategy just validates all of the provided requirements.
            if strategy.validate is None:
                strategy.validate = lambda reqs, val_ctx, output: self._validate(
                    reqs, output=output
                )

            # Default generation strategy just generates from context.
            if strategy.generate is None:
                strategy.generate = (
                    lambda sample_action,
                    gen_ctx,
                    g_logs: self.backend.generate_from_context(
                        sample_action,
                        ctx=gen_ctx,
                        format=format,
                        model_options=model_options,
                        tool_calls=tool_calls,
                        stream=stream,
                    )
                )

            if requirements is None:
                requirements = []

            sampling_result = await strategy.sample(
                action, self.ctx, requirements=requirements, generate_logs=generate_logs
            )

            assert sampling_result.sample_generations is not None
            for result in sampling_result.sample_generations:
                assert (
                    result._generate_log is not None
                )  # Cannot be None after generation.
                generate_logs.append(result._generate_log)

            result = sampling_result.result
            assert sampling_result.result._generate_log is not None
            assert sampling_result.result._generate_log.is_final_result, (
                "generate logs from the final result returned by the sampling strategy must be marked as final"
            )

        self.ctx.insert_turn(ContextTurn(action, result), generate_logs=generate_logs)

        if return_sampling_results:
            assert (
                sampling_result is not None
            )  # Needed for the type checker but should never happen.
            return sampling_result
        else:
            return result

    @overload
    def instruct(
        self,
        description: str,
        *,
        images: list[ImageBlock] | list[PILImage.Image] | None = None,
        requirements: list[Requirement | str] | None = None,
        icl_examples: list[str | CBlock] | None = None,
        grounding_context: dict[str, str | CBlock | Component] | None = None,
        user_variables: dict[str, str] | None = None,
        prefix: str | CBlock | None = None,
        output_prefix: str | CBlock | None = None,
        strategy: SamplingStrategy | None = None,
        return_sampling_results: Literal[False] = False,
        format: type[BaseModelSubclass] | None = None,
        model_options: dict | None = None,
        tool_calls: bool = False,
    ) -> ModelOutputThunk: ...

    @overload
    def instruct(
        self,
        description: str,
        *,
        images: list[ImageBlock] | list[PILImage.Image] | None = None,
        requirements: list[Requirement | str] | None = None,
        icl_examples: list[str | CBlock] | None = None,
        grounding_context: dict[str, str | CBlock | Component] | None = None,
        user_variables: dict[str, str] | None = None,
        prefix: str | CBlock | None = None,
        output_prefix: str | CBlock | None = None,
        strategy: SamplingStrategy | None = None,
        return_sampling_results: Literal[True],
        format: type[BaseModelSubclass] | None = None,
        model_options: dict | None = None,
        tool_calls: bool = False,
    ) -> SamplingResult: ...

    def instruct(
        self,
        description: str,
        *,
        images: list[ImageBlock] | list[PILImage.Image] | None = None,
        requirements: list[Requirement | str] | None = None,
        icl_examples: list[str | CBlock] | None = None,
        grounding_context: dict[str, str | CBlock | Component] | None = None,
        user_variables: dict[str, str] | None = None,
        prefix: str | CBlock | None = None,
        output_prefix: str | CBlock | None = None,
        strategy: SamplingStrategy | None = None,
        return_sampling_results: bool = False,
        format: type[BaseModelSubclass] | None = None,
        model_options: dict | None = None,
        tool_calls: bool = False,
    ) -> ModelOutputThunk | SamplingResult:
        """Generates from an instruction.

        Args:
            description: The description of the instruction.
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
        """

        requirements = [] if requirements is None else requirements
        icl_examples = [] if icl_examples is None else icl_examples
        grounding_context = dict() if grounding_context is None else grounding_context

        images = self._parse_and_clean_image_args(images)

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

        return self.act(
            i,
            requirements=i.requirements,
            strategy=strategy,
            return_sampling_results=return_sampling_results,
            format=format,
            model_options=model_options,
            tool_calls=tool_calls,
        )  # type: ignore[call-overload]

    def chat(
        self,
        content: str,
        role: Message.Role = "user",
        *,
        images: list[ImageBlock] | list[PILImage.Image] | None = None,
        user_variables: dict[str, str] | None = None,
        format: type[BaseModelSubclass] | None = None,
        model_options: dict | None = None,
        tool_calls: bool = False,
    ) -> Message:
        """Sends a simple chat message and returns the response. Adds both messages to the Context."""
        if user_variables is not None:
            content_resolved = Instruction.apply_user_dict_from_jinja(
                user_variables, content
            )
        else:
            content_resolved = content
        images = self._parse_and_clean_image_args(images)
        user_message = Message(role=role, content=content_resolved, images=images)

        result = self.act(
            user_message,
            format=format,
            model_options=model_options,
            tool_calls=tool_calls,
        )
        parsed_assistant_message = result.parsed_repr
        assert isinstance(parsed_assistant_message, Message)

        return parsed_assistant_message

    def validate(
        self,
        reqs: Requirement | list[Requirement],
        *,
        output: CBlock | None = None,
        format: type[BaseModelSubclass] | None = None,
        model_options: dict | None = None,
        generate_logs: list[GenerateLog] | None = None,
    ) -> list[ValidationResult]:
        """Validates a set of requirements over the output (if provided) or the current context (if the output is not provided)."""
        # Run everything in the specific event loop for this session.
        out = asyncio.run_coroutine_threadsafe(
            self._validate(
                reqs=reqs,
                output=output,
                format=format,
                model_options=model_options,
                generate_logs=generate_logs,
            ),
            self._event_loop,
        )

        # Wait for and return the result.
        return out.result()

    async def _validate(
        self,
        reqs: Requirement | list[Requirement],
        *,
        output: CBlock | None = None,
        format: type[BaseModelSubclass] | None = None,
        model_options: dict | None = None,
        generate_logs: list[GenerateLog] | None = None,
    ) -> list[ValidationResult]:
        """Validates a set of requirements over the output (if provided) or the current context (if the output is not provided)."""
        # TODO: JAL; fix this docstring
        # Turn a solitary requirement in to a list of requirements, and then reqify if needed.
        reqs = [reqs] if not isinstance(reqs, list) else reqs
        reqs = [Requirement(req) if type(req) is str else req for req in reqs]
        if output is None:
            validation_target_ctx = self.ctx
        else:
            validation_target_ctx = SimpleContext()
            validation_target_ctx.insert(output)

        rvs: list[ValidationResult] = []
        coroutines: list[Coroutine[Any, Any, ValidationResult]] = []
        for requirement in reqs:
            val_result_co = requirement.validate(
                self.backend,
                validation_target_ctx,
                format=format,
                model_options=model_options,
                generate_logs=generate_logs,
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

        return rvs

    def query(
        self,
        obj: Any,
        query: str,
        *,
        format: type[BaseModelSubclass] | None = None,
        model_options: dict | None = None,
        tool_calls: bool = False,
    ) -> ModelOutputThunk:
        """Query method for retrieving information from an object.

        Args:
            obj : The object to be queried. It should be an instance of MObject or can be converted to one if necessary.
            query:  The string representing the query to be executed against the object.
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

        answer = self.act(
            q, format=format, model_options=model_options, tool_calls=tool_calls
        )
        return answer

    def transform(
        self,
        obj: Any,
        transformation: str,
        *,
        format: type[BaseModelSubclass] | None = None,
        model_options: dict | None = None,
    ) -> ModelOutputThunk | Any:
        """Transform method for creating a new object with the transformation applied.

        Args:
            obj : The object to be queried. It should be an instance of MObject or can be converted to one if necessary.
            transformation:  The string representing the query to be executed against the object.

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
        transformed = self.act(
            t, format=format, model_options=model_options, tool_calls=True
        )

        tools = self._call_tools(transformed)

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

            self.ctx.insert(chosen_tool)
            FancyLogger.get_logger().info(
                "added a tool message from transform to the context"
            )
            return chosen_tool._tool_output

        return transformed

    def _call_tools(self, result: ModelOutputThunk) -> list[ToolMessage]:
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

                content = str(output)
                if isinstance(self.backend, FormatterBackend):
                    content = self.backend.formatter.print(output)  # type: ignore

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

    # ###############################
    #  Convenience functions
    # ###############################

    def last_prompt(self) -> str | list[dict] | None:
        """Returns the last prompt that has been called from the session context.

        Returns:
            A string if the last prompt was a raw call to the model OR a list of messages (as role-msg-dicts). Is None if none could be found.
        """
        _, log = self.ctx.last_output_and_logs()

        prompt = None
        if isinstance(log, GenerateLog):
            prompt = log.prompt
        elif isinstance(log, list):
            last_el = log[-1]
            if isinstance(last_el, GenerateLog):
                prompt = last_el.prompt
        return prompt


# Convenience functions that use the current session
def instruct(description: str, **kwargs) -> ModelOutputThunk | SamplingResult:
    """Instruct using the current session."""
    return get_session().instruct(description, **kwargs)


def chat(content: str, **kwargs) -> Message:
    """Chat using the current session."""
    return get_session().chat(content, **kwargs)


def validate(reqs, **kwargs):
    """Validate using the current session."""
    return get_session().validate(reqs, **kwargs)


def query(obj: Any, query_str: str, **kwargs) -> ModelOutputThunk:
    """Query using the current session."""
    return get_session().query(obj, query_str, **kwargs)


def transform(obj: Any, transformation: str, **kwargs):
    """Transform using the current session."""
    return get_session().transform(obj, transformation, **kwargs)
