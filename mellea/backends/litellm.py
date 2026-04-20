"""A generic LiteLLM compatible backend that wraps around the openai python sdk."""

import asyncio
import datetime
import functools
import json
from collections.abc import Callable, Coroutine, Sequence
from typing import Any, overload

try:
    import litellm
    import litellm.litellm_core_utils
    import litellm.litellm_core_utils.get_supported_openai_params
except ImportError as e:
    raise ImportError(
        "The LiteLLM backend requires extra dependencies. "
        'Please install them with: pip install "mellea[litellm]"'
    ) from e

from ..backends import model_ids
from ..core import (
    BaseModelSubclass,
    C,
    CBlock,
    Component,
    Context,
    GenerateLog,
    GenerateType,
    MelleaLogger,
    ModelOutputThunk,
    ModelToolCall,
)
from ..core.base import AbstractMelleaTool
from ..formatters import ChatFormatter, TemplateFormatter
from ..helpers import (
    chat_completion_delta_merge,
    extract_model_tool_requests,
    get_current_event_loop,
    message_to_openai_message,
    send_to_queue,
)
from ..stdlib.components import Message
from ..stdlib.requirements import ALoraRequirement
from ..telemetry.backend_instrumentation import (
    instrument_generate_from_raw,
    start_generate_span,
)
from ..telemetry.context import generate_request_id, with_context
from .backend import FormatterBackend
from .model_options import ModelOption
from .tools import (
    add_tools_from_context_actions,
    add_tools_from_model_options,
    convert_tools_to_json,
    validate_tool_arguments,
)

format: None = None  # typing this variable in order to shadow the global format function and ensure mypy checks for errors


class LiteLLMBackend(FormatterBackend):
    """A generic LiteLLM compatible backend.

    Args:
        model_id (str): The LiteLLM model identifier string; typically
            ``"<provider>/<model_creator>/<model_name>"``.
        formatter (ChatFormatter | None): Formatter for rendering components.
            Defaults to ``TemplateFormatter``.
        base_url (str | None): Base URL for the LLM API endpoint; defaults to
            the Ollama local endpoint.
        model_options (dict | None): Default model options for generation requests.

    Attributes:
        to_mellea_model_opts_map (dict): Mapping from backend-specific option names to
            Mellea ``ModelOption`` sentinel keys.
        from_mellea_model_opts_map (dict): Mapping from Mellea ``ModelOption`` sentinel
            keys to backend-specific option names.
    """

    def __init__(
        self,
        model_id: str = "ollama_chat/"
        + str(model_ids.IBM_GRANITE_4_MICRO_3B.ollama_name),
        formatter: ChatFormatter | None = None,
        base_url: str | None = "http://localhost:11434",
        model_options: dict | None = None,
    ):
        """Initialize a LiteLLM-compatible backend for the given model ID and endpoint."""
        super().__init__(
            model_id=model_id,
            formatter=(
                formatter
                if formatter is not None
                else TemplateFormatter(model_id=model_id)
            ),
            model_options=model_options,
        )

        assert isinstance(model_id, str), "Model ID must be a string."
        self._model_id = model_id

        if base_url is None:
            self._base_url = "http://localhost:11434/v1"  # ollama
        else:
            self._base_url = base_url

        # A mapping of common options for this backend mapped to their Mellea ModelOptions equivalent.
        # These are usually values that must be extracted before hand or that are common among backend providers.
        # OpenAI has some deprecated parameters. Those map to the same mellea parameter, but
        # users should only be specifying a single one in their request.
        self.to_mellea_model_opts_map = {
            "system": ModelOption.SYSTEM_PROMPT,
            "reasoning_effort": ModelOption.THINKING,
            "seed": ModelOption.SEED,
            "max_completion_tokens": ModelOption.MAX_NEW_TOKENS,
            "max_tokens": ModelOption.MAX_NEW_TOKENS,
            "tools": ModelOption.TOOLS,
            "functions": ModelOption.TOOLS,
            "stream": ModelOption.STREAM,
        }

        # A mapping of Mellea specific ModelOptions to the specific names for this backend.
        # These options should almost always be a subset of those specified in the `to_mellea_model_opts_map`.
        # Usually, values that are intentionally extracted while prepping for the backend generate call
        # will be omitted here so that they will be removed when model_options are processed
        # for the call to the model. For LiteLLM, this dict might change slightly depending on the provider.
        self.from_mellea_model_opts_map = {
            ModelOption.SEED: "seed",
            ModelOption.MAX_NEW_TOKENS: "max_completion_tokens",
            ModelOption.STREAM: "stream",
        }

        self._past_event_loops: set[int] = set()

    async def _generate_from_context(
        self,
        action: Component[C] | CBlock,
        ctx: Context,
        *,
        format: type[BaseModelSubclass] | None = None,
        model_options: dict | None = None,
        tool_calls: bool = False,
    ) -> tuple[ModelOutputThunk[C], Context]:
        """Generate a completion for ``action`` given ``ctx`` via the LiteLLM chat API.

        Delegates to ``_generate_from_chat_context_standard``. Only chat contexts are
        supported; raises ``NotImplementedError`` otherwise.

        Args:
            action (Component[C] | CBlock): The component or content block to generate
                a completion for.
            ctx (Context): The current generation context (must be a chat context).
            format (type[BaseModelSubclass] | None): Optional Pydantic model class for
                structured/constrained output decoding.
            model_options (dict | None): Per-call model options that override the
                backend's defaults.
            tool_calls (bool): If ``True``, expose available tools to the model and
                parse tool-call responses.

        Returns:
            tuple[ModelOutputThunk[C], Context]: A thunk holding the (lazy) model output
                and an updated context that includes ``action`` and the new output.
        """
        assert ctx.is_chat_context, NotImplementedError(
            "The Openai backend only supports chat-like contexts."
        )
        span = start_generate_span(
            backend=self, action=action, ctx=ctx, format=format, tool_calls=tool_calls
        )

        _model_id_str = str(getattr(self, "model_id", "unknown"))
        with with_context(request_id=generate_request_id(), model_id=_model_id_str):
            mot = await self._generate_from_chat_context_standard(
                action,
                ctx,
                _format=format,
                model_options=model_options,
                tool_calls=tool_calls,
            )

        # Store span for telemetry recording in post_processing
        if span is not None:
            mot._meta["_telemetry_span"] = span

        return mot, ctx.add(action).add(mot)

    def _simplify_and_merge(
        self, model_options: dict[str, Any] | None
    ) -> dict[str, Any]:
        """Simplifies model_options to use the Mellea specific ModelOption.Option and merges the backend's model_options with those passed into this call.

        Rules:
        - Within a model_options dict, existing keys take precedence. This means remapping to mellea specific keys will maintain the value of the mellea specific key if one already exists.
        - When merging, the keys/values from the dictionary passed into this function take precedence.

        Because this function simplifies and then merges, non-Mellea keys from the passed in model_options will replace
        Mellea specific keys from the backend's model_options.

        Args:
            model_options: the model_options for this call

        Returns:
            a new dict
        """
        backend_model_opts = ModelOption.replace_keys(
            self.model_options, self.to_mellea_model_opts_map
        )

        if model_options is None:
            return backend_model_opts

        generate_call_model_opts = ModelOption.replace_keys(
            model_options, self.to_mellea_model_opts_map
        )
        return ModelOption.merge_model_options(
            backend_model_opts, generate_call_model_opts
        )

    def _make_backend_specific_and_remove(
        self, model_options: dict[str, Any]
    ) -> dict[str, Any]:
        """Maps specified Mellea specific keys to their backend specific version and removes any remaining Mellea keys.

        Additionally, logs any params unknown to litellm and any params that are openai specific but not supported by this model/provider.

        Args:
            model_options: the model_options for this call

        Returns:
            a new dict
        """
        # We set `drop_params=True` which will drop non-supported openai params; check for non-openai
        # params that might cause errors and log which openai params aren't supported here.
        # See https://docs.litellm.ai/docs/completion/input.
        supported_params_list = litellm.litellm_core_utils.get_supported_openai_params.get_supported_openai_params(
            self._model_id
        )
        supported_params = (
            set(supported_params_list) if supported_params_list is not None else set()
        )

        # LiteLLM specific remappings (typically based on provider). There's a few cases where the provider accepts
        # different parameters than LiteLLM says it does. Here's a few rules that help in those scenarios.
        model_opts_remapping = self.from_mellea_model_opts_map.copy()
        if (
            "max_completion_tokens" not in supported_params
            and "max_tokens" in supported_params
        ):
            # Scenario hit by Watsonx. LiteLLM believes Watsonx doesn't accept "max_completion_tokens" even though
            # OpenAI compatible endpoints should accept both (and Watsonx does accept both).
            model_opts_remapping[ModelOption.MAX_NEW_TOKENS] = "max_tokens"

        backend_specific = ModelOption.replace_keys(model_options, model_opts_remapping)
        backend_specific = ModelOption.remove_special_keys(backend_specific)

        # Since LiteLLM has many different providers, we add some additional parameter logging here.
        # There's two sets of parameters we have to look at:
        #   - unsupported_openai_params: standard OpenAI parameters that LiteLLM will automatically drop for us when `drop_params=True` if the provider doesn't support them.
        #   - unknown_keys: parameters that LiteLLM doesn't know about, aren't standard OpenAI parameters, and might be used by the provider. We don't drop these.
        # We want to flag both for the end user.
        standard_openai_subset = litellm.get_standard_openai_params(backend_specific)
        unknown_keys = []  # Keys that are unknown to litellm.
        unsupported_openai_params = []  # OpenAI params that are known to litellm but not supported for this model/provider.
        for key in backend_specific.keys():
            if key not in supported_params:
                if key in standard_openai_subset:
                    # LiteLLM is pretty confident that this standard OpenAI parameter won't work.
                    unsupported_openai_params.append(key)
                else:
                    # LiteLLM doesn't make any claims about this parameter; we won't drop it but we will keep track of it..
                    unknown_keys.append(key)

        if len(unknown_keys) > 0:
            MelleaLogger.get_logger().warning(
                f"litellm allows for unknown / non-openai input params; mellea won't validate the following params that may cause issues: {', '.join(unknown_keys)}"
            )

        if len(unsupported_openai_params) > 0:
            MelleaLogger.get_logger().warning(
                f"litellm may drop the following openai keys that it doesn't seem to recognize as being supported by the current model/provider: {', '.join(unsupported_openai_params)}"
                "\nThere are sometimes false positives here."
            )

        return backend_specific

    async def _generate_from_chat_context_standard(
        self,
        action: Component[C] | CBlock,
        ctx: Context,
        *,
        _format: type[BaseModelSubclass]
        | None = None,  # Type[BaseModelSubclass] is a class object of a subclass of BaseModel
        model_options: dict | None = None,
        tool_calls: bool = False,
    ) -> ModelOutputThunk[C]:
        await self.do_generate_walk(action)

        model_opts = self._simplify_and_merge(model_options)
        linearized_context = ctx.view_for_generation()
        assert linearized_context is not None, (
            "Cannot generate from a non-linear context in a FormatterBackend."
        )
        # Convert our linearized context into a sequence of chat messages. Template formatters have a standard way of doing this.
        messages: list[Message] = self.formatter.to_chat_messages(linearized_context)

        # Add the final message.
        match action:
            case ALoraRequirement():
                raise Exception("The LiteLLM backend does not support activated LoRAs.")
            case _:
                messages.extend(self.formatter.to_chat_messages([action]))

        # TODO: the supports_vision function is not reliably predicting if models support vision. E.g., ollama/llava is not a vision model?
        # if any(m.images is not None for m in messages):
        #     # check if model can handle images
        #     assert litellm.supports_vision(
        #         model=self.model_id), f"Model {self.model_id} does not support vision. Please use a different model."

        conversation: list[dict] = []
        system_prompt = model_opts.get(ModelOption.SYSTEM_PROMPT, "")
        if system_prompt != "":
            conversation.append({"role": "system", "content": system_prompt})
        conversation.extend([message_to_openai_message(m) for m in messages])

        extra_params: dict[str, Any] = {}
        if _format is not None:
            extra_params["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": _format.__name__,
                    "schema": _format.model_json_schema(),  # type: ignore
                    "strict": True,
                },
            }

        # Request usage information in streaming responses
        if model_opts.get(ModelOption.STREAM, False):
            extra_params["stream_options"] = {"include_usage": True}

        thinking = model_opts.get(ModelOption.THINKING, None)
        if type(thinking) is bool and thinking:
            # OpenAI uses strings for its reasoning levels.
            thinking = "medium"

        # Append tool call information if applicable.
        tools = self._extract_tools(action, _format, model_opts, tool_calls, ctx)
        formatted_tools = convert_tools_to_json(tools) if len(tools) > 0 else None

        model_specific_options = self._make_backend_specific_and_remove(model_opts)

        if self._has_potential_event_loop_errors():
            MelleaLogger.get_logger().warning(
                "There is a known bug with litellm. This generation call may fail. If it does, you should ensure that you are either running only synchronous Mellea functions or running async Mellea functions from one asyncio.run() call."
            )

        chat_response: Coroutine[
            Any, Any, litellm.ModelResponse | litellm.ModelResponseStream  # type: ignore
        ] = litellm.acompletion(
            model=self._model_id,
            messages=conversation,
            tools=formatted_tools,
            reasoning_effort=thinking,  # type: ignore
            drop_params=True,  # See note in `_make_backend_specific_and_remove`.
            **extra_params,
            **model_specific_options,
        )

        output = ModelOutputThunk(None)
        output._start = datetime.datetime.now()
        output._context = linearized_context
        output._action = action
        output._model_options = model_opts

        # Processing functions only pass the ModelOutputThunk (and current chunk of response). Bind the other vars necessary for
        # each processing step.
        output._process = self.processing
        output._post_process = functools.partial(
            self.post_processing,
            conversation=conversation,
            tools=tools,
            thinking=thinking,
            _format=_format,
        )

        # Set model/provider early so they are available in the error path
        output.model = str(self.model_id)
        output.provider = "litellm"

        try:
            # To support lazy computation, will need to remove this create_task and store just the unexecuted coroutine.
            # We can also support synchronous calls by adding a flag and changing this ._generate function.

            # This function should always be called from a running event loop so we don't have to worry about
            # scheduling the task to a specific event loop here.
            output._generate = asyncio.create_task(
                send_to_queue(chat_response, output._async_queue)
            )
            output._generate_type = GenerateType.ASYNC
        except RuntimeError as e:
            # Most likely cause is running this function without an event loop present
            raise e

        return output

    async def processing(
        self,
        mot: ModelOutputThunk,
        chunk: litellm.ModelResponse | litellm.ModelResponseStream,  # type: ignore
    ):
        """Accumulate content and thinking tokens from a single LiteLLM response chunk.

        Called during generation for each ``ModelResponse`` (non-streaming) or
        ``ModelResponseStream`` chunk (streaming). Tool call parsing is deferred to
        ``post_processing``.

        Args:
            mot (ModelOutputThunk): The output thunk being populated.
            chunk (litellm.ModelResponse | litellm.ModelResponseStream): A single
                response object or streaming chunk from LiteLLM.
        """
        if mot._thinking is None:
            mot._thinking = ""
        if mot._underlying_value is None:
            mot._underlying_value = ""

        if isinstance(chunk, litellm.ModelResponse):  # type: ignore
            # choice should always be a `Choice`. There's some type weirdness going
            # on with how litellm have defined the `.choices` list.
            choice = chunk.choices[0]
            assert isinstance(choice, litellm.Choices)

            message = choice.message

            # Sometimes a message doesn't actually have this field.
            if hasattr(message, "reasoning_content"):
                thinking_chunk = message.reasoning_content
                if thinking_chunk is not None:
                    mot._thinking += thinking_chunk

            content_chunk = message.content
            if content_chunk is not None:
                mot._underlying_value += content_chunk

            # Store the full response (includes usage) as a dict
            mot._meta["litellm_full_response"] = chunk.model_dump()
            # Also store just the choice for backward compatibility
            mot._meta["litellm_chat_response"] = chunk.choices[0].model_dump()

        elif isinstance(chunk, litellm.ModelResponseStream):  # type: ignore
            message_delta = chunk.choices[0].delta

            # Sometimes a delta doesn't actually have this field.
            if hasattr(message_delta, "reasoning_content"):
                thinking_chunk = message_delta.reasoning_content
                if thinking_chunk is not None:
                    mot._thinking += thinking_chunk

            content_chunk = message_delta.content
            if content_chunk is not None:
                mot._underlying_value += content_chunk

            if mot._meta.get("litellm_chat_response_streamed", None) is None:
                mot._meta["litellm_chat_response_streamed"] = []
            mot._meta["litellm_chat_response_streamed"].append(
                chunk.choices[0].model_dump()
            )

            # Store usage information from the chunk if available (typically in the last chunk)
            if hasattr(chunk, "usage") and chunk.usage is not None:
                mot._meta["litellm_streaming_usage"] = chunk.usage.model_dump()

    async def post_processing(
        self,
        mot: ModelOutputThunk,
        conversation: list[dict],
        tools: dict[str, AbstractMelleaTool],
        thinking,
        _format,
    ):
        """Finalize the model output thunk after LiteLLM generation completes.

        Reconstructs a merged chat response from streaming chunks if applicable,
        extracts tool call requests, records token usage metrics, emits telemetry,
        and attaches the generate log to the output thunk.

        Args:
            mot (ModelOutputThunk): The output thunk to finalize.
            conversation (list[dict]): The chat conversation sent to the model,
                used for logging.
            tools (dict[str, AbstractMelleaTool]): Available tools, keyed by name.
            thinking: The thinking/reasoning effort level passed to the model, or
                ``None`` if reasoning mode was not enabled.
            _format: The structured output format class used during generation, if any.
        """
        # Reconstruct the chat_response from chunks if streamed.
        streamed_chunks = mot._meta.get("litellm_chat_response_streamed", None)
        if streamed_chunks is not None:
            # Must handle ollama differently due to: https://github.com/BerriAI/litellm/issues/14579.
            # Check that we are targeting ollama with the model_id prefix litellm uses.
            separate_tools = False
            if "ollama" in self._model_id.split("/")[0]:
                separate_tools = True
            mot._meta["litellm_chat_response"] = chat_completion_delta_merge(
                streamed_chunks, force_all_tool_calls_separate=separate_tools
            )

        assert mot._action is not None, (
            "ModelOutputThunks should have their action assigned during generation"
        )
        assert mot._model_options is not None, (
            "ModelOutputThunks should have their model_opts assigned during generation"
        )

        # OpenAI-like streamed responses potentially give you chunks of tool calls.
        # As a result, we have to store data between calls and only then
        # check for complete tool calls in the post_processing step.
        tool_chunk = extract_model_tool_requests(
            tools, mot._meta["litellm_chat_response"]
        )
        if tool_chunk is not None:
            if mot.tool_calls is None:
                mot.tool_calls = {}
            # Merge the tool_chunk dict.
            for key, val in tool_chunk.items():
                mot.tool_calls[key] = val

        # Generate the log for this ModelOutputThunk.
        generate_log = GenerateLog()
        generate_log.prompt = conversation
        generate_log.backend = f"litellm::{self.model_id!s}"
        generate_log.model_options = mot._model_options
        generate_log.date = datetime.datetime.now()
        generate_log.model_output = mot._meta["litellm_chat_response"]
        generate_log.extra = {
            "format": _format,
            "tools_available": tools,
            "tools_called": mot.tool_calls,
            "thinking": thinking,
        }
        generate_log.action = mot._action
        generate_log.result = mot
        mot._generate_log = generate_log

        # Extract token usage from full response dict or streaming usage
        full_response = mot._meta.get("litellm_full_response")
        usage = full_response.get("usage") if isinstance(full_response, dict) else None

        # For streaming responses, usage is stored separately
        if usage is None:
            usage = mot._meta.get("litellm_streaming_usage")

        # Populate standardized usage field (LiteLLM uses OpenAI format)
        if usage:
            mot.usage = usage

        # Populate model and provider metadata
        mot.model = str(self.model_id)
        mot.provider = "litellm"

        # Record telemetry now that response is available
        span = mot._meta.get("_telemetry_span")
        if span is not None:
            from ..telemetry import end_backend_span
            from ..telemetry.backend_instrumentation import (
                record_response_metadata,
                record_token_usage,
            )

            response = mot._meta.get("litellm_chat_response")
            if response:
                # LiteLLM responses have usage information
                if usage:
                    record_token_usage(span, usage)
                record_response_metadata(span, response)
            # Close the span now that async operation is complete
            end_backend_span(span)
            # Clean up the span reference
            del mot._meta["_telemetry_span"]

    @staticmethod
    def _extract_tools(
        action, _format, model_opts, tool_calls, ctx
    ) -> dict[str, AbstractMelleaTool]:
        tools: dict[str, AbstractMelleaTool] = dict()
        if tool_calls:
            if _format:
                MelleaLogger.get_logger().warning(
                    f"Tool calling typically uses constrained generation, but you have specified a `format` in your generate call. NB: tool calling is superseded by format; we will NOT call tools for your request: {action}"
                )
            else:
                add_tools_from_model_options(tools, model_opts)
                add_tools_from_context_actions(tools, ctx.actions_for_available_tools())

                # Add the tools from the action for this generation last so that
                # they overwrite conflicting names.
                add_tools_from_context_actions(tools, [action])
            MelleaLogger.get_logger().info(f"Tools for call: {tools.keys()}")
        return tools

    @overload
    async def generate_from_raw(
        self,
        actions: list[Component[C]],
        ctx: Context,
        *,
        format: type[BaseModelSubclass] | None = None,
        model_options: dict | None = None,
        tool_calls: bool = False,
    ) -> list[ModelOutputThunk[C]]: ...

    @overload
    async def generate_from_raw(
        self,
        actions: list[Component[C] | CBlock],
        ctx: Context,
        *,
        format: type[BaseModelSubclass] | None = None,
        model_options: dict | None = None,
        tool_calls: bool = False,
    ) -> list[ModelOutputThunk[C | str]]: ...

    async def generate_from_raw(
        self,
        actions: Sequence[Component[C] | CBlock],
        ctx: Context,
        *,
        format: type[BaseModelSubclass] | None = None,
        model_options: dict | None = None,
        tool_calls: bool = False,
    ) -> list[ModelOutputThunk]:
        """Generate completions for multiple actions without chat templating via LiteLLM.

        Passes formatted prompt strings directly to LiteLLM's text completion endpoint.
        Tool calling is not supported on this endpoint.

        Args:
            actions (Sequence[Component[C] | CBlock]): Actions to generate completions for.
            ctx (Context): The current generation context.
            format (type[BaseModelSubclass] | None): Optional Pydantic model for
                structured output; passed as ``guided_json`` in the request body.
            model_options (dict | None): Per-call model options.
            tool_calls (bool): Ignored; tool calling is not supported on this endpoint.

        Returns:
            list[ModelOutputThunk]: A list of model output thunks, one per action.
        """
        with instrument_generate_from_raw(
            backend=self, num_actions=len(actions), format=format, tool_calls=tool_calls
        ):
            await self.do_generate_walks(list(actions))
            extra_body = {}
            if format is not None:
                MelleaLogger.get_logger().warning(
                    "The official OpenAI completion api does not accept response format / structured decoding; "
                    "it will be passed as an extra arg."
                )

                # Some versions (like vllm's version) of the OpenAI API support structured decoding for completions requests.
                extra_body["guided_json"] = format.model_json_schema()  # type: ignore
            if tool_calls:
                MelleaLogger.get_logger().warning(
                    "The completion endpoint does not support tool calling."
                )

            # We don't do anything fancy for model_opts with generate from raw; litellm has too many potential options depending on provider.
            model_opts = self._simplify_and_merge(model_options)
            model_specific_options = self._make_backend_specific_and_remove(model_opts)

            if self._has_potential_event_loop_errors():
                MelleaLogger.get_logger().warning(
                    "There is a known bug with litellm. This generation call may fail. If it does, you should ensure that you are either running only synchronous Mellea functions or running async Mellea functions from one asyncio.run() call."
                )

            prompts = [self.formatter.print(action) for action in actions]

            completion_response = await litellm.atext_completion(
                model=self._model_id, prompt=prompts, **model_specific_options
            )

        # Necessary for type checker.
        assert isinstance(completion_response, litellm.TextCompletionResponse)  # type: ignore

        results = []
        date = datetime.datetime.now()
        responses = completion_response.choices
        if len(responses) != len(prompts):
            MelleaLogger.get_logger().error(
                "litellm appears to have sent your batch request as a single message; this typically happens with providers like ollama that don't support batching"
            )

        for res, action, prompt in zip(responses, actions, prompts):
            output = ModelOutputThunk(res.text)  # type: ignore
            output._context = None  # There is no context for generate_from_raw for now
            output._action = action
            output._model_options = model_opts
            output._meta = {
                "litellm_chat_response": res.model_dump(),
                "usage": completion_response.usage.model_dump()
                if completion_response.usage
                else None,
            }

            output.parsed_repr = (
                action.parse(output) if isinstance(action, Component) else output.value
            )

            generate_log = GenerateLog()
            generate_log.prompt = prompt
            generate_log.backend = f"litellm::{self.model_id!s}"
            generate_log.model_options = model_opts
            generate_log.date = date
            generate_log.model_output = completion_response
            generate_log.extra = {"seed": model_opts.get("seed", None)}
            generate_log.action = action
            output._generate_log = generate_log

            results.append(output)

        return results

    def _extract_model_tool_requests(
        self,
        tools: dict[str, AbstractMelleaTool],
        chat_response: litellm.ModelResponse,  # type: ignore
    ) -> dict[str, ModelToolCall] | None:
        model_tool_calls: dict[str, ModelToolCall] = {}
        choice_0 = chat_response.choices[0]
        assert isinstance(choice_0, litellm.utils.Choices), (  # type: ignore
            "Only works for non-streaming response for now"
        )
        calls = choice_0.message.tool_calls
        if calls:
            for tool_call in calls:
                tool_name = str(tool_call.function.name)
                tool_args = tool_call.function.arguments

                func = tools.get(tool_name)
                if func is None:
                    MelleaLogger.get_logger().warning(
                        f"model attempted to call a non-existing function: {tool_name}"
                    )
                    continue  # skip this function if we can't find it.

                # Returns the args as a string. Parse it here.
                args = json.loads(tool_args)

                # Validate and coerce argument types
                validated_args = validate_tool_arguments(func, args, strict=False)
                model_tool_calls[tool_name] = ModelToolCall(
                    tool_name, func, validated_args
                )

        if len(model_tool_calls) > 0:
            return model_tool_calls
        return None

    def _has_potential_event_loop_errors(self) -> bool:
        """In some cases litellm doesn't create a new async client. There doesn't appear to be any way for us to force that behavior. As a result, log a warning for known cases.

        This whole function can be removed once the bug is fixed: https://github.com/BerriAI/litellm/issues/15294.
        """
        # Async clients are tied to event loops.
        key = id(get_current_event_loop())

        has_potential_issue = False
        if (
            len(self._past_event_loops) > 0
            and key not in self._past_event_loops
            and "watsonx/" in str(self.model_id)
        ):
            has_potential_issue = True

        # Add this loop to the known set.
        self._past_event_loops.add(key)

        return has_potential_issue
