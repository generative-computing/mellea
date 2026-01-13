"""A generic OpenAI compatible backend that wraps around the openai python sdk."""

import asyncio
import datetime
import functools
import inspect
import os
from collections.abc import Callable, Coroutine, Sequence
from typing import TYPE_CHECKING, Any, overload

import granite_common
import openai
import requests
from openai.types.chat import ChatCompletion
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk
from openai.types.completion import Completion

from ..backends import ModelIdentifier, model_ids
from ..core import (
    BaseModelSubclass,
    C,
    CBlock,
    Component,
    Context,
    FancyLogger,
    GenerateLog,
    GenerateType,
    ModelOutputThunk,
    Requirement,
)
from ..formatters import ChatFormatter, TemplateFormatter
from ..helpers import (
    ClientCache,
    chat_completion_delta_merge,
    extract_model_tool_requests,
    get_current_event_loop,
    message_to_openai_message,
    messages_to_docs,
    send_to_queue,
)
from ..stdlib.components import Intrinsic, Message
from ..stdlib.requirements import ALoraRequirement, LLMaJRequirement
from .adapters import (
    AdapterMixin,
    AdapterType,
    GraniteCommonAdapter,
    OpenAIAdapter,
    get_adapter_for_intrinsic,
)
from .backend import FormatterBackend
from ..helpers import _server_type, _ServerType
from .model_options import ModelOption
from .tools import (
    add_tools_from_context_actions,
    add_tools_from_model_options,
    convert_tools_to_json,
)

if TYPE_CHECKING:
    from transformers.tokenization_utils import PreTrainedTokenizer

openai_ollama_batching_error = "json: cannot unmarshal array into Go struct field CompletionRequest.prompt of type string"

format: None = None  # typing this variable in order to shadow the global format function and ensure mypy checks for errors


class OpenAIBackend(FormatterBackend, AdapterMixin):
    """A generic OpenAI compatible backend."""

    def __init__(
        self,
        model_id: str | ModelIdentifier = model_ids.OPENAI_GPT_5_1,
        formatter: ChatFormatter | None = None,
        base_url: str | None = None,
        model_options: dict | None = None,
        *,
        default_to_constraint_checking_alora: bool = True,
        api_key: str | None = None,
        **kwargs,
    ):
        """Initialize and OpenAI compatible backend. For any additional kwargs that you need to pass the the client, pass them as a part of **kwargs.

        Args:
            model_id : A generic model identifier or OpenAI compatible string. Defaults to model_ids.IBM_GRANITE_3_3_8B.
            formatter: A custom formatter based on backend.If None, defaults to TemplateFormatter
            base_url : Base url for LLM API. Defaults to None.
            model_options : Generation options to pass to the LLM. Defaults to None.
            default_to_constraint_checking_alora: If set to False then aloras will be deactivated. This is primarily for performance benchmarking and debugging.
            api_key : API key for generation. Defaults to None.
            kwargs : additional kwargs to pass when creating the OpenAI client.
        """
        super().__init__(
            model_id=model_id,
            formatter=(
                formatter
                if formatter is not None
                else TemplateFormatter(model_id=model_id)
            ),
            model_options=model_options,
        )

        # A mapping of common options for this backend mapped to their Mellea ModelOptions equivalent.
        # These are usually values that must be extracted before hand or that are common among backend providers.
        # OpenAI has some deprecated parameters. Those map to the same mellea parameter, but
        # users should only be specifying a single one in their request.
        self.to_mellea_model_opts_map_chats = {
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
        # for the call to the model.
        self.from_mellea_model_opts_map_chats = {
            ModelOption.SEED: "seed",
            ModelOption.MAX_NEW_TOKENS: "max_completion_tokens",
            ModelOption.STREAM: "stream",
        }

        # See notes above.
        self.to_mellea_model_opts_map_completions = {
            "seed": ModelOption.SEED,
            "max_tokens": ModelOption.MAX_NEW_TOKENS,
            "stream": ModelOption.STREAM,
        }
        # See notes above.
        self.from_mellea_model_opts_map_completions = {
            ModelOption.SEED: "seed",
            ModelOption.MAX_NEW_TOKENS: "max_tokens",
            ModelOption.STREAM: "stream",
        }

        self.default_to_constraint_checking_alora = default_to_constraint_checking_alora

        match model_id:
            case str():
                self._model_id = model_id
            case ModelIdentifier():
                assert model_id.openai_name is not None, (
                    "model_id is None. This can also happen if the ModelIdentifier has no `openai_name` name set."
                )
                self._model_id = model_id.openai_name

        # Use provided parameters or fall back to environment variables
        self._api_key = api_key
        self._base_url = base_url

        # Validate that we have the required configuration
        if self._api_key is None and os.getenv("OPENAI_API_KEY") is None:
            raise ValueError(
                "OPENAI_API_KEY or api_key is required but not set. Please either:\n"
                "  1. Set the environment variable: export OPENAI_API_KEY='your-key-here'\n"
                "  2. Pass it as a parameter: OpenAIBackend(api_key='your-key-here')"
            )

        if self._base_url is None and os.getenv("OPENAI_BASE_URL") is None:
            FancyLogger.get_logger().warning(
                "OPENAI_BASE_URL or base_url is not set.\n"
                "The openai SDK is going to assume that the base_url is `https://api.openai.com/v1`"
            )

        self._server_type: _ServerType = (
            _server_type(self._base_url)
            if self._base_url is not None
            else _ServerType.OPENAI
        )  # type: ignore

        self._openai_client_kwargs = self.filter_openai_client_kwargs(**kwargs)

        self._client = openai.OpenAI(  # type: ignore
            api_key=self._api_key, base_url=self._base_url, **self._openai_client_kwargs
        )

        self._client_cache = ClientCache(2)

        # Call once to create an async_client and populate the cache.
        _ = self._async_client

        # Adapters can be made know to the backend (added) and
        # loaded / active.
        self._added_adapters: dict[str, OpenAIAdapter] = {}
        self._loaded_adapters: dict[str, OpenAIAdapter] = {}

    @property
    def _async_client(self) -> openai.AsyncOpenAI:
        """OpenAI's client usually handles changing event loops but explicitly handle it here for edge cases."""
        key = id(get_current_event_loop())

        _async_client = self._client_cache.get(key)
        if _async_client is None:
            _async_client = openai.AsyncOpenAI(
                api_key=self._api_key,
                base_url=self._base_url,
                **self._openai_client_kwargs,
            )
            self._client_cache.put(key, _async_client)
        return _async_client

    @staticmethod
    def filter_openai_client_kwargs(**kwargs) -> dict:
        """Filter kwargs to only include valid OpenAI client parameters."""
        openai_params = set(inspect.signature(openai.OpenAI.__init__).parameters.keys())  # type: ignore
        openai_params.discard("self")  # Remove 'self' parameter
        return {k: v for k, v in kwargs.items() if k in openai_params}

    def filter_chat_completions_kwargs(self, model_options: dict) -> dict:
        """Filter kwargs to only include valid OpenAI chat.completions.create parameters.

        https://platform.openai.com/docs/api-reference/chat/create
        """
        from openai.resources.chat.completions import Completions

        chat_params = set(inspect.signature(Completions.create).parameters.keys())
        chat_params.discard("self")
        return {k: v for k, v in model_options.items() if k in chat_params}

    def filter_completions_kwargs(self, model_options: dict) -> dict:
        """Filter kwargs to only include valid OpenAI completions.create parameters.

        https://platform.openai.com/docs/api-reference/completions
        """
        from openai.resources.completions import Completions

        completions_params = set(
            inspect.signature(Completions.create).parameters.keys()
        )
        completions_params.discard("self")  # Remove 'self' parameter
        return {k: v for k, v in model_options.items() if k in completions_params}

    def _simplify_and_merge(
        self, model_options: dict[str, Any] | None, is_chat_context: bool
    ) -> dict[str, Any]:
        """Simplifies model_options to use the Mellea specific ModelOption.Option and merges the backend's model_options with those passed into this call.

        Rules:
        - Within a model_options dict, existing keys take precedence. This means remapping to mellea specific keys will maintain the value of the mellea specific key if one already exists.
        - When merging, the keys/values from the dictionary passed into this function take precedence.

        Because this function simplifies and then merges, non-Mellea keys from the passed in model_options will replace
        Mellea specific keys from the backend's model_options.

        Args:
            model_options: the model_options for this call
            is_chat_context: set to True if using chat completion api

        Returns:
            a new dict
        """
        remap_dict = self.to_mellea_model_opts_map_chats
        if not is_chat_context:
            remap_dict = self.to_mellea_model_opts_map_completions

        backend_model_opts = ModelOption.replace_keys(self.model_options, remap_dict)

        if model_options is None:
            return backend_model_opts

        generate_call_model_opts = ModelOption.replace_keys(model_options, remap_dict)
        return ModelOption.merge_model_options(
            backend_model_opts, generate_call_model_opts
        )

    def _make_backend_specific_and_remove(
        self, model_options: dict[str, Any], is_chat_context: bool
    ) -> dict[str, Any]:
        """Maps specified Mellea specific keys to their backend specific version and removes any remaining Mellea keys.

        Args:
            model_options: the model_options for this call
            is_chat_context: set to True if using chat completion api

        Returns:
            a new dict
        """
        remap_dict = self.from_mellea_model_opts_map_chats
        if not is_chat_context:
            remap_dict = self.from_mellea_model_opts_map_completions

        backend_specific = ModelOption.replace_keys(model_options, remap_dict)

        # OpenAI Backend has specific filtering functionality.
        if is_chat_context:
            model_opts = self.filter_chat_completions_kwargs(backend_specific)
        else:
            model_opts = self.filter_completions_kwargs(backend_specific)

        return model_opts

    async def generate_from_context(
        self,
        action: Component[C] | CBlock,
        ctx: Context,
        *,
        format: type[BaseModelSubclass] | None = None,
        model_options: dict | None = None,
        tool_calls: bool = False,
    ) -> tuple[ModelOutputThunk[C], Context]:
        """See `generate_from_chat_context`."""
        assert ctx.is_chat_context, NotImplementedError(
            "The Openai backend only supports chat-like contexts."
        )
        return await self.generate_from_chat_context(
            action,
            ctx,
            _format=format,
            model_options=model_options,
            tool_calls=tool_calls,
        )

    async def generate_from_chat_context(
        self,
        action: Component[C] | CBlock,
        ctx: Context,
        *,
        _format: type[BaseModelSubclass]
        | None = None,  # Type[BaseModelSubclass] is a class object of a subclass of BaseModel
        model_options: dict | None = None,
        tool_calls: bool = False,
    ) -> tuple[ModelOutputThunk[C], Context]:
        """Generates a new completion from the provided Context using this backend's `Formatter`."""
        await self.do_generate_walk(action)

        # Requirements can be automatically rerouted to a requirement adapter.
        if isinstance(action, Requirement):
            # See docs/dev/requirement_aLoRA_rerouting.md
            reroute_to_alora = self.default_to_constraint_checking_alora
            adapter_name = "requirement_check"

            if isinstance(action, ALoraRequirement):
                reroute_to_alora = True
                adapter_name = action.intrinsic_name
                alora_action = action
            else:
                assert action.description is not None, (
                    "must have a description when generating from a requirement"
                )
                alora_action = ALoraRequirement(action.description, adapter_name)

            # Check if a requirement_check (or AloraRequirement specified) adapter exists.
            alora_req_adapter = get_adapter_for_intrinsic(
                adapter_name, [AdapterType.ALORA], self._added_adapters
            )
            if alora_req_adapter is None:
                # Log a warning if using an AloraRequirement but no adapter fit.
                if reroute_to_alora and isinstance(action, ALoraRequirement):
                    FancyLogger.get_logger().warning(
                        f"attempted to use an AloraRequirement but backend {self} doesn't have the specified adapter added {adapter_name}; defaulting to regular generation"
                    )
                reroute_to_alora = False

            if issubclass(type(action), LLMaJRequirement):
                reroute_to_alora = False

            if reroute_to_alora:
                # Keep the alora requirement handling separate for now.
                mot = await self._generate_from_intrinsic(
                    alora_action, ctx, model_options=model_options
                )
                return mot, ctx.add(alora_action).add(mot)

        elif isinstance(action, Intrinsic):
            mot = await self._generate_from_intrinsic(
                action, ctx, model_options=model_options
            )
            return mot, ctx.add(action).add(mot)

        mot = await self._generate_from_chat_context_standard(
            action,
            ctx,
            _format=_format,
            model_options=model_options,
            tool_calls=tool_calls,
        )
        return mot, ctx.add(action).add(mot)

    async def _generate_from_intrinsic(
        self, action: Intrinsic, ctx: Context, *, model_options: dict | None = None
    ) -> ModelOutputThunk:
        model_opts = self._simplify_and_merge(
            model_options, is_chat_context=ctx.is_chat_context
        )
        if len(model_opts.items()) > 0:
            FancyLogger.get_logger().info(
                "passing in model options when generating with an adapter; some model options may be overwritten / ignored"
            )

        linearized_context = ctx.view_for_generation()
        assert linearized_context is not None, (
            "Cannot generate from a non-linear context in a FormatterBackend."
        )
        if len(linearized_context) == 0:
            FancyLogger.get_logger().warning(
                f"generating with an intrinsic when the context is empty; this is typically incorrect: {action}"
            )

        # Convert our linearized context into a sequence of chat messages. Template formatters have a standard way of doing this.
        messages: list[Message] = self.formatter.to_chat_messages(linearized_context)

        conversation: list[dict] = []

        system_prompt = model_opts.get(ModelOption.SYSTEM_PROMPT, "")
        if system_prompt != "":
            conversation.append({"role": "system", "content": system_prompt})
        conversation.extend([message_to_openai_message(m) for m in messages])
        docs = messages_to_docs(messages)

        if model_opts.get(ModelOption.STREAM, None) is not None:
            # Intrinsics don't support streaming because of their post-processing step.
            FancyLogger.get_logger().warning(
                "intrinsics cannot use streaming; removing model option"
            )
            del model_opts[ModelOption.STREAM]

        adapter = get_adapter_for_intrinsic(
            action.intrinsic_name, action.adapter_types, self._added_adapters
        )
        if adapter is None:
            raise ValueError(
                f"backend ({self}) has no adapter for processing intrinsic: {action.intrinsic_name}"
            )

        # TODO: Code below this point is mostly specific to RagIntrinsics (and granite_common).
        #       It should be refactored into a specific adapter.transform() function.
        assert isinstance(adapter, GraniteCommonAdapter), (
            "currently Mellea only supports GraniteCommonAdapters and Intrinsics"
        )
        assert adapter.config is not None
        rewriter = granite_common.IntrinsicsRewriter(
            config_dict=adapter.config, model_name=adapter.qualified_name
        )
        result_processor = granite_common.IntrinsicsResultProcessor(
            config_dict=adapter.config
        )

        # Convert our conversation into a proper chat completions dict.
        # [{role: user, content: Hello}, {...}] -> {messages: [{role:user,...}, ...], model:..., ...}
        request_json: dict = {
            "messages": conversation,
            "extra_body": {"documents": docs},
        }

        # Convert other parameters from Mellea proprietary format to standard format.
        if model_options is not None:
            for model_option in model_options:
                if model_option == ModelOption.TEMPERATURE:
                    request_json["temperature"] = model_options[model_option]

        rewritten = rewriter.transform(request_json, **action.intrinsic_kwargs)

        self.load_adapter(adapter.qualified_name)
        chat_response: Coroutine[Any, Any, ChatCompletion] = (
            self._async_client.chat.completions.create(**rewritten.model_dump())
        )

        output = ModelOutputThunk(None)
        output._context = linearized_context
        output._action = action
        output._model_options = model_opts
        output._meta["granite_common_chat_response"] = rewritten

        # Add another step to the processing function.
        async def granite_common_processing(
            mot: ModelOutputThunk,
            chunk: ChatCompletion,
            rewritten: ChatCompletion,
            result_processor: granite_common.IntrinsicsResultProcessor,
        ):
            res = result_processor.transform(chunk, rewritten)  # type: ignore

            # processing expects a ChatCompletion object. Granite common differs slightly from this. Re-create the necessary object.
            full_res = ChatCompletion(
                id=chunk.id,
                choices=[],
                created=chunk.created,
                model=chunk.model,
                usage=chunk.usage,
                object="chat.completion",
            )

            # Set the choices here so that pydantic validation doesn't error out.
            full_res.choices = res.choices  # type: ignore

            return await self.processing(mot, full_res)

        output._process = functools.partial(
            granite_common_processing,
            rewritten=rewritten,  # type: ignore
            result_processor=result_processor,
        )

        output._post_process = functools.partial(
            self.post_processing,
            tools={},
            conversation=conversation,
            thinking=None,
            seed=model_opts.get(ModelOption.SEED, None),
            _format=None,
        )

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

    async def _generate_from_chat_context_standard(
        self,
        action: Component | CBlock,
        ctx: Context,
        *,
        _format: type[BaseModelSubclass]
        | None = None,  # Type[BaseModelSubclass] is a class object of a subclass of BaseModel
        model_options: dict | None = None,
        tool_calls: bool = False,
    ) -> ModelOutputThunk:
        model_opts = self._simplify_and_merge(
            model_options, is_chat_context=ctx.is_chat_context
        )
        linearized_context = ctx.view_for_generation()
        assert linearized_context is not None, (
            "Cannot generate from a non-linear context in a FormatterBackend."
        )
        # Convert our linearized context into a sequence of chat messages. Template formatters have a standard way of doing this.
        messages: list[Message] = self.formatter.to_chat_messages(linearized_context)
        # Add the final message.
        match action:
            case ALoraRequirement():
                raise Exception(
                    "The OpenAI backend does not support currently support activated LoRAs."
                )
            case _:
                messages.extend(self.formatter.to_chat_messages([action]))
        conversation: list[dict] = []

        system_prompt = model_opts.get(ModelOption.SYSTEM_PROMPT, "")
        if system_prompt != "":
            conversation.append({"role": "system", "content": system_prompt})
        conversation.extend([message_to_openai_message(m) for m in messages])

        extra_params: dict[str, Any] = {}
        if _format is not None:
            if self._server_type == _ServerType.OPENAI:
                # The OpenAI platform requires that additionalProperties=False on all response_format schemas.
                # However, not all schemas generates by Mellea include additionalProperties.
                # GenerativeSlot, in particular, does not add this property.
                # The easiest way to address this disparity between OpenAI and other inference providers is to
                # monkey-patch the response format exactly when we are actually using the OpenAI server.
                #
                # This only addresses the additionalProperties=False constraint.
                # Other constraints we should be checking/patching are described here:
                # https://platform.openai.com/docs/guides/structured-outputs?api-mode=chat
                monkey_patched_response_schema = _format.model_json_schema()  # type: ignore
                monkey_patched_response_schema["additionalProperties"] = False
                extra_params["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": _format.__name__,
                        "schema": monkey_patched_response_schema,
                        "strict": True,
                    },
                }
            else:
                FancyLogger().get_logger().warning(
                    "Mellea assumes you are NOT using the OpenAI platform, and that other model providers have less strict requirements on support JSON schemas passed into `format=`. If you encounter a server-side error following this message, then you found an exception to this assumption. Please open an issue at github.com/generative_computing/mellea with this stack trace and your inference engine / model provider."
                )
                extra_params["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": _format.__name__,
                        "schema": _format.model_json_schema(),  # type: ignore
                        "strict": True,
                    },
                }

        # Append tool call information if applicable.
        tools: dict[str, Callable] = dict()
        if tool_calls:
            if _format:
                FancyLogger.get_logger().warning(
                    f"Tool calling typically uses constrained generation, but you have specified a `format` in your generate call. NB: tool calling is superseded by format; we will NOT call tools for your request: {action}"
                )
            else:
                add_tools_from_model_options(tools, model_opts)
                add_tools_from_context_actions(tools, ctx.actions_for_available_tools())

                # Add the tools from the action for this generation last so that
                # they overwrite conflicting names.
                add_tools_from_context_actions(tools, [action])
            FancyLogger.get_logger().info(f"Tools for call: {tools.keys()}")

        thinking = model_opts.get(ModelOption.THINKING, None)
        if type(thinking) is bool and thinking:
            # OpenAI uses strings for its reasoning levels.
            thinking = "medium"

        formatted_tools = convert_tools_to_json(tools)
        use_tools = len(formatted_tools) > 0

        # Build optional reasoning parameters
        # NOTE: the openai SDK doesn't like it if you pass `reasoning_effort` param to a non-reasoning model e.g. gpt4o
        reasoning_params = {}
        if thinking is not None:
            reasoning_params["reasoning_effort"] = thinking

        chat_response: Coroutine[
            Any, Any, ChatCompletion | openai.AsyncStream[ChatCompletionChunk]
        ] = self._async_client.chat.completions.create(
            model=self._model_id,
            messages=conversation,  # type: ignore
            tools=formatted_tools if use_tools else None,  # type: ignore
            # parallel_tool_calls=False, # We only support calling one tool per turn. But we do the choosing on our side so we leave this False.
            **extra_params,
            **reasoning_params,  # type: ignore
            **self._make_backend_specific_and_remove(
                model_opts, is_chat_context=ctx.is_chat_context
            ),
        )  # type: ignore

        output = ModelOutputThunk(None)
        output._context = linearized_context
        output._action = action
        output._model_options = model_opts

        # Processing functions only pass the ModelOutputThunk (and current chunk of response). Bind the other vars necessary for
        # each processing step.
        output._process = self.processing
        output._post_process = functools.partial(
            self.post_processing,
            tools=tools,
            conversation=conversation,
            thinking=thinking,
            seed=model_opts.get(ModelOption.SEED, None),
            _format=_format,
        )

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
        self, mot: ModelOutputThunk, chunk: ChatCompletion | ChatCompletionChunk
    ):
        """Called during generation to add information from a single ChatCompletion or ChatCompletionChunk to the ModelOutputThunk.

        For OpenAI, tool call parsing is handled in the post processing step.
        """
        if mot._thinking is None:
            mot._thinking = ""
        if mot._underlying_value is None:
            mot._underlying_value = ""

        if isinstance(chunk, ChatCompletion):
            message = chunk.choices[0].message

            if hasattr(message, "reasoning_content"):
                thinking_chunk = message.reasoning_content  # type: ignore
                if thinking_chunk is not None:
                    mot._thinking += thinking_chunk

            content_chunk = message.content
            if content_chunk is not None:
                mot._underlying_value += content_chunk

            mot._meta["oai_chat_response"] = chunk.choices[0].model_dump()

        elif isinstance(chunk, ChatCompletionChunk):
            message_delta = chunk.choices[0].delta
            if hasattr(message_delta, "reasoning_content"):
                thinking_chunk = message_delta.reasoning_content  # type: ignore
                if thinking_chunk is not None:
                    mot._thinking += thinking_chunk

            content_chunk = message_delta.content
            if content_chunk is not None:
                mot._underlying_value += content_chunk

            if mot._meta.get("oai_chat_response_streamed", None) is None:
                mot._meta["oai_chat_response_streamed"] = []
            mot._meta["oai_chat_response_streamed"].append(
                chunk.choices[0].model_dump()
            )

    async def post_processing(
        self,
        mot: ModelOutputThunk,
        tools: dict[str, Callable],
        conversation: list[dict],
        thinking,
        seed,
        _format,
    ):
        """Called when generation is done."""
        # Reconstruct the chat_response from chunks if streamed.
        streamed_chunks = mot._meta.get("oai_chat_response_streamed", None)
        if streamed_chunks is not None:
            mot._meta["oai_chat_response"] = chat_completion_delta_merge(
                streamed_chunks
            )

        assert mot._action is not None, (
            "ModelOutputThunks should have their action assigned during generation"
        )
        assert mot._model_options is not None, (
            "ModelOutputThunks should have their model_opts assigned during generation"
        )

        # OpenAI streamed responses give you chunks of tool calls.
        # As a result, we have to store data between calls and only then
        # check for complete tool calls in the post_processing step.
        tool_chunk = extract_model_tool_requests(tools, mot._meta["oai_chat_response"])
        if tool_chunk is not None:
            if mot.tool_calls is None:
                mot.tool_calls = {}
            # Merge the tool_chunk dict.
            for key, val in tool_chunk.items():
                mot.tool_calls[key] = val

        # Generate the log for this ModelOutputThunk.
        generate_log = GenerateLog()
        generate_log.prompt = conversation
        generate_log.backend = f"openai::{self.model_id!s}"
        generate_log.model_options = mot._model_options
        generate_log.date = datetime.datetime.now()
        generate_log.model_output = mot._meta["oai_chat_response"]
        generate_log.extra = {
            "format": _format,
            "thinking": thinking,
            "tools_available": tools,
            "tools_called": mot.tool_calls,
            "seed": seed,
        }
        generate_log.action = mot._action
        generate_log.result = mot
        mot._generate_log = generate_log

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
        """Generate using the completions api. Gives the input provided to the model without templating."""
        await self.do_generate_walks(list(actions))

        extra_body = {}
        if format is not None:
            FancyLogger.get_logger().warning(
                "The official OpenAI completion api does not accept response format / structured decoding; "
                "it will be passed as an extra arg."
            )

            # Some versions (like vllm's version) of the OpenAI API support structured decoding for completions requests.
            extra_body["guided_json"] = format.model_json_schema()  # type: ignore
        if tool_calls:
            FancyLogger.get_logger().warning(
                "The completion endpoint does not support tool calling at the moment."
            )

        model_opts = self._simplify_and_merge(model_options, is_chat_context=False)

        prompts = [self.formatter.print(action) for action in actions]

        try:
            completion_response: Completion = (
                await self._async_client.completions.create(
                    model=self._model_id,
                    prompt=prompts,
                    extra_body=extra_body,
                    **self._make_backend_specific_and_remove(
                        model_opts, is_chat_context=False
                    ),
                )
            )  # type: ignore
        except openai.BadRequestError as e:
            if openai_ollama_batching_error in e.message:
                FancyLogger.get_logger().error(
                    "If you are trying to call `OpenAIBackend._generate_from_raw while targeting an ollama server, "
                    "your requests will fail since ollama doesn't support batching requests."
                )
            raise e

        # Necessary for type checker.
        assert isinstance(completion_response, Completion)

        results = []
        for response, action, prompt in zip(
            completion_response.choices, actions, prompts
        ):
            output = ModelOutputThunk(response.text)
            output._context = None  # There is no context for generate_from_raw for now
            output._action = action
            output._model_options = model_opts
            output._meta = {
                "oai_completion_response": response.model_dump(),
                "usage": completion_response.usage.model_dump()
                if completion_response.usage
                else None,
            }

            output.parsed_repr = (
                action.parse(output) if isinstance(action, Component) else output.value
            )

            generate_log = GenerateLog()
            generate_log.prompt = prompt
            generate_log.backend = f"openai::{self.model_id!s}"
            generate_log.model_options = model_opts
            generate_log.date = datetime.datetime.now()
            generate_log.model_output = completion_response
            generate_log.extra = {"seed": model_opts.get("seed", None)}
            generate_log.action = action
            output._generate_log = generate_log

            results.append(output)

        return results

    @property
    def base_model_name(self):
        """Returns the base_model_id of the model used by the backend. For example, `granite-3.3-8b-instruct` for `ibm-granite/granite-3.3-8b-instruct`."""
        if "/" in self._model_id:
            return self._model_id.split("/")[1]
        else:
            return self._model_id

    def add_adapter(self, adapter: OpenAIAdapter):
        """Adds the given adapter to the backend. Must not have been added to a different backend."""
        if adapter.backend is not None:
            if adapter.backend is self:
                FancyLogger.get_logger().warning(
                    f"attempted to add adapter {adapter.name} with type {adapter.adapter_type} to the same backend {adapter.backend}"
                )
                return
            else:
                raise Exception(
                    f"adapter {adapter.name} with type {adapter.adapter_type} has already been added to backend {adapter.backend}"
                )

        if self._added_adapters.get(adapter.qualified_name, None) is not None:
            FancyLogger.get_logger().warning(
                f"Client code attempted to add {adapter.name} with type {adapter.adapter_type} but it was already added to {self.__class__}. This attempt to add the adapter will be ignored."
            )
            return None

        adapter.path = adapter.get_open_ai_path(
            self.base_model_name, server_type=self._server_type
        )
        adapter.backend = self
        self._added_adapters[adapter.qualified_name] = adapter

    def load_adapter(self, adapter_qualified_name: str):
        """Loads the given adapter for the backend. Must have previously been added."""
        adapter = self._added_adapters.get(adapter_qualified_name, None)
        if adapter is None:
            raise ValueError(
                f"could not load adapter {adapter_qualified_name} for backend {self}: adapter was not previously added"
            )

        url = f"{self._base_url}/load_lora_adapter"
        response = requests.post(
            url,
            json={"lora_name": adapter_qualified_name, "lora_path": adapter.path},
            headers={"Content-Type": "application/json"},
        )

        err: str | None = None
        match response.status_code:
            case 200:
                FancyLogger.get_logger().info(
                    f"{url}: status {response.status_code} {response.text}"
                )
            case 400:
                if "has already been loaded." in str(response.content):
                    FancyLogger.get_logger().warning(
                        f"{url}: status {response.status_code} {response.text}"
                    )
                else:
                    err = f"{url}: status {response.status_code} {response.text}"
            case _:
                err = f"{url}: status {response.status_code} {response.text}"

        if err is not None:
            FancyLogger.get_logger().error(err)
            raise Exception(f"error loading adapter {adapter_qualified_name}: {err}")

        self._loaded_adapters[adapter.qualified_name] = adapter

    def unload_adapter(self, adapter_qualified_name: str):
        """Unloads the given adapter from the backend."""
        # Check if the backend knows about this adapter.
        adapter = self._loaded_adapters.get(adapter_qualified_name, None)
        if adapter is None:
            FancyLogger.get_logger().info(
                f"could not unload adapter {adapter_qualified_name} for backend {self}: adapter is not loaded"
            )
            return

        url = f"{self._base_url}/unload_lora_adapter"
        response = requests.post(
            url,
            json={"lora_name": adapter_qualified_name},
            headers={"Content-Type": "application/json"},
        )

        match response.status_code:
            case 200:
                FancyLogger.get_logger().info(
                    f"{url}: status {response.status_code} {response.text}"
                )
            case 404:
                # This response code indicates that the adapter isn't currently loaded;
                # which is the goal of this function. Log it but proceed as if successful.
                FancyLogger.get_logger().info(
                    f"{url}: status {response.status_code} {response.text}"
                )
            case _:
                # Unknown err.
                FancyLogger.get_logger().error(
                    f"{url}: status {response.status_code} {response.text}"
                )
                raise Exception(
                    f"error unloading adapter {adapter_qualified_name}: {url}: status {response.status_code} {response.text}"
                )

        # Remove the adapter from the list of loaded adapters.
        del self._loaded_adapters[adapter.qualified_name]

    def list_adapters(self) -> list[str]:
        """Lists the adapters added via add_adapter().

        :returns: list of adapter names that are currently registered with this backend
        """
        return list(self._loaded_adapters.keys())
