"""A generic OpenAI compatible backend that wraps around the openai python sdk."""

import abc
import asyncio
import datetime
import functools
import inspect
import json
from collections.abc import Callable, Coroutine
from copy import deepcopy
from typing import TYPE_CHECKING, Any, cast

import granite_common
import openai
import requests
from openai.types.chat import ChatCompletion
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk
from openai.types.completion import Completion

import mellea.backends.model_ids as model_ids
from mellea.backends import BaseModelSubclass
from mellea.backends.adapters.adapter import (
    GraniteCommonAdapter,
    OpenAIAdapter,
    get_adapter_for_intrinsic,
)
from mellea.backends.aloras import AdapterMixin, Alora, AloraBackendMixin
from mellea.backends.formatter import Formatter, FormatterBackend, TemplateFormatter
from mellea.backends.model_ids import ModelIdentifier
from mellea.backends.tools import (
    add_tools_from_context_actions,
    add_tools_from_model_options,
    convert_tools_to_json,
)
from mellea.backends.types import ModelOption, _server_type, _ServerType
from mellea.helpers.async_helpers import (
    ClientCache,
    get_current_event_loop,
    send_to_queue,
)
from mellea.helpers.fancy_logger import FancyLogger
from mellea.helpers.openai_compatible_helpers import (
    chat_completion_delta_merge,
    extract_model_tool_requests,
)
from mellea.stdlib.base import (
    CBlock,
    Component,
    Context,
    Document,
    GenerateLog,
    GenerateType,
    ModelOutputThunk,
)
from mellea.stdlib.chat import Message
from mellea.stdlib.intrinsics.intrinsic import Intrinsic
from mellea.stdlib.requirement import ALoraRequirement, LLMaJRequirement, Requirement

if TYPE_CHECKING:
    from transformers.tokenization_utils import PreTrainedTokenizer

openai_ollama_batching_error = "json: cannot unmarshal array into Go struct field CompletionRequest.prompt of type string"

format: None = None  # typing this variable in order to shadow the global format function and ensure mypy checks for errors


class OpenAIBackend(FormatterBackend, AdapterMixin):
    """A generic OpenAI compatible backend."""

    def __init__(
        self,
        model_id: str | ModelIdentifier = model_ids.IBM_GRANITE_4_MICRO_3B,
        formatter: Formatter | None = None,
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

        self._model_id = model_id
        match model_id:
            case str():
                self._hf_model_id = model_id
            case ModelIdentifier():
                assert model_id.hf_model_name is not None, (
                    "model_id is None. This can also happen if the ModelIdentifier has no hf_model_id name set."
                )
                self._hf_model_id = model_id.hf_model_name

        if base_url is None:
            self._base_url = "http://localhost:11434/v1"  # ollama
        else:
            self._base_url = base_url
        if api_key is None:
            self._api_key = "ollama"
        else:
            self._api_key = api_key

        self._server_type = _server_type(self._base_url)

        self._openai_client_kwargs = self.filter_openai_client_kwargs(**kwargs)

        self._client = openai.OpenAI(  # type: ignore
            api_key=self._api_key, base_url=self._base_url, **self._openai_client_kwargs
        )

        self._client_cache = ClientCache(2)

        # Call once to create an async_client and populate the cache.
        _ = self._async_client

        # ALoras that have been loaded for this model.
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

    def generate_from_context(
        self,
        action: Component | CBlock,
        ctx: Context,
        *,
        format: type[BaseModelSubclass] | None = None,
        model_options: dict | None = None,
        tool_calls: bool = False,
    ):
        """See `generate_from_chat_context`."""
        assert ctx.is_chat_context, NotImplementedError(
            "The Openai backend only supports chat-like contexts."
        )
        mot = self.generate_from_chat_context(
            action,
            ctx,
            _format=format,
            model_options=model_options,
            tool_calls=tool_calls,
        )
        return mot, ctx.add(action).add(mot)

    def generate_from_chat_context(
        self,
        action: Component | CBlock,
        ctx: Context,
        *,
        _format: type[BaseModelSubclass]
        | None = None,  # Type[BaseModelSubclass] is a class object of a subclass of BaseModel
        model_options: dict | None = None,
        tool_calls: bool = False,
    ) -> ModelOutputThunk:
        """Generates a new completion from the provided Context using this backend's `Formatter`."""
        if issubclass(type(action), Requirement):
            # The general rule is that we reroute to the alora if it exists.
            reroute_to_alora = self.get_alora("constraint") is not None
            # However, there are some exceptions:
            if not self.default_to_constraint_checking_alora:
                reroute_to_alora = False
            if issubclass(type(action), LLMaJRequirement):
                reroute_to_alora = False
            if issubclass(type(action), ALoraRequirement):
                reroute_to_alora = True
            if reroute_to_alora:
                return self._generate_from_chat_context_alora(
                    action, ctx, _format=_format, model_options=model_options
                )

        elif isinstance(action, Intrinsic):
            return self._generate_from_intrinsic(
                action, ctx, model_options=model_options
            )

        return self._generate_from_chat_context_standard(
            action,
            ctx,
            _format=_format,
            model_options=model_options,
            tool_calls=tool_calls,
        )

    def _generate_from_chat_context_alora(
        self,
        action: Component | CBlock,
        ctx: Context,
        *,
        _format: type[BaseModelSubclass]
        | None = None,  # Type[BaseModelSubclass] is a class object of a subclass of BaseModel
        model_options: dict | None = None,
    ) -> ModelOutputThunk:
        match action:
            case ALoraRequirement():
                alora_for_this_request = (
                    self.get_alora("constraint")
                    if action.alora is None
                    else action.alora
                )
            case _:
                alora_for_this_request = self.get_alora("constraint")
                assert alora_for_this_request is not None, (
                    "This code block should not execute unless there is a 'constraint' alora loaded."
                )

        # Construct the linearized context. This is very similar to normal generation.
        linearized_ctx = ctx.view_for_generation()
        assert linearized_ctx is not None and len(linearized_ctx) > 1
        msgs = self.formatter.to_chat_messages(linearized_ctx)
        user_message, assistant_message = msgs[-2].content, msgs[-1].content
        assert alora_for_this_request is not None
        assert type(user_message) is str
        assert type(assistant_message) is str
        assert _format is None, "Structured outputs are not supported by ALoRAs."

        model_opts = self._simplify_and_merge(model_options, is_chat_context=True)

        alora_output = alora_for_this_request.generate_using_strings(
            input=user_message,
            response=assistant_message,
            constraint=action.description,  # type: ignore
            stream=model_opts.get(ModelOption.STREAM, False),
        )

        # The alora function doesn't set up all the fields.
        alora_output._context = linearized_ctx
        alora_output._action = action
        alora_output._model_options = model_options

        # TODO: Figure out what info we want to populate for aloras here.
        alora_output._generate_log = GenerateLog()

        return alora_output

    def _generate_from_intrinsic(
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
        )  # TODO: JAL. Log a warning if this is empty?...

        # Convert our linearized context into a sequence of chat messages. Template formatters have a standard way of doing this.
        messages: list[Message] = self.formatter.to_chat_messages(linearized_context)

        conversation: list[dict] = []

        # TODO: JAL. Need to handle system prompts? Confirm what granite common does.
        system_prompt = model_opts.get(ModelOption.SYSTEM_PROMPT, "")
        if system_prompt != "":
            conversation.append({"role": "system", "content": system_prompt})
        conversation.extend([self.message_to_openai_message(m) for m in messages])
        docs = self.messages_to_docs(messages)  # TODO: JAL. Improve docs interface.

        adapter = get_adapter_for_intrinsic(
            action.intrinsic_name, action.adapter_types, self._added_adapters
        )
        if adapter is None:
            raise ValueError(
                f"backend ({self}) has no adapter for processing intrinsic: {action.intrinsic_name}"
            )

        # TODO: Code below this point is mostly specific to RagIntrinsics (and granite_common).
        #       It should be refactored into a specific adapter.transform() function.
        assert isinstance(adapter, GraniteCommonAdapter)

        intrinsic_config = adapter.config
        if intrinsic_config is None:
            # If the adapter wasn't initialized with a config, grab one here based off the backend's model.
            intrinsic_config_file = granite_common.intrinsics.util.obtain_io_yaml(
                action.intrinsic_name, self._hf_model_id.split("/")[-1]
            )
            intrinsic_config = granite_common.intrinsics.util.make_config_dict(
                config_file=intrinsic_config_file
            )
            intrinsic_config = cast(
                dict, intrinsic_config
            )  # TODO: Can remove if util function gets exported properly.

        rewriter = granite_common.IntrinsicsRewriter(
            config_dict=intrinsic_config, model_name=adapter.qualified_name
        )
        result_processor = granite_common.IntrinsicsResultProcessor(
            config_dict=intrinsic_config
        )

        # Convert our conversation into a proper chat completions dict.
        # [{role: user, content: Hello}, {...}] -> {messages: [{role:user,...}, ...], model:..., ...}
        request_json: dict = {
            "messages": conversation,
            "extra_body": {"documents": docs},
        }

        rewritten = rewriter.transform(request_json, **action.intrinsic_kwargs)
        # TODO: JAL. Move this comment to hugging face.
        # TODO: Handle caching here. Need to see if granite_common gives us any indication
        #       of what messages have changed. We will also have to support caching at a
        #       Component / message level.

        # TODO: JAL. This needs to be made async with processing and post_processing.
        # chat_response: Coroutine[
        #     Any, Any, openai.AsyncStream[Completion] | Completion
        # ] = self._async_client.chat.completions.create(
        #     **rewritten.model_dump()
        # )
        self.load_adapter(adapter.qualified_name)
        chat_response = self._client.chat.completions.create(**rewritten.model_dump())

        processed_chat_completion = result_processor.transform(chat_response, rewritten)

        # TODO: JAL. Put into ModelOutputThunk, parse, etc...
        mot = ModelOutputThunk(processed_chat_completion.choices[0].message.content)
        mot._generate_log = GenerateLog()
        return mot

    @staticmethod
    def message_to_openai_message(msg: Message):
        """Serializes a mellea Message object to the message format required by OpenAI compatible api providers."""
        if msg.images is not None:
            img_list = [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{img}"},
                }
                for img in msg.images
            ]

            return {
                "role": msg.role,
                "content": [{"type": "text", "text": msg.content}, *img_list],
            }
        else:
            return {"role": msg.role, "content": msg.content}
            # Target format:
            # {
            #     "role": "user",
            #     "content": [
            #       {
            #         "type": "text",
            #         "text": "What's in this picture?"
            #       },
            #       {
            #         "type": "image_url",
            #         "image_url": {
            #           "url": "data:image/jpeg;base64,<base64_string>"
            #         }
            #       }
            #     ]
            #   }

    @staticmethod
    def messages_to_docs(msgs: list[Message]) -> list[dict[str, str]]:
        docs: list[Document] = []
        for message in msgs:
            if message._docs is not None:
                docs.extend(message._docs)

        # TODO: We can add doc_ids here for vllm.
        json_docs: list[dict[str, str]] = []
        for doc in docs:
            json_doc = {"text": doc.text}
            if doc.title is not None:
                json_doc["title"] = doc.title
            json_docs.append(json_doc)
        return json_docs

    def _generate_from_chat_context_standard(
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
        conversation.extend([self.message_to_openai_message(m) for m in messages])

        if _format is not None:
            response_format = {
                "type": "json_schema",
                "json_schema": {
                    "name": _format.__name__,
                    "schema": _format.model_json_schema(),
                    "strict": True,
                },
            }
        else:
            response_format = {"type": "text"}

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

        chat_response: Coroutine[
            Any, Any, ChatCompletion | openai.AsyncStream[ChatCompletionChunk]
        ] = self._async_client.chat.completions.create(
            model=self._hf_model_id,
            messages=conversation,  # type: ignore
            reasoning_effort=thinking,  # type: ignore
            response_format=response_format,  # type: ignore
            tools=formatted_tools if use_tools else None,  # type: ignore
            # parallel_tool_calls=False, # We only support calling one tool per turn. But we do the choosing on our side so we leave this False.
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

        self.formatter.parse(mot._action, mot)

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

    def _generate_from_raw(
        self,
        actions: list[Component | CBlock],
        *,
        format: type[BaseModelSubclass] | None = None,
        model_options: dict | None = None,
        generate_logs: list[GenerateLog] | None = None,
    ) -> list[ModelOutputThunk]:
        """Generate using the completions api. Gives the input provided to the model without templating."""
        extra_body = {}
        if format is not None:
            FancyLogger.get_logger().warning(
                "The official OpenAI completion api does not accept response format / structured decoding; "
                "it will be passed as an extra arg."
            )

            # Some versions (like vllm's version) of the OpenAI API support structured decoding for completions requests.
            extra_body["guided_json"] = format.model_json_schema()

        model_opts = self._simplify_and_merge(model_options, is_chat_context=False)

        prompts = [self.formatter.print(action) for action in actions]

        try:
            completion_response: Completion = self._client.completions.create(
                model=self._hf_model_id,
                prompt=prompts,
                extra_body=extra_body,
                **self._make_backend_specific_and_remove(
                    model_opts, is_chat_context=False
                ),
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

        results = [
            ModelOutputThunk(
                value=response.text,
                meta={"oai_completion_response": response.model_dump()},
            )
            for response in completion_response.choices
        ]

        for i, result in enumerate(results):
            self.formatter.parse(actions[i], result)

        if generate_logs is not None:
            assert isinstance(generate_logs, list)
            date = datetime.datetime.now()

            for i in range(len(prompts)):
                generate_log = GenerateLog()
                generate_log.prompt = prompts[i]
                generate_log.backend = f"openai::{self.model_id!s}"
                generate_log.model_options = model_opts
                generate_log.date = date
                generate_log.model_output = completion_response
                generate_log.extra = {"seed": model_opts.get("seed", None)}
                generate_log.action = actions[i]
                generate_log.result = results[i]
                generate_logs.append(generate_log)

        return results

    def add_adapter(self, adapter: OpenAIAdapter):
        # Gets the path / downloads files... ie does setup.
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

        base_model_name = self._hf_model_id.split("/")[-1]
        adapter.path = adapter.get_open_ai_path(
            base_model_name, server_type=self._server_type
        )

        if self._added_adapters.get(adapter.qualified_name, None) is not None:
            FancyLogger.get_logger().warning(
                f"Client code attempted to add {adapter.name} with type {adapter.adapter_type} but it was already added to {self.__class__}. This attempt to add the adapter will be ignored."
            )
            return None

        self._added_adapters[adapter.qualified_name] = adapter

    def load_adapter(self, adapter_qualified_name: str):
        # Actually loads the adapter...

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
        # Unloads the adapter...

        # Check if the backend knows about this adapter.
        adapter = self._added_adapters.get(adapter_qualified_name, None)
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

        # Remove the alora from the list of loaded adapters.
        del self._loaded_adapters[adapter.qualified_name]

    # TODO: JAL. Remove these functions.
    # def add_alora(self, alora: "OpenAIAlora"):
    #     """Loads an ALora for this backend.

    #     Args:
    #         alora (str): identifier for the ALora adapter
    #     """
    #     assert issubclass(alora.__class__, OpenAIAlora), (
    #         f"cannot add an ALora of type {alora.__class__} to model; must inherit from {OpenAIAlora.__class__}"
    #     )
    #     assert alora._backend == self, "Cannot load an ALora into the wrong backend."

    #     if self.get_alora(alora.name) is not None:
    #         FancyLogger.get_logger().warning(
    #             f"Client code attempted to add {alora.name} but {alora.name} was already added to {self.__class__}. The backend is refusing to do this, because ALora loading is not idempotent."
    #         )
    #         return None

    #     assert _server_type(self._base_url) == _ServerType.LOCALHOST, (
    #         "alora is supported only for locally running vllm instances"
    #     )

    #     # TODO: JAL. Make sure all hf model ids fit this way.
    #     # base_model_name = self._hf_model_id.split("/")[1]
    #     # TODO: JAL. change snapshot path to this...?
    #     # snapshot_path = granite_common.intrinsics.util.obtain_lora(
    #     #     alora.name, base_model_name, alora=True
    #     # )
    #     snapshot_path = f"/u/jakelorocco/eiger-user-folder/mellea-public/test/backends/test_openai_vllm/rag-intrinsics-lib/{alora.name}/alora/granite-3.3-8b-instruct"

    #     # https://docs.vllm.ai/en/stable/features/lora.html#using-api-endpoints
    #     # curl -X POST http://localhost:8000/v1/load_lora_adapter \
    #     #     -H "Content-Type: application/json" \
    #     #     -d '{
    #     #     "lora_name": "sql_adapter",
    #     #     "lora_path": "/path/to/sql-lora-adapter"
    #     #     }'

    #     url = f"{self._base_url}/load_lora_adapter"
    #     response = requests.post(
    #         url,
    #         json={"lora_name": alora.name, "lora_path": snapshot_path},
    #         headers={"Content-Type": "application/json"},
    #     )

    #     # TODO: Add a check here for the lora/alora already being loaded.
    #     # TODO: See what happens if you try load the lora with the same name...
    #     # TODO: If the alora isn't loaded, we should raise an error or at least not
    #     #       add it to the list of aloras.
    #     match response.status_code:
    #         case 200:
    #             FancyLogger.get_logger().info(
    #                 f"{url}: status {response.status_code} {response.text}"
    #             )
    #             self._aloras[alora.name] = alora
    #         case _:
    #             FancyLogger.get_logger().error(
    #                 f"{url}: status {response.status_code} {response.text}"
    #             )

    #     self._aloras[alora.name] = alora

    #     return None

    # def get_alora(self, alora_name: str) -> Alora | None:
    #     """Returns the ALora by name, or None if that ALora isn't loaded."""
    #     return self._aloras.get(alora_name)

    # def get_aloras(self) -> list[Alora]:
    #     """Returns a list of all loaded ALora adapters."""
    #     return list(self._aloras.values())

    def apply_chat_template(self, chat: list[dict[str, str]]):
        """Apply the chat template for the model, if such a model is available (e.g., when it can deduce the huggingface model id)."""
        from transformers import AutoTokenizer

        if not hasattr(self, "_tokenizer"):
            match _server_type(self._base_url):
                case _ServerType.LOCALHOST:
                    self._tokenizer: "PreTrainedTokenizer" = (  # noqa: UP037
                        AutoTokenizer.from_pretrained(self._hf_model_id)
                    )
                case _ServerType.OPENAI:
                    raise Exception(
                        "apply_chat_template is called while targeting a server at openai.com. "
                        "This is not supported --- openai.com does not support Activated Lora. "
                        "Use a locally served vllm instance. "
                    )

        return self._tokenizer.apply_chat_template(chat, tokenize=False)


class OpenAIAlora(Alora, abc.ABC):
    """ALoras that work with OpenAI backend."""

    def __init__(
        self, name: str, path: str, generation_prompt: str, backend: OpenAIBackend
    ):
        """Initialize an ALora that should work with OpenAI backends that support ALoras.

        Args:
            name (str): An arbitrary name/label to assign to an ALora. This is irrelevant from the alora's (huggingface) model id.
            path (str): A local path to ALora's weights or a Huggingface model_id to an ALora.
            generation_prompt (str): A prompt used to "activate" the Lora. This string goes between the pre-activation context and the aLora generate call. This needs to be provided by the entity that trained the ALora.
            backend (OpenAIBackend): Mained as a pointer to the backend to which this this ALora is attached.
        """
        super().__init__(name)
        self.path = path
        self._backend = backend
        self._generation_prompt = generation_prompt
