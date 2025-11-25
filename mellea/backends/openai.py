"""A generic OpenAI compatible backend that wraps around the openai python sdk."""

import abc
import asyncio
import datetime
import functools
import inspect
import json
from collections.abc import Callable, Coroutine, Sequence
from enum import Enum
from typing import TYPE_CHECKING, Any, overload
from urllib.parse import urlparse

import openai
import requests
from huggingface_hub import snapshot_download
from openai.types.chat import ChatCompletion
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk
from openai.types.completion import Completion

import mellea.backends.model_ids as model_ids
from mellea.backends import BaseModelSubclass
from mellea.backends.aloras import Alora, AloraBackendMixin
from mellea.backends.formatter import Formatter, FormatterBackend, TemplateFormatter
from mellea.backends.model_ids import ModelIdentifier
from mellea.backends.tools import (
    add_tools_from_context_actions,
    add_tools_from_model_options,
    convert_tools_to_json,
)
from mellea.backends.types import ModelOption
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
    GenerateLog,
    GenerateType,
    ModelOutputThunk,
)
from mellea.stdlib.chat import Message
from mellea.stdlib.requirement import ALoraRequirement, LLMaJRequirement, Requirement

if TYPE_CHECKING:
    from transformers.tokenization_utils import PreTrainedTokenizer

openai_ollama_batching_error = "json: cannot unmarshal array into Go struct field CompletionRequest.prompt of type string"

format: None = None  # typing this variable in order to shadow the global format function and ensure mypy checks for errors


class _ServerType(Enum):
    LOCALHOST = 1
    OPENAI = 2


def _server_type(url: str) -> _ServerType | None:
    try:
        parsed = urlparse(url)
        hostname = parsed.hostname
        if hostname in ("localhost", "127.0.0.1", "::1"):
            return _ServerType.LOCALHOST
        elif hostname == "api.openai.com":
            return _ServerType.OPENAI
    except Exception as e:
        print(f"Error parsing URL: {e}")
    return None


class OpenAIBackend(FormatterBackend, AloraBackendMixin):
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

        self._openai_client_kwargs = self.filter_openai_client_kwargs(**kwargs)

        self._client = openai.OpenAI(  # type: ignore
            api_key=self._api_key, base_url=self._base_url, **self._openai_client_kwargs
        )

        self._client_cache = ClientCache(2)

        # Call once to create an async_client and populate the cache.
        _ = self._async_client

        # ALoras that have been loaded for this model.
        self._aloras: dict[str, OpenAIAlora] = {}

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
        action: Component | CBlock | None,
        ctx: Context,
        *,
        format: type[BaseModelSubclass] | None = None,
        model_options: dict | None = None,
        tool_calls: bool = False,
        labels: Sequence[str] | None = None,
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
            labels=labels,
        )
        #
        if action is not None:
            ctx = ctx.add(action, labels=labels)

        # return
        return mot, ctx.add(mot, labels=labels)

    def generate_from_chat_context(
        self,
        action: Component | CBlock | None,
        ctx: Context,
        *,
        _format: (
            type[BaseModelSubclass] | None
        ) = None,  # Type[BaseModelSubclass] is a class object of a subclass of BaseModel
        model_options: dict | None = None,
        tool_calls: bool = False,
        labels: Sequence[str] | None = None,
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

        return self._generate_from_chat_context_standard(
            action,
            ctx,
            _format=_format,
            model_options=model_options,
            tool_calls=tool_calls,
            labels=labels,
        )

    def _generate_from_chat_context_alora(
        self,
        action: Component | CBlock,
        ctx: Context,
        *,
        _format: (
            type[BaseModelSubclass] | None
        ) = None,  # Type[BaseModelSubclass] is a class object of a subclass of BaseModel
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

    def _generate_from_chat_context_standard(
        self,
        action: Component | CBlock | None,
        ctx: Context,
        *,
        _format: (
            type[BaseModelSubclass] | None
        ) = None,  # Type[BaseModelSubclass] is a class object of a subclass of BaseModel
        model_options: dict | None = None,
        tool_calls: bool = False,
        labels: Sequence[str] | None = None,
    ) -> ModelOutputThunk:
        model_opts = self._simplify_and_merge(
            model_options, is_chat_context=ctx.is_chat_context
        )
        linearized_context = ctx.view_for_generation(labels=labels)
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
            case None:
                action = ctx.node_data  # action defaults to node_data if None provided
            case _:
                messages.extend(self.formatter.to_chat_messages([action]))
        conversation: list[dict] = []

        system_prompt = model_opts.get(ModelOption.SYSTEM_PROMPT, "")
        if system_prompt != "":
            conversation.append({"role": "system", "content": system_prompt})
        conversation.extend([self.message_to_openai_message(m) for m in messages])

        extra_params: dict[str, Any] = {}
        if _format is not None:
            extra_params["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": _format.__name__,
                    "schema": _format.model_json_schema(),
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

        chat_response: Coroutine[
            Any, Any, ChatCompletion | openai.AsyncStream[ChatCompletionChunk]
        ] = self._async_client.chat.completions.create(
            model=self._hf_model_id,
            messages=conversation,  # type: ignore
            reasoning_effort=thinking,  # type: ignore
            tools=formatted_tools if use_tools else None,  # type: ignore
            # parallel_tool_calls=False, # We only support calling one tool per turn. But we do the choosing on our side so we leave this False.
            **extra_params,
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

    def generate_from_raw(
        self,
        actions: list[Component | CBlock],
        ctx: Context,
        *,
        format: type[BaseModelSubclass] | None = None,
        model_options: dict | None = None,
        tool_calls: bool = False,
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
        if tool_calls:
            FancyLogger.get_logger().warning(
                "The completion endpoint does not support tool calling at the moment."
            )

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

        results = []
        for response, action, prompt in zip(
            completion_response.choices, actions, prompts
        ):
            output = ModelOutputThunk(None)
            output.value = response.text
            output._context = None  # There is no context for generate_from_raw for now
            output._action = action
            output._model_options = model_opts
            output._meta = {
                "oai_completion_response": response.model_dump(),
                "usage": (
                    completion_response.usage.model_dump()
                    if completion_response.usage
                    else None
                ),
            }

            self.formatter.parse(action, output)

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

    def add_alora(self, alora: "OpenAIAlora"):
        """Loads an ALora for this backend.

        Args:
            alora (str): identifier for the ALora adapter
        """
        assert issubclass(alora.__class__, OpenAIAlora), (
            f"cannot add an ALora of type {alora.__class__} to model; must inherit from {OpenAIAlora.__class__}"
        )
        assert alora._backend == self, "Cannot load an ALora into the wrong backend."

        if self.get_alora(alora.name) is not None:
            FancyLogger.get_logger().warning(
                f"Client code attempted to add {alora.name} but {alora.name} was already added to {self.__class__}. The backend is refusing to do this, because ALora loading is not idempotent."
            )
            return None

        assert _server_type(self._base_url) == _ServerType.LOCALHOST, (
            "alora is supported only for locally running vllm instances"
        )

        snapshot_path = snapshot_download(alora.path)

        # https://docs.vllm.ai/en/stable/features/lora.html#using-api-endpoints
        # curl -X POST http://localhost:8000/v1/load_lora_adapter \
        #     -H "Content-Type: application/json" \
        #     -d '{
        #     "lora_name": "sql_adapter",
        #     "lora_path": "/path/to/sql-lora-adapter"
        #     }'

        url = f"{self._base_url}/load_lora_adapter"
        response = requests.post(
            url,
            json={"lora_name": alora.name, "lora_path": snapshot_path},
            headers={"Content-Type": "application/json"},
        )

        match response.status_code:
            case 200:
                FancyLogger.get_logger().info(
                    f"{url}: status {response.status_code} {response.text}"
                )
                self._aloras[alora.name] = alora
            case _:
                FancyLogger.get_logger().error(
                    f"{url}: status {response.status_code} {response.text}"
                )

        self._aloras[alora.name] = alora

        return None

    def get_alora(self, alora_name: str) -> Alora | None:
        """Returns the ALora by name, or None if that ALora isn't loaded."""
        return self._aloras.get(alora_name)

    def get_aloras(self) -> list[Alora]:
        """Returns a list of all loaded ALora adapters."""
        return list(self._aloras.values())

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
