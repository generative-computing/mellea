# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""A generic WatsonX.ai compatible backend that wraps around the watson_machine_learning library."""

import asyncio
import datetime
import functools
import json
import os
import warnings
from collections.abc import AsyncGenerator, Coroutine, Sequence
from dataclasses import fields
from typing import Any

try:
    from ibm_watsonx_ai import APIClient, Credentials
    from ibm_watsonx_ai.foundation_models import ModelInference
    from ibm_watsonx_ai.foundation_models.schema import TextChatParameters
except ImportError as e:
    raise ImportError(
        "The Watsonx backend requires extra dependencies. "
        'Please install them with: pip install "mellea[watsonx]"'
    ) from e

from ..backends import ModelIdentifier, model_ids
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
    RawProviderResponse,
)
from ..core.base import AbstractMelleaTool
from ..formatters import ChatFormatter, TemplateFormatter
from ..helpers import (
    DEFAULT_CHUNK_TIMEOUT,
    ClientCache,
    chat_completion_delta_merge,
    extract_model_tool_requests,
    get_current_event_loop,
    send_to_queue,
    should_replay_reasoning,
)
from ..stdlib.components import Message
from ..stdlib.requirements import ALoraRequirement
from ..telemetry.context import generate_request_id, with_context
from .backend import FormatterBackend
from .model_options import ModelOption
from .tools import (
    add_tools_from_context_actions,
    add_tools_from_model_options,
    convert_tools_to_json,
    validate_tool_arguments,
)
from .utils import populate_response_metadata_openai_shape

format: None = None  # typing this variable in order to shadow the global format function and ensure mypy checks for errors


class WatsonxAIBackend(FormatterBackend):
    """A generic backend class for watsonx SDK.

    Args:
        model_id (str | ModelIdentifier): WatsonX model identifier. Defaults to
            `model_ids.IBM_GRANITE_4_HYBRID_SMALL`.
        formatter (ChatFormatter | None): Formatter for rendering components.
            Defaults to `TemplateFormatter`.
        base_url (str | None): URL for the WatsonX ML deployment endpoint;
            defaults to the `WATSONX_URL` environment variable.
        model_options (dict | None): Default model options for generation requests.
        api_key (str | None): WatsonX API key; defaults to the
            `WATSONX_API_KEY` environment variable.
        project_id (str | None): WatsonX project ID; defaults to the
            `WATSONX_PROJECT_ID` environment variable.

    Attributes:
        to_mellea_model_opts_map_chats (dict): Mapping from chat-endpoint option names
            to Mellea `ModelOption` sentinel keys.
        from_mellea_model_opts_map_chats (dict): Mapping from Mellea sentinel keys to
            chat-endpoint option names.
        to_mellea_model_opts_map_completions (dict): Mapping from completions-endpoint
            option names to Mellea `ModelOption` sentinel keys.
        from_mellea_model_opts_map_completions (dict): Mapping from Mellea sentinel
            keys to completions-endpoint option names.
    """

    def __init__(
        self,
        model_id: str | ModelIdentifier = model_ids.IBM_GRANITE_4_HYBRID_SMALL,
        formatter: ChatFormatter | None = None,
        base_url: str | None = None,
        model_options: dict | None = None,
        *,
        api_key: str | None = None,
        project_id: str | None = None,
        **kwargs,
    ):
        """Initialize a WatsonX AI backend using the ibm_watsonx_ai SDK."""
        # There are bugs with the Watsonx python sdk related to async event loops;
        # using the same watsonx backend across multiple event loops causes errors.
        warnings.warn(
            "Watsonx Backend is deprecated, use 'LiteLLM' or 'OpenAI' Backends instead",
            DeprecationWarning,
            2,
        )

        super().__init__(
            model_id=model_id,
            formatter=(
                formatter
                if formatter is not None
                else TemplateFormatter(model_id=model_id)
            ),
            model_options=model_options,
        )

        # Resolve to a concrete watsonx model name; assertion fires if no watsonx_name.
        watsonx_model_id = (
            model_id.watsonx_name if isinstance(model_id, ModelIdentifier) else model_id
        )
        assert watsonx_model_id is not None, (
            "model_id is None: the ModelIdentifier has no watsonx_name configured, or this model is not available in watsonx."
        )
        self._model_id: str = watsonx_model_id
        self._provider: str = "watsonx"

        if base_url is None:
            base_url = os.environ.get("WATSONX_URL")
        if api_key is None:
            api_key = os.environ.get("WATSONX_API_KEY")

        if project_id is None:
            project_id = os.environ.get("WATSONX_PROJECT_ID")
        self._project_id = project_id
        self._base_url = base_url
        self._api_key = api_key

        self._creds = Credentials(url=base_url, api_key=api_key)
        self._kwargs = kwargs

        self._client_cache = ClientCache(2)

        # Call once to set up the model inference and prepopulate the cache.
        _ = self._model

        # A mapping of common options for this backend mapped to their Mellea ModelOptions equivalent.
        # These are usually values that must be extracted before hand or that are common among backend providers.
        self.to_mellea_model_opts_map_chats = {
            "system": ModelOption.SYSTEM_PROMPT,
            "max_tokens": ModelOption.MAX_NEW_TOKENS,  # Is being deprecated in favor of `max_completion_tokens.`
            "max_completion_tokens": ModelOption.MAX_NEW_TOKENS,
            "tools": ModelOption.TOOLS,
            "stream": ModelOption.STREAM,
            "stop": ModelOption.STOP_SEQUENCES,
        }
        # A mapping of Mellea specific ModelOptions to the specific names for this backend.
        # These options should almost always be a subset of those specified in the `to_mellea_model_opts_map`.
        # Usually, values that are intentionally extracted while prepping for the backend generate call
        # will be omitted here so that they will be removed when model_options are processed
        # for the call to the model.
        self.from_mellea_model_opts_map_chats = {
            ModelOption.MAX_NEW_TOKENS: "max_completion_tokens",
            ModelOption.STOP_SEQUENCES: "stop",
        }

        # See notes above.
        self.to_mellea_model_opts_map_completions = {
            "random_seed": ModelOption.SEED,
            "max_new_tokens": ModelOption.MAX_NEW_TOKENS,
            "stream": ModelOption.STREAM,
            "stop_sequences": ModelOption.STOP_SEQUENCES,
        }
        # See notes above.
        self.from_mellea_model_opts_map_completions = {
            ModelOption.SEED: "random_seed",
            ModelOption.MAX_NEW_TOKENS: "max_new_tokens",
            ModelOption.STOP_SEQUENCES: "stop_sequences",
        }

    def __repr__(self) -> str:
        """Mask the API key to prevent accidental exposure in logs."""
        key_repr = "'***'" if self._api_key is not None else "None"
        return (
            f"{self.__class__.__name__}("
            f"model_id={self._model_id!r}, "
            f"base_url={self._base_url!r}, "
            f"_api_key={key_repr})"
        )

    def __str__(self) -> str:
        """Mask the API key to prevent accidental exposure in logs."""
        return repr(self)

    @property
    def _model(self) -> ModelInference:
        """Watsonx's client gets tied to a specific event loop. Reset it if needed here."""
        key = id(get_current_event_loop())

        _model_inference = self._client_cache.get(key)
        if _model_inference is None:
            _client = APIClient(credentials=self._creds)
            _model_inference = ModelInference(
                model_id=self._model_id,
                api_client=_client,
                credentials=self._creds,
                project_id=self._project_id,
                params=self.model_options,
                **self._kwargs,
            )
            self._client_cache.put(key, _model_inference)
        return _model_inference

    def filter_chat_completions_kwargs(self, model_options: dict) -> dict:
        """Filter kwargs to only include valid watsonx chat.completions.create parameters.

        Args:
            model_options (dict): Model options dict that may contain non-chat keys.

        Returns:
            dict: A dict containing only keys accepted by the WatsonX chat endpoint.
        """
        # TextChatParameters.get_sample_params().keys() can't be completely trusted. It doesn't always contain all
        # all of the accepted keys. In version 1.3.39, max_tokens was removed even though it's still accepted.
        # It's a dataclass so use the fields function to get the names.
        chat_params = {field.name for field in fields(TextChatParameters)}
        return {k: v for k, v in model_options.items() if k in chat_params}

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
            is_chat_context: set to True if used for chat completion apis

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
        merged = ModelOption.merge_model_options(
            backend_model_opts, generate_call_model_opts
        )
        return merged

    def _make_backend_specific_and_remove(
        self, model_options: dict[str, Any], is_chat_context: bool
    ) -> dict[str, Any]:
        """Maps specified Mellea specific keys to their backend specific version and removes any remaining Mellea keys.

        Args:
            model_options: the model_options for this call
            is_chat_context: set to True if used for chat completion apis

        Returns:
            a new dict
        """
        remap_dict = self.from_mellea_model_opts_map_chats
        if not is_chat_context:
            remap_dict = self.from_mellea_model_opts_map_completions

        for opt, field in (
            (ModelOption.LOGITS, "generation.logits"),
            (ModelOption.RAW_LOGITS, "generation.raw_logits"),
        ):
            if model_options.get(opt) and opt not in self._warned_about:
                self._warned_about.add(opt)
                MelleaLogger.get_logger().warning(
                    f"{opt!r} is not supported by the WatsonX backend; {field} will be None."
                )

        backend_specific = ModelOption.replace_keys(model_options, remap_dict)

        if is_chat_context:
            model_opts = self.filter_chat_completions_kwargs(backend_specific)
        else:
            model_opts = ModelOption.remove_special_keys(backend_specific)

        return model_opts

    async def _generate_from_context(
        self,
        action: Component[C] | CBlock | ModelOutputThunk,
        ctx: Context,
        *,
        format: type[BaseModelSubclass] | None = None,
        model_options: dict | None = None,
        tool_calls: bool = False,
    ) -> tuple[ModelOutputThunk[C], Context]:
        """Generate a completion for `action` given `ctx` via the WatsonX chat API.

        Delegates to `generate_from_chat_context`. Only chat contexts are
        supported; raises `NotImplementedError` otherwise.

        Args:
            action (Component[C] | CBlock): The component or content block to generate
                a completion for.
            ctx (Context): The current generation context (must be a chat context).
            format (type[BaseModelSubclass] | None): Optional Pydantic model class for
                structured/constrained output decoding.
            model_options (dict | None): Per-call model options that override the
                backend's defaults.
            tool_calls (bool): If `True`, expose available tools to the model and
                parse tool-call responses.

        Returns:
            tuple[ModelOutputThunk[C], Context]: A thunk holding the (lazy) model output
                and an updated context that includes `action` and the new output.
        """
        assert ctx.is_chat_context, NotImplementedError(
            "The watsonx.ai backend only supports chat-like contexts."
        )

        _model_id_str = str(getattr(self, "model_id", "unknown"))
        with with_context(request_id=generate_request_id(), model_id=_model_id_str):
            mot = await self.generate_from_chat_context(
                action,
                ctx,
                _format=format,
                model_options=model_options,
                tool_calls=tool_calls,
            )

        return mot, ctx.add(action).add(mot)

    async def generate_from_chat_context(
        self,
        action: Component[C] | CBlock | ModelOutputThunk,
        ctx: Context,
        *,
        _format: type[BaseModelSubclass]
        | None = None,  # Type[BaseModelSubclass] is a class object of a subclass of BaseModel
        model_options: dict | None = None,
        tool_calls: bool = False,
    ) -> ModelOutputThunk[C]:
        """Generate a new completion from the provided context using this backend's formatter.

        Formats the context and action into WatsonX-compatible chat messages, submits
        the request asynchronously, and returns a thunk that lazily resolves the output.

        Args:
            action (Component[C] | CBlock): The component or content block to generate
                a completion for.
            ctx (Context): The current generation context.
            _format (type[BaseModelSubclass] | None): Optional Pydantic model class for
                structured output decoding.
            model_options (dict | None): Per-call model options.
            tool_calls (bool): If `True`, expose available tools and parse responses.

        Returns:
            ModelOutputThunk[C]: A thunk holding the (lazy) model output.

        Raises:
            Exception: If `action` is an `ALoraRequirement`, which is not
                supported by this backend.
            RuntimeError: If not called from a thread with a running event loop.
        """
        await self.do_generate_walk(action)

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
                    "The watsonx backend does not currently support aLoRA adapters."
                )
            case _:
                messages.extend(self.formatter.to_chat_messages([action]))

        conversation: list[dict] = []
        system_prompt = model_opts.get(ModelOption.SYSTEM_PROMPT, "")
        if system_prompt != "":
            conversation.append({"role": "system", "content": system_prompt})

        # NOTE: `self.formatter.to_chat_messages` explicitly skips `Message` objects. However, we need
        # to print `Message`s to correctly serialize any documents with the message. Do the printing here.
        replay_flags = should_replay_reasoning(messages, self._provider)
        for m, replay in zip(messages, replay_flags):
            if m.images:
                raise ValueError(
                    "WatsonxAIBackend does not support image inputs (ImageBlock/ImageUrlBlock). "
                    "Remove image blocks before passing messages to Watsonx."
                )
            if m.audio:
                raise ValueError(
                    "WatsonxAIBackend does not support audio inputs (AudioBlock/AudioUrlBlock). "
                    "Remove audio blocks before passing messages to Watsonx."
                )
            message_dict: dict[str, Any] = {
                "role": m.role,
                "content": self.formatter.print(m),
            }
            if replay and m.thinking:
                message_dict["reasoning_content"] = m.thinking
            conversation.append(message_dict)

        if _format is not None:
            model_opts["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": _format.__name__,
                    "schema": _format.model_json_schema(),  # type: ignore
                    "strict": True,
                },
            }
        else:
            model_opts["response_format"] = {"type": "text"}

        # Append tool call information if applicable.
        tools: dict[str, AbstractMelleaTool] = {}
        if tool_calls:
            if _format:
                MelleaLogger.get_logger().warning(
                    f"tool calling is superseded by format; will not call tools for request: {action}"
                )
            else:
                add_tools_from_model_options(tools, model_opts)
                add_tools_from_context_actions(tools, ctx.actions_for_available_tools())

                # Add the tools from the action for this generation last so that
                # they overwrite conflicting names.
                add_tools_from_context_actions(tools, [action])
            MelleaLogger.get_logger().info(f"Tools for call: {tools.keys()}")

        formatted_tools = convert_tools_to_json(tools)

        chat_response: (
            Coroutine[Any, Any, AsyncGenerator] | Coroutine[Any, Any, dict] | None
        ) = None

        stream = model_opts.get(ModelOption.STREAM, False)
        if stream:
            chat_response = self._model.achat_stream(
                messages=conversation,
                tools=formatted_tools,
                tool_choice_option=(
                    "auto" if formatted_tools and len(formatted_tools) > 0 else "none"
                ),
                params=self._make_backend_specific_and_remove(
                    model_opts, is_chat_context=ctx.is_chat_context
                ),
            )
        else:
            chat_response = self._model.achat(
                messages=conversation,
                tools=formatted_tools,
                tool_choice_option=(
                    "auto" if formatted_tools and len(formatted_tools) > 0 else "none"
                ),
                params=self._make_backend_specific_and_remove(
                    model_opts, is_chat_context=ctx.is_chat_context
                ),
            )

        output = ModelOutputThunk(None)
        output._gen.start = datetime.datetime.now()
        output._call.context = linearized_context
        output._call.action = action
        output._call.model_options = model_opts

        # Processing functions only pass the ModelOutputThunk (and current chunk of response). Bind the other vars necessary for
        # each processing step.
        output._gen.process = self.processing
        output._gen.post_process = functools.partial(
            self.post_processing,
            conversation=conversation,
            tools=tools,
            seed=model_opts.get(ModelOption.SEED, None),
            _format=_format,
        )

        # Set model/provider early so they are available in the error path
        output.generation.model = self._model_id
        output.generation.provider = self._provider

        try:
            # To support lazy computation, will need to remove this create_task and store just the unexecuted coroutine.
            # We can also support synchronous calls by adding a flag and changing this ._gen.generate function.

            # This function should always be called from a running event loop so we don't have to worry about
            # scheduling the task to a specific event loop here.
            output._gen.generate = asyncio.create_task(
                send_to_queue(
                    chat_response,
                    output._gen.queue,
                    chunk_timeout=model_opts.get(
                        ModelOption.STREAM_TIMEOUT, DEFAULT_CHUNK_TIMEOUT
                    ),
                )
            )
            output._gen.generate_type = GenerateType.ASYNC
        except RuntimeError as e:
            # Most likely cause is running this function without an event loop present
            raise e

        return output

    async def processing(self, mot: ModelOutputThunk, chunk: dict):
        """Accumulate content from a single WatsonX response dict into the output thunk.

        Called for each non-streaming chat dict (with a `"message"` key) or
        streaming delta dict (with a `"delta"` key). Tool call parsing is
        handled in the post-processing step.

        Args:
            mot (ModelOutputThunk): The output thunk being populated.
            chunk (dict): A single response dict or streaming delta from the WatsonX API.
        """
        if mot.thinking is None:
            mot.thinking = ""
        if mot._underlying_value is None:
            mot._underlying_value = ""

        if len(chunk["choices"]) < 1:
            return  # Empty chunk. Note: this has some metadata information, but ignoring for now.

        # Watsonx returns dicts. Distinguish streaming and non-streaming based on their fields.
        not_streaming = chunk["choices"][0].get("message", None) is not None
        if not_streaming:
            message: dict = chunk["choices"][0].get("message", dict())

            thinking_chunk = message.get("reasoning_content", None)
            if thinking_chunk is not None:
                mot.thinking += thinking_chunk

            content_chunk = message.get("content", "")
            if content_chunk is not None:
                mot._underlying_value += content_chunk

            # Store full chunk (includes usage information).
            mot.raw.response = chunk

        else:  # Streaming.
            message_delta: dict = chunk["choices"][0].get("delta", dict())

            thinking_chunk = message_delta.get("reasoning_content", None)
            if thinking_chunk is not None:
                mot.thinking += thinking_chunk

            content_chunk = message_delta.get("content", None)
            if content_chunk is not None:
                mot._underlying_value += content_chunk

            if mot.raw.streamed_chunks is None:
                mot.raw.streamed_chunks = []
            mot.raw.streamed_chunks.append(chunk["choices"][0])

    async def post_processing(
        self,
        mot: ModelOutputThunk,
        conversation: list[dict],
        tools: dict[str, AbstractMelleaTool],
        seed,
        _format,
    ):
        """Finalize the output thunk after WatsonX generation completes.

        Reconstructs a merged chat response from streaming chunks if applicable,
        extracts any tool call requests, records token usage metrics, emits telemetry,
        and attaches the generate log.

        Args:
            mot (ModelOutputThunk): The output thunk to finalize.
            conversation (list[dict]): The chat conversation sent to the model,
                used for logging.
            tools (dict[str, AbstractMelleaTool]): Available tools, keyed by name.
            seed: The random seed used during generation, or `None`.
            _format: The structured output format class used during generation, if any.
        """
        # Reconstruct the top-level response from chunks if streamed.
        if mot.raw.streamed_chunks is not None:
            mot.raw.response = chat_completion_delta_merge(mot.raw.streamed_chunks)

        assert mot._call.action is not None, (
            "ModelOutputThunks should have their action assigned during generation"
        )
        assert mot._call.model_options is not None, (
            "ModelOutputThunks should have their model_opts assigned during generation"
        )

        # OpenAI streamed responses give you chunks of tool calls.
        # As a result, we have to store data between calls and only then
        # check for complete tool calls in the post_processing step.
        # Non-streaming stores a top-level response (index into choices); streaming
        # stores the already-merged choice dict (use directly).
        response = mot.raw.response
        assert response is not None
        choice_response = (
            response["choices"][0]
            if isinstance(response, dict) and "choices" in response
            else response
        )
        tool_chunk = extract_model_tool_requests(tools, choice_response)
        if tool_chunk is not None:
            if mot.tool_calls is None:
                mot.tool_calls = {}
            # Merge the tool_chunk dict.
            for key, val in tool_chunk.items():
                mot.tool_calls[key] = val

        # Populate usage when the response carries it (WatsonX uses OpenAI format).
        if usage := response.get("usage"):
            mot.generation.usage = usage

        # Populate model and provider metadata
        mot.generation.model = self._model_id
        mot.generation.provider = self._provider
        mot.raw.provider = self._provider

        # Populate response-side metadata for telemetry
        populate_response_metadata_openai_shape(mot, response)

        # Generate the log for this ModelOutputThunk.
        generate_log = GenerateLog()
        generate_log.prompt = conversation
        generate_log.backend = f"watsonx::{self.model_id!s}"
        generate_log.model_options = mot._call.model_options
        generate_log.date = datetime.datetime.now()
        generate_log.model_output = response
        generate_log.extra = {
            "format": _format,
            "tools_available": tools,
            "tools_called": mot.tool_calls,
            "seed": seed,
        }
        generate_log.result = mot
        generate_log.action = mot._call.action
        mot._generate_log = generate_log

    async def _generate_from_raw(
        self,
        actions: Sequence[Component[C] | CBlock],
        ctx: Context,
        *,
        format: type[BaseModelSubclass] | None = None,
        model_options: dict | None = None,
        tool_calls: bool = False,
    ) -> tuple[list[ModelOutputThunk], dict[str, Any] | None]:
        """Generate completions for multiple actions without chat templating via WatsonX.

        Passes formatted prompt strings directly to WatsonX's generate endpoint.
        The `format` parameter is not supported and will be ignored with a warning.

        Args:
            actions (Sequence[Component[C] | CBlock]): Actions to generate completions for.
            ctx (Context): The current generation context.
            format (type[BaseModelSubclass] | None): Not supported; ignored with a warning.
            model_options (dict | None): Per-call model options.
            tool_calls (bool): Ignored; tool calling is not supported on this endpoint.

        Returns:
            tuple[list[ModelOutputThunk], dict | None]: `(results, usage)` where
                `results` is a list of model output thunks, one per action, and
                `usage` is the aggregate token-usage dict for the batch.
        """
        await self.do_generate_walks(list(actions))

        if format is not None:
            MelleaLogger.get_logger().warning(
                "WatsonxAI completion api does not accept response format, ignoring it for this request."
            )

        model_opts = self._simplify_and_merge(model_options, is_chat_context=False)

        prompts = [self.formatter.print(action) for action in actions]

        responses = await asyncio.to_thread(
            self._model.generate,
            prompt=prompts,
            params=self._make_backend_specific_and_remove(
                model_opts, is_chat_context=False
            ),
        )

        results = []
        date = datetime.datetime.now()
        # All-or-nothing per-MOT usage; missing counts contribute 0 to the aggregate.
        agg_prompt = 0
        agg_completion = 0

        for i, response in enumerate(responses):
            output = response["results"][0]
            n_in = output.get("input_token_count")
            n_out = output.get("generated_token_count")
            per_mot_usage: dict[str, Any] | None
            if n_in is not None and n_out is not None:
                agg_prompt += n_in
                agg_completion += n_out
                per_mot_usage = {
                    "prompt_tokens": n_in,
                    "completion_tokens": n_out,
                    "total_tokens": n_in + n_out,
                }
            else:
                per_mot_usage = None
            result = ModelOutputThunk(value=output["generated_text"])
            result.raw = RawProviderResponse(
                provider=self._provider, response=response["results"][0]
            )
            result.generation.usage = per_mot_usage
            result.generation.model = self._model_id
            result.generation.provider = self._provider

            action = actions[i]
            result.parsed_repr = (
                action.parse(result) if isinstance(action, Component) else result.value
            )

            generate_log = GenerateLog()
            generate_log.prompt = prompts[i]
            generate_log.backend = f"watsonx::{self.model_id!s}"
            generate_log.model_options = model_opts
            generate_log.date = date
            generate_log.model_output = responses
            generate_log.extra = {
                "format": format,
                "seed": model_opts.get(ModelOption.SEED, None),
            }
            generate_log.action = action

            result._generate_log = generate_log

            results.append(result)

        usage: dict[str, Any] | None = (
            {
                "prompt_tokens": agg_prompt,
                "completion_tokens": agg_completion,
                "total_tokens": agg_prompt + agg_completion,
            }
            if (agg_prompt or agg_completion)
            else None
        )
        return results, usage

    def _extract_model_tool_requests(
        self, tools: dict[str, AbstractMelleaTool], chat_response: dict
    ) -> dict[str, ModelToolCall] | None:
        model_tool_calls: dict[str, ModelToolCall] = {}
        for tool_call in chat_response["choices"][0]["message"].get("tool_calls", []):
            tool_name = tool_call["function"]["name"]
            tool_args = tool_call["function"]["arguments"]

            func = tools.get(tool_name)
            if func is None:
                MelleaLogger.get_logger().warning(
                    f"model attempted to call a non-existing function: {tool_name}"
                )
                continue  # skip this function if we can't find it.

            # Watsonx returns the args as a string. Parse it here.
            args = json.loads(tool_args)

            # Validate and coerce argument types
            validated_args = validate_tool_arguments(func, args, strict=False)
            model_tool_calls[tool_name] = ModelToolCall(tool_name, func, validated_args)

        if len(model_tool_calls) > 0:
            return model_tool_calls
        return None
