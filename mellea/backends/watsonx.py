"""A generic WatsonX.ai compatible backend that wraps around the watson_machine_learning library."""

import asyncio
import datetime
import functools
import json
import os
import warnings
from collections.abc import AsyncGenerator, Callable, Coroutine, Sequence
from dataclasses import fields
from typing import Any, overload

from ibm_watsonx_ai import APIClient, Credentials
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.foundation_models.schema import TextChatParameters

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
    ModelToolCall,
)
from ..core.base import AbstractMelleaTool
from ..formatters import ChatFormatter, TemplateFormatter
from ..helpers import (
    ClientCache,
    chat_completion_delta_merge,
    extract_model_tool_requests,
    get_current_event_loop,
    send_to_queue,
)
from ..security import SecLevel, taint_sources
from ..stdlib.components import Message
from ..stdlib.requirements import ALoraRequirement
from ..telemetry.backend_instrumentation import (
    instrument_generate_from_raw,
    start_generate_span,
)
from .backend import FormatterBackend
from .model_options import ModelOption
from .tools import (
    add_tools_from_context_actions,
    add_tools_from_model_options,
    convert_tools_to_json,
    validate_tool_arguments,
)

format: None = None  # typing this variable in order to shadow the global format function and ensure mypy checks for errors


class WatsonxAIBackend(FormatterBackend):
    """A generic backend class for watsonx SDK."""

    def __init__(
        self,
        model_id: str | ModelIdentifier = model_ids.IBM_GRANITE_3_3_8B,
        formatter: ChatFormatter | None = None,
        base_url: str | None = None,
        model_options: dict | None = None,
        *,
        api_key: str | None = None,
        project_id: str | None = None,
        **kwargs,
    ):
        """A generic watsonx backend that wraps around the ibm_watsonx_ai sdk.

        Args:
            model_id  : Model id. Defaults to model_ids.IBM_GRANITE_3_3_8B.
            formatter : input formatter. Defaults to TemplateFormatter in __init__.
            base_url  : url for watson ML deployment. Defaults to env(WATSONX_URL).
            model_options : Global model options to pass to the model. Defaults to None.
            api_key : watsonx API key. Defaults to None.
            project_id : watsonx project ID. Defaults to None.
            kwargs : extra kwargs passed to model inference creation.
        """
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
        self._model_id = model_id

        if base_url is None:
            base_url = os.environ.get("WATSONX_URL")
        if api_key is None:
            api_key = os.environ.get("WATSONX_API_KEY")

        if project_id is None:
            project_id = os.environ.get("WATSONX_PROJECT_ID")
        self._project_id = project_id

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
        }
        # A mapping of Mellea specific ModelOptions to the specific names for this backend.
        # These options should almost always be a subset of those specified in the `to_mellea_model_opts_map`.
        # Usually, values that are intentionally extracted while prepping for the backend generate call
        # will be omitted here so that they will be removed when model_options are processed
        # for the call to the model.
        self.from_mellea_model_opts_map_chats = {
            ModelOption.MAX_NEW_TOKENS: "max_completion_tokens"
        }

        # See notes above.
        self.to_mellea_model_opts_map_completions = {
            "random_seed": ModelOption.SEED,
            "max_new_tokens": ModelOption.MAX_NEW_TOKENS,
            "stream": ModelOption.STREAM,
        }
        # See notes above.
        self.from_mellea_model_opts_map_completions = {
            ModelOption.SEED: "random_seed",
            ModelOption.MAX_NEW_TOKENS: "max_new_tokens",
        }

    @property
    def _model(self) -> ModelInference:
        """Watsonx's client gets tied to a specific event loop. Reset it if needed here."""
        key = id(get_current_event_loop())

        _model_inference = self._client_cache.get(key)
        if _model_inference is None:
            _client = APIClient(credentials=self._creds)
            _model_inference = ModelInference(
                model_id=self._get_watsonx_model_id(),
                api_client=_client,
                credentials=self._creds,
                project_id=self._project_id,
                params=self.model_options,
                **self._kwargs,
            )
            self._client_cache.put(key, _model_inference)
        return _model_inference

    def _get_watsonx_model_id(self) -> str:
        """Gets the watsonx model id from the model_id that was provided in the constructor. Raises AssertionError if the ModelIdentifier does not provide a watsonx_name."""
        watsonx_model_id = (
            self.model_id.watsonx_name
            if isinstance(self.model_id, ModelIdentifier)
            else self.model_id
        )
        assert watsonx_model_id is not None, (
            "model_id is None. This can also happen if the ModelIdentifier has no watsonx name set or this model is not available in watsonx."
        )
        return watsonx_model_id

    def filter_chat_completions_kwargs(self, model_options: dict) -> dict:
        """Filter kwargs to only include valid watsonx chat.completions.create parameters."""
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
        return ModelOption.merge_model_options(
            backend_model_opts, generate_call_model_opts
        )

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

        backend_specific = ModelOption.replace_keys(model_options, remap_dict)

        if is_chat_context:
            model_opts = self.filter_chat_completions_kwargs(backend_specific)
        else:
            model_opts = ModelOption.remove_special_keys(backend_specific)

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
            "The watsonx.ai backend only supports chat-like contexts."
        )
        span = start_generate_span(
            backend=self, action=action, ctx=ctx, format=format, tool_calls=tool_calls
        )
        mot = await self.generate_from_chat_context(
            action,
            ctx,
            _format=format,
            model_options=model_options,
            tool_calls=tool_calls,
        )

        # Store span in metadata for post_processing to record telemetry
        if span is not None:
            mot._meta["_telemetry_span"] = span

        return mot, ctx.add(action).add(mot)

    async def generate_from_chat_context(
        self,
        action: Component[C] | CBlock,
        ctx: Context,
        *,
        _format: type[BaseModelSubclass]
        | None = None,  # Type[BaseModelSubclass] is a class object of a subclass of BaseModel
        model_options: dict | None = None,
        tool_calls: bool = False,
    ) -> ModelOutputThunk[C]:
        """Generates a new completion from the provided Context using this backend's `Formatter`."""
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
                    "The watsonx backend does not support currently support activated LoRAs."
                )
            case _:
                messages.extend(self.formatter.to_chat_messages([action]))

        conversation: list[dict] = []
        system_prompt = model_opts.get(ModelOption.SYSTEM_PROMPT, "")
        if system_prompt != "":
            conversation.append({"role": "system", "content": system_prompt})
        conversation.extend([{"role": m.role, "content": m.content} for m in messages])

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
                FancyLogger.get_logger().warning(
                    f"tool calling is superseded by format; will not call tools for request: {action}"
                )
            else:
                add_tools_from_model_options(tools, model_opts)
                add_tools_from_context_actions(tools, ctx.actions_for_available_tools())

                # Add the tools from the action for this generation last so that
                # they overwrite conflicting names.
                add_tools_from_context_actions(tools, [action])
            FancyLogger.get_logger().info(f"Tools for call: {tools.keys()}")

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

        # Compute taint sources from action and context
        sources = taint_sources(action, ctx)
        sec_level = SecLevel.tainted_by(sources) if sources else SecLevel.none()

        output = ModelOutputThunk(value=None, sec_level=sec_level, meta={})
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

    async def processing(self, mot: ModelOutputThunk, chunk: dict):
        """Called during generation to add information from a single ChatCompletion or ChatCompletionChunk to the ModelOutputThunk.

        For OpenAI-like APIs, tool call parsing is handled in the post processing step.
        """
        if mot._thinking is None:
            mot._thinking = ""
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
                mot._thinking += thinking_chunk

            content_chunk = message.get("content", "")
            if content_chunk is not None:
                mot._underlying_value += content_chunk

            mot._meta["oai_chat_response"] = chunk["choices"][0]

        else:  # Streaming.
            message_delta: dict = chunk["choices"][0].get("delta", dict())

            thinking_chunk = message_delta.get("reasoning_content", None)
            if thinking_chunk is not None:
                mot._thinking += thinking_chunk

            content_chunk = message_delta.get("content", None)
            if content_chunk is not None:
                mot._underlying_value += content_chunk

            if mot._meta.get("oai_chat_response_streamed", None) is None:
                mot._meta["oai_chat_response_streamed"] = []
            mot._meta["oai_chat_response_streamed"].append(chunk["choices"][0])

    async def post_processing(
        self,
        mot: ModelOutputThunk,
        conversation: list[dict],
        tools: dict[str, AbstractMelleaTool],
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

        # Record telemetry if span is available
        span = mot._meta.get("_telemetry_span")
        if span is not None:
            from ..telemetry import end_backend_span
            from ..telemetry.backend_instrumentation import (
                record_response_metadata,
                record_token_usage,
            )

            response = mot._meta.get("oai_chat_response")
            if response is not None:
                # Watsonx responses may have usage information
                usage = (
                    response.get("usage")
                    if isinstance(response, dict)
                    else getattr(response, "usage", None)
                )
                if usage:
                    record_token_usage(span, usage)
                record_response_metadata(span, response)

            # Close the span now that async operation is complete
            end_backend_span(span)
            # Clean up span reference
            del mot._meta["_telemetry_span"]

        # Generate the log for this ModelOutputThunk.
        generate_log = GenerateLog()
        generate_log.prompt = conversation
        generate_log.backend = f"watsonx::{self.model_id!s}"
        generate_log.model_options = mot._model_options
        generate_log.date = datetime.datetime.now()
        generate_log.model_output = mot._meta["oai_chat_response"]
        generate_log.extra = {
            "format": _format,
            "tools_available": tools,
            "tools_called": mot.tool_calls,
            "seed": seed,
        }
        generate_log.result = mot
        generate_log.action = mot._action
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
        """Generates a completion text. Gives the input provided to the model without templating."""
        with instrument_generate_from_raw(
            backend=self, num_actions=len(actions), format=format, tool_calls=tool_calls
        ):
            await self.do_generate_walks(list(actions))

            if format is not None:
                FancyLogger.get_logger().warning(
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

        for i, response in enumerate(responses):
            output = response["results"][0]
            sources = taint_sources(actions[i], ctx)
            sec_level = SecLevel.tainted_by(sources) if sources else SecLevel.none()

            result = ModelOutputThunk(
                value=output["generated_text"],
                sec_level=sec_level,
                meta={
                    "oai_completion_response": response["results"][0],
                    "usage": {
                        "prompt_tokens": output.get("input_token_count", 0),
                        "completion_tokens": output.get("generated_token_count", 0),
                        "total_tokens": output.get("input_token_count", 0)
                        + output.get("generated_token_count", 0),
                    },
                },
            )

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

        return results

    def _extract_model_tool_requests(
        self, tools: dict[str, AbstractMelleaTool], chat_response: dict
    ) -> dict[str, ModelToolCall] | None:
        model_tool_calls: dict[str, ModelToolCall] = {}
        for tool_call in chat_response["choices"][0]["message"].get("tool_calls", []):
            tool_name = tool_call["function"]["name"]
            tool_args = tool_call["function"]["arguments"]

            func = tools.get(tool_name)
            if func is None:
                FancyLogger.get_logger().warning(
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
