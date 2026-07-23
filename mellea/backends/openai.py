# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""A generic OpenAI compatible backend that wraps around the openai python sdk."""

import asyncio
import datetime
import functools
import inspect
import os
from collections.abc import Coroutine, Sequence
from typing import Any

import openai
from openai.types.chat import ChatCompletion
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk
from openai.types.completion import Completion

from mellea.stdlib.requirements.requirement import ALoraRequirement

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
    RawProviderResponse,
    Requirement,
)
from ..core.base import AbstractMelleaTool
from ..formatters import ChatFormatter, TemplateFormatter, granite as granite_formatters
from ..helpers import (
    DEFAULT_CHUNK_TIMEOUT,
    ClientCache,
    _server_type,
    _ServerType,
    chat_completion_delta_merge,
    extract_model_tool_requests,
    get_current_event_loop,
    is_vllm_server_with_structured_output,
    message_to_openai_message,
    messages_to_docs,
    send_to_queue,
    should_replay_reasoning,
)
from ..stdlib.components import Intrinsic, Message
from ..stdlib.requirements import LLMaJRequirement
from ..telemetry.context import generate_request_id, with_context
from .adapters._core import Adapter
from .adapters.adapter import AdapterMixin, EmbeddedIntrinsicAdapter
from .backend import FormatterBackend
from .model_options import ModelOption
from .tools import (
    add_tools_from_context_actions,
    add_tools_from_model_options,
    convert_tools_to_json,
)
from .utils import populate_response_metadata_openai_shape

openai_ollama_batching_error = "json: cannot unmarshal array into Go struct field CompletionRequest.prompt of type string"

format: None = None  # typing this variable in order to shadow the global format function and ensure mypy checks for errors


class OpenAIBackend(FormatterBackend, AdapterMixin):
    """A generic OpenAI compatible backend.

    Args:
        model_id (str | ModelIdentifier): OpenAI-compatible model identifier.
            Defaults to `model_ids.OPENAI_GPT_5_1`.
        formatter (ChatFormatter | None): Formatter for rendering components.
            Defaults to `TemplateFormatter`.
        base_url (str | None): Base URL for the API endpoint; defaults to the
            standard OpenAI endpoint if not set.
        model_options (dict | None): Default model options for generation requests.
        default_to_constraint_checking_alora (bool): If `False`, deactivates aLoRA
            constraint checking; primarily for benchmarking and debugging.
        load_embedded_adapters (bool): If `True`, automatically registers
            embedded intrinsic adapters from *adapter_source* (or *model_id* if
            *adapter_source* is not set). Looks first for a local directory
            and then for a Hugging Face hub repo.
        adapter_source (str | None): Local directory path or Hugging Face hub
            repo ID from which to load embedded adapter configs. When `None`,
            falls back to *model_id*. Use this when the vLLM served model name
            differs from the adapter config location.
        api_key (str | None): API key; falls back to `OPENAI_API_KEY` env var.
        kwargs: Additional keyword arguments forwarded to the OpenAI client.

    Attributes:
        to_mellea_model_opts_map_chats (dict): Mapping from chat-endpoint option names
            to Mellea `ModelOption` sentinel keys.
        from_mellea_model_opts_map_chats (dict): Mapping from Mellea sentinel keys to
            chat-endpoint option names.
        to_mellea_model_opts_map_completions (dict): Mapping from completions-endpoint
            option names to Mellea `ModelOption` sentinel keys.
        from_mellea_model_opts_map_completions (dict): Mapping from Mellea sentinel keys
            to completions-endpoint option names.
    """

    def __init__(
        self,
        model_id: str | ModelIdentifier = model_ids.OPENAI_GPT_5_1,
        formatter: ChatFormatter | None = None,
        base_url: str | None = None,
        model_options: dict | None = None,
        *,
        default_to_constraint_checking_alora: bool = True,
        load_embedded_adapters: bool = False,
        adapter_source: str | None = None,
        api_key: str | None = None,
        **kwargs,
    ):
        """Initialize an OpenAI-compatible backend with the given model ID and API credentials."""
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
            "stop": ModelOption.STOP_SEQUENCES,
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
            ModelOption.STOP_SEQUENCES: "stop",
        }

        # See notes above.
        self.to_mellea_model_opts_map_completions = {
            "seed": ModelOption.SEED,
            "max_tokens": ModelOption.MAX_NEW_TOKENS,
            "stream": ModelOption.STREAM,
            "stop": ModelOption.STOP_SEQUENCES,
        }
        # See notes above.
        self.from_mellea_model_opts_map_completions = {
            ModelOption.SEED: "seed",
            ModelOption.MAX_NEW_TOKENS: "max_tokens",
            ModelOption.STREAM: "stream",
            ModelOption.STOP_SEQUENCES: "stop",
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

        self._provider: str = "openai"

        self._adapter_source = adapter_source

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
            MelleaLogger.get_logger().warning(
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

        # Attempt to detect vllm so that we can pass the correct structured output payload based on vllm version.
        # This is only necessary when passing format to generate_from_raw.
        self._use_structured_output_for_raw = is_vllm_server_with_structured_output(
            base_url=str(self._client.base_url), headers=self._client._custom_headers
        )

        self._client_cache = ClientCache(2)

        self._added_adapters: dict[str, EmbeddedIntrinsicAdapter] = {}

        # Call once to create an async_client and populate the cache.
        _ = self._async_client

        # TODO: We should change this logic once we have a better protocol for "auto-loading"
        # adapters during call_intrinsic, or once we support other types of adapters for
        # OpenAIBackends.
        # OpenAI Backends only support embedded_adapters.
        self._uses_embedded_adapters = True
        if load_embedded_adapters:
            self.register_embedded_adapter_model(self._adapter_source or self._model_id)

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

    # ------------------------------------------------------------------
    # AdapterMixin implementation
    # ------------------------------------------------------------------

    def add_adapter(self, adapter: Adapter) -> None:
        """Register an adapter with this backend.

        Currently only :class:`EmbeddedIntrinsicAdapter` is supported.

        Args:
            adapter: The adapter to register.

        Raises:
            TypeError: If *adapter* is not an `EmbeddedIntrinsicAdapter`.
        """
        if not isinstance(adapter, EmbeddedIntrinsicAdapter):
            raise TypeError(
                f"OpenAIBackend currently only supports EmbeddedIntrinsicAdapter. "
                f"Got: {type(adapter).__name__}"
            )
        adapter.backend = self
        self._added_adapters[adapter.qualified_name] = adapter

    def load_adapter(self, adapter_qualified_name: str) -> None:
        """No-op for embedded adapters — weights are baked into the model."""
        MelleaLogger.get_logger().debug(
            "load_adapter is a no-op for OpenAIBackends (adapter: %s)",
            adapter_qualified_name,
        )

    def unload_adapter(self, adapter_qualified_name: str) -> None:
        """No-op for embedded adapters — weights are baked into the model."""
        MelleaLogger.get_logger().debug(
            "unload_adapter is a no-op for OpenAIBackends (adapter: %s)",
            adapter_qualified_name,
        )

    def list_adapters(self) -> list[str]:
        """Return qualified names of all registered adapters.

        Returns:
            list[str]: Qualified adapter names.
        """
        return list(self._added_adapters.keys())

    # ------------------------------------------------------------------
    # Convenience registration helpers
    # ------------------------------------------------------------------

    def register_embedded_adapter_model(
        self, source: str, *, revision: str = "main", cache_dir: str | None = None
    ) -> list[str]:
        """Register all embedded adapters from an Embedded Adapter model.

        Args:
            source (str): A local model directory path or Hugging Face Hub repo ID.
            revision (str): Git revision when loading from Hugging Face Hub.
            cache_dir (str | None): Cache directory for HF downloads.

        Returns:
            list[str]: Names of the registered intrinsics.
        """
        import os

        adapters = EmbeddedIntrinsicAdapter.from_source(
            source, revision=revision, cache_dir=cache_dir
        )

        for adapter in adapters:
            self.add_adapter(adapter)

        return [a.intrinsic_name for a in adapters]

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
        """Filter kwargs to only include valid OpenAI client constructor parameters.

        Args:
            kwargs: Arbitrary keyword arguments to filter.

        Returns:
            dict: A dict containing only keys accepted by `openai.OpenAI.__init__`.
        """
        openai_params = set(inspect.signature(openai.OpenAI.__init__).parameters.keys())  # type: ignore
        openai_params.discard("self")  # Remove 'self' parameter
        return {k: v for k, v in kwargs.items() if k in openai_params}

    def filter_chat_completions_kwargs(self, model_options: dict) -> dict:
        """Filter model options to only include valid OpenAI chat completions parameters.

        See https://platform.openai.com/docs/api-reference/chat/create for the full
        list of accepted parameters.

        Args:
            model_options (dict): Model options dict that may contain non-chat keys.

        Returns:
            dict: A dict containing only keys accepted by `chat.completions.create`.
        """
        from openai.resources.chat.completions import Completions

        chat_params = set(inspect.signature(Completions.create).parameters.keys())
        chat_params.discard("self")
        return {k: v for k, v in model_options.items() if k in chat_params}

    def filter_completions_kwargs(self, model_options: dict) -> dict:
        """Filter model options to only include valid OpenAI completions parameters.

        See https://platform.openai.com/docs/api-reference/completions for the full
        list of accepted parameters.

        Args:
            model_options (dict): Model options dict that may contain non-completions keys.

        Returns:
            dict: A dict containing only keys accepted by `completions.create`.
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
            is_chat_context: set to True if using chat completion api

        Returns:
            a new dict
        """
        remap_dict = self.from_mellea_model_opts_map_chats
        if not is_chat_context:
            remap_dict = self.from_mellea_model_opts_map_completions

        backend_specific = ModelOption.replace_keys(model_options, remap_dict)

        for opt, field in (
            (ModelOption.LOGITS, "generation.logits"),
            (ModelOption.RAW_LOGITS, "generation.raw_logits"),
        ):
            if model_options.get(opt) and opt not in self._warned_about:
                self._warned_about.add(opt)
                MelleaLogger.get_logger().warning(
                    f"{opt!r} is not supported by the OpenAI backend; {field} will be None."
                )

        # OpenAI Backend has specific filtering functionality.
        if is_chat_context:
            model_opts = self.filter_chat_completions_kwargs(backend_specific)
        else:
            model_opts = self.filter_completions_kwargs(backend_specific)

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
        """Generate a completion for `action` given `ctx` via the OpenAI chat API.

        Delegates to `generate_from_chat_context`. Only chat contexts are supported.

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
            "The Openai backend only supports chat-like contexts."
        )

        _model_id_str = str(getattr(self, "model_id", "unknown"))
        with with_context(request_id=generate_request_id(), model_id=_model_id_str):
            await self.do_generate_walk(action)

            model_opts = self._simplify_and_merge(
                model_options, is_chat_context=ctx.is_chat_context
            )

            # Requirements can be automatically rerouted to a requirement adapter.
            if isinstance(action, Requirement):
                reroute_to_alora = self.default_to_constraint_checking_alora
                adapter_name = "requirement-check"

                if isinstance(action, ALoraRequirement):
                    reroute_to_alora = True
                    adapter_name = action.intrinsic_name
                    alora_action = action
                else:
                    assert action.description is not None, (
                        "must have a description when generating from a requirement"
                    )
                    alora_action = ALoraRequirement(action.description, adapter_name)

                alora_req_adapter = self._find_adapter(adapter_name, ("alora",))
                if alora_req_adapter is None:
                    if reroute_to_alora and isinstance(action, ALoraRequirement):
                        MelleaLogger.get_logger().warning(
                            f"attempted to use an AloraRequirement but backend {self} "
                            f"doesn't have the specified adapter added {adapter_name}; "
                            f"defaulting to regular generation"
                        )
                    reroute_to_alora = False

                if issubclass(type(action), LLMaJRequirement):
                    reroute_to_alora = False

                if reroute_to_alora:
                    mot = await self._generate_from_intrinsic(
                        alora_action,
                        ctx,
                        model_options=model_opts,
                        tool_calls=tool_calls,
                    )
                    return mot, ctx.add(alora_action).add(mot)

            elif isinstance(action, Intrinsic):
                mot = await self._generate_from_intrinsic(
                    action, ctx, model_options=model_opts, tool_calls=tool_calls
                )
                return mot, ctx.add(action).add(mot)

            result = await self.generate_from_chat_context(
                action,
                ctx,
                _format=format,
                model_options=model_options,
                tool_calls=tool_calls,
            )

        return result

    async def _generate_from_intrinsic(
        self,
        action: Intrinsic,
        ctx: Context,
        *,
        model_options: dict[str, Any],
        tool_calls: bool = False,
    ) -> ModelOutputThunk:
        """Generate a completion for an intrinsic action using an embedded adapter.

        Applies the intrinsic's I/O rewriter to transform the conversation,
        injects `intrinsic_name` into `chat_template_kwargs` so that the
        Granite Switch chat template activates the correct adapter, and
        post-processes the model output through the intrinsic's result
        processor.

        Intrinsics default to options provided by `io.yaml`. Model options
        override these defaults. All model options besides streaming are
        respected.

        Args:
            action (Intrinsic): The intrinsic component to execute.
            ctx (Context): The current generation context (must be a chat context).
            model_options (dict[str, Any]): Merged model options for this call.
            tool_calls (bool): If `True`, expose available tools to the model
                and parse tool-call responses.

        Returns:
            ModelOutputThunk: A thunk that lazily resolves to the processed
            intrinsic output.

        Raises:
            ValueError: If no embedded adapter is registered for the requested
                intrinsic.
            TypeError: If the adapter isn't an EmbeddedIntrinsicAdapter.
        """
        if not ctx.is_chat_context:
            raise NotImplementedError("Intrinsics require a chat context.")

        # Intrinsics don't support streaming because of their post-processing step.
        if model_options.get(ModelOption.STREAM, False):
            raise NotImplementedError(
                "Intrinsics do not support streaming due to structured output parsing."
            )

        # --- adapter lookup ------------------------------------------------
        allowed_types = tuple(at.value for at in action.adapter_types)
        adapter = self._find_adapter(action.intrinsic_name, allowed_types)
        if adapter is None:
            raise ValueError(
                f"backend ({self}) has no adapter for processing adapter function: "
                f"{action.intrinsic_name}"
            )

        # TODO: OpenAIBackend only supports EmbeddedAdapters.
        #       It should be refactored into a specific adapter.transform() function.
        if not isinstance(adapter, EmbeddedIntrinsicAdapter):
            raise TypeError(
                f"OpenAIBackend only supports EmbeddedIntrinsicAdapter, got: {type(adapter).__name__}"
            )

        intrinsic_config = adapter.config
        assert intrinsic_config is not None

        rewriter = granite_formatters.IntrinsicsRewriter(
            config_dict=intrinsic_config, model_name=adapter.name
        )
        result_processor = granite_formatters.IntrinsicsResultProcessor(
            config_dict=intrinsic_config
        )

        # --- linearize context and build conversation ----------------------
        linearized_context = ctx.view_for_generation()
        assert linearized_context is not None, (
            "If ctx.is_chat_context, then the context should be linearizable."
        )

        # NOTE: Explicitly do not add the action to the context here.
        #       Intrinsics modify the context through their rewriters.
        messages: list[Message] = self.formatter.to_chat_messages(linearized_context)

        # Extract system prompt and prepend to conversation.
        system_prompt = model_options.get(ModelOption.SYSTEM_PROMPT, "")
        conversation: list[dict] = []
        if system_prompt != "":
            conversation.append({"role": "system", "content": system_prompt})
        # Intrinsic/adapter calls are single-shot evaluations over a rewritten
        # conversation, not multi-turn generation, so reasoning is never replayed
        # here (no `replay_reasoning=`) — unlike the chat path in
        # `_generate_from_context`, which applies `should_replay_reasoning`.
        conversation.extend([message_to_openai_message(m) for m in messages])

        docs = messages_to_docs(messages)

        # Convert our conversation into a proper chat completions dict.
        request_json: dict = {
            "messages": conversation,
            "extra_body": {"documents": docs},
        }

        rewritten = rewriter.transform(request_json, **action.intrinsic_kwargs)

        # --- prepare extra_body and api_params --------------------------------
        extra_body = {}
        if rewritten.extra_body is not None:
            extra_body = rewritten.extra_body.model_dump(exclude_unset=True)

        # Start with rewriter parameters (io.yaml defaults).
        api_params: dict[str, Any] = {}
        if rewriter.parameters:
            api_params.update(rewriter.parameters)

        # Embedded adapters activate via control tokens in the chat template.
        if isinstance(adapter, EmbeddedIntrinsicAdapter):
            chat_template_kwargs = extra_body.pop("chat_template_kwargs", {}) or {}
            chat_template_kwargs["adapter_name"] = action.intrinsic_name
            extra_body["chat_template_kwargs"] = chat_template_kwargs
            # The rewriter config may set `model` to the adapter name, but
            # for embedded adapters the actual model is self._model_id.
            api_params.pop("model", None)

        # Collect tools if tool_calls is enabled.
        tools: dict[str, AbstractMelleaTool] = dict()
        if tool_calls:
            add_tools_from_model_options(tools, model_options)
            add_tools_from_context_actions(tools, ctx.actions_for_available_tools())
            MelleaLogger.get_logger().info(f"Tools for call: {tools.keys()}")

        formatted_tools = convert_tools_to_json(tools)
        use_tools = len(formatted_tools) > 0

        # Remap and filter remaining model options, then overlay onto api_params
        # so user values override rewriter/io.yaml defaults.
        user_api_params = self._make_backend_specific_and_remove(
            model_options, is_chat_context=True
        )
        user_extra_body = user_api_params.pop("extra_body", None)
        api_params.update(user_api_params)

        # Map THINKING to the correct backend parameter(s). Two mechanisms:
        # - chat_template_kwargs.enable_thinking: vLLM/Qwen3 (bool toggle)
        # - reasoning_effort: OpenAI/DeepSeek (string level, or True → "medium")
        # Both are set for True so the right server picks up whichever it understands.
        thinking = model_options.get(ModelOption.THINKING)
        if thinking is not None:  # False is a valid value — cannot use `if thinking`
            if type(thinking) is bool:
                ctk = extra_body.get("chat_template_kwargs", {}) or {}
                ctk["enable_thinking"] = thinking
                extra_body["chat_template_kwargs"] = ctk
                if thinking:
                    api_params["reasoning_effort"] = "medium"
                # False: don't send reasoning_effort — OpenAI disables reasoning by
                # default when the param is absent; passing False would be invalid.
            else:
                api_params["reasoning_effort"] = thinking

        if user_extra_body is not None:
            user_extra_body = dict(user_extra_body)
            merged_extra_body = dict(extra_body)
            user_ctk = user_extra_body.pop("chat_template_kwargs", None)
            merged_extra_body.update(user_extra_body)
            if user_ctk is not None:
                merged_extra_body["chat_template_kwargs"] = {
                    **merged_extra_body.get("chat_template_kwargs", {}),
                    **user_ctk,
                }
            extra_body = merged_extra_body

        # --- call the OpenAI-compatible API --------------------------------
        # The rewriter may add instruction messages where 'role' is a default
        # (e.g. UserMessage with role="user").  exclude_unset would drop it,
        # so we always force 'role' into the serialized dict.
        messages_dicts = []
        for m in rewritten.messages:
            d = m.model_dump(exclude_unset=True)
            if "role" not in d:
                d["role"] = m.role
            messages_dicts.append(d)

        chat_response = self._async_client.chat.completions.create(
            model=self._model_id,
            messages=messages_dicts,  # type: ignore
            tools=formatted_tools if use_tools else None,  # type: ignore
            extra_body=extra_body,
            **api_params,
        )

        # --- wire up ModelOutputThunk with intrinsic post-processing ------
        output = ModelOutputThunk(None)
        output._gen.start = datetime.datetime.now()
        output._call.context = linearized_context
        output._call.action = action
        output._call.model_options = model_options

        async def granite_formatters_processing(
            mot: ModelOutputThunk,
            chunk: ChatCompletion,
            rewritten: granite_formatters.ChatCompletion,
            result_processor: granite_formatters.IntrinsicsResultProcessor,
        ):
            """Accumulate content and apply intrinsic result processing."""
            import json as _json

            # Delegate standard metadata storage to the shared processing method.
            await self.processing(mot, chunk)

            # Apply intrinsic-specific result transformation on top.
            response_dict = chunk.model_dump()
            try:
                res = result_processor.transform(response_dict, rewritten)
            except _json.JSONDecodeError as e:
                raise Exception(
                    f"Intrinsic did not return a JSON: "
                    f"{chunk.choices[0].message.content}"
                ) from e

            # Overwrite the value accumulated by processing() with the
            # post-processed intrinsic output.
            mot._underlying_value = res.choices[0].message.content

        # Processing functions only pass the ModelOutputThunk (and current chunk
        # of response). Bind the other vars necessary for each processing step.
        output._gen.process = functools.partial(
            granite_formatters_processing,
            rewritten=rewritten,
            result_processor=result_processor,
        )

        output._gen.post_process = functools.partial(
            self.post_processing,
            tools=tools,
            conversation=conversation,
            thinking=thinking,
            seed=model_options.get(ModelOption.SEED, None),
            _format=None,
        )

        try:
            # To support lazy computation, will need to remove this create_task
            # and store just the unexecuted coroutine.
            # We can also support synchronous calls by adding a flag and changing
            # this ._gen.generate function.

            # This function should always be called from a running event loop so
            # we don't have to worry about scheduling the task to a specific
            # event loop here.
            output._gen.generate = asyncio.create_task(
                send_to_queue(
                    chat_response,
                    output._gen.queue,
                    chunk_timeout=model_options.get(
                        ModelOption.STREAM_TIMEOUT, DEFAULT_CHUNK_TIMEOUT
                    ),
                )
            )
            output._gen.generate_type = GenerateType.ASYNC
        except RuntimeError as e:
            # Most likely cause is running this function without an event loop present.
            raise e

        return output

    async def generate_from_chat_context(
        self,
        action: Component[C] | CBlock | ModelOutputThunk,
        ctx: Context,
        *,
        _format: type[BaseModelSubclass]
        | None = None,  # Type[BaseModelSubclass] is a class object of a subclass of BaseModel
        model_options: dict | None = None,
        tool_calls: bool = False,
    ) -> tuple[ModelOutputThunk[C], Context]:
        """Generate a new completion from the provided Context using this backend's `Formatter`.

        Formats the context and action into OpenAI-compatible chat messages, submits the
        request asynchronously, and returns a thunk that lazily resolves the output.

        Args:
            action (Component[C] | CBlock): The component or content block to generate
                a completion for.
            ctx (Context): The current generation context.
            _format (type[BaseModelSubclass] | None): Optional Pydantic model class for
                structured output decoding.
            model_options (dict | None): Per-call model options.
            tool_calls (bool): If `True`, expose available tools and parse responses.

        Returns:
            tuple[ModelOutputThunk[C], Context]: A thunk holding the (lazy) model output
                and an updated context that includes `action` and the new output.
        """
        await self.do_generate_walk(action)

        mot = await self._generate_from_chat_context_standard(
            action,
            ctx,
            _format=_format,
            model_options=model_options,
            tool_calls=tool_calls,
        )
        return mot, ctx.add(action).add(mot)

    async def _generate_from_chat_context_standard(
        self,
        action: Component | CBlock | ModelOutputThunk,
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
        messages.extend(self.formatter.to_chat_messages([action]))
        # ALoraRequirement may arrive here when no adapter is registered;
        # _generate is responsible for logging a warning in that case.

        conversation: list[dict] = []

        system_prompt = model_opts.get(ModelOption.SYSTEM_PROMPT, "")
        if system_prompt != "":
            conversation.append({"role": "system", "content": system_prompt})
        replay_flags = should_replay_reasoning(messages, self._provider)
        conversation.extend(
            [
                message_to_openai_message(m, self.formatter, replay_reasoning=replay)
                for m, replay in zip(messages, replay_flags)
            ]
        )

        extra_params: dict[str, Any] = {}
        if _format is not None:
            if self._server_type == _ServerType.OPENAI:
                # The OpenAI platform requires that additionalProperties=False on all response_format schemas.
                # However, not all schemas generates by Mellea include additionalProperties.
                # GenerativeStub, in particular, does not add this property.
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
                MelleaLogger.get_logger().info(
                    "Mellea assumes you are NOT using the OpenAI platform, and that other model providers have less strict requirements on supporting JSON schemas passed into `format=`. If you encounter a server-side error following this message, then you found an exception to this assumption. Please open an issue at github.com/generative_computing/mellea with this stack trace and your inference engine / model provider."
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

        formatted_tools = convert_tools_to_json(tools)
        use_tools = len(formatted_tools) > 0

        # Map THINKING to the correct backend parameter(s). Two mechanisms:
        # - chat_template_kwargs.enable_thinking: vLLM/Qwen3 (bool toggle)
        # - reasoning_effort: OpenAI/DeepSeek (string level, or True → "medium")
        # NOTE: don't pass reasoning_effort to non-reasoning models (e.g. gpt-4o).
        thinking = model_opts.get(ModelOption.THINKING)
        reasoning_params: dict[str, Any] = {}
        if thinking is not None:  # False is a valid value — cannot use `if thinking`
            if type(thinking) is bool:
                ctk_body: dict[str, Any] = extra_params.get("extra_body", {}) or {}
                ctk = ctk_body.get("chat_template_kwargs", {}) or {}
                ctk["enable_thinking"] = thinking
                ctk_body["chat_template_kwargs"] = ctk
                extra_params["extra_body"] = ctk_body
                if thinking:
                    reasoning_params["reasoning_effort"] = "medium"
                # False: don't send reasoning_effort — OpenAI disables reasoning by
                # default when the param is absent; passing False would be invalid.
            else:
                reasoning_params["reasoning_effort"] = thinking

        # Request usage information in streaming responses
        if model_opts.get(ModelOption.STREAM, False):
            extra_params["stream_options"] = {"include_usage": True}

        # Build the final backend-specific params and merge any user-supplied
        # extra_body into extra_params so there is a single extra_body source.
        # Two spreads each containing extra_body raises TypeError at call time.
        backend_specific = self._make_backend_specific_and_remove(
            model_opts, is_chat_context=ctx.is_chat_context
        )
        user_extra_body = backend_specific.pop("extra_body", None)
        if user_extra_body is not None:
            # shallow copy so .pop() below doesn't mutate the caller's dict
            user_extra_body = dict(user_extra_body)
            eb = dict(extra_params.get("extra_body") or {})
            user_ctk = user_extra_body.pop("chat_template_kwargs", None)
            # shallow merge is safe: chat_template_kwargs is the only nested dict
            # key Mellea writes into extra_body; it is deep-merged separately below
            eb.update(user_extra_body)
            if user_ctk is not None:
                eb["chat_template_kwargs"] = {
                    **eb.get("chat_template_kwargs", {}),
                    **user_ctk,
                }
            extra_params["extra_body"] = eb

        chat_response: Coroutine[
            Any, Any, ChatCompletion | openai.AsyncStream[ChatCompletionChunk]
        ] = self._async_client.chat.completions.create(
            model=self._model_id,
            messages=conversation,  # type: ignore
            tools=formatted_tools if use_tools else None,  # type: ignore
            # parallel_tool_calls=False, # We only support calling one tool per turn. But we do the choosing on our side so we leave this False.
            **extra_params,
            **reasoning_params,  # type: ignore
            **backend_specific,
        )  # type: ignore

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
            tools=tools,
            conversation=conversation,
            thinking=thinking,
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

    async def processing(
        self, mot: ModelOutputThunk, chunk: ChatCompletion | ChatCompletionChunk
    ):
        """Accumulate content from a single OpenAI response object into the output thunk.

        Called for each `ChatCompletion` (non-streaming) or `ChatCompletionChunk`
        (streaming). Tool call parsing is deferred to `post_processing`.

        Args:
            mot (ModelOutputThunk): The output thunk being populated.
            chunk (ChatCompletion | ChatCompletionChunk): A single response object or
                streaming delta from the OpenAI API.
        """
        if mot.thinking is None:
            mot.thinking = ""
        if mot._underlying_value is None:
            mot._underlying_value = ""

        if isinstance(chunk, ChatCompletion):
            message = chunk.choices[0].message

            # reasoning_content (Anthropic/DeepSeek attribute path) takes priority;
            # fall back to the "reasoning" extra field used by vLLM and compatible servers.
            thinking_chunk = getattr(message, "reasoning_content", None)
            if thinking_chunk is None:
                thinking_chunk = (message.model_extra or {}).get("reasoning")
            if thinking_chunk is not None:
                mot.thinking += thinking_chunk

            content_chunk = message.content
            if content_chunk is not None:
                mot._underlying_value += content_chunk

            # Store the full response (includes usage) as a dict.
            mot.raw.response = chunk.model_dump()

        elif isinstance(chunk, ChatCompletionChunk):
            # Usage arrives on its own chunk (typically the last); record it now.
            if hasattr(chunk, "usage") and chunk.usage is not None:
                mot.generation.usage = chunk.usage.model_dump()

            # Some chunks (like the final usage chunk) may not have choices
            if len(chunk.choices) == 0:
                return

            message_delta = chunk.choices[0].delta
            thinking_chunk = getattr(message_delta, "reasoning_content", None)
            if thinking_chunk is None:
                thinking_chunk = (message_delta.model_extra or {}).get("reasoning")
            if thinking_chunk is not None:
                mot.thinking += thinking_chunk

            content_chunk = message_delta.content
            if content_chunk is not None:
                mot._underlying_value += content_chunk

            if mot.raw.streamed_chunks is None:
                mot.raw.streamed_chunks = []
            mot.raw.streamed_chunks.append(chunk.choices[0].model_dump())

    async def post_processing(
        self,
        mot: ModelOutputThunk,
        tools: dict[str, AbstractMelleaTool],
        conversation: list[dict],
        thinking,
        seed,
        _format,
    ):
        """Finalize the output thunk after OpenAI generation completes.

        Reconstructs a merged chat response from streaming chunks if applicable,
        extracts any tool call requests, records token usage metrics, emits telemetry,
        and attaches the generate log.

        Args:
            mot (ModelOutputThunk): The output thunk to finalize.
            tools (dict[str, AbstractMelleaTool]): Available tools, keyed by name.
            conversation (list[dict]): The chat conversation sent to the model,
                used for logging.
            thinking: The reasoning value passed to the model: a string level
                (``"low"``, ``"medium"``, ``"high"``) for explicit effort strings,
                ``True``/``False`` for the bool toggle, or ``None`` if reasoning
                was not enabled.
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

        # Generate the log for this ModelOutputThunk.
        generate_log = GenerateLog()
        generate_log.prompt = conversation
        generate_log.backend = f"openai::{self.model_id!s}"
        generate_log.model_options = mot._call.model_options
        generate_log.date = datetime.datetime.now()
        # Store the full response (includes usage info)
        generate_log.model_output = response
        generate_log.extra = {
            "format": _format,
            "thinking": thinking,
            "tools_available": tools,
            "tools_called": mot.tool_calls,
            "seed": seed,
        }
        generate_log.action = mot._call.action
        generate_log.result = mot
        mot._generate_log = generate_log

        # Non-streaming carries usage on the response; streaming already set it.
        if usage := response.get("usage"):
            mot.generation.usage = usage

        # Populate model and provider metadata
        mot.generation.model = self._model_id
        mot.generation.provider = self._provider
        mot.raw.provider = self._provider

        # Populate response-side metadata for telemetry
        if isinstance(response, dict):
            populate_response_metadata_openai_shape(mot, response)

    async def _generate_from_raw(
        self,
        actions: Sequence[Component[C] | CBlock],
        ctx: Context,
        *,
        format: type[BaseModelSubclass] | None = None,
        model_options: dict | None = None,
        tool_calls: bool = False,
    ) -> tuple[list[ModelOutputThunk], dict[str, Any] | None]:
        """Generate completions for multiple actions without chat templating via the OpenAI completions API.

        Passes formatted prompt strings directly to the completions endpoint.
        Tool calling is not supported on this endpoint. Per-MOT `mot.generation.usage`
        stays `None` because the OpenAI completions API only reports whole-batch usage.

        Args:
            actions (Sequence[Component[C] | CBlock]): Actions to generate completions for.
            ctx (Context): The current generation context.
            format (type[BaseModelSubclass] | None): Optional Pydantic model for
                structured output; passed as a guided-decoding parameter.
            model_options (dict | None): Per-call model options.
            tool_calls (bool): Ignored; tool calling is not supported on this endpoint.

        Returns:
            tuple[list[ModelOutputThunk], dict | None]: `(results, usage)` where
                `results` is a list of model output thunks, one per action, and
                `usage` is the whole-batch token-usage dict or `None`.

        Raises:
            openai.BadRequestError: If the request is invalid (e.g. when targeting an
                Ollama server that does not support batched completion requests).
        """
        await self.do_generate_walks(list(actions))

        extra_body = {}
        if format is not None:
            MelleaLogger.get_logger().warning(
                "The official OpenAI completion api does not accept response format / structured decoding; "
                "it will be passed as an extra arg."
            )

            # Some versions (like vllm's version) of the OpenAI API support structured decoding for completions requests.
            # It's dependent on the vllm version though. We check at backend init.
            if self._use_structured_output_for_raw:
                extra_body["structured_outputs"] = {"json": format.model_json_schema()}  # type: ignore
            else:
                extra_body["guided_json"] = format.model_json_schema()  # type: ignore
        if tool_calls:
            MelleaLogger.get_logger().warning(
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
                MelleaLogger.get_logger().error(
                    "If you are trying to call `OpenAIBackend._generate_from_raw while targeting an ollama server, "
                    "your requests will fail since ollama doesn't support batching requests."
                )
            raise

        # Necessary for type checker.
        assert isinstance(completion_response, Completion)

        usage_dump = (
            completion_response.usage.model_dump()
            if completion_response.usage
            else None
        )

        results = []
        for response, action, prompt in zip(
            completion_response.choices, actions, prompts
        ):
            output = ModelOutputThunk(response.text)
            # There is no context for generate_from_raw for now
            output._call.context = None
            output._call.action = action
            output._call.model_options = model_opts
            output.raw = RawProviderResponse(
                provider=self._provider, response=response.model_dump()
            )
            output.generation.model = self._model_id
            output.generation.provider = self._provider

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

        return results, usage_dump

    @property
    def base_model_name(self):
        """Returns the base_model_id of the model used by the backend. For example, `granite-3.3-8b-instruct` for `ibm-granite/granite-3.3-8b-instruct`."""
        if "/" in self._model_id:
            return self._model_id.split("/")[1]
        else:
            return self._model_id
