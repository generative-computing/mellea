# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""A generic OpenAI compatible backend that wraps around the openai python sdk."""

import asyncio
import datetime
import functools
import hashlib
import inspect
import json
import os
from collections.abc import Coroutine, Sequence
from typing import Any

import httpx
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
    PreTokenizedCBlock,
    RawProviderResponse,
    Requirement,
    Span,
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
from ..plugins.manager import has_plugins, invoke_hook
from ..plugins.types import HookType
from ..stdlib.components import Intrinsic, Message
from ..stdlib.context.chat import ChatContext
from ..stdlib.requirements import LLMaJRequirement
from ..telemetry.context import generate_request_id, with_context
from ._options import resolve_model_options
from .adapters import EmbeddedActivationRequest, EmbeddedBinding
from .adapters.adapter import AdapterInput, AdapterMixin, EmbeddedIntrinsicAdapter
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


# --- token-id history: server-side tokenization and prefix subtraction ------
#
# `/tokenize` is a vLLM extension (not the OpenAI API), reached through the client's
# generic request path. It renders a chat conversation through the server's own chat
# template and returns the ids, letting this backend work in id space with no local
# tokenizer.


class TokenizeUnavailable(RuntimeError):
    """Raised when the server cannot tokenize: no route, or an unusable reply."""


class DeltaNotDerivable(RuntimeError):
    """Raised when a full render does not extend the ids already sent."""


def derive_delta(prev_ids: list[int], full_ids: list[int]) -> list[int]:
    """Return the ids `full_ids` adds to `prev_ids`.

    BOTH arguments must be FRESH renders from the same tokenizer (`prev_ids` = the
    already-sent side, `full_ids` = the whole conversation). Do NOT pass the retained
    ids as `prev_ids`: those are what the server saw, and they diverge from a fresh
    re-render exactly on the conversation retaining ids exists to survive. The caller
    SPLICES the retained ids onto this delta; it does not compare against them.

    Two independent reasons a re-render diverges from what was sent. (1) Template:
    for `ibm-granite/granite-switch-4.1-3b-preview`, an adapter control token
    substitutes for the role marker (`100356` in place of `100264`) -- same length,
    so a length check misses it, yet every cache block after it is invalidated.
    (2) Tokenizer: `encode(decode(ids))` is not the identity -- the model can emit
    `[71, 4896]` (`'h'`,`'ello'`) where canonical `'hello'` is `[15339]`.

    Args:
        prev_ids (list[int]): Fresh render of the already-sent messages, without a
            generation prompt.
        full_ids (list[int]): Fresh render of every message including the new turn,
            with a generation prompt.

    Returns:
        list[int]: The ids the new turn adds. Empty when the two are identical, and
            the whole of `full_ids` when `prev_ids` is empty -- so a first turn needs
            no special case.

    Raises:
        DeltaNotDerivable: If `full_ids` does not start with `prev_ids`. Two fresh
            renders can only disagree if the earlier messages were rendered
            differently this time, so no suffix describes the new turn alone. Usual
            causes: a changed system block (documents introduced mid-conversation),
            changed template kwargs, or an adapter whose control token is emitted at
            sequence position 0.
    """
    if len(full_ids) < len(prev_ids):
        raise DeltaNotDerivable(
            f"the full render is shorter than the already-sent side "
            f"({len(full_ids)} < {len(prev_ids)} ids), so history shrank rather than "
            "grew. Compaction dropping a turn will do this. No suffix describes the "
            "new turn alone, so the retained ids cannot be extended."
        )
    for i, (was, now) in enumerate(zip(prev_ids, full_ids, strict=False)):
        if was != now:
            raise DeltaNotDerivable(
                f"the full render diverges from the already-sent side at index {i} "
                f"({was} != {now}), so it re-renders earlier turns instead of "
                "extending them. Common causes: documents or other template kwargs "
                "introduced mid-conversation, or an adapter whose control token is "
                "emitted at sequence position 0. No suffix describes the new turn "
                "alone, so the retained ids cannot be extended."
            )
    return list(full_ids[len(prev_ids) :])


def _prompt_digest(messages: list[dict]) -> tuple[str, ...]:
    """Return a per-message fingerprint of `messages`, one opaque string each.

    Turns "the first N messages" into "the first N messages PROVEN unchanged": a
    backend fingerprints the leading messages it is about to send and compares against
    `ChatContext.sent_prompt_digest`, refusing reuse on a mismatch. The comparison is
    over message TEXT, so it is immune to the `encode(decode(ids))` non-identity this
    policy exists to survive (see `derive_delta`) while still catching an edited
    historical turn or dropped oldest turns that leave `sent_message_count` intact.

    Only `role`, `content`, and `tool_calls` are fingerprinted, canonicalized rather
    than hashed as the raw dict, so the SAME turn fingerprints identically whether it
    was shaped by the chat path's serializer or the intrinsic path's
    `ChatMessage.model_dump()` (which differ in incidental keys) -- keeping a valid
    `Chat -> Intrinsic` reuse from being needlessly refused.

    Args:
        messages (list[dict]): OpenAI-shaped chat messages, in order.

    Returns:
        tuple[str, ...]: One digest per message, positionally aligned with `messages`.
    """
    digests: list[str] = []
    for message in messages:
        # Path-independent projection: content coerced to a string (multimodal lists
        # JSON-canonicalized), plus role and any tool_calls; sort_keys so key order
        # and incidental serializer differences never matter.
        raw_content = message.get("content")
        content = (
            raw_content
            if isinstance(raw_content, str) or raw_content is None
            else json.dumps(raw_content, sort_keys=True, ensure_ascii=False)
        )
        projection = {"role": message.get("role"), "content": content}
        if message.get("tool_calls"):
            projection["tool_calls"] = message["tool_calls"]
        canonical = json.dumps(projection, sort_keys=True, ensure_ascii=False)
        digests.append(hashlib.sha256(canonical.encode("utf-8")).hexdigest())
    return tuple(digests)


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
        default_extra_body (dict | None): Construction-time `extra_body` fields
            that are merged into every request this backend makes. Per-call
            `extra_body` values (from `model_options`) take precedence.
            `chat_template_kwargs` is deep-merged across all layers so that,
            for example, a construction-time `enable_thinking` flag is not
            silently dropped when the request also carries an `adapter_name`.
            Defaults to `{}` (no extra fields).
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

    Raises:
        TypeError: If `model_id` is neither a `str` nor a `ModelIdentifier` — most often
            `None`, from forwarding a `ModelIdentifier` field that this model does not set.
        ValueError: If `model_id` is an empty string, if neither `api_key` nor
            `OPENAI_API_KEY` is set, or if `model_id` is a `ModelIdentifier` with no `openai_name` set.
    """

    # Derived lazily by `_turn_terminator` and cached: the value is a property of the
    # served chat template, so one probe serves every turn. `[]` records a failed
    # probe so it is not retried on each turn.
    _turn_terminator_ids: list[int] | None = None

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
        default_extra_body: dict | None = None,
        **kwargs,
    ):
        """Initialize an OpenAI-compatible backend with the given model ID and API credentials."""
        # Resolve the served model name first: an unusable model_id must fail here rather
        # than as a missing `self._model_id` at generation time. `None` reaches this branch
        # whenever a caller forwards a ModelIdentifier field that is unset for this model,
        # e.g. `SOME_MODEL.hf_model_name`.
        match model_id:
            case str():
                if not model_id.strip():
                    raise ValueError(
                        "model_id is an empty string. Pass the model name your endpoint "
                        "serves it under, or a ModelIdentifier from "
                        "`mellea.backends.model_ids`."
                    )
                self._model_id = model_id
            case ModelIdentifier():
                if model_id.openai_name is None:
                    raise ValueError(
                        "The ModelIdentifier passed as model_id has no `openai_name` set "
                        f"(hf_model_name={model_id.hf_model_name!r}), so there is no model name "
                        "to send to an OpenAI-compatible endpoint. Either use a ModelIdentifier "
                        "whose provider hosts the model, or pass the name your server serves the "
                        "model under as a string -- for self-hosted vLLM/SGLang that is usually "
                        "`hf_model_name` -- along with a matching `base_url`."
                    )
                self._model_id = model_id.openai_name
            case _:
                raise TypeError(
                    "model_id must be a str or ModelIdentifier, got "
                    f"{type(model_id).__name__}. ModelIdentifier fields such as "
                    "`hf_model_name` are `None` when the model has no name for that "
                    "provider; check the constant in `mellea.backends.model_ids`."
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
        self._default_extra_body: dict = default_extra_body or {}

        self._provider: str = "openai"

        self._adapter_source = adapter_source

        # Use provided parameters or fall back to environment variables
        self._api_key = api_key
        # Resolve env here (not only in the SDK) so _server_type / init logging
        # see the same host the client will actually call.
        self._base_url = base_url or os.getenv("OPENAI_BASE_URL")

        # Validate that we have the required configuration
        if self._api_key is None and os.getenv("OPENAI_API_KEY") is None:
            raise ValueError(
                "OPENAI_API_KEY or api_key is required but not set. Please either:\n"
                "  1. Set the environment variable: export OPENAI_API_KEY='your-key-here'\n"
                "  2. Pass it as a parameter: OpenAIBackend(api_key='your-key-here')"
            )

        if self._base_url is None:
            MelleaLogger.get_logger().warning(
                "OPENAI_BASE_URL or base_url is not set.\n"
                "The openai SDK is going to assume that the base_url is `https://api.openai.com/v1`"
            )

        self._server_type: _ServerType = (
            _server_type(self._base_url)
            if self._base_url is not None
            else _ServerType.OPENAI
        )  # type: ignore
        if self._server_type != _ServerType.OPENAI:
            MelleaLogger.get_logger().info(
                "Mellea assumes you are NOT using the OpenAI platform, and that "
                "other model providers have less strict requirements on supporting "
                "JSON schemas passed into `format=`. If you encounter a server-side "
                "error when using format=, then you found an exception to this "
                "assumption. Please open an issue at "
                "github.com/generative-computing/mellea with the stack trace and "
                "your inference engine / model provider."
            )

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

    def add_adapter(self, adapter: AdapterInput) -> None:
        """Register an adapter with this backend.

        Accepts the full `AdapterInput` union to honour the mixin contract, but
        currently only `EmbeddedIntrinsicAdapter` (the Embedded/Granite Switch
        reality) is supported; other realities are rejected at runtime. As a
        side effect, an `EmbeddedBinding` weights handler is stamped with this
        backend's `base_model_name` in its `source` field.

        Args:
            adapter (AdapterInput): The adapter to register. Must be an
                `EmbeddedIntrinsicAdapter`.

        Raises:
            TypeError: If `adapter` is not an `EmbeddedIntrinsicAdapter`.
        """
        if not isinstance(adapter, EmbeddedIntrinsicAdapter):
            raise TypeError(
                f"OpenAIBackend currently only supports EmbeddedIntrinsicAdapter. "
                f"Got: {type(adapter).__name__}"
            )
        adapter.backend = self
        if isinstance(adapter.weights, EmbeddedBinding):
            adapter.weights.source = self.base_model_name
        self._added_adapters[adapter.qualified_name] = adapter

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

        Raises:
            ValueError: If `model_options` attempts to select a model. An
                OpenAIBackend's model is fixed when the backend is constructed.
        """
        remap_dict = self.to_mellea_model_opts_map_chats
        if not is_chat_context:
            remap_dict = self.to_mellea_model_opts_map_completions

        resolved_options = resolve_model_options(
            backend_defaults=self.model_options,
            remap=remap_dict,
            call_options=model_options,
        )
        if "model" in resolved_options:
            raise ValueError(
                "model cannot be set via model_options on OpenAIBackend — model "
                "selection happens at the backend/session level (construct a backend "
                "per model, or start a session against the chosen model_id)."
            )
        return resolved_options

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

    def _merge_user_extra_body(
        self, base: dict[str, Any], user: dict[str, Any] | None
    ) -> dict[str, Any]:
        """Merges default_extra_body, Mellea-assembled extra_body, and caller-supplied extra_body.

        Merge order (lowest → highest priority):
          1. `self._default_extra_body` — set at construction time
          2. `base` — assembled by Mellea for this request (documents, structured_outputs, …)
          3. `user` — from the caller's per-call `model_options`

        Both must end up in a single `extra_body` value; passing two spreads
        that each contain one raises `TypeError` at call time.

        `chat_template_kwargs` is the only nested dict Mellea writes into
        `extra_body` and is deep-merged across all three layers so that, for
        example, a construction-time `{"enable_thinking": True}` is not silently
        dropped when a per-call `{"adapter_name": "foo"}` is also present.

        Args:
            base: the `extra_body` Mellea assembled for this request.
            user: `extra_body` taken from the caller's model_options, or None.

        Returns:
            a new dict; `base`, `user`, and `self._default_extra_body` are
            left unmodified.
        """
        # Start from construction-time defaults, then overlay Mellea-built values.
        # Work on copies throughout so no caller dict is mutated.
        merged = dict(self._default_extra_body)
        default_ctk = merged.pop("chat_template_kwargs", None)

        base = dict(base) if base else {}
        base_ctk = base.pop("chat_template_kwargs", None)
        merged.update(base)

        # Merge chat_template_kwargs from default and base layers.
        merged_ctk: dict = {}
        if default_ctk is not None:
            merged_ctk.update(default_ctk)
        if base_ctk is not None:
            merged_ctk.update(base_ctk)
        if merged_ctk:
            merged["chat_template_kwargs"] = merged_ctk

        if user is None:
            return merged

        # Overlay caller-supplied values last (highest priority).
        user = dict(user)
        user_ctk = user.pop("chat_template_kwargs", None)
        merged.update(user)
        if user_ctk is not None:
            merged["chat_template_kwargs"] = {
                **merged.get("chat_template_kwargs", {}),
                **user_ctk,
            }
        return merged

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
            NotImplementedError: If the context isn't a chat context, or if
                streaming is requested (intrinsic post-processing requires
                the complete response).
            ValueError: If no embedded adapter is registered for the requested
                intrinsic.
            TypeError: If the adapter isn't an EmbeddedIntrinsicAdapter, or its
                `weights` isn't an EmbeddedBinding (only reachable if a caller
                reassigns `.weights` after construction).
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
        if user_extra_body is not None:
            protected_extra_body_keys = {
                "messages",
                "model",
                "parallel_tool_calls",
                "stream",
                "stream_options",
                "tool_choice",
                "tools",
            }
            overridden_keys = protected_extra_body_keys.intersection(user_extra_body)
            if overridden_keys:
                raise ValueError(
                    "extra_body cannot override intrinsic request fields: "
                    + ", ".join(sorted(overridden_keys))
                )
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

        extra_body = self._merge_user_extra_body(extra_body, user_extra_body)

        # Embedded adapters activate via control tokens in the chat template;
        # the binding owns the final request edit so callers cannot override
        # the adapter selected for this intrinsic. `adapter.weights` is always
        # an EmbeddedBinding here — EmbeddedIntrinsicAdapter.__init__
        # constructs one unconditionally — but the shim permits attribute
        # mutation, so a caller reassigning `.weights` must fail loudly here
        # rather than silently skip activation and send an unactivated request.
        if not isinstance(adapter.weights, EmbeddedBinding):
            raise TypeError(
                f"EmbeddedIntrinsicAdapter.weights must be an EmbeddedBinding; "
                f"got {type(adapter.weights).__name__}. Activation cannot proceed."
            )
        activation_request = EmbeddedActivationRequest(
            extra_body=extra_body, api_params=api_params
        )
        await adapter.weights.apply_activation(activation_request, adapter.identity)

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

        # `Chat -> Intrinsic` cache hit: if the caller retains ids and this intrinsic
        # only APPENDED to an unchanged prefix (instruction-style adapters), send the
        # exact retained ids + this turn's suffix to /v1/completions -- REUSING the
        # prefix without COMMITTING the rewritten request as history (the caller still
        # records only the canonical action/output). Otherwise `reuse_prompt_ids` is
        # None and the ordinary chat send runs (a correct cache miss).
        reuse_prompt_ids = await self._reuse_intrinsic_prefix_ids(
            ctx, messages_dicts, extra_body, api_params, use_tools
        )
        if reuse_prompt_ids is not None:
            chat_response: Coroutine[Any, Any, ChatCompletion] = (
                self._intrinsic_completion_as_chat(
                    reuse_prompt_ids, api_params, extra_body
                )
            )
        else:
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

    async def _reuse_intrinsic_prefix_ids(
        self,
        ctx: Context,
        messages_dicts: list[dict],
        extra_body: dict[str, Any],
        api_params: dict[str, Any],
        use_tools: bool,
    ) -> list[int] | None:
        """Return spliced prompt ids for an intrinsic that reuses an unchanged prefix, else `None`.

        Asks, against the REWRITTEN conversation (`messages_dicts`, after the io.yaml
        rewriter and adapter activation): does it still begin with the exact messages
        whose ids the server holds? If so, `_build_prompt_ids` returns those ids plus
        this turn's suffix -- a genuine cache hit. If the intrinsic rewrote or prepended
        to the prefix, the digest guard raises and this returns `None`.

        `None` is also returned when the completions endpoint cannot honour the request
        without a response-shape translation this path does not perform. Each is a
        documented fallback, not a silent degrade -- the chat send still runs correctly,
        only the cache hit is forgone:

        - tools: no `tools` parameter on the completions endpoint.
        - `logprobs`: score adapters (certainty, answerability) get logprobs in a
          different shape from /v1/completions than the result processor expects.
        - `reasoning_effort`: a string reasoning level is chat-only.
        - `documents`: rendered server-side by the chat template, but not given to
          `/tokenize`, so pre-tokenized ids would omit them.

        Args:
            ctx (Context): The generation context. Reuse is attempted only for a
                `ChatContext` that opted into `retain_token_ids` and already holds ids.
            messages_dicts (list[dict]): The rewritten conversation being sent.
            extra_body (dict[str, Any]): The intrinsic request's `extra_body`, read for
                `chat_template_kwargs` (carries the adapter control token into
                `/tokenize`) and the `documents` gate.
            api_params (dict[str, Any]): The intrinsic request's top-level params, read
                for the `logprobs` and `reasoning_effort` gates.
            use_tools (bool): Whether tools were assembled for this turn.

        Returns:
            list[int] | None: The retained ids plus this turn's suffix, or `None` to
                fall back to the chat endpoint.
        """
        if not (isinstance(ctx, ChatContext) and ctx.retains_token_ids):
            return None
        if not ctx.sent_token_ids:
            # Nothing retained yet, and intrinsics never commit ids, so the chat
            # endpoint is fine.
            return None
        if (
            use_tools
            or api_params.get("logprobs")
            or api_params.get("reasoning_effort")
        ):
            return None
        if extra_body.get("documents"):
            return None
        try:
            return await self._build_prompt_ids(
                ctx, messages_dicts, extra_body.get("chat_template_kwargs")
            )
        except (DeltaNotDerivable, TokenizeUnavailable) as e:
            # The intrinsic changed the prefix, or the server cannot tokenize: the chat
            # send re-renders (correct for a changed prefix, documented fallback otherwise).
            MelleaLogger.get_logger().debug(
                "intrinsic token-id reuse declined, falling back to chat endpoint: %s",
                e,
            )
            return None

    async def _intrinsic_completion_as_chat(
        self,
        prompt_ids: list[int],
        api_params: dict[str, Any],
        extra_body: dict[str, Any],
    ) -> ChatCompletion:
        """Send `prompt_ids` to `/v1/completions` and adapt the reply to a `ChatCompletion`.

        The intrinsic result processor consumes a chat-shaped response, but exact ids
        can only be sent through the completions endpoint, whose reply is text-shaped.
        This bridges the two so the existing pipeline runs unchanged.

        Args:
            prompt_ids (list[int]): The exact ids to send as the prompt.
            api_params (dict[str, Any]): The intrinsic's top-level params;
                `max_completion_tokens` is translated to `max_tokens`, the rest forwarded.
            extra_body (dict[str, Any]): The intrinsic's `extra_body`. `chat_template_kwargs`
                and `documents` are dropped (no chat template applies to a pre-tokenized
                prompt); guided-decoding keys are forwarded.

        Returns:
            ChatCompletion: A chat-shaped response carrying the completion text as the
                assistant message content.
        """
        params = dict(api_params)
        params.pop("model", None)
        if "max_completion_tokens" in params:
            params["max_tokens"] = params.pop("max_completion_tokens")
        body = {
            k: v
            for k, v in (extra_body or {}).items()
            if k not in ("chat_template_kwargs", "documents")
        }
        completion: Completion = await self._async_client.completions.create(
            model=self._model_id,
            prompt=[prompt_ids],  # type: ignore[arg-type]
            extra_body=body,
            **params,
        )
        choice = completion.choices[0]
        return ChatCompletion.model_validate(
            {
                "id": completion.id,
                "object": "chat.completion",
                "created": completion.created,
                "model": completion.model,
                "choices": [
                    {
                        "index": choice.index,
                        "message": {"role": "assistant", "content": choice.text},
                        "finish_reason": choice.finish_reason or "stop",
                    }
                ],
                "usage": completion.usage.model_dump() if completion.usage else None,
            }
        )

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
        new_ctx = ctx.add(action).add(mot)
        # The id path stashes the sequence the server has now seen on the thunk;
        # this is the only place that knows the next context, so it records them.
        retained = mot._meta.get("retained_token_ids")
        if retained is not None and isinstance(new_ctx, ChatContext):
            new_ctx = new_ctx.with_sent_token_ids(
                retained,
                model_id=mot._meta.get("retained_model_id"),
                message_count=mot._meta.get("retained_message_count", 0),
                prompt_digest=mot._meta.get("retained_prompt_digest", ()),
            )
        return mot, new_ctx

    def _server_root_url(self, route: str) -> str:
        """Return an absolute URL for `route`, served at the server root.

        vLLM serves `/tokenize` beside `/v1/completions`, not inside `/v1`. The SDK
        resolves a relative path against `base_url` (which ends in `/v1`), so a plain
        `"/tokenize"` would build `/v1/tokenize` and 404; an absolute url is used
        verbatim. Only a trailing `v1` segment is dropped, so a hosting prefix (gateway,
        notebook proxy) is preserved.

        Args:
            route (str): Route name without a leading slash, e.g. `"tokenize"`.

        Returns:
            str: An absolute URL the SDK will use verbatim.
        """
        base = self._async_client.base_url
        segments = [s for s in base.path.split("/") if s]
        if segments and segments[-1] == "v1":
            segments.pop()
        prefix = "".join(f"/{s}" for s in segments)
        return str(base.copy_with(raw_path=f"{prefix}/{route.lstrip('/')}".encode()))

    async def _tokenize_chat(
        self,
        messages: list[dict],
        *,
        add_generation_prompt: bool = True,
        chat_template_kwargs: dict[str, Any] | None = None,
    ) -> list[int]:
        """Tokenize a chat conversation server-side and return the token ids.

        Goes through this backend's own client, so base URL, auth, and timeouts
        match every other call it makes.

        Args:
            messages (list[dict]): OpenAI-shaped chat messages.
            add_generation_prompt (bool): Append the assistant generation prompt.
                `True` for the prompt being generated; `False` when tokenizing an
                already-closed history.
            chat_template_kwargs (dict[str, Any] | None): Extra template variables,
                e.g. `{"adapter_name": "uncertainty"}`. Omitted from the body when
                `None`, so a server that rejects unknown keys is unaffected.

        Returns:
            list[int]: Token ids for the rendered conversation.

        Raises:
            TokenizeUnavailable: If the route is missing, the request fails, or the
                reply carries no `tokens` list. All mean the same thing to a caller
                -- ids cannot be obtained -- so they are one exception rather than
                three.
        """
        body: dict[str, Any] = {
            "model": self._model_id,
            "messages": messages,
            "add_generation_prompt": add_generation_prompt,
        }
        if chat_template_kwargs is not None:
            body["chat_template_kwargs"] = chat_template_kwargs

        try:
            response = await self._async_client.post(
                self._server_root_url("tokenize"), body=body, cast_to=httpx.Response
            )
            payload = response.json()
        except Exception as e:
            raise TokenizeUnavailable(
                "the server could not tokenize this conversation, so token-id "
                "history cannot be built against it. Use a context without "
                "`retain_token_ids`."
            ) from e

        # Guard the `.get` below so every way ids can fail to arrive raises the one
        # documented exception type rather than an AttributeError.
        if not isinstance(payload, dict):
            raise TokenizeUnavailable(
                f"/tokenize replied with {type(payload).__name__}, not a JSON "
                "object, so its ids cannot be trusted."
            )
        tokens = payload.get("tokens")
        if not isinstance(tokens, list):
            raise TokenizeUnavailable(
                f"/tokenize replied without a 'tokens' list (got keys "
                f"{sorted(payload)}), so its ids cannot be trusted."
            )
        return [int(t) for t in tokens]

    async def _turn_terminator(self) -> list[int] | None:
        r"""Return the ids the chat template puts after an assistant turn.

        A model's reported ids stop at the end of its content; the template then adds
        a terminator (Granite: `<|end_of_text|>\n`). Those ids were in the prompt the
        server saw, so the retained sequence must include them, or the next turn runs
        an answer straight into the following role marker. Derived once (it is a
        property of the template) by subtracting an open render from a closed one.

        Returns:
            list[int] | None: The terminator ids, or `None` if the server could not
                be asked or the probe produced nothing usable -- in which case this
                turn cannot be retained.
        """
        if self._turn_terminator_ids is not None:
            return self._turn_terminator_ids or None
        probe = [{"role": "user", "content": "x"}]
        try:
            open_ids = await self._tokenize_chat(probe, add_generation_prompt=True)
            closed_ids = await self._tokenize_chat(
                [*probe, {"role": "assistant", "content": ""}],
                add_generation_prompt=False,
            )
        except TokenizeUnavailable:
            # Cache the failure as empty: a server with no tokenize route will not
            # acquire one mid-session, so don't re-probe every turn.
            self._turn_terminator_ids = []
            return None
        if len(closed_ids) <= len(open_ids) or closed_ids[: len(open_ids)] != open_ids:
            # The closed render does not extend the open one; the probe told us nothing.
            self._turn_terminator_ids = []
            return None
        self._turn_terminator_ids = closed_ids[len(open_ids) :]
        return self._turn_terminator_ids

    async def _build_prompt_ids(
        self,
        ctx: ChatContext,
        conversation: list[dict],
        chat_template_kwargs: dict[str, Any] | None,
    ) -> list[int]:
        """Splice the retained id prefix onto freshly-derived ids for the rest of `conversation`.

        The reusable core of the policy: reads the retained state off `ctx` and reuses
        the exact ids of a genuinely-unchanged prefix. `conversation` is the messages
        actually being sent -- the canonical history on the chat path, or the REWRITTEN
        conversation `_reuse_intrinsic_prefix_ids` hands over on the intrinsic path. The
        prefix is identified by `sent_prompt_digest`, not `sent_message_count`, which
        cannot see a prefix whose content changed under it.

        The new turn's ids come from subtracting two FRESH renders (the already-sent
        side, and the whole conversation) -- NOT by comparing against the retained ids,
        which differ from a re-render exactly on the conversation the policy exists to
        survive (see `derive_delta`). So the retained ids are spliced onto the delta,
        never compared with it, and their length is free to differ from the fresh render.

        The model, shrink, and digest guards run BEFORE any `/tokenize` round trip, so a
        request that cannot reuse is refused without paying for two renders. Reuse costs
        a second `/tokenize` round trip -- the price of the subtraction being sound.

        Args:
            ctx (ChatContext): The context carrying the ids already sent, how many
                messages they cover, the model that produced them, and the per-message
                digest proving which.
            conversation (list[dict]): The whole conversation actually being sent,
                newest turn last.
            chat_template_kwargs (dict[str, Any] | None): Template variables for the
                turn being generated, e.g. `{"adapter_name": ...}`.

        Returns:
            list[int]: The ids already sent, plus the new turn's ids.

        Raises:
            DeltaNotDerivable: If the retained prefix cannot be reused -- ids from a
                different model, a history that shrank below the retained boundary, a
                leading prefix whose content no longer matches the recorded digest, or
                two renders that disagree on the already-sent side.
            TokenizeUnavailable: If the server cannot tokenize the conversation.
        """
        retained_ids = list(ctx.sent_token_ids)
        retained_count = ctx.sent_message_count
        retained_digest = ctx.sent_prompt_digest
        retained_model = ctx.sent_model_id

        # Ids are not portable across models: reusing across vocabularies produces a
        # silently wrong prompt. Checked first, before any tokenize round trip.
        if retained_model is not None and retained_model != self._model_id:
            raise DeltaNotDerivable(
                f"the retained ids were produced by model {retained_model!r}, but this "
                f"backend serves {self._model_id!r}. Token ids are not portable across "
                "models; start a fresh context for a different model."
            )

        if not retained_ids or retained_count <= 0:
            # Nothing sent yet: the whole render is the prompt, no subtraction needed.
            return await self._tokenize_chat(
                conversation,
                add_generation_prompt=True,
                chat_template_kwargs=chat_template_kwargs,
            )

        # The `conversation[:retained_count]` slice below CLAMPS rather than raises, so
        # a shrunken history would silently subtract the newest turn into the prefix and
        # `derive_delta` (comparing two renders of the same clamped list) could not catch
        # it. Checked before tokenizing so a shrunk history costs no round trip.
        if retained_count > len(conversation):
            raise DeltaNotDerivable(
                f"the retained ids cover {retained_count} messages but this "
                f"conversation now has {len(conversation)}, so history shrank rather "
                "than grew. Compaction dropping turns will do this, as will the "
                "token-budget truncation `view_for_generation()` applies once a "
                "model_id is bound. The already-sent side can no longer be identified, "
                "so no suffix describes the new turn alone; this turn re-renders as "
                "chat messages and only its prefix-cache hit is lost."
            )

        # Prove the leading prefix is still the SAME messages (over TEXT, so immune to
        # the `encode(decode(ids))` non-identity) before reusing its ids: `retained_count`
        # cannot see a rewritten earlier turn or dropped oldest turns. Compared over the
        # digest's own length, before tokenizing so a changed prefix costs no round trip.
        if retained_digest and (
            _prompt_digest(conversation[: len(retained_digest)]) != retained_digest
        ):
            raise DeltaNotDerivable(
                "the retained ids' leading prefix no longer matches this conversation: "
                "an earlier message's content changed, or the oldest turns were "
                "dropped, while the message count stayed at or above the retained "
                "boundary. Splicing the old ids would send a prompt whose prefix the "
                "server never cached, so the ids are not reused; this turn re-renders "
                "as chat messages and only its prefix-cache hit is lost."
            )

        full_ids = await self._tokenize_chat(
            conversation,
            add_generation_prompt=True,
            chat_template_kwargs=chat_template_kwargs,
        )
        # The adapter goes on the NEW turn only. Rendering the already-sent side with
        # this turn's adapter would put its control token inside the region being
        # subtracted, so it would cancel out of the delta and never reach the server.
        prev_kwargs = {
            k: v for k, v in (chat_template_kwargs or {}).items() if k != "adapter_name"
        } or None
        prev_ids = await self._tokenize_chat(
            conversation[:retained_count],
            add_generation_prompt=False,
            chat_template_kwargs=prev_kwargs,
        )
        return list(retained_ids) + derive_delta(prev_ids, full_ids)

    def _retained_ids(
        self, prompt_ids: list[int], output: ModelOutputThunk, terminator: list[int]
    ) -> list[int] | None:
        """Return `prompt_ids` plus the ids the model emitted, or `None`.

        `None` means the server did not report ids, so this turn cannot be retained
        as ids. That is a meaningful answer rather than a failure: re-encoding the
        returned text would not reproduce them, and one wrong id invalidates every
        cache block after it.

        Args:
            prompt_ids (list[int]): Ids that were sent.
            output (ModelOutputThunk): The thunk the request produced.
            terminator (list[int]): Ids the template puts after an assistant turn,
                from `_turn_terminator`. Appended because they were part of the
                prompt the server saw even though the model did not emit them.

        Returns:
            list[int] | None: The full id sequence the server has now seen, or `None`
                when the emitted ids were not reported. Ask for them with
                `extra_body={"return_token_ids": True}`.
        """
        # `raw.response` is typed `Any` and defaults to None; a non-dict means the ids
        # cannot be read, which `None` communicates (the caller warns and succeeds).
        response = output.raw.response
        if not isinstance(response, dict):
            return None
        # `token_ids` is always a key but reads null without the flag; check the VALUE.
        emitted = response.get("token_ids")
        if not isinstance(emitted, list) or not emitted:
            return None
        # Two vLLM id-reporting shapes: the `return_token_ids` body flag yields plain
        # ints; the server-side `--return-tokens-as-token-ids` flag yields
        # `"token_id:NNNN"` strings. Accept both; anything else -> None (warn, continue).
        ids: list[int] = []
        for t in emitted:
            if isinstance(t, bool):
                return None
            if isinstance(t, int):
                ids.append(t)
            elif isinstance(t, str) and t.startswith("token_id:"):
                suffix = t.removeprefix("token_id:")
                if not suffix.isdigit():
                    return None
                ids.append(int(suffix))
            else:
                return None
        return list(prompt_ids) + ids + list(terminator)

    async def _generate_via_token_ids(
        self,
        ctx: ChatContext,
        conversation: list[dict],
        chat_template_kwargs: dict[str, Any] | None,
        *,
        action: Span | None = None,
        linearized_context: list[Span] | None = None,
        _format: type[BaseModelSubclass] | None = None,
        model_options: dict | None = None,
        has_tools: bool = False,
    ) -> ModelOutputThunk:
        """Send this turn as token ids rather than as chat messages.

        Args:
            ctx (ChatContext): Context carrying the ids already sent.
            conversation (list[dict]): The whole conversation, newest turn last.
            action (Span | None): The originating action, restored onto the thunk (the
                raw path returns one carrying the synthetic id block instead).
            linearized_context (list[Span] | None): The context the turn was rendered
                from, recorded on the thunk.
            chat_template_kwargs (dict[str, Any] | None): Template variables for the turn.
            _format (type[BaseModelSubclass] | None): Structured-output schema, forwarded.
            model_options (dict | None): Per-call model options, forwarded unchanged.
            has_tools (bool): Whether tools were assembled; used only to refuse.

        Returns:
            ModelOutputThunk: The generated output. When the server reported emitted ids,
                `_meta["retained_token_ids"]` carries the full sequence for
                `generate_from_chat_context` to record on the context.

        Raises:
            DeltaNotDerivable: If history was re-rendered rather than extended. Raised
                before anything is sent. `_generate_from_chat_context_standard` catches
                it and falls back to the chat send; direct callers get the exception.
            TokenizeUnavailable: If the server cannot tokenize. Not caught by that
                fallback, since no usable `/tokenize` route means id retention cannot
                work against this server at all.
            NotImplementedError: If tools or streaming were requested -- chat-endpoint
                features this transport cannot honour.
        """
        opts = model_options or {}
        if has_tools:
            raise NotImplementedError(
                "tool calling is not available when a context retains token ids: the "
                "prompt is sent to the completions endpoint, which has no tools "
                "parameter. Use a plain ChatContext for tool-calling turns."
            )
        thinking = opts.get(ModelOption.THINKING)
        if thinking is not None and type(thinking) is not bool:
            # A string level routes only to `reasoning_effort`, which
            # `completions.create` has no parameter for. A bool is accepted: it travels
            # as `chat_template_kwargs.enable_thinking`, which this path forwards.
            raise NotImplementedError(
                f"ModelOption.THINKING={thinking!r} is not available when a context "
                "retains token ids: a string reasoning level is sent as "
                "`reasoning_effort`, which the completions endpoint does not accept. "
                "Pass `True` instead (it travels in chat_template_kwargs), or use a "
                "context without `retain_token_ids`."
            )
        if opts.get(ModelOption.STREAM, False):
            raise NotImplementedError(
                "streaming is not yet available when a context retains token ids. "
                "The completions endpoint supports it, but this path does not "
                "consume the stream, so requesting it would fail deeper with no "
                "usable message. Use a context without `retain_token_ids` to stream."
            )
        # Start the terminator probe as a task so its round trips overlap the prompt
        # build and completion (it is only read once the answer is back); later turns
        # hit the cache and resolve immediately. Started after the refusals so a refusal
        # never orphans it. Not `asyncio.gather` (it leaves siblings running on error) --
        # hence the try/except below cancels it explicitly.
        terminator_task = asyncio.ensure_future(self._turn_terminator())
        try:
            return await self._generate_via_token_ids_inner(
                ctx,
                conversation,
                chat_template_kwargs,
                terminator_task,
                action=action,
                linearized_context=linearized_context,
                _format=_format,
                opts=opts,
            )
        except BaseException:
            # An abandoned task logs "exception was never retrieved" and can outlive
            # the request that started it.
            terminator_task.cancel()
            raise

    async def _generate_via_token_ids_inner(
        self,
        ctx: ChatContext,
        conversation: list[dict],
        chat_template_kwargs: dict[str, Any] | None,
        terminator_task: asyncio.Future[list[int] | None],
        *,
        action: Span | None,
        linearized_context: list[Span] | None,
        _format: type[BaseModelSubclass] | None,
        opts: dict,
    ) -> ModelOutputThunk:
        """Build the prompt, generate, and retain -- with `terminator_task` in flight.

        A separate method from `_generate_via_token_ids` so that its single `try/except`
        covers every path that would otherwise abandon `terminator_task`.

        Args:
            ctx (ChatContext): Context carrying the ids already sent.
            conversation (list[dict]): The whole conversation, newest turn last.
            chat_template_kwargs (dict[str, Any] | None): Template variables for the
                turn being generated.
            terminator_task (asyncio.Future[list[int] | None]): The in-flight turn
                terminator probe, awaited after generation.
            action (Span | None): The originating action, restored onto the thunk.
            linearized_context (list[Span] | None): The context the turn was rendered
                from, recorded on the thunk.
            _format (type[BaseModelSubclass] | None): Structured-output schema.
            opts (dict): Per-call model options, already merged.

        Returns:
            ModelOutputThunk: The generated output, with retained ids on `_meta` when
                the turn could be retained.
        """
        prompt_ids = await self._build_prompt_ids(
            ctx, conversation, chat_template_kwargs
        )
        # Ask the server to report emitted ids; without it the prefix never grows and
        # every turn re-sends the whole render AND pays a /tokenize round trip -- worse
        # than not opting in. Merged so a caller's own extra_body survives.
        forwarded = dict(opts)
        forwarded["extra_body"] = self._merge_user_extra_body(
            {"return_token_ids": True}, forwarded.get("extra_body")
        )
        # The PRIVATE method, deliberately: the public `generate_from_raw` is `@final`
        # and fires its own GENERATION_BATCH_PRE/POST_CALL hooks, which -- nested inside
        # the chat path's already-fired GENERATION_PRE/POST_CALL -- would double-count
        # metrics and nest a `text_completion` span in a `chat` span. It also returns
        # `tuple[list, dict | None]` rather than a bare list.
        results, usage = await self._generate_from_raw(
            [PreTokenizedCBlock(prompt_ids)],
            ctx,
            format=_format,
            model_options=forwarded,
        )
        output = results[0]

        # `_generate_from_raw` builds a thunk shaped for batch use; repair two things
        # for a chat turn.
        #
        # 1. Its action is the synthetic `PreTokenizedCBlock`, so `parsed_repr` is the
        #    raw string. Callers depend on the real action's parse (e.g. `mfuncs.chat`
        #    asserts `isinstance(parsed_repr, Message)`). Only the Component branch is
        #    reproduced; other actions keep the raw string the raw path left.
        if action is not None:
            output._call.action = action
            output._call.context = linearized_context
            if isinstance(action, Component):
                output.parsed_repr = action._parse(output)
        # 2. Per-thunk usage is None on the raw path (the API reports batch usage only);
        #    this batch is one prompt, so the batch usage IS this turn's. Dropping it
        #    would silence TokenMetricsPlugin and break the AGENTS.md backend contract.
        if usage is not None:
            output.generation.usage = usage

        # Two independent things can stop this turn being retained, and they have
        # different remedies. Reported separately: a single message covering both sends
        # half its readers to fix something that is not broken, and non-retention is
        # not benign -- the turn still paid its /tokenize round trips and the next one
        # re-sends the whole render, i.e. strictly worse than never opting in. This log
        # line is all a caller has to work from.
        terminator = await terminator_task
        retained = None
        if terminator is None:
            # Without it the retained sequence stops at the end of the model's content,
            # so the next prompt would run a role marker straight onto an answer.
            MelleaLogger.get_logger().warning(
                "the chat template's turn terminator could not be derived, so this "
                "turn cannot be retained as ids. The emitted ids may well have been "
                "reported; it is the terminator probe that failed, so check the "
                "server's /tokenize route rather than the generation request."
            )
        else:
            retained = self._retained_ids(prompt_ids, output, terminator)
            if retained is None:
                MelleaLogger.get_logger().warning(
                    "the server did not report the ids it emitted, so this turn "
                    "cannot be retained as ids; pass "
                    "extra_body={'return_token_ids': True} to keep id history exact. "
                    "Re-encoding the text would break the prefix cache from the first "
                    "differing token onward."
                )
        if retained is not None:
            # `_CallInfo.context` cannot carry a Context, so hand the ids to the caller
            # on _meta (the channel litellm uses for logprobs) for
            # `generate_from_chat_context` to move onto the context.
            output._meta["retained_token_ids"] = retained
            output._meta["retained_model_id"] = self._model_id
            # Covers every rendered message plus the assistant turn just produced.
            output._meta["retained_message_count"] = len(conversation) + 1
            # Fingerprint of the sent messages so the next turn can PROVE the prefix
            # unchanged before splicing. The assistant reply has no dict form yet; the
            # count still covers it, and it is spliced verbatim from `retained` anyway.
            output._meta["retained_prompt_digest"] = _prompt_digest(conversation)

        # The returned thunk is ALREADY computed, so `avalue()` short-circuits and the
        # compute path's post-call block never runs -- but GENERATION_PRE_CALL fired for
        # this turn, and GenerationTracingPlugin closes its span only on the post-call
        # hook. Fire it here (last, so the payload sees the finished thunk) or the span
        # and its OTel token leak and the metrics plugins record nothing.
        if has_plugins(HookType.GENERATION_POST_CALL):
            from ..plugins.hooks.generation import GenerationPostCallPayload

            glog = output._generate_log
            await invoke_hook(
                HookType.GENERATION_POST_CALL,
                GenerationPostCallPayload(
                    prompt=glog.prompt if glog and glog.prompt else "",
                    model_output=output,
                    latency_ms=(
                        (datetime.datetime.now() - output._gen.start).total_seconds()
                        * 1000
                        if output._gen.start
                        else -1
                    ),
                    generation_id=output._call.generation_id,
                ),
            )
        return output

    async def _generate_from_chat_context_standard(
        self,
        action: Span,
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
        extra_params["extra_body"] = self._merge_user_extra_body(
            extra_params.get("extra_body") or {}, user_extra_body
        )

        # Placed AFTER extra_body is merged so /tokenize sees the same
        # chat_template_kwargs the turn is generated under (a wrong render otherwise).
        #
        # This is the standard chat dispatch, which both REUSES a retained prefix and
        # COMMITS the produced ids as the next canonical prefix. `_generate_from_intrinsic`
        # never reaches here (`_generate_from_context` returns earlier for an Intrinsic);
        # it must NOT commit, since its io.yaml rewriter REPLACES the conversation -- so
        # it only reuses, via `_reuse_intrinsic_prefix_ids`. Both call the same
        # `_build_prompt_ids`; only the commit differs.
        if isinstance(ctx, ChatContext) and ctx.retains_token_ids:
            try:
                return await self._generate_via_token_ids(
                    ctx,
                    conversation,
                    (extra_params.get("extra_body") or {}).get("chat_template_kwargs"),
                    action=action,
                    linearized_context=linearized_context,
                    _format=_format,
                    model_options=model_opts,
                    has_tools=use_tools,
                )
            except DeltaNotDerivable as e:
                # The retained prefix cannot describe this turn: earlier messages were
                # re-rendered rather than extended, history shrank, or the digest no
                # longer matches. Raised before any request, so falling through to the
                # chat send below is a clean first attempt -- correct output, no reuse.
                #
                # Retained ids are left in place rather than cleared, so a later turn
                # that lines up with the prefix again resumes reuse.
                MelleaLogger.get_logger().warning(
                    "token-id history could not be extended for this turn, so it was "
                    "sent as chat messages instead: %s The turn itself is unaffected; "
                    "the server's prefix cache is re-primed from this render, and any "
                    "control tokens in earlier turns are dropped from it.",
                    e,
                )

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
                (`"low"`, `"medium"`, `"high"`) for explicit effort strings,
                `True`/`False` for the bool toggle, or `None` if reasoning
                was not enabled.
            seed: The random seed used during generation, or `None`.
            _format: The structured output format class used during generation, if any.
        """
        # Reconstruct the top-level response from chunks if streamed.
        if mot.raw.streamed_chunks is not None:
            merged = chat_completion_delta_merge(mot.raw.streamed_chunks)
            mot.raw.response = {"choices": [merged], "usage": mot.generation.usage}

        assert mot._call.action is not None, (
            "ModelOutputThunks should have their action assigned during generation"
        )
        assert mot._call.model_options is not None, (
            "ModelOutputThunks should have their model_opts assigned during generation"
        )

        # OpenAI streamed responses give you chunks of tool calls.
        # As a result, we have to store data between calls and only then
        # check for complete tool calls in the post_processing step.
        response = mot.raw.response
        assert response is not None
        choice_response = response["choices"][0]
        tool_chunk = extract_model_tool_requests(tools, choice_response)
        if tool_chunk is not None:
            if mot.tool_calls is None:
                mot.tool_calls = []
            # Extend the tool_chunk list.
            mot.tool_calls.extend(tool_chunk)

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
            ValueError: If `actions` mixes `PreTokenizedCBlock` entries with text
                actions. The completions endpoint accepts `prompt` as a list of
                strings or a list of token-id lists, never a mixture.
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

        # A PreTokenizedCBlock carries ids that must reach the server verbatim, so it
        # bypasses the formatter. The endpoint accepts `prompt` as a list of strings OR
        # of id lists, not a mixture -- refuse a mixed batch rather than send garbage.
        tokenized = [isinstance(action, PreTokenizedCBlock) for action in actions]
        if any(tokenized) and not all(tokenized):
            raise ValueError(
                "cannot mix PreTokenizedCBlock actions with text actions in one "
                "_generate_from_raw call: the completions endpoint takes either a list "
                "of strings or a list of id lists for `prompt`, not both. Split them "
                "into separate calls."
            )
        # Two homogeneous branches, not one comprehension: `list[str | list[int]]` is not
        # assignable to the endpoint's `list[str] | list[list[int]]` prompt type.
        prompts: list[str] | list[list[int]]
        if any(tokenized):
            prompts = [
                action.token_ids
                for action in actions
                if isinstance(action, PreTokenizedCBlock)
            ]
        else:
            prompts = [self.formatter.print(action) for action in actions]

        backend_specific = self._make_backend_specific_and_remove(
            model_opts, is_chat_context=False
        )
        extra_body = self._merge_user_extra_body(
            extra_body, backend_specific.pop("extra_body", None)
        )

        try:
            completion_response: Completion = (
                await self._async_client.completions.create(
                    model=self._model_id,
                    prompt=prompts,
                    extra_body=extra_body,
                    **backend_specific,
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
            # For a PreTokenizedCBlock `prompt` is a list of ids; log its repr, since
            # this flows into `GenerationPostCallPayload.prompt` (typed
            # `str | list[dict]`) and a `list[int]` would widen that public contract.
            generate_log.prompt = prompt if isinstance(prompt, str) else repr(prompt)
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
