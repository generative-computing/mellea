# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""A generic OpenAI compatible backend that wraps around the openai python sdk."""

import asyncio
import contextlib
import datetime
import functools
import hashlib
import inspect
import json
import os
import threading
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
from ..stdlib.components import Intrinsic, Message
from ..stdlib.context.chat import ChatContext
from ..stdlib.requirements import LLMaJRequirement
from ..telemetry.context import generate_request_id, with_context
from ._options import resolve_model_options
from .adapters import EmbeddedActivationRequest, EmbeddedBinding, Identity
from .adapters._core import (
    Adapter as _AdapterCore,
    _await_embedded_generation,
    _fire_embedded_invocation_complete,
)
from .adapters.adapter import (
    AdapterInput,
    AdapterMixin,
    EmbeddedIntrinsicAdapter,
    _composed_adapter_key,
    _discover_embedded_adapters,
)
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


# The coded switch recovers a control token's write address from a 1/(1+n) attention
# signal in the model dtype. In bf16 that inverts exactly only below this count; past
# it, addresses alias and routing silently degrades. Retaining ids is what makes n grow,
# so the ceiling belongs to whichever layer does the retaining.
#
# 188 is one below the functional cliff, not at it. In bf16 n=189 recovers as 190, but
# the SAME recovered address writes the codeword and reads it back, so 189 still routes
# correctly. n=190 is where 189 and 190 both recover as 190: two control tokens key one
# codeword and the memory head returns the mean of their two expert ids -- an arbitrary
# adapter after rounding, with no error in the output. 188 therefore leaves exactly one
# token of headroom for a control token the model emits mid-answer, which joins the same
# request during decode. That headroom is one token, not two.
#
# Both the re-baseline and the raise compare with `>`, so this count is legal and only
# the next one is not.
MAX_RETAINED_CONTROL_TOKENS = 188


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

    EVERY field on the message is fingerprinted: these dicts are what goes on the wire,
    so any field the chat template renders is part of the prompt, and one left out of the
    fingerprint is a prompt change this guard cannot see. `reasoning_content` is included
    for that reason -- the chat path emits it for a tool-calling turn
    (`should_replay_reasoning`) and the intrinsic path never does, so the same history
    renders differently on the two paths -- as is `tool_call_id`.

    Values are canonicalized rather than hashed as the raw dict, and empty ones are
    dropped, so the SAME turn fingerprints identically whether the chat path's serializer
    (which omits a field it has no value for) or the intrinsic path's
    `ChatMessage.model_dump()` (which can carry it as `None`) produced it -- keeping a
    valid `Chat -> Intrinsic` reuse from being needlessly refused.

    Args:
        messages (list[dict]): OpenAI-shaped chat messages, in order.

    Returns:
        tuple[str, ...]: One digest per message, positionally aligned with `messages`.
    """
    digests: list[str] = []
    for message in messages:
        # Every key, with non-strings JSON-canonicalized (multimodal content lists,
        # tool_calls) and empty values dropped so absent-vs-`None` is not a difference.
        # sort_keys so key order never matters.
        projection = {
            key: value
            if isinstance(value, str)
            else json.dumps(value, sort_keys=True, ensure_ascii=False)
            for key, value in message.items()
            if value is not None and value != "" and value != [] and value != {}
        }
        canonical = json.dumps(projection, sort_keys=True, ensure_ascii=False)
        digests.append(hashlib.sha256(canonical.encode("utf-8")).hexdigest())
    return tuple(digests)


def _completion_choice_as_chat_response(
    choice: dict[str, Any],
    usage: dict[str, Any] | None = None,
    *,
    response_id: str | None = None,
    response_model: str | None = None,
) -> dict[str, Any]:
    """Return a `/v1/completions` choice in chat-completion shape.

    The id transport sends to the completions endpoint, whose reply is text-shaped
    (`text`), while every consumer of `ModelOutputThunk.raw.response` on a chat turn
    dispatches on `provider` rather than on shape -- `Message._parse` most of all,
    which reads `response["choices"][0]["message"]` for provider `"openai"`. Without
    this the two disagree and the read raises `KeyError`.

    The target is the shape a real vLLM chat reply has when `return_token_ids` is on:
    `message` and `token_ids` both sit on the CHOICE, not at top level. Producing
    exactly that keeps the transport swap invisible to everything downstream,
    including `_retained_ids`, instead of teaching each consumer a second shape.

    Rebuilt from the choice alone, the result would silently lose what only the
    enclosing response carries -- the provider's response id and served model -- and the
    choice's own `logprobs`. `raw.response` is what `generate_log.model_output` and any
    provider-shape consumer reads, so those are carried through explicitly rather than
    dropped by the transport swap.

    Args:
        choice (dict[str, Any]): One element of a completions response's `choices`.
        usage (dict[str, Any] | None): Token usage for the request, if reported.
        response_id (str | None): The enclosing response's `id`, so a retained turn's
            log can still be traced to the provider's response.
        response_model (str | None): The model the enclosing response reported serving.

    Returns:
        dict[str, Any]: A chat-shaped response carrying the completion text as the
            assistant message content, with `token_ids` preserved on the choice.
    """
    return {
        "id": response_id,
        "model": response_model,
        "object": "chat.completion",
        "choices": [
            {
                "index": choice.get("index", 0),
                "message": {"role": "assistant", "content": choice.get("text") or ""},
                "finish_reason": choice.get("finish_reason") or "stop",
                # Per-token detail the completions endpoint reports on the choice; a
                # consumer reading it from a chat-shaped reply finds it in the same place.
                "logprobs": choice.get("logprobs"),
                # Kept on the choice, where a genuine chat reply puts them, so
                # `_retained_ids` reads one location for both transports.
                "token_ids": choice.get("token_ids"),
            }
        ],
        "usage": usage,
    }


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

    _supports_composed_adapters = True

    # Class-level defaults only, so a subclass or a `__new__`-built instance reads
    # something sane; `__init__` replaces both with per-instance containers. They must
    # NOT be shared: a terminator is a property of one server's chat template, and a
    # control-token id is a property of one served vocabulary.
    _turn_terminator_ids: dict[str, list[int]] = {}
    _control_token_id_set: set[int] = set()

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

        # Construction-time `extra_body` passed via the generic `model_options`
        # dict (as opposed to the dedicated `default_extra_body` param) must not
        # outrank a per-call `ModelOption.THINKING` the way `default_extra_body`
        # doesn't. Fold it into `_default_extra_body`'s tier now and exclude it
        # from `self.model_options` in `_simplify_and_merge`, so it can no longer
        # re-enter the per-call precedence chain as if it were caller-supplied
        # `extra_body` (#1617).
        construction_extra_body = self.model_options.get("extra_body")
        if isinstance(construction_extra_body, dict):
            self._default_extra_body = ModelOption._merge_extra_body(
                construction_extra_body, self._default_extra_body
            )

        self._provider: str = "openai"

        # Token-id retention caches, per instance (see the class-level defaults).
        # `_turn_terminator_ids` is keyed by the canonicalized chat-template kwargs the
        # probe ran under, because Granite 4.2 closes an assistant turn differently
        # depending on `enable_thinking`; `[]` records a failed probe so it is not
        # retried every turn. `_control_token_ids_by_adapter` caches one `/tokenize`
        # diff per adapter name and feeds `_control_token_id_set`, the union used to
        # count a prompt against `MAX_RETAINED_CONTROL_TOKENS`.
        self._turn_terminator_ids: dict[str, list[int]] = {}
        self._control_token_ids_by_adapter: dict[str, list[int]] = {}
        self._control_token_id_set: set[int] = set()
        self._token_id_reprefills: int = 0

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

        # EmbeddedIntrinsicAdapter is itself an _AdapterCore subclass, so this
        # single type covers both the shim and composed-Adapter realities.
        self._added_adapters: dict[str, _AdapterCore] = {}
        # Raw io.yaml config for composed Adapter instances (Epic #929, issue
        # #1144), keyed by _composed_adapter_key(). A composed Adapter has no
        # `.config` field (that's shim-only state); see
        # _intrinsic_adapter_name_and_config.
        self._composed_adapter_configs: dict[str, dict] = {}
        self._adapter_lock = threading.RLock()
        # Backs _adapter_resolve_lock() — deliberately separate from
        # _adapter_lock (`_adapter_activation_lock()`); see that method's
        # docstring for the deadlock a shared lock would risk.
        self._adapter_resolve_lock_obj = threading.RLock()

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

    def add_adapter(self, adapter: AdapterInput, *, config: dict | None = None) -> None:
        """Register an adapter with this backend.

        Accepts the full `AdapterInput` union to honour the mixin contract, but
        currently only `EmbeddedIntrinsicAdapter` (the Embedded/Granite Switch
        reality) is supported; other realities are rejected at runtime. As a
        side effect, an `EmbeddedBinding` weights handler is stamped with this
        backend's `base_model_name` in its `source` field. Refuses a name
        already registered instead of silently overwriting it.

        Args:
            adapter (AdapterInput): The adapter to register. Must be an
                `EmbeddedIntrinsicAdapter` or a composed `Adapter` whose
                `weights` is an `EmbeddedBinding`.
            config (dict | None): Raw io.yaml config for a composed `Adapter`.
                Required for that shape (its config cannot be cheaply
                re-derived later); rejected for `EmbeddedIntrinsicAdapter`,
                which already carries its own `.config`.

        Raises:
            TypeError: If `adapter` is not a supported Embedded adapter, or
                `config` is given for an `EmbeddedIntrinsicAdapter`.
            ValueError: If `adapter` is a composed `Adapter` and `config` is
                not given — registering it without one would make it
                discoverable but permanently unable to generate.
        """
        if isinstance(adapter, EmbeddedIntrinsicAdapter):
            if config is not None:
                raise TypeError(
                    "config= is not accepted for an EmbeddedIntrinsicAdapter; "
                    "it already carries its own .config."
                )
            if adapter.qualified_name in self._added_adapters:
                MelleaLogger.get_logger().warning(
                    f"attempted to add adapter {adapter.qualified_name!r} but it is "
                    "already registered; refusing to overwrite it."
                )
                return
            adapter.backend = self
            if isinstance(adapter.weights, EmbeddedBinding):
                adapter.weights.source = self.base_model_name
            self._added_adapters[adapter.qualified_name] = adapter
            return

        if isinstance(adapter, _AdapterCore):
            if not isinstance(adapter.weights, EmbeddedBinding):
                raise TypeError(
                    "OpenAIBackend only supports the Embedded/Granite Switch "
                    f"reality for a composed Adapter; got {type(adapter.weights).__name__}."
                )
            # Validated ahead of the duplicate-registration guard below: a
            # malformed config= argument must raise even when the call would
            # otherwise be a silently-refused duplicate.
            if config is None:
                raise ValueError(
                    f"No io.yaml config given for composed embedded adapter "
                    f"{adapter.identity.name!r}; registering it without one "
                    "would leave it discoverable but unable to generate. "
                    "Pass config=, or register it via "
                    "register_embedded_adapter_model() or resolve_adapter()."
                )
            key = _composed_adapter_key(adapter)
            # Locked: a direct add_adapter() call (unlike resolve_adapter(),
            # which already holds this lock around its own add_adapter()
            # calls) is otherwise an unguarded check-then-write across
            # _added_adapters/_composed_adapter_configs — two concurrent
            # callers registering under the same key could each pass the
            # duplicate check below before either writes, then pair one
            # call's adapter with the other's config. Reentrant, so a caller
            # already holding it (resolve_adapter()) is unaffected.
            with self._adapter_activation_lock():
                if key in self._added_adapters:
                    MelleaLogger.get_logger().warning(
                        f"attempted to add adapter {key!r} but it is already "
                        "registered; refusing to overwrite it."
                    )
                    return
                adapter.weights.source = self.base_model_name
                self._added_adapters[key] = adapter
                self._composed_adapter_configs[key] = config
            return

        raise TypeError(
            "OpenAIBackend currently only supports EmbeddedIntrinsicAdapter or a "
            f"composed Adapter. Got: {type(adapter).__name__}"
        )

    def list_adapters(self) -> list[str]:
        """Return qualified names of all registered adapters.

        Returns:
            list[str]: Qualified adapter names.
        """
        return list(self._added_adapters.keys())

    def _adapter_activation_lock(
        self,
    ) -> contextlib.AbstractContextManager[bool | None]:
        """Serialize `add_adapter()`'s registration-dict writes for this backend."""
        return self._adapter_lock

    def _adapter_resolve_lock(self) -> contextlib.AbstractContextManager[bool | None]:
        """A separate lock for `resolve_adapter()`/registration orchestration.

        Not `_adapter_lock` (the lock `_adapter_activation_lock()` returns):
        see that method's docstring, and the base class's
        `_adapter_resolve_lock()` docstring, for the deadlock a shared lock
        would risk.
        """
        return self._adapter_resolve_lock_obj

    # ------------------------------------------------------------------
    # Convenience registration helpers
    # ------------------------------------------------------------------

    def _intrinsic_adapter_name_and_config(
        self, adapter: "EmbeddedIntrinsicAdapter | _AdapterCore"
    ) -> tuple[str, dict]:
        """Return the adapter-function name and raw io.yaml config for an adapter.

        `EmbeddedIntrinsicAdapter` carries both directly (`.name`/`.config`).
        A composed `Adapter` carries neither — `io_contract` does not yet
        drive `IntrinsicsRewriter`/`IntrinsicsResultProcessor` (Epic #929,
        issue #1144) — so its config is looked up from
        `_composed_adapter_configs`, cached at registration time (see
        `add_adapter`/`register_embedded_adapter_model`) since it comes from
        `adapter_index.json`/`io.yaml` and cannot be cheaply re-derived.

        Args:
            adapter: The adapter to resolve a name and config for.

        Returns:
            tuple[str, dict]: The adapter-function name and its parsed
            io.yaml config.

        Raises:
            ValueError: A composed adapter has no cached config (never
                registered via `add_adapter`/`register_embedded_adapter_model`).
        """
        if isinstance(adapter, EmbeddedIntrinsicAdapter):
            return adapter.name, adapter.config
        key = _composed_adapter_key(adapter)
        config = self._composed_adapter_configs.get(key)
        if config is None:
            raise ValueError(
                f"No io.yaml config cached for composed adapter {key!r}; register "
                "it via register_embedded_adapter_model() or resolve_adapter()."
            )
        return adapter.identity.name, config

    def register_embedded_adapter_model(
        self,
        source: str,
        *,
        revision: str = "main",
        cache_dir: str | None = None,
        intrinsic_name: str | None = None,
    ) -> list[str]:
        """Register embedded adapters from an Embedded Adapter model.

        Args:
            source (str): A local model directory path or Hugging Face Hub repo ID.
            revision (str): Git revision when loading from Hugging Face Hub.
            cache_dir (str | None): Cache directory for HF downloads.
            intrinsic_name (str | None): If provided, register only the adapter
                matching this adapter function name. `None` registers all
                adapters found in `source`.

        Returns:
            list[str]: Names of the registered intrinsics.

        Raises:
            ImportError: If Hugging Face Hub support is not installed.
            PermissionError: If the model repository is private or gated.
            FileNotFoundError: If the source has no adapter index.
            ValueError: If the source has no matching embedded adapter functions.
            TypeError: If an adapter from the source has an unsupported binding.
        """
        # Locked (unlike __init__'s call to this, which still runs
        # single-threaded during construction, before the backend is exposed
        # to any other thread): this method is now also the documented
        # post-construction replacement for `EmbeddedIntrinsicAdapter.from_hub()`
        # on a live backend, so a caller here can race another thread's
        # `add_adapter`/`resolve_adapter` — both `add_adapter`'s
        # read-then-write across `_added_adapters` and `_discover_embedded_adapters`'
        # mutation of global `warnings` filter state need the same lock
        # `add_adapter` itself would take via `resolve_adapter`.
        # `_adapter_resolve_lock()`, not `_adapter_activation_lock()`: see the
        # lock-order note on the latter — this method's own I/O (this Hub
        # discovery call) must not run under the lock `add_adapter()` briefly
        # takes around its dict writes.
        with self._adapter_resolve_lock():
            discovered = _discover_embedded_adapters(
                source,
                revision=revision,
                cache_dir=cache_dir,
                intrinsic_name=intrinsic_name,
            )
            names = []
            for adapter, config in discovered:
                key = _composed_adapter_key(adapter)
                # add_adapter() caches config atomically with registration now, so
                # a refused duplicate (a different object already holding `key`)
                # never reaches that write — no separate clobber guard needed for
                # the config. The identity check below is still required, though:
                # add_adapter() returns None on both success and silent refusal,
                # so this is the only way to know whether *this* adapter is the
                # one that actually ended up registered, for the `names` result.
                self.add_adapter(adapter, config=config)
                with self._adapter_activation_lock():
                    registered = self._added_adapters.get(key) is adapter
                if not registered:
                    continue
                names.append(adapter.identity.name)
            return names

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
            # `extra_body` is excluded here: it was already folded into
            # `_default_extra_body` at construction time (see `__init__`), so
            # merging it again would let it re-enter the per-call precedence
            # chain and outrank a per-call `ModelOption.THINKING`.
            backend_defaults={
                k: v for k, v in self.model_options.items() if k != "extra_body"
            },
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

    def _map_thinking_option(
        self, thinking: Any, extra_body: dict[str, Any]
    ) -> dict[str, Any]:
        """Maps `ModelOption.THINKING` to the correct backend parameter(s).

        Two mechanisms, both set (when applicable) so the right server picks
        up whichever it understands:
          - `extra_body["chat_template_kwargs"]["enable_thinking"]`: vLLM/Qwen3
          - `reasoning_effort`: OpenAI/DeepSeek/Ollama (string level; True →
            "medium", False → "none")

        Ollama-served models (e.g. granite4.2) think by default unless
        `reasoning_effort="none"` is sent (Ollama >= 0.33.1); real OpenAI
        rejects `"none"`, so that value is scoped to non-OpenAI servers.

        Args:
            thinking: the raw `ModelOption.THINKING` value (bool, string
                reasoning-effort level, or None).
            extra_body: the in-progress `extra_body` dict for this request;
                mutated in place to add `chat_template_kwargs` if `thinking`
                is a bool.

        Returns:
            dict[str, Any]: `reasoning_effort` params to merge into the
            request's top-level kwargs, or `{}` if `thinking` is None.
        """
        reasoning_params: dict[str, Any] = {}
        if thinking is None:  # False is a valid value — cannot use `if thinking`
            return reasoning_params
        if type(thinking) is bool:
            ctk = extra_body.get("chat_template_kwargs", {}) or {}
            ctk["enable_thinking"] = thinking
            extra_body["chat_template_kwargs"] = ctk
            if thinking:
                reasoning_params["reasoning_effort"] = "medium"
            elif self._server_type != _ServerType.OPENAI:
                reasoning_params["reasoning_effort"] = "none"
        else:
            reasoning_params["reasoning_effort"] = thinking
        return reasoning_params

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

                # An explicit adapter_types override (Epic #929, issue #1144)
                # is honoured here — e.g. a custom, LoRA-only adapter would
                # never be found by the ("alora",)-only default search,
                # silently falling back to regular generation regardless of
                # what the caller asked for.
                explicit_types = getattr(alora_action, "_adapter_types", None)
                search_types = (
                    tuple(t.value for t in explicit_types)
                    if explicit_types
                    else ("alora",)
                )
                alora_req_adapter = self._find_adapter(adapter_name, search_types)
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
                intrinsic, or a composed `Adapter` has no io.yaml config
                cached (see `_intrinsic_adapter_name_and_config`).
            TypeError: If the adapter is neither an `EmbeddedIntrinsicAdapter`
                nor a composed `Adapter`, or its `weights` isn't an
                `EmbeddedBinding` (for the shim, only reachable if a caller
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
        # EmbeddedIntrinsicAdapter is itself an _AdapterCore subclass, so checking
        # for _AdapterCore alone covers both the shim and composed-Adapter realities.
        if not isinstance(adapter, _AdapterCore):
            raise TypeError(
                "OpenAIBackend only supports EmbeddedIntrinsicAdapter or a composed "
                f"Adapter, got: {type(adapter).__name__}"
            )

        adapter_name, intrinsic_config = self._intrinsic_adapter_name_and_config(
            adapter
        )

        rewriter = granite_formatters.IntrinsicsRewriter(
            config_dict=intrinsic_config, model_name=adapter_name
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
        conversation.extend(
            [message_to_openai_message(m, provider=self._provider) for m in messages]
        )

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

        thinking = model_options.get(ModelOption.THINKING)
        api_params.update(self._map_thinking_option(thinking, extra_body))

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
        # None and the ordinary chat send runs (a correct cache miss). Either way the
        # coroutine is wrapped in `_await_embedded_generation` so a generation failure
        # fires the adapter invocation's `outcome="error"` event.
        reuse_prompt_ids = await self._reuse_intrinsic_prefix_ids(
            ctx, messages_dicts, extra_body, api_params, use_tools
        )
        if reuse_prompt_ids is not None:
            chat_response: Coroutine[Any, Any, ChatCompletion] = (
                _await_embedded_generation(
                    self._intrinsic_completion_as_chat(
                        reuse_prompt_ids, api_params, extra_body
                    ),
                    adapter.identity,
                )
            )
        else:
            chat_response = _await_embedded_generation(
                self._async_client.chat.completions.create(
                    model=self._model_id,
                    messages=messages_dicts,  # type: ignore
                    tools=formatted_tools if use_tools else None,  # type: ignore
                    extra_body=extra_body,
                    **api_params,
                ),
                adapter.identity,
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
            identity: Identity,
        ):
            """Accumulate content and apply intrinsic result processing."""
            import json as _json

            # Kept in one try, including the self.processing() call: a
            # response with an empty choices list raises IndexError there
            # (openai.py's processing() indexes chunk.choices[0].message
            # unguarded), and that must still fire outcome="error" rather
            # than skip the fire site entirely.
            try:
                # Delegate standard metadata storage to the shared processing method.
                await self.processing(mot, chunk)

                response_dict = chunk.model_dump()
                res = result_processor.transform(response_dict, rewritten)
                # Kept inside the try: a malformed `res` (e.g. an empty
                # choices list) must still fire outcome="error", not success
                # then raise — the exact bug #1559 reverted.
                mot._underlying_value = res.choices[0].message.content
            except _json.JSONDecodeError as e:
                await _fire_embedded_invocation_complete(
                    identity=identity, outcome="schema_error", error=e
                )
                raise Exception(
                    f"Intrinsic did not return a JSON: "
                    f"{chunk.choices[0].message.content}"
                ) from e
            except Exception as e:
                await _fire_embedded_invocation_complete(
                    identity=identity, outcome="error", error=e
                )
                raise
            else:
                await _fire_embedded_invocation_complete(
                    identity=identity, outcome="success", error=None
                )

        # Processing functions only pass the ModelOutputThunk (and current chunk
        # of response). Bind the other vars necessary for each processing step.
        output._gen.process = functools.partial(
            granite_formatters_processing,
            rewritten=rewritten,
            result_processor=result_processor,
            identity=adapter.identity,
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
                template_kwargs=mot._meta.get("retained_template_kwargs"),
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
        # Validated, not coerced. `int()` accepts things that are not ids and turns them
        # into plausible ones -- `True` -> 1, `2.9` -> 2, `"12"` -> 12 -- while
        # `int("abc")` raises a bare `ValueError` that escapes the callers catching
        # `TokenizeUnavailable` to fall back to a chat send. These ids BECOME the prompt,
        # and both sides of `derive_delta` come through this same reader, so a consistent
        # corruption cancels out of the subtraction and reaches the server with every
        # guard passing. Same rule `PreTokenizedCBlock` applies on the way out
        # (`core/base.py`):
        # an int, and `bool` is not one despite subclassing it.
        non_ids = [t for t in tokens if not isinstance(t, int) or isinstance(t, bool)]
        if non_ids:
            raise TokenizeUnavailable(
                f"/tokenize replied with {len(non_ids)} of {len(tokens)} entries that "
                f"are not exact token ids (e.g. {non_ids[0]!r}, a "
                f"{type(non_ids[0]).__name__}), so its ids cannot be trusted. Coercing "
                "them would build a different prompt than the server rendered."
            )
        return list(tokens)

    @property
    def token_id_reprefills(self) -> int:
        """How many retained turns dropped their prefix to stay inside the switch's range.

        Non-zero means at least one turn re-rendered from scratch, so the control tokens
        of every earlier turn are gone from the prompt and those regions are interpreted
        under base from that turn on. Nothing else distinguishes such a conversation from
        one that never needed it, which is why it is counted as well as logged.
        """
        return self._token_id_reprefills

    @staticmethod
    def _kwargs_cache_key(chat_template_kwargs: dict[str, Any] | None) -> str:
        """A stable cache key for a chat-template kwargs dict."""
        return json.dumps(chat_template_kwargs or {}, sort_keys=True, default=repr)

    async def _control_token_ids(self, adapter_name: str) -> list[int]:
        """Return the control-token ids the chat template adds for `adapter_name`.

        Learned rather than configured: unlike a local checkpoint, a served model exposes
        no `adapter_token_ids`, so the ids are recovered by rendering one probe with and
        without `adapter_name` and taking the positional difference. A control token
        SUBSTITUTES for the role marker rather than being inserted (for
        `ibm-granite/granite-switch-4.1-3b-preview`, `100356` in place of `100264`), so
        the two renders are the same length and only a positional diff finds them.

        Cached per adapter name and unioned into `_control_token_id_set`, which
        `_control_count` reads. An adapter never used by this backend contributes nothing
        to the set, so a control token the MODEL emits for such an adapter is not counted
        -- the ceiling is a floor on the true count, never an overestimate.

        Args:
            adapter_name (str): The adapter as passed in `chat_template_kwargs`.

        Returns:
            list[int]: The ids that differ, or `[]` when the server cannot be asked or
                the two renders disagree in length (nothing can be attributed then).
        """
        cached = self._control_token_ids_by_adapter.get(adapter_name)
        if cached is not None:
            return cached
        probe = [{"role": "user", "content": "x"}]
        try:
            plain = await self._tokenize_chat(probe, add_generation_prompt=True)
            adapted = await self._tokenize_chat(
                probe,
                add_generation_prompt=True,
                chat_template_kwargs={"adapter_name": adapter_name},
            )
        except TokenizeUnavailable:
            self._control_token_ids_by_adapter[adapter_name] = []
            return []
        if len(plain) != len(adapted):
            # Not a substitution: the diff cannot be attributed to a control token, and
            # guessing would make the count wrong in the direction that matters.
            self._control_token_ids_by_adapter[adapter_name] = []
            return []
        found = [now for was, now in zip(plain, adapted, strict=True) if was != now]
        self._control_token_ids_by_adapter[adapter_name] = found
        self._control_token_id_set.update(found)
        return found

    def _seed_control_tokens_from_adapters(self) -> None:
        """Seed `_control_token_id_set` from EVERY registered adapter's metadata.

        The ceiling guard is only correct if `_control_count` recognizes every control
        token the served model can emit -- not just the adapters this backend happens to
        have invoked. The per-adapter probe learns lazily and so undercounts (a control
        token for an un-probed adapter is not counted, leaving the ceiling a floor rather
        than a bound). Each registered adapter already carries its id from
        `adapter_index.json` on `identity.control_token_id`, so union them all up front.

        Idempotent and cheap (a dict scan); safe to call before every count. An adapter
        whose metadata supplied no id contributes nothing and falls back to the probe.
        """
        for adapter in self._added_adapters.values():
            cid = getattr(adapter.identity, "control_token_id", None)
            if cid is None:
                continue
            name = adapter.identity.name
            # Populate the per-name cache too, so `_control_token_ids` treats this
            # adapter as already learned and never probes for it.
            self._control_token_ids_by_adapter.setdefault(name, [cid])
            self._control_token_id_set.add(cid)

    async def _learn_control_tokens(self, adapter_name: object) -> None:
        """Ensure `_control_token_id_set` covers every adapter this model can emit.

        Seeds the COMPLETE set from registered adapters' metadata first (so the count is
        a true bound, not a floor), then falls back to the per-adapter `/tokenize` probe
        only for an `adapter_name` passed as a raw template kwarg with no registered
        adapter to read the id from. Called only after every guard that can refuse a
        request without a round trip, so a turn that will not reuse never pays the probe.

        Args:
            adapter_name (object): The `adapter_name` template kwarg, or a falsy value
                when this turn uses no adapter.
        """
        self._seed_control_tokens_from_adapters()
        if adapter_name and str(adapter_name) not in self._control_token_ids_by_adapter:
            await self._control_token_ids(str(adapter_name))

    def _control_count(self, ids: list[int]) -> int:
        """Control tokens in `ids`; `0` while none have been learned."""
        if not self._control_token_id_set:
            return 0
        return sum(1 for t in ids if t in self._control_token_id_set)

    async def _turn_terminator(
        self, chat_template_kwargs: dict[str, Any] | None = None
    ) -> list[int] | None:
        r"""Return the ids the chat template puts after an assistant turn.

        A model's reported ids stop at the end of its content; the template then adds
        a terminator (Granite: `<|end_of_text|>\n`). Those ids were in the prompt the
        server saw, so the retained sequence must include them, or the next turn runs
        an answer straight into the following role marker. Derived by subtracting an
        open render from a closed one.

        Probed under THIS turn's template kwargs and cached per kwargs, not once
        globally: Granite 4.2 renders the assistant boundary differently depending on
        `enable_thinking`, so a terminator derived under the template defaults would be
        spliced into a sequence closed the other way -- one wrong id mid-conversation,
        which `derive_delta` cannot catch because it compares two fresh renders and
        never looks at the retained ids.

        Args:
            chat_template_kwargs (dict[str, Any] | None): Template variables for the
                turn being closed. `adapter_name` is dropped: it applies to the turn's
                own region, not to its terminator, and keying on it would re-probe for
                every adapter.

        Returns:
            list[int] | None: The terminator ids, or `None` if the server could not
                be asked or the probe produced nothing usable -- in which case this
                turn cannot be retained.
        """
        kwargs = {
            k: v for k, v in (chat_template_kwargs or {}).items() if k != "adapter_name"
        } or None
        key = self._kwargs_cache_key(kwargs)
        cached = self._turn_terminator_ids.get(key)
        if cached is not None:
            return cached or None
        probe = [{"role": "user", "content": "x"}]
        try:
            open_ids = await self._tokenize_chat(
                probe, add_generation_prompt=True, chat_template_kwargs=kwargs
            )
            closed_ids = await self._tokenize_chat(
                [*probe, {"role": "assistant", "content": ""}],
                add_generation_prompt=False,
                chat_template_kwargs=kwargs,
            )
        except TokenizeUnavailable:
            # Cache the failure as empty: a server with no tokenize route will not
            # acquire one mid-session, so don't re-probe every turn.
            self._turn_terminator_ids[key] = []
            return None
        if len(closed_ids) <= len(open_ids) or closed_ids[: len(open_ids)] != open_ids:
            # The closed render does not extend the open one, so the probe yielded
            # nothing that can be attributed to a terminator.
            self._turn_terminator_ids[key] = []
            return None
        self._turn_terminator_ids[key] = closed_ids[len(open_ids) :]
        return self._turn_terminator_ids[key]

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
                digest that fingerprints fewer messages than the count claims, a leading
                prefix whose content no longer matches the recorded digest, or two
                renders that disagree on the already-sent side.
            TokenizeUnavailable: If the server cannot tokenize the conversation.
        """
        retained_ids = list(ctx.sent_token_ids)
        retained_count = ctx.sent_message_count
        retained_digest = ctx.sent_prompt_digest
        retained_model = ctx.sent_model_id
        retained_kwargs = ctx.sent_template_kwargs
        # `adapter_name` belongs to the turn being generated, never to the prefix.
        turn_kwargs = {
            k: v for k, v in (chat_template_kwargs or {}).items() if k != "adapter_name"
        }
        adapter_name = (chat_template_kwargs or {}).get("adapter_name")

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
            await self._learn_control_tokens(adapter_name)
            fresh = await self._tokenize_chat(
                conversation,
                add_generation_prompt=True,
                chat_template_kwargs=chat_template_kwargs,
            )
            self._assert_control_budget(fresh)
            return fresh

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
        # cannot see a rewritten earlier turn or dropped oldest turns. Checked before
        # tokenizing, so a changed prefix costs no round trip.
        #
        # The count and the digest describe the same messages, so they must agree
        # exactly. FEWER fingerprints than the count proves less than the count claims,
        # and the unproven tail is the newest end -- the assistant reply included -- so
        # an edit there would pass while the retained ids still carry the original text.
        # MORE fingerprints reach past the retained boundary, where an edit cannot
        # corrupt anything, and would refuse reuse for nothing. Either way the state was
        # not recorded by this backend and is not trusted; an empty digest stays the
        # explicit "no proof recorded" opt-out it has always been.
        if retained_digest and len(retained_digest) != retained_count:
            raise DeltaNotDerivable(
                f"the retained ids cover {retained_count} messages but their digest "
                f"fingerprints {len(retained_digest)}. The two must describe the same "
                "messages, so this retained state cannot be verified; this turn "
                "re-renders as chat messages and only its prefix-cache hit is lost."
            )
        # Compared over the count -- the messages the ids actually cover -- now that the
        # digest is guaranteed to be exactly that long.
        if retained_digest and (
            _prompt_digest(conversation[:retained_count]) != retained_digest
        ):
            raise DeltaNotDerivable(
                "the retained ids' leading prefix no longer matches this conversation: "
                "an earlier message's content changed, an already-sent turn was edited "
                "in place, or the oldest turns were dropped, while the message count "
                "stayed at or above the retained boundary. Splicing the old ids would "
                "send a prompt whose prefix the server never cached, so the ids are not "
                "reused; this turn re-renders as chat messages and only its prefix-cache "
                "hit is lost."
            )

        # A kwarg that re-renders the already-sent region cannot be represented as a
        # delta, and -- unlike a changed message -- it would not show up as one: it
        # appears in BOTH renders of the subtraction and cancels out, so the server
        # would receive a prompt containing it nowhere. Refused by name, before any
        # round trip. Compared with `adapter_name` excluded on both sides, since that
        # one legitimately differs per turn.
        if turn_kwargs != retained_kwargs:
            changed = sorted(
                set(turn_kwargs) ^ set(retained_kwargs)
                | {
                    k
                    for k in set(turn_kwargs) & set(retained_kwargs)
                    if turn_kwargs[k] != retained_kwargs[k]
                }
            )
            raise DeltaNotDerivable(
                f"chat template kwargs changed mid-conversation "
                f"({', '.join(changed)}), and the new values re-render turns whose ids "
                "are already retained. Such a kwarg appears in both renders of the "
                "subtraction and cancels out of the delta, so it would reach neither "
                "the reused prefix nor the new turn -- a request that silently omits "
                "it entirely. Pass the same chat_template_kwargs on every turn (supply "
                "documents=[...] from the first turn, empty if there are none yet), or "
                "use a context without `retain_token_ids`. This turn re-renders as chat "
                "messages and only its prefix-cache hit is lost."
            )

        # Learned only now: every guard above refuses without a round trip, and probing
        # for the ceiling before them would spend two on a request that is not sent.
        await self._learn_control_tokens(adapter_name)

        full_ids = await self._tokenize_chat(
            conversation,
            add_generation_prompt=True,
            chat_template_kwargs=chat_template_kwargs,
        )
        # The already-sent side is rendered under the kwargs it was ACTUALLY sent with,
        # and without this turn's adapter: its control token belongs to the new turn's
        # region, and rendering it into the subtracted region would cancel it out.
        prev_ids = await self._tokenize_chat(
            conversation[:retained_count],
            add_generation_prompt=False,
            chat_template_kwargs=retained_kwargs or None,
        )
        spliced = list(retained_ids) + derive_delta(prev_ids, full_ids)

        # Retaining ids is what makes the control-token count grow, so the ceiling is
        # enforced here. Over it, drop the prefix and re-render: earlier control tokens
        # are lost and those regions fall back to base, which is strictly better than
        # aliased write addresses silently routing to the mean of two experts.
        count = self._control_count(spliced)
        if count > MAX_RETAINED_CONTROL_TOKENS:
            rebaselined = await self._tokenize_chat(
                conversation,
                add_generation_prompt=True,
                chat_template_kwargs=chat_template_kwargs,
            )
            self._token_id_reprefills += 1
            MelleaLogger.get_logger().warning(
                "the retained prompt reached %d control tokens, past the %d the coded "
                "switch can address exactly in bf16, so its prefix was dropped and the "
                "transcript re-rendered (%d ids instead of %d). Earlier turns are "
                "interpreted under base from now on, and this conversation's prefix "
                "cache is recomputed once. Re-prefills so far: %d.",
                count,
                MAX_RETAINED_CONTROL_TOKENS,
                len(rebaselined),
                len(spliced),
                self._token_id_reprefills,
            )
            self._assert_control_budget(rebaselined)
            return rebaselined
        return spliced

    def _assert_control_budget(self, ids: list[int]) -> None:
        """Raise when a FULL render is itself past the addressable control-token count.

        Reached only for a prompt that is already a fresh render of the transcript, so
        dropping a retained prefix cannot reduce it: the count is coming from the current
        turn, or from control-token text recorded into a message.

        Args:
            ids (list[int]): The prompt about to be sent.

        Raises:
            RuntimeError: If `ids` holds more than `MAX_RETAINED_CONTROL_TOKENS`
                control tokens.
        """
        count = self._control_count(ids)
        if count > MAX_RETAINED_CONTROL_TOKENS:
            raise RuntimeError(
                f"{count} control tokens in one request exceeds "
                f"{MAX_RETAINED_CONTROL_TOKENS}, the range over which the coded switch "
                "recovers a write address exactly in bf16; past it addresses alias and "
                "two control tokens key one codeword, so routing degrades with no error "
                "in the output. This prompt is already a full render of the transcript, "
                "so dropping the retained prefix cannot reduce it -- the count comes "
                "from the current turn, or from control-token text recorded into a "
                "message. Shorten the transcript, or send this turn without "
                "`retain_token_ids`."
            )

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
        # On the CHOICE, which is where vLLM puts them on a chat reply and where
        # `_completion_choice_as_chat_response` therefore keeps them -- so this reads
        # one location whichever endpoint served the turn.
        choices = response.get("choices")
        if not isinstance(choices, list) or not choices:
            return None
        choice = choices[0]
        if not isinstance(choice, dict):
            return None
        # `token_ids` is always a key but reads null without the flag; check the VALUE.
        emitted = choice.get("token_ids")
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
        terminator_task = asyncio.ensure_future(
            self._turn_terminator(chat_template_kwargs)
        )
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
        #
        # Stamp the generation start BEFORE the request, matching the standard chat path
        # (which sets `_gen.start` just before `chat.completions.create`). `_generate_from_raw`
        # never sets it, so without this `_elapsed_ms()` returns -1 and LatencyMetricsPlugin
        # records a negative duration for every retained turn.
        gen_start = datetime.datetime.now()
        results, usage = await self._generate_from_raw(
            [PreTokenizedCBlock(prompt_ids)],
            ctx,
            format=_format,
            model_options=forwarded,
        )
        output = results[0]
        output._gen.start = gen_start

        # `_generate_from_raw` builds a thunk shaped for batch use; repair three things
        # for a chat turn.
        #
        # 0. Its `raw.response` is the completions CHOICE (`text`/`token_ids`, no
        #    `message`), but consumers on a chat turn dispatch on provider and expect
        #    chat shape -- `Message._parse` reads `["choices"][0]["message"]` and would
        #    raise KeyError. Normalized FIRST, so the `_parse` below and
        #    `_retained_ids` further down both see one shape.
        if isinstance(output.raw.response, dict):
            output.raw.response = _completion_choice_as_chat_response(
                output.raw.response,
                usage,
                # Recovered from the thunk, where `_generate_from_raw` recorded what the
                # enclosing completion reported; the per-choice dump does not carry them.
                response_id=output.generation.response_id,
                response_model=output.generation.response_model,
            )
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
            # Every rendered message PLUS the assistant turn just produced -- serialized
            # through the same `to_chat_messages` -> `message_to_openai_message` pipeline
            # the next turn renders its history with, so its fingerprint matches then.
            # (`_prompt_digest` projects only role/content/tool_calls, so a differing
            # `replay_reasoning` decision on the next turn cannot perturb it.)
            retained_messages = list(conversation) + [
                message_to_openai_message(m, self.formatter, provider=self._provider)
                for m in self.formatter.to_chat_messages([output])
            ]
            # Count and digest are derived from ONE list, so they cannot disagree about
            # which messages the ids cover: a count reaching past the digest would leave
            # the uncovered tail -- the assistant reply most of all -- reused on trust,
            # and an edit to it would splice the ORIGINAL reply back into the prompt.
            output._meta["retained_message_count"] = len(retained_messages)
            output._meta["retained_prompt_digest"] = _prompt_digest(retained_messages)
            # The kwargs this prefix was rendered under, so the next turn can
            # re-render the already-sent side the way it was actually sent and
            # refuse a kwarg that would cancel out of the subtraction.
            output._meta["retained_template_kwargs"] = dict(chat_template_kwargs or {})

        # This thunk is ALREADY computed (the reply had to be materialized to derive the
        # retained ids), so `avalue()` short-circuits and the post-call hook astream
        # normally fires never runs -- yet GENERATION_PRE_CALL already fired for this turn.
        # This method cannot fire the matching post-call itself: `_call.generation_id` is
        # assigned by the public `generate_from_context` wrapper only after this returns,
        # so a hook fired here would carry `None` and no plugin could match the open span.
        # Flag the thunk instead; the wrapper fires the post-call once the id is set. See
        # `_CallInfo.fire_post_call_on_return`.
        output._call.fire_post_call_on_return = True
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
                message_to_openai_message(
                    m, self.formatter, replay_reasoning=replay, provider=self._provider
                )
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

        # NOTE: don't pass THINKING to non-reasoning models (e.g. gpt-4o).
        thinking = model_opts.get(ModelOption.THINKING)
        ctk_body: dict[str, Any] = extra_params.get("extra_body", {}) or {}
        reasoning_params = self._map_thinking_option(thinking, ctk_body)
        extra_params["extra_body"] = ctk_body

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
            # Request shapes the completions transport cannot honour. Each is checked
            # before any round trip and costs only the cache hit, and each mirrors a gate
            # `_reuse_intrinsic_prefix_ids` already applies -- the two paths must decline
            # on the same grounds, or the same request succeeds differently depending on
            # whether an intrinsic or a chat turn issued it.
            decline_reason: str | None = None
            if (extra_params.get("extra_body") or {}).get("documents"):
                # `documents` is rendered into the system block by the chat template but
                # is not a `/tokenize` parameter, so pre-tokenized ids would omit it
                # entirely -- a RAG request answered against no context, with no error.
                decline_reason = (
                    "token-id history is not used for this turn because `documents` "
                    "were supplied: they are rendered server-side by the chat template "
                    "and cannot be tokenized into the prompt, so retained ids would "
                    "omit them. The turn is sent as chat messages instead; only its "
                    "prefix-cache hit is lost."
                )
            elif backend_specific.get("logprobs") or (
                extra_params.get("extra_body") or {}
            ).get("logprobs"):
                # The two endpoints report logprobs in incompatible shapes: a
                # completions choice as `tokens`/`token_logprobs`/`top_logprobs`/
                # `text_offset`, a chat choice as `content`, a list of
                # `{token, logprob, top_logprobs}`. Consumers read the chat shape
                # (`make_begin_to_token_table` takes `logprobs.content`, and
                # `TokenToFloat` requires a `ChatCompletionLogProbs`, not a dict), and
                # this path adapts only the reply's ENVELOPE -- so the id transport would
                # hand them a payload they cannot read inside a reply labelled a chat
                # completion.
                #
                # Both channels are read because the SDK sends `extra_body` keys at the
                # top level of the request: `logprobs=True` and
                # `extra_body={"logprobs": True}` are the same request to the server, so
                # a gate watching only the model option would leave the second one open.
                decline_reason = (
                    "token-id history is not used for this turn because `logprobs` "
                    "were requested: /v1/completions reports them in a different shape "
                    "than a chat reply, which consumers expect. The turn is sent as "
                    "chat messages instead; only its prefix-cache hit is lost."
                )
            if decline_reason is not None:
                MelleaLogger.get_logger().warning(decline_reason)
            else:
                try:
                    return await self._generate_via_token_ids(
                        ctx,
                        conversation,
                        (extra_params.get("extra_body") or {}).get(
                            "chat_template_kwargs"
                        ),
                        action=action,
                        linearized_context=linearized_context,
                        _format=_format,
                        model_options=model_opts,
                        has_tools=use_tools,
                    )
                except DeltaNotDerivable as e:
                    # The retained prefix cannot describe this turn: earlier messages
                    # were re-rendered rather than extended, history shrank, the digest
                    # no longer matches, or the template kwargs drifted. Raised before
                    # any request, so falling through to the chat send below is a clean
                    # first attempt -- correct output, no reuse.
                    #
                    # Retained ids are left in place rather than cleared, so a later
                    # turn that lines up with the prefix again resumes reuse.
                    MelleaLogger.get_logger().warning(
                        "token-id history could not be extended for this turn, so it "
                        "was sent as chat messages instead: %s The turn itself is "
                        "unaffected; the server's prefix cache is re-primed from this "
                        "render, and any control tokens in earlier turns are dropped "
                        "from it.",
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
            # Response-side metadata, which lives on the ENCLOSING completion rather
            # than the per-choice dump above. Set here because this endpoint has no
            # `post_processing()` step to populate it, so without this every batch
            # completion -- and every token-id-retained chat turn, which borrows this
            # method as its transport -- reports `None` to telemetry.
            output.generation.response_id = completion_response.id
            output.generation.response_model = completion_response.model
            if response.finish_reason:
                # This thunk's own choice, not every choice in the batch.
                output.generation.finish_reasons = [response.finish_reason]

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
