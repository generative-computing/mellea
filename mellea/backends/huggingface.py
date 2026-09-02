# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""A backend that uses the Hugging Face Transformers library.

The purpose of the Hugging Face backend is to provide a setting for implementing experimental features. If you want a performance local backend, and do not need experimental features such as Span-based context or aLoRA adapters, consider using Ollama backends instead.
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import datetime
import functools
import importlib
import json
import pathlib
import threading
import warnings
from collections import OrderedDict
from collections.abc import Callable, Coroutine, Sequence
from importlib import metadata
from typing import Any, ClassVar, TypeVar, cast

import jinja2
import jinja2.meta
from packaging.requirements import Requirement as PackagingRequirement
from packaging.version import InvalidVersion, Version

try:
    import llguidance
    import llguidance.hf
    import llguidance.torch
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
    from transformers.cache_utils import DynamicCache
    from transformers.generation.logits_process import LogitsProcessorList
    from transformers.generation.stopping_criteria import (
        StoppingCriteria,
        StoppingCriteriaList,
    )
    from transformers.generation.streamers import AsyncTextIteratorStreamer
    from transformers.generation.utils import (
        GenerateBeamDecoderOnlyOutput,
        GenerateDecoderOnlyOutput,
        GenerationMixin,
    )
    from transformers.modeling_utils import PreTrainedModel
    from transformers.tokenization_utils_base import PreTrainedTokenizerBase
    from transformers.trainer_utils import set_seed
except ImportError as e:
    raise ImportError(
        "The Hugging Face backend requires extra dependencies. "
        'Please install them with: pip install "mellea[hf]"'
    ) from e

from ..backends import kv_block_helpers
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
    Requirement,
    Span,
    get_audio_from_component,
    get_images_from_component,
)
from ..core.base import AbstractMelleaTool
from ..formatters import ChatFormatter, TemplateFormatter, granite as granite_formatters
from ..formatters.granite.base.util import (
    _LLGUIDANCE_GRAMMAR_DEFAULTS,
    _GuidanceLogitsProcessor,
)
from ..helpers import (
    DEFAULT_CHUNK_TIMEOUT,
    message_to_openai_message,
    messages_to_docs,
    send_to_queue,
)
from ..stdlib.components import Intrinsic, Message
from ..stdlib.requirements import ALoraRequirement, LLMaJRequirement
from ..telemetry.context import generate_request_id, with_context
from ._options import resolve_model_options
from .adapters import (
    AdapterMixin,
    EmbeddedActivationRequest,
    EmbeddedBinding,
    IntrinsicAdapter,
    LocalHFAdapter,
)
from .adapters._core import (
    Adapter as _AdapterCore,
    Identity,
    IOContract,
    LocalFileBinding,
    WeightsBinding,
    _fire_embedded_invocation_complete,
)
from .adapters.adapter import AdapterInput, EmbeddedIntrinsicAdapter
from .backend import FormatterBackend
from .cache import Cache, SimpleLRUCache
from .model_ids import ModelIdentifier
from .model_options import ModelOption
from .tools import (
    add_tools_from_context_actions,
    add_tools_from_model_options,
    convert_tools_to_json,
)
from .utils import to_chat, to_tool_calls


class _EventStoppingCriteria(StoppingCriteria):
    """StoppingCriteria that signals the model to stop when a threading.Event is set.

    Used by LocalHFBackend to implement cooperative cancellation: when
    `cancel_generation` is called, it sets the backing event via
    `_cancel_hook` before cancelling the asyncio task, giving the HF
    `model.generate` thread a chance to exit cleanly rather than running
    to completion.
    """

    def __init__(self, event: threading.Event) -> None:
        self._event = event

    def __call__(self, input_ids: Any, scores: Any, **kwargs: Any) -> bool:  # type: ignore[override]
        return self._event.is_set()


def _install_cancel_stopping_criteria(
    generate_options: dict[str, Any], streaming_kwargs: dict[str, Any]
) -> threading.Event:
    """Wire a cooperative-cancel event into the generate call's stopping criteria.

    Pops any caller-supplied `stopping_criteria` from *generate_options* (to
    avoid passing it twice via both `**generate_options` and
    `**streaming_kwargs`), prepends an :class:`_EventStoppingCriteria` backed
    by a fresh `threading.Event`, and stores the merged list in
    *streaming_kwargs*.  Returns the event so the caller can arm
    `output._gen.cancel_hook = event.set`.
    """
    cancel_event = threading.Event()
    user_sc = generate_options.pop("stopping_criteria", None)
    streaming_kwargs["stopping_criteria"] = StoppingCriteriaList(
        [_EventStoppingCriteria(cancel_event)]
        + (list(user_sc) if user_sc is not None else [])
    )
    return cancel_event


"""A configuration type for the unhappy path: Tokenizer * Model * torch device string

Hugging Face backends can initialize themselves from a model string if the transformers `Auto*` classes can be used. Therefore, a TransformersTorchConfig usually isn't required. However, sometimes a model needs special care to instantiate properly, or a custom device type needs to bse used. Instead of trying to do a lot of partial magic, we basically have two modaliites: either the constructor can figure out everything from the model_id, or the user has to provide an entire config.
"""
TransformersTorchConfig = tuple[PreTrainedTokenizerBase, PreTrainedModel, torch.device]

format: None = None  # typing this variable in order to shadow the global format function and ensure mypy checks for errors

_T = TypeVar("_T")


@dataclasses.dataclass
class HFAloraCacheInfo:
    """A dataclass for holding a KV cache and associated generation metadata.

    Used by `LocalHFBackend` to store intermediate model state that can be
    reused across generation requests via an LRU cache.

    Args:
        kv_cache (DynamicCache | None): The Hugging Face `DynamicCache` holding
            precomputed key/value tensors, or `None` if not available.
        merged_token_ids (Any): Token IDs corresponding to the cached prefix.
        merged_attention (Any): Attention mask for the cached prefix tokens.
        q_end (int): Index of the last prompt token in the merged token sequence;
            defaults to `-1`.
        scores (Any): Optional logit scores from the generation step; defaults to
            `None`.

    Attributes:
        kv_cache (DynamicCache | None): The cached key/value tensors.
        merged_token_ids (Any): Token IDs for the cached prefix.
        merged_attention (Any): Attention mask for the cached prefix.
        q_end (int): End index of the prompt portion in merged token IDs.
        scores (Any): Logit scores from generation, or `None`.
    """

    kv_cache: DynamicCache | None
    merged_token_ids: Any
    merged_attention: Any
    q_end: int = -1
    scores: Any = None


def _cleanup_kv_cache(cache_info: HFAloraCacheInfo) -> None:
    """Free GPU memory when KV cache is evicted from LRU.

    This function is called by SimpleLRUCache when an entry is evicted.
    It explicitly deletes tensor references and calls torch.cuda.empty_cache()
    to return pooled CUDA memory to the device.

    Args:
        cache_info: The HFAloraCacheInfo being evicted from cache.
    """
    import gc

    if cache_info is None:
        return

    kv = cache_info.kv_cache
    if kv is not None:
        # Delete individual tensors from each layer
        if hasattr(kv, "key_cache"):
            for tensor in kv.key_cache:
                del tensor
            kv.key_cache.clear()
        if hasattr(kv, "value_cache"):
            for tensor in kv.value_cache:
                del tensor
            kv.value_cache.clear()
        del cache_info.kv_cache

    # Delete other tensors
    if cache_info.merged_attention is not None:
        del cache_info.merged_attention

    # Delete score tensors if present
    if cache_info.scores is not None:
        for tensor in cache_info.scores:
            del tensor
        del cache_info.scores

    # Force Python garbage collection and return CUDA memory to device
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# Variables that Hugging Face injects into the Jinja template namespace automatically.
# These must be excluded from _chat_template_allowlist because they are provided by
# apply_chat_template itself — forwarding a model_option with the same name would
# either cause a "got multiple values for keyword argument" TypeError (for the named-param
# group) or silently replace a function injected by the Jinja environment (for the globals group).
#
# ⚠️  REVIEW NOTE: verify this set against the transformers source whenever upgrading.
# Namespace vars: transformers.utils.chat_template_utils.render_jinja_template
#                 (the compiled_template.render(...) call — every kwarg there becomes
#                 a Jinja variable and must be excluded from the allowlist)
# Jinja globals: transformers.utils.chat_template_utils._cached_compile_jinja_template
#                (the jinja_env.globals[...] assignments near the bottom)
_HF_INTERNAL_TEMPLATE_VARS: frozenset[str] = frozenset(
    {
        # --- Variables injected into the Jinja namespace by render_jinja_template ---
        # These come from `compiled_template.render(messages=chat, tools=..., documents=...,
        # add_generation_prompt=..., **kwargs)` in transformers' chat_template_utils.
        # Forwarding any of these from model_options to apply_chat_template would
        # cause a duplicate-kwarg TypeError or silently shadow the value HF supplies.
        "messages",  # the conversation; HF binds `chat` to this name in render()
        "tools",  # tool schemas; our call sites pass tools=convert_tools_to_json(...) explicitly
        "documents",  # RAG documents; injected even when None
        "add_generation_prompt",  # bool; passed explicitly at the standard-generation call site
        # --- Jinja environment globals set by _compile_jinja_template ---
        # find_undeclared_variables cannot see env globals, so these appear as "undeclared"
        # in templates that reference them and must be excluded manually.
        "raise_exception",  # callable: raises jinja2.exceptions.TemplateError
        "strftime_now",  # callable: returns datetime.now().strftime(format)
    }
)

_CHAT_TEMPLATE_THINKING_VARS: tuple[str, ...] = ("think", "thinking", "enable_thinking")


def _compute_generate_kwargs_allowlist() -> frozenset[str]:
    """Names that `transformers`' `model.generate` accepts as keyword arguments.

    Combines two sources of truth:

    - The explicit named parameters of `GenerationMixin.generate`
      (`stopping_criteria`, `streamer`, `synced_gpus`, …).
    - Every public field on `GenerationConfig`
      (`temperature`, `max_new_tokens`, `return_dict_in_generate`, …).

    Anything outside this set, passed as a free kwarg to `generate`, is either
    absorbed by `**kwargs` and forwarded to the model forward pass (rare, but
    valid for forward-only kwargs Mellea sets explicitly — `attention_mask`,
    `past_key_values`, `cache_position`, `token_type_ids`, `inputs_embeds`)
    or — far more commonly — raises `TypeError` because the chat template
    variable name (`thinking`, `enable_thinking`, `custom_tools` …) is not
    a generation parameter. This allowlist is the self-maintaining filter that
    drops those before they reach `generate`.

    Computed at import time; depends only on the installed `transformers`
    version.
    """
    import inspect

    sig = inspect.signature(GenerationMixin.generate)
    sig_kw = {
        p.name
        for p in sig.parameters.values()
        if p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)
    } - {"self", "inputs"}
    gc_fields = {a for a in vars(GenerationConfig()) if not a.startswith("_")}
    return frozenset(sig_kw | gc_fields)


_GENERATE_KWARGS_ALLOWLIST: frozenset[str] = _compute_generate_kwargs_allowlist()
_SWITCH_TRANSFORMERS_WARNING_EMITTED = False


def _warn_if_granite_switch_transformers_override_is_active() -> None:
    """Warn once when Granite Switch metadata excludes the installed Transformers.

    Mellea intentionally overrides Granite Switch's restrictive Transformers
    metadata for the local embedded-adapter path. The warning is self-retiring:
    an upstream package release that widens its declared range makes the
    installed version satisfy the requirement and prevents the warning.
    """
    global _SWITCH_TRANSFORMERS_WARNING_EMITTED
    if _SWITCH_TRANSFORMERS_WARNING_EMITTED:
        return

    try:
        installed_version = Version(metadata.version("transformers"))
        requirements = metadata.requires("granite-switch") or []
    except (metadata.PackageNotFoundError, InvalidVersion):
        return

    try:
        for requirement_text in requirements:
            requirement = PackagingRequirement(requirement_text)
            if requirement.name != "transformers":
                continue
            if requirement.marker is not None and not requirement.marker.evaluate():
                continue
            if installed_version not in requirement.specifier:
                warnings.warn(
                    "granite-switch declares a Transformers range that excludes "
                    f"{installed_version}; Mellea is using its explicit compatibility "
                    "override for local Granite Switch support. This warning will "
                    "disappear once Granite Switch publishes widened metadata.",
                    UserWarning,
                    stacklevel=3,
                )
                _SWITCH_TRANSFORMERS_WARNING_EMITTED = True
            return
    except Exception:
        return


def _check_no_multimodal_blocks(action: Span | None, ctx: Context | None) -> None:
    """Raise ValueError if any component in ctx or action carries image/audio blocks.

    Args:
        action: The generation action component, or None for intrinsic paths.
        ctx: The current conversation context, or None to skip the context scan
            (used on paths where ctx content is never rendered, e.g. `_generate_from_raw`).

    Raises:
        ValueError: If any component contains images or audio.
            `LocalHFBackend` does not support multimodal inputs.
    """
    linearized = (ctx.view_for_generation() or []) if ctx is not None else []
    candidates: list[Any] = list(linearized)
    if action is not None:
        candidates.append(action)
    for c in candidates:
        if get_images_from_component(c) is not None:
            raise ValueError(
                "LocalHFBackend does not support images. "
                "Remove image content before passing messages to the Hugging Face backend."
            )
        if get_audio_from_component(c) is not None:
            raise ValueError(
                "LocalHFBackend does not support audio. "
                "Remove audio content before passing messages to the Hugging Face backend."
            )


class LocalHFBackend(FormatterBackend, AdapterMixin):
    """The LocalHFBackend uses Hugging Face's transformers library for inference, and uses a Formatter to convert `Component`s into prompts. This backend also supports [aLoRA adapters](https://arxiv.org/pdf/2504.12397).

    This backend is designed for running an HF model for small-scale inference locally on your machine.

    This backend is NOT designed for inference scaling on CUDA-enabled hardware.

    Args:
        model_id (str | ModelIdentifier): Used to load the model and tokenizer via
            Hugging Face `Auto*` classes.
        formatter (ChatFormatter | None): Formatter for rendering components into
            prompts. Defaults to `TemplateFormatter`.
        use_caches (bool): If `False`, KV caching is disabled even if a `Cache`
            is provided.
        cache (Cache | None): Caching strategy; defaults to
            `SimpleLRUCache(0, on_evict=_cleanup_kv_cache)`.
        custom_config (TransformersTorchConfig | None): Override for
            tokenizer/model/device; if provided, `model_id` is not used for loading.
        default_to_constraint_checking_alora (bool): If `False`, aLoRA constraint
            checking is deactivated; mainly for benchmarking and debugging.
        load_embedded_adapters (bool): If `True`, register adapter functions
            embedded in the Granite Switch checkpoint named by `adapter_source`
            (or `model_id` when `adapter_source` is not set). Requires the
            `hf` extra.
        adapter_source (str | None): Local checkpoint directory or Hugging Face
            Hub repository used to discover embedded adapter functions.
        model_options (dict | None): Default model options for generation requests.

    Attributes:
        to_mellea_model_opts_map (dict): Mapping from HF-specific option names to
            Mellea `ModelOption` sentinel keys.
        from_mellea_model_opts_map (dict): Mapping from Mellea sentinel keys to
            HF-specific option names.

    Raises:
        ImportError: If embedded adapter loading is requested without the `hf`
            extra installed.
        OSError: If the model cannot be loaded from Hugging Face Hub (bad ID,
            missing access, or local filesystem/cache error).
    """

    _cached_blocks: dict[str, DynamicCache] = dict()

    def __init__(
        self,
        model_id: str | ModelIdentifier,
        formatter: ChatFormatter | None = None,
        *,
        use_caches: bool = True,
        cache: Cache | None = None,
        custom_config: TransformersTorchConfig | None = None,
        default_to_constraint_checking_alora: bool = True,
        load_embedded_adapters: bool = False,
        adapter_source: str | None = None,
        model_options: dict | None = None,
    ):
        """Load model weights from the given model ID, or from a custom config if provided."""
        formatter = (
            formatter if formatter is not None else TemplateFormatter(model_id=model_id)
        )

        super().__init__(model_id, formatter, model_options=model_options)

        # A mapping of common options for this backend mapped to their Mellea ModelOptions equivalent.
        # These are usually values that must be extracted before hand or that are common among backend providers
        self.to_mellea_model_opts_map = {
            "system": ModelOption.SYSTEM_PROMPT,
            "max_new_tokens": ModelOption.MAX_NEW_TOKENS,
            "seed": ModelOption.SEED,
            "tools": ModelOption.TOOLS,
            "stream": ModelOption.STREAM,
            "stop_strings": ModelOption.STOP_SEQUENCES,
            "think": ModelOption.THINKING,
            "thinking": ModelOption.THINKING,
            "enable_thinking": ModelOption.THINKING,
        }

        # A mapping of Mellea specific ModelOptions to the specific names for this backend.
        # These options should almost always be a subset of those specified in the `to_mellea_model_opts_map`.
        # Usually, values that are intentionally extracted while prepping for the backend generate call
        # will be omitted here so that they will be removed when model_options are processed
        # for the call to the model.
        self.from_mellea_model_opts_map = {
            ModelOption.MAX_NEW_TOKENS: "max_new_tokens",
            ModelOption.STOP_SEQUENCES: "stop_strings",
        }

        self.default_to_constraint_checking_alora = default_to_constraint_checking_alora
        self._provider: str = "huggingface"

        # Either use the custom config or load the model from its model_id
        match model_id:
            case str():
                self._model_id: str = model_id
            case ModelIdentifier():
                assert model_id.hf_model_name is not None, (
                    "model_id is None. This can also happen if the ModelIdentifier has no hf_model_id name set."
                )
                self._model_id = model_id.hf_model_name
        if load_embedded_adapters:
            try:
                importlib.import_module("granite_switch.hf")
            except ImportError as e:
                raise ImportError(
                    "Loading Granite Switch checkpoints with LocalHFBackend "
                    'requires the Hugging Face extra. Install it with: pip install "mellea[hf]"'
                ) from e
            _warn_if_granite_switch_transformers_override_is_active()

        match custom_config:
            case None:
                # Choose a device.
                self._device = torch.device(
                    "cuda"
                    if torch.cuda.is_available()
                    else "mps"
                    if torch.backends.mps.is_available()
                    else "cpu"
                )
                # Get the model and tokenizer.
                try:
                    self._model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
                        self._model_id, device_map=str(self._device)
                    )
                    self._tokenizer: PreTrainedTokenizerBase = (
                        AutoTokenizer.from_pretrained(self._model_id)
                    )
                except OSError as e:
                    raise OSError(
                        f"Model '{self._model_id}' could not be loaded from Hugging Face Hub. "
                        "Check that the model ID is correct and that you have access to it. "
                        "If the model is gated, set the HF_TOKEN environment variable. "
                        "To browse available models, visit https://huggingface.co/models "
                        f"(Original error: {e})"
                    ) from e
            case _:
                self._tokenizer, self._model, self._device = custom_config

        # Preemptively fix vocab size discrepancies between the tokenizer and model if needed.
        n_vocab = max(
            self._tokenizer.vocab_size, len(self._tokenizer), self._model.vocab_size
        )
        self._llguidance_tokenizer: llguidance.LLTokenizer = (
            llguidance.hf.from_tokenizer(self._tokenizer, n_vocab=n_vocab)  # type:ignore
        )
        assert (
            self._llguidance_tokenizer.vocab_size
            == self._tokenizer._tokenizer.get_vocab_size(with_added_tokens=True)
        ), "vocab size mismatch between llguidance and Hugging Face tokenizers"

        self._use_caches = use_caches
        self._cache = (
            cache
            if cache is not None
            else SimpleLRUCache(0, on_evict=_cleanup_kv_cache)
        )

        # Adapters can be made known to the backend (added) and loaded.
        self._added_adapters: dict[
            str, LocalHFAdapter | LocalFileBinding | EmbeddedIntrinsicAdapter
        ] = {}
        self._loaded_adapters: dict[str, LocalHFAdapter | LocalFileBinding] = {}
        self._adapter_source = adapter_source
        self._uses_embedded_adapters = load_embedded_adapters

        self._generation_lock = threading.RLock()
        """Forces generation requests to be non-concurrent, and guards adapter
        activation (`_adapter_activation_lock()` returns this same lock).

        Reentrant (`threading.RLock`, not `threading.Lock`) because
        `_generate_intrinsic_with_adapter_scope` holds it for a whole
        prepare -> activate -> generate -> deactivate critical section, and
        `adapter_scope()`'s `activate()`/`deactivate()` re-acquire it from
        inside that section, on the same thread, via `_adapter_activation_lock()`.
        A plain `Lock` deadlocks on that same-thread re-acquisition (#1465).
        """

        if load_embedded_adapters:
            self.register_embedded_adapter_model(self._adapter_source or self._model_id)

    def _make_dc_cache(self, toks, **model_options):
        dc = DynamicCache()
        with torch.no_grad():
            dc = self._model(
                toks["input_ids"].to(self._device),
                attention_mask=toks["attention_mask"].to(self._device),
                past_key_values=dc,
                **model_options,
            ).past_key_values
        return dc

    async def _generate_from_context(
        self,
        action: Component[C] | CBlock | ModelOutputThunk,
        ctx: Context,
        *,
        format: type[BaseModelSubclass] | None = None,
        model_options: dict | None = None,
        tool_calls: bool = False,
    ) -> tuple[ModelOutputThunk[C], Context]:
        """Generate a completion for `action` given `ctx` using the Hugging Face model.

        Automatically routes `Requirement` and `Intrinsic` actions to their
        corresponding aLoRA adapters when available.

        Args:
            action (Component[C] | CBlock): The component or content block to generate
                a completion for.
            ctx (Context): The current generation context (must be a chat context).
            format (type[BaseModelSubclass] | None): Optional Pydantic model class for
                structured/constrained output decoding via llguidance.
            model_options (dict | None): Per-call model options that override the
                backend's defaults.
            tool_calls (bool): If `True`, expose available tools to the model and
                parse tool-call responses.

        Returns:
            tuple[ModelOutputThunk[C], Context]: A thunk holding the (lazy) model output
                and an updated context that includes `action` and the new output.
        """
        with with_context(
            request_id=generate_request_id(),
            model_id=str(getattr(self, "model_id", "unknown")),
        ):
            await self.do_generate_walk(action)

            # Upsert model options.
            model_opts = self._simplify_and_merge(model_options)

            # Requirements can be automatically rerouted to a requirement adapter.
            if isinstance(action, Requirement):
                # See "How automatic routing works" in
                # docs/docs/advanced/lora-and-alora-adapters.md for the three
                # exceptions to this rule.
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

                # Check if a requirement-check (or AloraRequirement specified) adapter
                # exists.
                alora_req_adapter = self._find_adapter(adapter_name, ("alora",))
                if alora_req_adapter is None:
                    # Log a warning if using an AloraRequirement but no adapter fit.
                    if reroute_to_alora and isinstance(action, ALoraRequirement):
                        MelleaLogger.get_logger().warning(
                            f"attempted to use an AloraRequirement but backend {self} doesn't have the specified adapter added {adapter_name}; defaulting to regular generation"
                        )
                    reroute_to_alora = False

                if issubclass(type(action), LLMaJRequirement):
                    reroute_to_alora = False

                if reroute_to_alora:
                    # Keep the alora requirement handling separate for now.
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

            mot = await self._generate_from_context_standard(
                action,
                ctx,
                _format=format,
                model_options=model_opts,
                tool_calls=tool_calls,
            )

            return mot, ctx.add(action).add(mot)

    def _generate_with_adapter_lock(self, generate_func: Callable, *args, **kwargs):
        """Exclusive generation against the base model for the standard path.

        Standard (non-intrinsic) generation runs without adapters: any active
        adapter is deactivated before the model call, and the model's active
        state is asserted before and after. Adapter-active generation goes
        elsewhere — `_generate_intrinsic_with_adapter_scope` for runtime PEFT
        adapters (routed through `adapter_scope`, with its lifecycle hooks) and
        `_generate_embedded_with_generation_lock` for Granite Switch adapters,
        which are activated while rendering the chat template.

        Args:
            generate_func: The synchronous generation callable to invoke.
            *args: Positional arguments forwarded to `generate_func`.
            **kwargs: Keyword arguments forwarded to `generate_func`.

        Returns:
            Whatever `generate_func` returns.
        """
        with self._generation_lock:
            self.deactivate_peft_adapter("")

            _assert_correct_adapters("", self._model)
            out = generate_func(*args, **kwargs)
            _assert_correct_adapters("", self._model)
            return out

    def _generate_embedded_with_generation_lock(
        self, generate_func: Callable[..., _T], *args: Any, **kwargs: Any
    ) -> _T:
        """Run embedded-adapter generation while serialising model access.

        Embedded adapter functions select their control token while rendering
        the chat template. They must not use the PEFT lifecycle, but local
        Transformers generation still shares the backend's model and therefore
        remains serialised with every other generation path.

        Args:
            generate_func: The synchronous generation callable to invoke.
            *args: Positional arguments forwarded to `generate_func`.
            **kwargs: Keyword arguments forwarded to `generate_func`.

        Returns:
            Whatever `generate_func` returns.
        """
        with self._generation_lock:
            self.deactivate_peft_adapter("")
            _assert_correct_adapters("", self._model)
            return generate_func(*args, **kwargs)

    def _generate_intrinsic_with_adapter_scope(
        self,
        adapter: IntrinsicAdapter,
        generate_func: Callable[..., _T],
        *args: Any,
        **kwargs: Any,
    ) -> _T:
        """Runs `generate_func` with `adapter` active, inside `AdapterMixin.adapter_scope`.

        The model call now runs inside `adapter_scope`, so activation and
        deactivation fire the `ADAPTER_FUNCTION_PHASE_COMPLETE`/
        `ADAPTER_FUNCTION_INVOCATION_COMPLETE` hooks (`phase="activate"` and
        `"deactivate"` only — `"generate"`/`"parse"` are not emitted; adding
        those is out of scope here, see #1466), and deactivation is guaranteed
        even if `generate_func` raises. Those hooks dispatch while
        `_generation_lock` is held below: the deadlock and stall consequences
        in `adapter_scope()`'s hook-dispatch paragraph apply here in full,
        since this backend's `_generation_lock` is what the dispatching thread
        holds, and the `RLock` cannot rescue a subscriber that re-enters from
        the event-loop thread — reentrance only helps the owning thread.
        `ADAPTER_FUNCTION_*` subscribers must therefore be non-blocking and
        must never re-enter this backend; a merely slow one stalls all
        generation on this backend for that dispatch's duration.

        `IntrinsicAdapter` (the legacy shim `adapter` is expected to be) carries
        an inert `_ShimWeightsBinding` that raises on every verb, so it can't be
        handed to `adapter_scope()` directly. This method builds a fresh
        `_AdapterCore` around `adapter`'s own `identity` (not a rebuilt one —
        rebuilding it would re-run `Identity.__post_init__`'s capability-registry
        check on every call) wrapping a `_IntrinsicPeftBinding`, which drives
        this backend's own `load_peft_adapter`/`activate_peft_adapter`/
        `deactivate_peft_adapter` verbs — the same verbs
        `_generate_with_adapter_lock` used to call directly.

        Holds `_generation_lock` for the *entire* prepare -> activate ->
        generate -> deactivate section, not just around activation, so a
        concurrent caller's `activate()` can never interleave with this call's
        generation — the atomicity gap `adapter_scope()`'s own docstring flags
        as unresolved for a caller that doesn't do this. Safe here because each
        invocation of this method runs entirely on one thread (in production it
        is only ever invoked via a single `asyncio.to_thread` call — see its
        call site in `_generate_from_intrinsic`; the tests call it directly,
        one invocation per thread) and never re-enters generation on
        another thread: the inner re-acquisition of `_generation_lock` inside
        `adapter_scope()`'s `activate()`/`deactivate()` (via
        `_adapter_activation_lock()`) is therefore same-thread and reentrant.
        Concurrent invocations land on different `asyncio.to_thread` pool
        threads and simply serialise on the lock.

        Args:
            adapter: The (legacy) `IntrinsicAdapter` to activate for this call.
            generate_func: The synchronous generation callable to invoke while
                `adapter` is active.
            *args: Positional arguments forwarded to `generate_func`.
            **kwargs: Keyword arguments forwarded to `generate_func`.

        Returns:
            Whatever `generate_func` returns.
        """
        weights = _IntrinsicPeftBinding(
            self, adapter.qualified_name, adapter.intrinsic_metadata.revision
        )
        scope_adapter = _AdapterCore(
            identity=adapter.identity, io_contract=_UnusedIOContract(), weights=weights
        )
        with self._generation_lock:
            weights.prepare()
            with self.adapter_scope(scope_adapter):
                _assert_correct_adapters(adapter.qualified_name, self._model)
                out = generate_func(*args, **kwargs)
                _assert_correct_adapters(adapter.qualified_name, self._model)
                return out

    async def _generate_from_intrinsic(
        self,
        action: Intrinsic,
        ctx: Context,
        *,
        model_options: dict[str, Any],
        tool_calls: bool = False,
    ) -> ModelOutputThunk:
        """Generate a completion for an intrinsic action using an adapter.

        Applies the intrinsic's I/O rewriter to transform the conversation,
        injects `intrinsic_name` into `chat_template_kwargs` so that the
        Granite Switch chat template activates the correct adapter, and
        post-processes the model output through the intrinsic's result
        processor.

        Intrinsics default to options provided by `io.yaml`. Model options
        override these defaults. All model options besides streaming are
        respected. We add `do_sample=True` if `temperature != 0.0` and `temperature is not None`.

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
            ValueError: If no adapter is registered for the requested intrinsic,
                or if the context contains images or audio; `LocalHFBackend`
                does not support multimodal inputs.
            TypeError: If the adapter is neither an `IntrinsicAdapter` nor an
                `EmbeddedIntrinsicAdapter`, or if an embedded adapter does not
                have an `EmbeddedBinding`.
        """
        if not ctx.is_chat_context:
            raise Exception("Does not yet support non-chat contexts.")

        _check_no_multimodal_blocks(None, ctx)

        seed = model_options.get(ModelOption.SEED, None)
        if seed is not None:
            set_seed(seed)

        # Collect tools if tool_calls is enabled.
        tools: dict[str, AbstractMelleaTool] = dict()
        if tool_calls:
            add_tools_from_model_options(tools, model_options)
            add_tools_from_context_actions(tools, ctx.actions_for_available_tools())
            MelleaLogger.get_logger().info(f"Tools for call: {tools.keys()}")

        # Intrinsics don't support streaming because of their post-processing step.
        if model_options.get(ModelOption.STREAM, False):
            raise NotImplementedError(
                "Intrinsics do not support streaming due to structured output parsing."
            )

        linearized_ctx = ctx.view_for_generation()
        assert linearized_ctx is not None, (
            "If ctx.is_chat_context, then the context should be linearizable."
        )
        ctx_as_message_list: list[Message] = self.formatter.to_chat_messages(
            linearized_ctx
        )

        # NOTE: Explicitly do not add the action to the context here. Intrinsics modify the context
        #       through their rewriters.

        # Extract system prompt and prepend to conversation.
        system_prompt = model_options.get(ModelOption.SYSTEM_PROMPT, "")
        conversation: list[dict] = []
        if system_prompt != "":
            conversation.append({"role": "system", "content": system_prompt})
        # Intrinsic/adapter calls are single-shot evaluations over a rewritten
        # conversation, not multi-turn generation, so reasoning is never replayed
        # here (no `replay_reasoning=`). The HF chat path serializes via
        # `to_chat`/`apply_chat_template`, not this helper.
        conversation.extend(
            [
                message_to_openai_message(m, provider=self._provider)
                for m in ctx_as_message_list
            ]
        )

        docs = messages_to_docs(ctx_as_message_list)

        allowed_types = tuple(at.value for at in action.adapter_types)
        adapter = self._find_adapter(action.intrinsic_name, allowed_types)
        if adapter is None:
            raise ValueError(
                f"backend ({self}) has no adapter for processing intrinsic: {action.intrinsic_name}"
            )

        # TODO: Code below this point is mostly specific to RagIntrinsics
        #       It should be refactored into a specific adapter.transform() function.
        if not isinstance(adapter, (IntrinsicAdapter, EmbeddedIntrinsicAdapter)):
            raise TypeError(
                "LocalHFBackend only supports IntrinsicAdapter or "
                f"EmbeddedIntrinsicAdapter, got: {type(adapter).__name__}"
            )

        intrinsic_config = adapter.config
        assert intrinsic_config is not None

        rewriter = granite_formatters.IntrinsicsRewriter(
            config_dict=intrinsic_config, model_name=adapter.name
        )
        result_processor = granite_formatters.IntrinsicsResultProcessor(
            config_dict=intrinsic_config
        )

        # The pydantic models used by the intrinsic rewriter are stricter than the actual OpenAI SDK.
        # Extract the "function" fields from each json tool which contains `{"name":..., "description":..., "parameters":...}`.
        formatted_tools = [tool["function"] for tool in convert_tools_to_json(tools)]
        # Convert our conversation into a proper chat completions dict.
        # [{role: user, content: Hello}, {...}] -> {messages: [{role:user,...}, ...], model:..., ...}
        request_json: dict = {
            "messages": conversation,
            "extra_body": {"documents": docs},
            "tools": formatted_tools if len(formatted_tools) > 0 else None,
        }

        rewritten = rewriter.transform(request_json, **action.intrinsic_kwargs)

        # Extract temperature and apply it to the rewritten request so that
        # chat_completion_request_to_transformers_inputs handles the
        # do_sample/temperature logic correctly.
        temperature = model_options.get(ModelOption.TEMPERATURE, None)
        if temperature is not None:
            rewritten = rewritten.model_copy(update={"temperature": temperature})

        # TODO: Handle caching here. granite_formatters doesn't tell us what changed,
        #       so we will have to invalidate the cache on our side. This requires
        #       us having specific caching for each Component/Message.

        rewritten_request = rewritten.model_dump()
        if isinstance(adapter, EmbeddedIntrinsicAdapter):
            if not isinstance(adapter.weights, EmbeddedBinding):
                raise TypeError(
                    "EmbeddedIntrinsicAdapter.weights must be an EmbeddedBinding; "
                    f"got {type(adapter.weights).__name__}. Activation cannot proceed."
                )
            extra_body = rewritten_request.setdefault("extra_body", {})
            if not isinstance(extra_body, dict):
                raise TypeError(
                    "Embedded adapter generation requires extra_body to be a dict."
                )
            await adapter.weights.apply_activation(
                EmbeddedActivationRequest(
                    extra_body=extra_body, api_params=rewritten_request
                ),
                adapter.identity,
            )

        generate_input, other_input = (
            granite_formatters.base.util.chat_completion_request_to_transformers_inputs(  # type: ignore
                rewritten_request,
                self._tokenizer,
                self._model,
                ll_tokenizer=self._llguidance_tokenizer,
            )
        )

        # Apply remaining user model options directly to generate_input,
        # overwriting any values set by the util function or io.yaml defaults.
        # We don't update other_input since those inputs are specific to `generate_with_transformers`
        # and not covered by model options.
        user_params = self._make_backend_specific_and_remove(model_options)
        if temperature == 0.0:
            # Preserve the formatter's greedy do_sample=False setup; temperature=0 is invalid once sampling is enabled.
            user_params.pop("temperature", None)
        if "stop_strings" in user_params and "tokenizer" not in user_params:
            user_params["tokenizer"] = self._tokenizer
        generate_input.update(user_params)

        want_scores = bool(
            model_options.get(ModelOption.LOGITS)
            or model_options.get(ModelOption.RAW_LOGITS)
        )
        if model_options.get(ModelOption.LOGITS):
            generate_input["output_scores"] = True
        if model_options.get(ModelOption.RAW_LOGITS):
            generate_input["output_logits"] = True

        # When logits are requested, intercept the raw GenerateDecoderOnlyOutput that
        # generate_with_transformers produces internally but never returns (it wraps
        # everything into a ChatCompletionResponse). We proxy self._model so that
        # .generate() stores the raw output in raw_hf_output_cell before returning it;
        # granite_formatters_processing then writes it to mot.raw.response for
        # _surface_logits in post_processing.
        raw_hf_output_cell: list[GenerateDecoderOnlyOutput | None] = [None]

        model_arg = self._model
        if want_scores:
            _real_model = self._model

            # Two load-bearing assumptions: (1) __getattr__ falls through for attributes
            # accessed by generate_with_transformers (model.device, model.vocab_size,
            # model.generation_config). (2) chat_completion_request_to_transformers_inputs
            # always sets return_dict_in_generate=True, so .generate() always returns a
            # GenerateDecoderOnlyOutput — if that ever changes, the cell stays None and
            # logits silently won't be populated.
            class _CapturingModelProxy:
                def generate(self_proxy, *args: Any, **kwargs: Any) -> Any:
                    result = cast(Callable[..., Any], _real_model.generate)(
                        *args, **kwargs
                    )
                    if isinstance(result, GenerateDecoderOnlyOutput):
                        raw_hf_output_cell[0] = result
                    return result

                def __getattr__(self_proxy, name: str) -> Any:
                    return getattr(_real_model, name)

            model_arg = _CapturingModelProxy()  # type: ignore[assignment]

        if isinstance(adapter, IntrinsicAdapter):
            chat_response = asyncio.to_thread(
                self._generate_intrinsic_with_adapter_scope,
                adapter,
                granite_formatters.base.util.generate_with_transformers,  # type: ignore
                # Passed as args/kwargs to generate.
                self._tokenizer,
                model_arg,
                generate_input,
                other_input,
            )
        else:
            chat_response = asyncio.to_thread(
                self._generate_embedded_with_generation_lock,
                granite_formatters.base.util.generate_with_transformers,  # type: ignore
                # Passed as args/kwargs to generate.
                self._tokenizer,
                model_arg,
                generate_input,
                other_input,
            )

        output = ModelOutputThunk(None)
        output._gen.start = datetime.datetime.now()
        output._call.context = ctx.view_for_generation()
        output._call.action = action
        output._call.model_options = model_options

        # Add another step to the processing function.
        async def granite_formatters_processing(
            mot: ModelOutputThunk,
            chunk: granite_formatters.ChatCompletionResponse,
            rewritten: granite_formatters.ChatCompletion,
            result_processor: granite_formatters.IntrinsicsResultProcessor,
            input_ids,
            embedded_identity: Identity | None,
        ):
            # embedded_identity is set only for EmbeddedIntrinsicAdapter calls.
            # This is the single synchronous point where the Embedded call's
            # real outcome becomes known, so it also fires
            # adapter_function_invocation_complete for that case (apply_activation
            # could not have known the outcome earlier). The legacy PEFT path
            # (embedded_identity=None) already gets this signal from
            # adapter_scope, so it must not be fired again here.
            try:
                res = result_processor.transform(chunk, rewritten)  # type: ignore
            except json.JSONDecodeError as e:
                if embedded_identity is not None:
                    await _fire_embedded_invocation_complete(
                        identity=embedded_identity, outcome="schema_error", error=e
                    )
                raise Exception(f"Intrinsic did not return a JSON: {chunk}") from e
            except Exception as e:
                if embedded_identity is not None:
                    await _fire_embedded_invocation_complete(
                        identity=embedded_identity, outcome="error", error=e
                    )
                raise
            else:
                if embedded_identity is not None:
                    await _fire_embedded_invocation_complete(
                        identity=embedded_identity, outcome="success", error=None
                    )

            # If logits were requested, stash the intercepted raw output so that
            # post_processing/_surface_logits can populate generation.logits/raw_logits.
            if want_scores:
                if raw_hf_output_cell[0] is not None:
                    mot.raw.response = raw_hf_output_cell[0]
                    raw_hf_output_cell[0] = None
                else:
                    warn_key = "intrinsic_no_hf_output"
                    if warn_key not in self._warned_about:
                        self._warned_about.add(warn_key)
                        MelleaLogger.get_logger().warning(
                            "ModelOption.LOGITS/RAW_LOGITS requested on intrinsic path but "
                            "generate_with_transformers did not return a GenerateDecoderOnlyOutput; "
                            "generation.logits and generation.raw_logits will be None."
                        )

            # processing expects a str or a GenerateDecoderOnlyOutput. Extract the str.
            return await self.processing(
                mot, res.choices[0].message.content, input_ids=input_ids
            )

        output._gen.process = functools.partial(
            granite_formatters_processing,
            rewritten=rewritten,
            result_processor=result_processor,
            input_ids=generate_input["input_tokens"],
            embedded_identity=(
                adapter.identity
                if isinstance(adapter, EmbeddedIntrinsicAdapter)
                else None
            ),
        )

        # TODO: Post-processing should release the lock for this generation.
        output._gen.post_process = functools.partial(
            self.post_processing,
            conversation=conversation,
            input_ids=generate_input["input_tokens"],
            _format=None,
            tool_calls=tool_calls,
            tools=tools,
            seed=seed,
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
                    chunk_timeout=model_options.get(
                        ModelOption.STREAM_TIMEOUT, DEFAULT_CHUNK_TIMEOUT
                    ),
                )  # type: ignore
            )
            output._gen.generate_type = GenerateType.ASYNC
        except RuntimeError as e:
            # Most likely cause is running this function without an event loop present.
            raise e

        return output

    # TODO make this async.
    def _make_merged_kv_cache(
        self,
        linearized_ctx: list[Span],
        ctx_as_conversation: Any,
        model_options: Any,
        tools: Any,
    ):
        # Explanation for code blocks inside of use_kv_cache checks:
        # 1. cache every CBlock that is marked with `cache=True` and store in _cached_blocks.
        # 2. Mark each "hit" by adding the string (tokenized?) value to `cached_block_keys`.
        # 3. apply the chat template (without?) tokenization
        # 4. split on cache hits
        # 5. prefill + smash together everything.
        # 6. generate

        # 1. cache every CBlock that is marked with `cache=True` and store in _cached_blocks.
        # AND
        # 2. Mark each "hit" by adding the string (tokenized?) value to `cached_block_keys`.
        cached_block_keys = []
        for c in linearized_ctx:
            match c:
                case CBlock() if c.cache:
                    assert c.value is not None
                    if c.value in self._cached_blocks:
                        MelleaLogger.get_logger().info(
                            f"KV CACHE HIT for: {hash(c.value)} ({c.value[:3]}..{c.value[-3:]})"  # type: ignore
                        )
                    else:
                        MelleaLogger.get_logger().debug(
                            f"HF backend is caching a CBlock with hashed contents: {hash(c.value)} ({c.value[:3]}..{c.value[-3:]})"
                        )
                        tokens = self._tokenizer(c.value, return_tensors="pt")
                        self._cached_blocks[c.value] = self._make_dc_cache(tokens)
                        cached_block_keys.append(c.value)
                case ModelOutputThunk():
                    # MOTs are treated the same as non-cached CBlocks: no KV-cache entry.
                    continue
                case _:
                    continue

        # 3. apply the chat template WITHOUT tokenization.
        # Doing this without tokenization and then gluing together the tokens is necessary because
        # things that KV cache together must tokenize together.
        # Note: add_generation_prompt is in _HF_INTERNAL_TEMPLATE_VARS, so any
        # user-supplied value is intentionally dropped by _filter_for_chat_template.
        # The KV-cache path formats context (not a generation turn), so
        # add_generation_prompt=False (the HF default) is correct here.
        input_text = self._tokenizer.apply_chat_template(  # type: ignore
            ctx_as_conversation,
            tools=convert_tools_to_json(tools),  # type: ignore
            **self._filter_for_chat_template(model_options),
            tokenize=False,
        )

        # 4. split the input_text back up again, re-using DC where it exists.
        str_parts = []
        tok_parts = []
        dc_parts = []
        current_suffix = input_text
        for key in cached_block_keys:
            assert key is not None, (
                "Some input CBlock must not have bee ncomputed yet? The error comes far before this line."
            )
            assert key in current_suffix, (
                "Could happen but would be rare. related to the other assert in this block."
            )
            parts = current_suffix.split(key)  # type: ignore
            assert len(parts) == 2, (
                "Known issue: cached substring might occur more than once. We need to handle this situation earlier. Notice if this happens and keep a count."
            )
            prefix, suffix = parts
            # Add the prefix, if any, to str+tok+dc parts.
            if prefix != "":
                MelleaLogger.get_logger().debug(
                    f"Doing a forward pass on uncached block which is prefix to a cached CBlock: {prefix[:3]}.{len(prefix)}.{prefix[-3:]}"
                )
                str_parts.append(prefix)
                tok_parts.append(self._tokenizer(prefix, return_tensors="pt"))
                dc_parts.append(self._make_dc_cache(tok_parts[-1]))
            # Add the cached CBlock to str+tok+dc parts.
            MelleaLogger.get_logger().debug(
                f"Replacing a substring with previously computed/retrieved cache with hahs value {hash(key)} ({key[:3]}..{key[-3:]})"
            )
            # str_parts.append(key)
            # tok_parts.append(self._tokenizer(key, return_tensors="pt"))
            # dc_parts.append(self._make_dc_cache(tok_parts[-1])) # TODO this is wrong.
            str_parts.append(key)
            tok_parts.append(self._tokenizer(key, return_tensors="pt"))
            dc_parts.append(self._cached_blocks[key])
            # set the suffix for the next loop iteration.
            current_suffix = suffix
        # "base" case: the final suffix.
        if current_suffix != "":
            MelleaLogger.get_logger().debug(  # type: ignore
                f"Doing a forward pass on final suffix, an uncached block: {current_suffix[:3]}.{len(current_suffix)}.{current_suffix[-3:]}"  # type: ignore
            )  # type: ignore
            str_parts.append(current_suffix)
            tok_parts.append(self._tokenizer(current_suffix, return_tensors="pt"))
            dc_parts.append(self._make_dc_cache(tok_parts[-1]))

        # Smash together the caches, the input_ids, and the attention masks.
        assert "".join(str_parts) == input_text, (
            "Should've ended up with the same input text!"
        )
        input_ids = torch.cat([toks["input_ids"] for toks in tok_parts], dim=1)
        attention_mask = torch.cat(
            [toks["attention_mask"] for toks in tok_parts], dim=1
        )
        assert input_ids.shape == attention_mask.shape
        merged_cache: DynamicCache = kv_block_helpers.merge_dynamic_caches_v5(dc_parts)
        # TODO: also assert that the merged cached is the correct shape given the input_ids and attention_mask shapes.
        # rewind merged cache by 1 for safety.
        merged_cache.crop(-1)  # type: ignore
        # Return the merged cache.
        return input_text, input_ids, merged_cache, attention_mask

    async def _generate_from_context_with_kv_cache(
        self,
        action: Component[C] | CBlock,
        ctx: Context,
        *,
        _format: type[BaseModelSubclass] | None = None,
        model_options: dict[str, Any],
        tool_calls: bool = False,
    ) -> ModelOutputThunk[C]:
        """Generate a completion using the KV cache path.

        Raises:
            ValueError: If the context or action contains images or audio;
                `LocalHFBackend` does not support multimodal inputs.
        """
        # Construct input.
        # If the Context is a ChatHistory then we will pretty-print each content as a message and then use apply_chat_template.
        # Otherwise, we will linearize the context and treat it as a raw input.
        if ctx.is_chat_context:
            _check_no_multimodal_blocks(action, ctx)
            system_prompt = model_options.get(ModelOption.SYSTEM_PROMPT, None)
            ctx_as_chat = to_chat(action, ctx, self.formatter, system_prompt)

            # Append tool call information if applicable.
            tools: dict[str, AbstractMelleaTool] = dict()
            if tool_calls:
                if _format:
                    MelleaLogger.get_logger().warning(
                        f"Tool calling typically uses constrained generation, but you have specified a `format` in your generate call. NB: tool calling is superseded by format; we will NOT call tools for your request: {action}"
                    )
                else:
                    add_tools_from_model_options(tools, model_options)
                    add_tools_from_context_actions(
                        tools, ctx.actions_for_available_tools()
                    )

                    # Add the tools from the action for this generation last so that
                    # they overwrite conflicting names.
                    add_tools_from_context_actions(tools, [action])
                MelleaLogger.get_logger().info(f"Tools for call: {tools.keys()}")

            seed = model_options.get(ModelOption.SEED, None)
            if seed is not None:
                set_seed(seed)

            format_kwargs = {}
            if _format:
                schema: dict[str, Any] = _format.model_json_schema()
                grammar: str = llguidance.LLMatcher.grammar_from_json_schema(
                    schema, overrides=_LLGUIDANCE_GRAMMAR_DEFAULTS
                )
                logits_processor = _GuidanceLogitsProcessor(
                    grammar, self._llguidance_tokenizer
                )
                format_kwargs["logits_processor"] = LogitsProcessorList(
                    [logits_processor]
                )

            streaming_kwargs = {}
            streamer = None
            stream = model_options.get(ModelOption.STREAM, False)
            if stream:
                try:
                    # Hugging Face uses a streaming interface that you pass to the generate call.
                    # Must be called from a running event loop. This should always be the case given the same
                    # requirement of the ._gen.generate function below.
                    streamer = AsyncTextIteratorStreamer(
                        self._tokenizer,  # type: ignore
                        skip_prompt=True,
                        skip_special_tokens=True,
                    )
                    streaming_kwargs["streamer"] = streamer
                except RuntimeError as e:
                    # Most likely cause is creating this object without an event loop present.
                    raise e

            # Create a separate thread to handle the processing. Make it awaitable
            # for non-streaming cases and to get the final output.
            # Details: https://huggingface.co/docs/transformers/en/internal/generation_utils#transformers.AsyncTextIteratorStreamer

            # Rename Mellea sentinels and filter to kwargs that generate() accepts.
            # Chat-template-only variables (thinking, enable_thinking, custom_tools, …)
            # are dropped here so they cannot crash generate() with TypeError.
            generate_kwargs = self._make_backend_specific_and_remove(model_options)

            # Only install cooperative-cancel plumbing on the streaming path.
            # Non-streaming calls have no orchestrator calling cancel_generation(),
            # so the hook would be dead code and the StoppingCriteria would silently
            # wrap any user-supplied stopping_criteria on every decode step.
            if stream:
                _cancel_event = _install_cancel_stopping_criteria(
                    generate_kwargs, streaming_kwargs
                )

            linearized_ctx = ctx.view_for_generation()
            assert linearized_ctx is not None
            _input_text, input_ids, merged_cache, attention_mask = (
                self._make_merged_kv_cache(
                    linearized_ctx=linearized_ctx,
                    ctx_as_conversation=ctx_as_chat,
                    model_options=model_options,
                    tools=tools,
                )
            )

            if "stop_strings" in generate_kwargs and "tokenizer" not in generate_kwargs:
                # transformers' generate() requires a tokenizer to decode stop_strings.
                generate_kwargs["tokenizer"] = self._tokenizer

            kv_scores_kwargs: dict[str, Any] = {}
            if model_options.get(ModelOption.LOGITS):
                kv_scores_kwargs["output_scores"] = True
            if model_options.get(ModelOption.RAW_LOGITS):
                kv_scores_kwargs["output_logits"] = True

            chat_response = asyncio.to_thread(
                self._generate_with_adapter_lock,
                self._model.generate,  # type: ignore
                # Passed as args/kwargs to generate.
                input_ids.to(self._device),
                use_cache=True,
                past_key_values=merged_cache,
                attention_mask=attention_mask.to(self._device),
                return_dict_in_generate=True,
                **kv_scores_kwargs,
                **generate_kwargs,
                **streaming_kwargs,  # type: ignore
                **format_kwargs,  # type: ignore
            )

            output = ModelOutputThunk(None)
            # Arm the cancel hook before creating tasks so a cancel racing
            # task creation still finds the hook set.
            if stream:
                output._gen.cancel_hook = _cancel_event.set
            output._gen.start = datetime.datetime.now()
            output._call.context = ctx.view_for_generation()
            output._call.action = action
            output._call.model_options = model_options

            # Processing functions only pass the ModelOutputThunk (and current chunk of response). Bind the other vars necessary for
            # each processing step.
            output._gen.process = functools.partial(
                self.processing, input_ids=input_ids
            )
            output._gen.post_process = functools.partial(
                self.post_processing,
                conversation=ctx_as_chat,
                input_ids=input_ids,
                _format=_format,
                tool_calls=tool_calls,
                tools=tools,
                seed=seed,
            )

            # Set model/provider early so they are available in the error path
            output.generation.model = self._model_id
            output.generation.provider = self._provider

            try:
                # To support lazy computation, will need to remove this create_task and store just the unexecuted coroutine.
                # We can also support synchronous calls by adding a flag and changing this ._gen.generate function.

                response: AsyncTextIteratorStreamer | Coroutine = chat_response
                if stream and streamer is not None:
                    # For streaming, we want to pass the AsyncIterator to the function. Unlike other backends,
                    # this isn't returned by the chat_response coroutine. So we handle it here.
                    response = streamer

                    # Since the async iterator isn't returned by the chat_response coroutine, we have to create a separate
                    # task for it here so that it runs in the background. Attach it to the ModelOutputThunk.
                    output._gen.generate_extra = asyncio.create_task(chat_response)

                # This function should always be called from a running event loop so we don't have to worry about
                # scheduling the task to a specific event loop here.
                output._gen.generate = asyncio.create_task(
                    send_to_queue(
                        response,
                        output._gen.queue,
                        chunk_timeout=model_options.get(
                            ModelOption.STREAM_TIMEOUT, DEFAULT_CHUNK_TIMEOUT
                        ),
                        on_timeout=output._gen.cancel_hook,
                    )  # type: ignore
                )
                output._gen.generate_type = GenerateType.ASYNC
            except RuntimeError as e:
                # Most likely cause is running this function without an event loop present.
                raise e

            return output

        else:
            raise Exception("Does not yet support non-chat contexts.")

    async def _generate_from_context_standard(
        self,
        action: Span,
        ctx: Context,
        *,
        _format: type[BaseModelSubclass] | None = None,
        model_options: dict[str, Any],
        tool_calls: bool = False,
    ) -> ModelOutputThunk:
        """Generate a completion using the standard (non-KV-cache) path.

        Raises:
            ValueError: If the context or action contains images or audio;
                `LocalHFBackend` does not support multimodal inputs.
        """
        # Construct input.
        # If the Context is a ChatHistory then we will pretty-print each content as a message and then use apply_chat_template.
        # Otherwise, we will linearize the context and treat it as a raw input.
        if ctx.is_chat_context:
            _check_no_multimodal_blocks(action, ctx)
            system_prompt = model_options.get(ModelOption.SYSTEM_PROMPT, None)
            ctx_as_chat = to_chat(action, ctx, self.formatter, system_prompt)

            # Append tool call information if applicable.
            tools: dict[str, AbstractMelleaTool] = dict()
            if tool_calls:
                if _format:
                    MelleaLogger.get_logger().warning(
                        f"Tool calling typically uses constrained generation, but you have specified a `format` in your generate call. NB: tool calling is superseded by format; we will NOT call tools for your request: {action}"
                    )
                else:
                    add_tools_from_model_options(tools, model_options)
                    add_tools_from_context_actions(
                        tools, ctx.actions_for_available_tools()
                    )

                    # Add the tools from the action for this generation last so that
                    # they overwrite conflicting names.
                    add_tools_from_context_actions(tools, [action])
                MelleaLogger.get_logger().info(f"Tools for call: {tools.keys()}")

            seed = model_options.get(ModelOption.SEED, None)
            if seed is not None:
                set_seed(seed)

            input_ids = self._tokenizer.apply_chat_template(  # type: ignore
                ctx_as_chat,
                tools=convert_tools_to_json(tools),  # type: ignore
                add_generation_prompt=True,  # If we change this, must modify Hugging Face granite guardian.
                return_tensors="pt",
                return_dict=True,
                **self._filter_for_chat_template(model_options),
            ).to(self._device)  # type: ignore

            format_kwargs = {}
            if _format:
                schema: dict[str, Any] = _format.model_json_schema()
                grammar: str = llguidance.LLMatcher.grammar_from_json_schema(
                    schema, overrides=_LLGUIDANCE_GRAMMAR_DEFAULTS
                )
                logits_processor = _GuidanceLogitsProcessor(
                    grammar, self._llguidance_tokenizer
                )
                format_kwargs["logits_processor"] = LogitsProcessorList(
                    [logits_processor]
                )

            streaming_kwargs = {}
            streamer = None
            stream = model_options.get(ModelOption.STREAM, False)
            if stream:
                try:
                    # Hugging Face uses a streaming interface that you pass to the generate call.
                    # Must be called from a running event loop. This should always be the case given the same
                    # requirement of the ._gen.generate function below.
                    streamer = AsyncTextIteratorStreamer(
                        self._tokenizer,  # type: ignore
                        skip_prompt=True,
                        skip_special_tokens=True,
                    )
                    streaming_kwargs["streamer"] = streamer
                except RuntimeError as e:
                    # Most likely cause is creating this object without an event loop present.
                    raise e

            # Create a separate thread to handle the processing. Make it awaitable
            # for non-streaming cases and to get the final output.
            # Details: https://huggingface.co/docs/transformers/en/internal/generation_utils#transformers.AsyncTextIteratorStreamer

            # Rename Mellea sentinels and filter to kwargs that generate() accepts.
            # Chat-template-only variables (thinking, enable_thinking, custom_tools, …)
            # are dropped here so they cannot crash generate() with TypeError.
            generate_kwargs = self._make_backend_specific_and_remove(model_options)

            # Only install cooperative-cancel plumbing on the streaming path.
            # Non-streaming calls have no orchestrator calling cancel_generation(),
            # so the hook would be dead code and the StoppingCriteria would silently
            # wrap any user-supplied stopping_criteria on every decode step.
            if stream:
                _cancel_event = _install_cancel_stopping_criteria(
                    generate_kwargs, streaming_kwargs
                )

            if "stop_strings" in generate_kwargs and "tokenizer" not in generate_kwargs:
                # transformers' generate() requires a tokenizer to decode stop_strings.
                generate_kwargs["tokenizer"] = self._tokenizer

            scores_kwargs: dict[str, Any] = {}
            if model_options.get(ModelOption.LOGITS):
                scores_kwargs["output_scores"] = True
            if model_options.get(ModelOption.RAW_LOGITS):
                scores_kwargs["output_logits"] = True

            chat_response = asyncio.to_thread(
                self._generate_with_adapter_lock,
                self._model.generate,  # type: ignore
                # Passed as args/kwargs to generate.
                inputs=input_ids["input_ids"],
                attention_mask=input_ids["attention_mask"],
                return_dict_in_generate=True,
                use_cache=self._use_caches,  # Only create KV cache if caching is enabled
                **scores_kwargs,
                **generate_kwargs,
                **streaming_kwargs,  # type: ignore
                **format_kwargs,  # type: ignore
            )

            output = ModelOutputThunk(None)
            # Arm the cancel hook before creating tasks so a cancel racing
            # task creation still finds the hook set.
            if stream:
                output._gen.cancel_hook = _cancel_event.set
            output._gen.start = datetime.datetime.now()
            output._call.context = ctx.view_for_generation()
            output._call.action = action
            output._call.model_options = model_options

            # Processing functions only pass the ModelOutputThunk (and current chunk of response). Bind the other vars necessary for
            # each processing step.
            output._gen.process = functools.partial(
                self.processing, input_ids=input_ids
            )
            output._gen.post_process = functools.partial(
                self.post_processing,
                conversation=ctx_as_chat,
                input_ids=input_ids,
                _format=_format,
                tool_calls=tool_calls,
                tools=tools,
                seed=seed,
            )

            # Set model/provider early so they are available in the error path
            output.generation.model = self._model_id
            output.generation.provider = self._provider

            try:
                # To support lazy computation, will need to remove this create_task and store just the unexecuted coroutine.
                # We can also support synchronous calls by adding a flag and changing this ._gen.generate function.

                response: AsyncTextIteratorStreamer | Coroutine = chat_response
                if stream and streamer is not None:
                    # For streaming, we want to pass the AsyncIterator to the function. Unlike other backends,
                    # this isn't returned by the chat_response coroutine. So we handle it here.
                    response = streamer

                    # Since the async iterator isn't returned by the chat_response coroutine, we have to create a separate
                    # task for it here so that it runs in the background. Attach it to the ModelOutputThunk.
                    output._gen.generate_extra = asyncio.create_task(chat_response)

                # This function should always be called from a running event loop so we don't have to worry about
                # scheduling the task to a specific event loop here.
                output._gen.generate = asyncio.create_task(
                    send_to_queue(
                        response,
                        output._gen.queue,
                        chunk_timeout=model_options.get(
                            ModelOption.STREAM_TIMEOUT, DEFAULT_CHUNK_TIMEOUT
                        ),
                        on_timeout=output._gen.cancel_hook,
                    )  # type: ignore
                )
                output._gen.generate_type = GenerateType.ASYNC
            except RuntimeError as e:
                # Most likely cause is running this function without an event loop present.
                raise e

            return output

        else:
            raise Exception("Does not yet support non-chat contexts.")

    async def processing(
        self, mot: ModelOutputThunk, chunk: str | GenerateDecoderOnlyOutput, input_ids
    ):
        """Accumulate decoded text from a streaming chunk or full generation output.

        For streaming responses the chunk is an already-decoded string from
        `AsyncTextIteratorStreamer`; for non-streaming it is a
        `GenerateDecoderOnlyOutput` that is decoded here.

        Args:
            mot (ModelOutputThunk): The output thunk being populated.
            chunk (str | GenerateDecoderOnlyOutput): A decoded text chunk (streaming)
                or a full Hugging Face generation output object (non-streaming).
            input_ids: The prompt token IDs used for decoding; required to slice off
                the prompt portion from the generated sequences.
        """
        input_ids_tensor = (
            input_ids if isinstance(input_ids, torch.Tensor) else input_ids["input_ids"]
        )

        if mot._underlying_value is None:
            mot._underlying_value = ""

        # Because we use the AsyncTextIteratorStreamer, streaming responses are of type str;
        # and already decoded.
        if isinstance(chunk, str):
            mot._underlying_value += chunk
        elif isinstance(chunk, GenerateDecoderOnlyOutput):
            # Otherwise, it's a non-streaming request. Decode it here.
            mot.raw.response = chunk
            mot._underlying_value += cast(
                str,
                self._tokenizer.decode(
                    chunk.sequences[0, input_ids_tensor.shape[1] :],
                    skip_special_tokens=True,
                ),
            )

    def _surface_logits(
        self, mot: ModelOutputThunk, hf_output: GenerateDecoderOnlyOutput
    ) -> None:
        """Populate mot.generation.logits and/or raw_logits from hf_output when requested.

        Checks ModelOption.LOGITS (processed scores) and ModelOption.RAW_LOGITS (raw
        LM-head logits) in mot._call.model_options. Skips population when
        num_return_sequences > 1, logging a warning the first time per backend instance.

        Args:
            mot: The output thunk whose generation metadata will be updated.
            hf_output: The HuggingFace generation output containing scores/logits.
        """
        hf_field = {ModelOption.LOGITS: "scores", ModelOption.RAW_LOGITS: "logits"}
        for opt, tensors, attr in (
            (ModelOption.LOGITS, hf_output.scores, "logits"),
            (ModelOption.RAW_LOGITS, hf_output.logits, "raw_logits"),
        ):
            if not (mot._call.model_options and mot._call.model_options.get(opt)):
                continue
            if tensors is None:
                MelleaLogger.get_logger().debug(
                    "%s requested but hf_output.%s is None; generation.%s will not be populated.",
                    opt,
                    hf_field[opt],
                    attr,
                )
                continue
            warn_key = f"nrs_{opt}"
            if tensors[0].shape[0] > 1:
                if warn_key not in self._warned_about:
                    self._warned_about.add(warn_key)
                    MelleaLogger.get_logger().warning(
                        "%s is set but num_return_sequences > 1; "
                        "logit tensors are ambiguous across sequences and will not be populated.",
                        opt,
                    )
            else:
                setattr(
                    mot.generation,
                    attr,
                    tuple(s.squeeze(0).detach().clone() for s in tensors),
                )

    async def post_processing(
        self,
        mot: ModelOutputThunk,
        conversation: list[dict],
        _format: type[BaseModelSubclass] | None,
        tool_calls: bool,
        tools: dict[str, AbstractMelleaTool],
        seed,
        input_ids,
    ):
        """Finalize the output thunk after Hugging Face generation completes.

        Stores the KV cache for future reuse, parses tool calls if applicable,
        records token usage metrics, emits telemetry, and attaches the generate log.

        Args:
            mot (ModelOutputThunk): The output thunk to finalize.
            conversation (list[dict]): The chat conversation sent to the model,
                used for logging.
            _format (type[BaseModelSubclass] | None): The structured output format
                class used during generation, if any.
            tool_calls (bool): Whether tool calling was enabled for this request.
            tools (dict[str, AbstractMelleaTool]): Available tools, keyed by name.
            seed: The random seed used during generation, or `None`.
            input_ids: The prompt token IDs; used to compute token counts and for
                KV cache bookkeeping.
        """
        if mot.raw.response is None:
            if mot._gen.generate_extra is not None:
                full_output = await mot._gen.generate_extra
                assert isinstance(full_output, GenerateDecoderOnlyOutput)
                mot.raw.response = full_output

        # The ModelOutputThunk must be computed by this point.
        assert mot.value is not None

        # Store KV cache in LRU separately (not on the MOT) to enable proper cleanup on eviction.
        # This prevents GPU memory from being held by ModelOutputThunk references.
        hf_output = mot.raw.response

        # Surface logits before the caching branch clears hf_output.scores/logits.
        if isinstance(hf_output, GenerateDecoderOnlyOutput) and mot._call.model_options:
            self._surface_logits(mot, hf_output)

        if (
            self._use_caches
            and isinstance(hf_output, GenerateDecoderOnlyOutput)
            and (hf_output.past_key_values is not None or hf_output.scores is not None)
        ):
            output_complete = hf_output.sequences[0]
            kv_cache: DynamicCache | None = hf_output.past_key_values  # type: ignore

            cache_info = HFAloraCacheInfo(
                kv_cache=kv_cache,
                merged_token_ids=output_complete,
                merged_attention=torch.ones_like(output_complete).to(self._device),
                q_end=(
                    input_ids
                    if isinstance(input_ids, torch.Tensor)
                    else input_ids["input_ids"]
                ).shape[1],
                scores=hf_output.scores,
            )

            cache_key = id(mot.value)
            self.cache_put(cache_key, cache_info)

            # Clear KV cache and scores from HF output; retained via LRU cache above.
            # `ModelOutput` (`OrderedDict` subclass) does not sync `None` writes back
            # to the mapping, so plain attribute assignment leaves the dict entry — and
            # its tensor — alive.
            for _field in ("past_key_values", "scores"):
                if _field in hf_output:
                    OrderedDict.__delitem__(hf_output, _field)
                setattr(hf_output, _field, None)

        # Clear the raw logits tensor (scores already cleared above if cached).
        if isinstance(hf_output, GenerateDecoderOnlyOutput):
            if "logits" in hf_output:
                OrderedDict.__delitem__(hf_output, "logits")
            hf_output.logits = None

        # Only scan for tools if we are not doing structured output and tool calls were provided to the model.
        if _format is None and tool_calls:
            mot.tool_calls = to_tool_calls(tools, mot.value)

        assert mot._call.action is not None, (
            "ModelOutputThunks should have their action assigned during generation"
        )
        assert mot._call.model_options is not None, (
            "ModelOutputThunks should have their model_opts assigned during generation"
        )

        # Derive token counts from the output sequences (HF models have no usage object).
        hf_output = mot.raw.response
        n_prompt, n_completion = None, None
        if isinstance(hf_output, GenerateDecoderOnlyOutput):
            try:
                if input_ids is not None and hf_output.sequences is not None:
                    n_prompt = (
                        input_ids
                        if isinstance(input_ids, torch.Tensor)
                        else input_ids["input_ids"]
                    ).shape[1]
                    n_completion = hf_output.sequences[0].shape[0] - n_prompt
            except Exception:
                pass

        if n_prompt is not None and n_completion is not None:
            mot.generation.usage = {
                "prompt_tokens": n_prompt,
                "completion_tokens": n_completion,
                "total_tokens": n_prompt + n_completion,
            }

        # Derive finish reason: "stop" if last token is EOS or the output ends
        # with a configured stop string, "length" if we hit max_new_tokens. HF
        # has no provider response object; this is synthesised from local state
        # (sequences + tokenizer.eos_token_id + model_options).
        if (
            isinstance(hf_output, GenerateDecoderOnlyOutput)
            and hf_output.sequences is not None
        ):
            try:
                # Chat path is single-action; sequences[0] is the only sequence.
                last_token = hf_output.sequences[0][-1].item()
                eos = self._tokenizer.eos_token_id
                eos_set = set(eos) if isinstance(eos, list) else {eos}
                max_new_tokens = mot._call.model_options.get(ModelOption.MAX_NEW_TOKENS)
                stop_strings = (
                    mot._call.model_options.get(ModelOption.STOP_SEQUENCES) or []
                )
                ends_with_stop_string = isinstance(mot.value, str) and any(
                    mot.value.endswith(s) for s in stop_strings
                )
                if last_token in eos_set or ends_with_stop_string:
                    mot.generation.finish_reasons = ["stop"]
                elif (
                    max_new_tokens is not None
                    and n_completion is not None
                    and n_completion >= max_new_tokens
                ):
                    mot.generation.finish_reasons = ["length"]
            except (IndexError, AttributeError, TypeError) as e:
                MelleaLogger.get_logger().debug(
                    f"Could not derive finish_reasons from HF output: {e}"
                )

        # Populate model and provider metadata
        mot.generation.model = self._model_id
        mot.generation.provider = self._provider
        mot.raw.provider = self._provider

        # When caching is disabled, clear hf_output from raw to free GPU memory.
        # The sequences tensor is on GPU and accumulates if not cleared.
        if not self._use_caches and isinstance(
            mot.raw.response, GenerateDecoderOnlyOutput
        ):
            hf_out = mot.raw.response
            # GenerateDecoderOnlyOutput is a ModelOutput (OrderedDict subclass).
            # ModelOutput.__setattr__ skips the dict write when value is None, and
            # ModelOutput defines no __delattr__, so both `out.f = None` and
            # `del out.f` leave the mapping entry — and its tensor — alive.
            # Clear both the dict entry and the instance attribute to release them.
            for field in ("sequences", "scores", "logits", "past_key_values"):
                if field in hf_out:
                    OrderedDict.__delitem__(hf_out, field)
                setattr(hf_out, field, None)

            mot.raw.response = None

            if torch.cuda.is_available():
                import gc

                gc.collect()
                torch.cuda.empty_cache()

        # Generate the log for this ModelOutputThunk.
        generate_log = GenerateLog()
        generate_log.prompt = conversation
        generate_log.backend = f"hf::{self.model_id!s}"
        generate_log.model_options = mot._call.model_options
        generate_log.date = datetime.datetime.now()
        generate_log.model_output = mot.value
        generate_log.extra = {
            "format": _format,
            "tools_available": tools,
            "tools_called": mot.tool_calls,
            "seed": seed,
        }
        generate_log.action = mot._call.action
        generate_log.result = mot

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
        """Generate completions for multiple actions without chat templating.

        Passes formatted prompt strings directly to the Hugging Face model's
        `generate` method as a batch. Tool calling is not supported.

        Args:
            actions (Sequence[Component[C] | CBlock]): Actions to generate completions for.
            ctx (Context): The current generation context.
            format (type[BaseModelSubclass] | None): Optional Pydantic model for
                structured output decoding via llguidance.
            model_options (dict | None): Per-call model options. Supports
                `ModelOption.MAX_NEW_TOKENS`, `ModelOption.TEMPERATURE`,
                `ModelOption.SEED`, `ModelOption.STOP_SEQUENCES`, and
                `ModelOption.LOGITS` (populate `mot.generation.logits`
                with per-token scores of shape `(vocab_size,)`).
            tool_calls (bool): Ignored; tool calling is not supported on this endpoint.

        Returns:
            tuple[list[ModelOutputThunk], dict | None]: `(results, usage)` where
                `results` is a list of model output thunks, one per action, and
                `usage` is the aggregate token-usage dict for the batch.
        """
        for action in actions:
            _check_no_multimodal_blocks(action, None)
        await self.do_generate_walks(list(actions))

        if tool_calls:
            MelleaLogger.get_logger().warning(
                "The raw endpoint does not support tool calling at the moment."
            )

        if self._model.device.type == "mps":
            # TODO: Remove this when we are able to update the torch package.
            #       Test this by ensuring all outputs from this call are populated when running on mps.
            #       https://github.com/pytorch/pytorch/pull/157727
            MelleaLogger.get_logger().warning(
                "utilizing device mps with a `generate_from_raw` request; you may see issues when submitting batches of prompts to a Hugging Face backend; ensure all ModelOutputThunks have non-empty values."
            )

        model_opts = self._simplify_and_merge(model_options)
        seed = model_opts.get(ModelOption.SEED, None)
        if seed is not None:
            set_seed(seed)

        prompts = [self.formatter.print(action) for action in actions]

        # batch-encoding call is deprecated in favor of this
        inputs = self._tokenizer(prompts, return_tensors="pt", padding=True).to(
            self._device
        )

        format_kwargs = {}
        if format:
            schema: dict[str, Any] = format.model_json_schema()
            grammar: str = llguidance.LLMatcher.grammar_from_json_schema(
                schema, overrides=_LLGUIDANCE_GRAMMAR_DEFAULTS
            )
            logits_processor = _GuidanceLogitsProcessor(
                grammar, self._llguidance_tokenizer
            )
            format_kwargs["logits_processor"] = LogitsProcessorList([logits_processor])

        generate_kwargs = self._make_backend_specific_and_remove(model_opts)
        if "stop_strings" in generate_kwargs and "tokenizer" not in generate_kwargs:
            # transformers' generate() requires a tokenizer to decode stop_strings.
            generate_kwargs["tokenizer"] = self._tokenizer

        raw_scores_kwargs: dict[str, Any] = {}
        if model_opts.get(ModelOption.LOGITS):
            raw_scores_kwargs["output_scores"] = True
        if model_opts.get(ModelOption.RAW_LOGITS):
            raw_scores_kwargs["output_logits"] = True

        outputs = await asyncio.to_thread(
            self._generate_with_adapter_lock,
            self._model.generate,  # type: ignore
            # Passed as args/kwargs to generate.
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            return_dict_in_generate=True,
            **raw_scores_kwargs,
            **generate_kwargs,
            **format_kwargs,
        )

        sequences_to_decode = [
            sequence[inputs["input_ids"][i].size(0) :]  # type: ignore
            for i, sequence in enumerate(outputs.sequences)
        ]

        try:
            decoded_results = self._tokenizer.batch_decode(
                sequences_to_decode, skip_special_tokens=True
            )

            results = []
            agg_prompt = 0
            agg_completion = 0
            want_logits = bool(model_opts and model_opts.get(ModelOption.LOGITS))
            want_raw_logits = bool(
                model_opts and model_opts.get(ModelOption.RAW_LOGITS)
            )
            for i, decoded_result in enumerate(decoded_results):
                n_prompt_tokens = int(inputs["input_ids"][i].size(0))
                n_completion_tokens = len(sequences_to_decode[i])
                agg_prompt += n_prompt_tokens
                agg_completion += n_completion_tokens
                per_mot_usage: dict[str, Any] = {
                    "prompt_tokens": n_prompt_tokens,
                    "completion_tokens": n_completion_tokens,
                    "total_tokens": n_prompt_tokens + n_completion_tokens,
                }
                result = ModelOutputThunk(value=decoded_result)
                result.generation.usage = per_mot_usage
                result.generation.model = self._model_id
                result.generation.provider = self._provider
                result.raw.provider = self._provider

                # Extract per-MOT tensors once. These are shared by generation
                # and, where supported, raw.response.
                mot_scores: tuple[torch.Tensor, ...] | None = None
                if outputs.scores is not None:
                    mot_scores = tuple(
                        score[i].detach().clone() for score in outputs.scores
                    )

                mot_logits_raw: tuple[torch.Tensor, ...] | None = None
                if outputs.logits is not None:
                    mot_logits_raw = tuple(
                        logits[i].detach().clone() for logits in outputs.logits
                    )

                # Construct a per-MOT GenerateDecoderOnlyOutput slice holding clones of row i.
                # past_key_values, attentions, and hidden_states are shared across the batch
                # and cannot be sliced per MOT, so they are omitted.
                # Note: beam-search outputs (GenerateBeamDecoderOnlyOutput) are not supported here;
                # mot.raw.response will be None.
                if (
                    self._use_caches
                    and isinstance(outputs, GenerateDecoderOnlyOutput)
                    and isinstance(outputs.sequences, torch.Tensor)
                ):
                    response_scores = (
                        tuple(score.unsqueeze(0) for score in mot_scores)
                        if mot_scores is not None
                        else None
                    )
                    response_logits = (
                        tuple(logits.unsqueeze(0) for logits in mot_logits_raw)
                        if mot_logits_raw is not None
                        else None
                    )

                    if "raw_batch_response_fields_omitted" not in self._warned_about:
                        self._warned_about.add("raw_batch_response_fields_omitted")
                        MelleaLogger.get_logger().debug(
                            "mot.raw.response.past_key_values, .attentions, and "
                            ".hidden_states are not available on the raw batch path "
                            "and will always be None."
                        )

                    result.raw.response = GenerateDecoderOnlyOutput(
                        sequences=cast(
                            "torch.LongTensor",
                            outputs.sequences[i : i + 1, :].detach().clone(),
                        ),
                        scores=cast("tuple[torch.FloatTensor] | None", response_scores),
                        logits=cast("tuple[torch.FloatTensor] | None", response_logits),
                        attentions=None,
                        hidden_states=None,
                        past_key_values=None,
                    )
                elif self._use_caches:
                    if (
                        isinstance(outputs, GenerateBeamDecoderOnlyOutput)
                        and "raw_batch_beam_search_unsupported"
                        not in self._warned_about
                    ):
                        self._warned_about.add("raw_batch_beam_search_unsupported")
                        MelleaLogger.get_logger().debug(
                            "mot.raw.response is not available for beam-search outputs on the "
                            "raw batch path and will be None."
                        )
                    elif (
                        not isinstance(outputs.sequences, torch.Tensor)
                        and "raw_batch_non_tensor_sequences_unsupported"
                        not in self._warned_about
                    ):
                        self._warned_about.add(
                            "raw_batch_non_tensor_sequences_unsupported"
                        )
                        MelleaLogger.get_logger().debug(
                            "mot.raw.response is not available because raw batch output "
                            "sequences are not a torch.Tensor and will be None."
                        )
                    else:
                        # defensive fallback
                        if "raw_batch_response_unsupported" not in self._warned_about:
                            self._warned_about.add("raw_batch_response_unsupported")
                            MelleaLogger.get_logger().debug(
                                "mot.raw.response is not available for this raw batch output "
                                "and will be None."
                            )

                # Reuse the cloned tensors.
                if want_logits and mot_scores is not None:
                    result.generation.logits = mot_scores

                if want_raw_logits and mot_logits_raw is not None:
                    result.generation.raw_logits = mot_logits_raw

                action = actions[i]
                result.parsed_repr = (
                    action.parse(result)
                    if isinstance(action, Component)
                    else result.value
                )

                generate_log = GenerateLog()
                generate_log.prompt = self.formatter.print(actions[i])
                generate_log.backend = f"hf::{self.model_id!s}"
                generate_log.model_options = model_opts
                generate_log.date = datetime.datetime.now()
                generate_log.model_output = decoded_result
                generate_log.extra = {"format": format, "seed": seed}
                generate_log.action = action

                result._generate_log = generate_log
                results.append(result)
        finally:
            # Drop all references that might pin the shared batch tensors.
            # `sequences_to_decode` holds slice views of `outputs.sequences`, and
            # `outputs` is a `GenerateDecoderOnlyOutput` — a `ModelOutput`, which
            # subclasses `OrderedDict`. `del obj.attr` only removes the `__dict__`
            # slot; the `OrderedDict` entry keeps a strong reference to the tensor.
            # Setting `outputs = None` drops the whole container at once
            del sequences_to_decode
            outputs = None
        if torch.cuda.is_available():
            import gc

            MelleaLogger.get_logger().debug(
                "GPU memory before raw batch cleanup: %d bytes reserved",
                torch.cuda.memory_reserved(),
            )
            gc.collect()
            torch.cuda.empty_cache()
            MelleaLogger.get_logger().debug(
                "GPU memory after raw batch cleanup: %d bytes reserved",
                torch.cuda.memory_reserved(),
            )

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

    # region cache management
    def cache_get(self, id: str | int) -> HFAloraCacheInfo | None:
        """Retrieve a cached `HFAloraCacheInfo` entry by its key.

        Args:
            id (str | int): The cache key to look up.

        Returns:
            HFAloraCacheInfo | None: The cached entry, or `None` if not found.
        """
        v = self._cache.get(id)
        assert v is None or type(v) is HFAloraCacheInfo
        return v

    def cache_put(self, id: str | int, v: HFAloraCacheInfo):
        """Store an `HFAloraCacheInfo` entry in the cache under the given key.

        Args:
            id (str | int): The cache key to store the entry under.
            v (HFAloraCacheInfo): The cache entry containing KV cache state
                and associated generation metadata.
        """
        self._cache.put(id, v)

    # endregion

    def _simplify_and_merge(
        self, model_options: dict[str, Any] | None
    ) -> dict[str, Any]:
        """Simplifies model_options to use the Mellea specific ModelOption.Option and merges the backend's model_options with those passed into this call.

        Rules:
        - Within a model_options dict, existing keys take precedence. This means remapping to mellea specific keys will maintain the value of the mellea specific key if one already exists.
        - When merging, the keys/values from the dictionary passed into this function take precedence.

        Because this function simplifies and then merges, non-Mellea keys from the passed in model_options will replace
        Mellea specific keys from the backend's model_options.

        Common model options: https://huggingface.co/docs/transformers/en/llm_tutorial#common-options

        Args:
            model_options: the model_options for this call

        Returns:
            a new dict
        """
        return resolve_model_options(
            backend_defaults=self.model_options,
            remap=self.to_mellea_model_opts_map,
            call_options=model_options,
        )

    def _make_backend_specific_and_remove(
        self, model_options: dict[str, Any], *, for_generate: bool = True
    ) -> dict[str, Any]:
        """Maps specified Mellea specific keys to their backend specific version and removes any remaining Mellea keys.

        If the caller supplied a `SEED` or a non-zero `TEMPERATURE` but did
        not explicitly set `do_sample`, `do_sample` is forced to `True` so
        the underlying transformers `generate` call respects those parameters
        (they are silently ignored under the default greedy `do_sample=False`).

        An explicit `TEMPERATURE` of `0.0` always means greedy decoding and
        suppresses this override even when a seed is set — pairing
        `do_sample=True` with `temperature=0` would crash transformers
        ("temperature has to be a strictly positive float").

        When `for_generate` is True (the default — and the right choice for any
        kwargs about to be splatted into `model.generate`), the result is filtered
        through the module-level `_GENERATE_KWARGS_ALLOWLIST` so chat-template-only
        variables (`thinking`, `enable_thinking`, `custom_tools` …) cannot leak
        in and crash `generate` with `TypeError`.  Pass `for_generate=False` from
        `_filter_for_chat_template` to skip that filter — that path applies the
        chat-template allowlist instead.

        Args:
            model_options: the model_options for this call
            for_generate: whether the result will be passed to
                `model.generate`; controls whether the generate-kwargs filter
                runs.

        Returns:
            a new dict
        """
        seed = model_options.get(ModelOption.SEED, None)
        temperature = model_options.get(ModelOption.TEMPERATURE, None)
        backend_specific = ModelOption.replace_keys(
            model_options, self.from_mellea_model_opts_map
        )
        backend_specific = ModelOption.remove_special_keys(backend_specific)
        temp_allows_sampling = temperature is None or temperature != 0.0
        if (
            "do_sample" not in backend_specific
            and temp_allows_sampling
            and (seed is not None or temperature is not None)
        ):
            backend_specific["do_sample"] = True
        if for_generate:
            backend_specific = {
                k: v
                for k, v in backend_specific.items()
                if k in _GENERATE_KWARGS_ALLOWLIST
            }
        return backend_specific

    @functools.cached_property
    def _chat_template_allowlist(self) -> frozenset[str]:
        """Variable names the tokenizer's chat template can accept from caller-supplied kwargs.

        Parses the Jinja2 template with :func:`jinja2.meta.find_undeclared_variables` to
        find every variable the template references but does not define itself, then
        subtracts :data:`_HF_INTERNAL_TEMPLATE_VARS` (variables Hugging Face provides
        automatically). What remains is the exact set of keys a caller may legitimately
        forward from `model_options` to `apply_chat_template`.

        This is the self-maintaining alternative to a hand-written denylist: it adapts
        automatically as the loaded model's Jinja template changes without any manual
        enumeration of generate()-only option names.

        Returns:
            frozenset of kwarg names valid for `apply_chat_template` on this tokenizer,
            minus HF-provided variables. Empty if the tokenizer has no `chat_template`
            (`apply_chat_template` would raise before kwargs matter in that case).
        """
        template_str = getattr(self._tokenizer, "chat_template", None)
        if not isinstance(template_str, str) or not template_str:
            # Non-string (None, list of alternates, dict) or empty — cannot parse.
            # apply_chat_template handles those formats internally.
            if template_str is not None and template_str:
                # A non-empty, non-string value (e.g. list of alternates, dict) means
                # we cannot inspect the template's variable names.  Any caller-supplied
                # model_options that the template would have accepted are silently dropped.
                MelleaLogger.get_logger().warning(
                    f"Chat template for {self._model_id} is not a plain string "
                    f"(got {type(template_str).__name__}); cannot inspect variable names. "
                    "model_options kwargs will not be forwarded to apply_chat_template."
                )
            return frozenset()
        # loopcontrols enables {% break %} / {% continue %} used in some HF templates.
        env = jinja2.Environment(extensions=["jinja2.ext.loopcontrols"])
        try:
            ast = env.parse(template_str)
        except jinja2.TemplateSyntaxError as e:
            # Templates using unsupported extensions (e.g. {% generation %} from
            # transformers' AssistantTracker) cannot be parsed with a plain Jinja2
            # environment. Fall back to forwarding nothing rather than crashing.
            MelleaLogger.get_logger().warning(
                f"Could not parse chat template for {self._model_id} ({e}); "
                "forwarding no model_options to apply_chat_template. "
                "Template-referenced kwargs (think, guardian_config, etc.) will be ignored."
            )
            return frozenset()
        all_vars = jinja2.meta.find_undeclared_variables(ast)
        return frozenset(all_vars - _HF_INTERNAL_TEMPLATE_VARS)

    def _filter_for_chat_template(
        self, model_options: dict[str, Any]
    ) -> dict[str, Any]:
        """Return only the model_options that the chat template can actually consume.

        Renames Mellea sentinels via :meth:`_make_backend_specific_and_remove`, then
        keeps only keys that appear in :attr:`_chat_template_allowlist` — the set of
        variables the tokenizer's Jinja template actually references. Generation-only
        options that the template does not reference are silently dropped, so they
        cannot pollute the Jinja template variable namespace.

        Args:
            model_options: raw model options (may contain Mellea sentinel keys)

        Returns:
            dict of kwargs safe to splat into `apply_chat_template`
        """
        backend_opts = self._make_backend_specific_and_remove(
            model_options, for_generate=False
        )
        thinking_template_var = next(
            (
                variable
                for variable in _CHAT_TEMPLATE_THINKING_VARS
                if variable in self._chat_template_allowlist
            ),
            None,
        )
        # Deliberately read THINKING from model_options, not backend_opts:
        # _make_backend_specific_and_remove already stripped the @@@thinking@@@
        # sentinel (HF has no from_mellea_model_opts_map entry for it, so
        # remove_special_keys discards it), but model_options still has it.
        # The resolved sentinel must win outright over whatever alias name is
        # already in backend_opts, to match the other backends' precedence
        # (e.g. Ollama reads ModelOption.THINKING directly at the generate
        # call site rather than deferring to a backend-specific key).
        thinking_value = model_options.get(ModelOption.THINKING)
        if thinking_template_var is not None and type(thinking_value) is bool:
            backend_opts[thinking_template_var] = thinking_value
        return {
            k: v for k, v in backend_opts.items() if k in self._chat_template_allowlist
        }

    # region Adapter loading, unloading, and utility functions.
    @property
    def base_model_name(self):
        """Returns the base_model_id of the model used by the backend. For example, `granite-3.3-8b-instruct` for `ibm-granite/granite-3.3-8b-instruct`."""
        return pathlib.PurePath(self._model_id).name

    def add_adapter(self, adapter: AdapterInput) -> None:
        """Register an adapter function with this backend.

        Downloads the adapter weights (via `adapter.get_local_hf_path`) and records
        the adapter in the backend's registry. The adapter must not already be
        registered with a different backend.

        Accepts the full `AdapterInput` union to honour the mixin contract, but
        LocalFile/PEFT and Embedded/Granite Switch realities are supported.
        Embedded adapter functions are already present in the model, so they
        are registered without downloading or loading PEFT weights.

        Args:
            adapter (AdapterInput): The adapter to register. Must be a
                `LocalHFAdapter`, `LocalFileBinding`, or
                `EmbeddedIntrinsicAdapter`; other adapter realities are
                rejected.

        Raises:
            TypeError: If `adapter` is not a supported local or embedded adapter.
            Exception: If `adapter` has already been added to a different backend.
        """
        if not isinstance(
            adapter, (LocalHFAdapter, LocalFileBinding, EmbeddedIntrinsicAdapter)
        ):
            raise TypeError(
                "LocalHFBackend requires a LocalHFAdapter, LocalFileBinding, or "
                f"EmbeddedIntrinsicAdapter; got {type(adapter).__name__}."
            )
        if adapter.backend is not None:
            if adapter.backend is self:
                MelleaLogger.get_logger().warning(
                    f"attempted to add adapter {adapter.name} with type {adapter.adapter_type} to the same backend {adapter.backend}"
                )
                return
            else:
                raise Exception(
                    f"adapter {adapter.name} with type {adapter.adapter_type} has already been added to backend {adapter.backend}"
                )

        existing = self._added_adapters.get(adapter.qualified_name)
        if existing is not None:
            MelleaLogger.get_logger().warning(
                f"Client code attempted to add {adapter.name} with type {adapter.adapter_type} "
                f"but {adapter.qualified_name!r} is already registered as a "
                f"{type(existing).__name__} on {self.__class__.__name__}. The backend is "
                "refusing to do this, because adapter loading is not idempotent. "
                "LocalFileBinding and IntrinsicAdapter/resolve_adapter() registrations "
                "share this qualified-name key space and cannot both claim the same name."
            )
            return None

        if isinstance(adapter, EmbeddedIntrinsicAdapter):
            if not isinstance(adapter.weights, EmbeddedBinding):
                raise TypeError(
                    "EmbeddedIntrinsicAdapter.weights must be an EmbeddedBinding; "
                    f"got {type(adapter.weights).__name__}."
                )
            adapter.backend = self
            adapter.weights.source = self.base_model_name
            self._added_adapters[adapter.qualified_name] = adapter
            return

        adapter.path = adapter.get_local_hf_path(self.base_model_name)
        adapter.backend = self
        self._added_adapters[adapter.qualified_name] = adapter

    def register_embedded_adapter_model(
        self, source: str, *, revision: str = "main", cache_dir: str | None = None
    ) -> list[str]:
        """Register embedded adapter functions from a Granite Switch checkpoint.

        Args:
            source: Local checkpoint directory or Hugging Face Hub repository ID.
            revision: Git revision when loading from the Hub.
            cache_dir: Cache directory for Hub downloads.

        Returns:
            Names of the registered adapter functions.

        Raises:
            ImportError: If Hugging Face Hub support is not installed.
            PermissionError: If the model repository is private or gated.
            FileNotFoundError: If the source has no adapter index.
            ValueError: If the source has no matching embedded adapter functions.
            TypeError: If an adapter from the source has an unsupported binding.
        """
        adapters = EmbeddedIntrinsicAdapter.from_source(
            source, revision=revision, cache_dir=cache_dir
        )
        for adapter in adapters:
            self.add_adapter(adapter)
        return [adapter.intrinsic_name for adapter in adapters]

    def load_peft_adapter(self, adapter_qualified_name: str) -> None:
        """Load a previously registered adapter into the underlying Hugging Face model.

        The adapter must have been registered via `add_adapter` first. Do not call
        this method while generation requests are in progress.

        Args:
            adapter_qualified_name (str): The `adapter.qualified_name` of the adapter
                to load (i.e. `"<name>_<adapter_type>"`)

        Raises:
            ValueError: If no adapter with the given qualified name has been added to
                this backend.
            TypeError: If the named adapter is embedded in the model and therefore
                activated through the chat template rather than loaded through PEFT.
        """
        adapter = self._added_adapters.get(adapter_qualified_name, None)
        if adapter is None:
            raise ValueError(
                f"could not load adapter {adapter_qualified_name} for backend {self}: adapter was not previously added"
            )
        if isinstance(adapter, EmbeddedIntrinsicAdapter):
            raise TypeError(
                f"cannot load embedded adapter {adapter_qualified_name} through PEFT; "
                "it is activated by the chat template"
            )

        try:
            # self._model is a HF PreTrainedModel — .load_adapter() here is
            # transformers/PEFT's own method, not the renamed AdapterMixin verb.
            # v5: adapter_kwargs is forwarded to download_kwargs only; device is
            # derived automatically from self.device, so we don't pass it here —
            # find_adapter_config_file() no longer accepts a 'device' argument.
            self._model.load_adapter(
                adapter.path, adapter.qualified_name, adapter_kwargs={}
            )
        except ValueError as e:
            # If it's just that it's already loaded, ignore it.
            if f"Adapter with name {adapter_qualified_name} already exists." not in str(
                e
            ):
                raise e

        # Loading an adapter activates it. We disable adapters immediately after.
        # Prefer this over `.disable_adapters()`; the disable function doesn't always
        # seem to work.
        self._model.set_adapter([])
        # self._model.disable_adapters()
        self._loaded_adapters[adapter.qualified_name] = adapter

    def unload_peft_adapter(self, adapter_qualified_name: str) -> None:
        """Unload a previously loaded adapter from the underlying Hugging Face model.

        If the adapter is not currently loaded, a log message is emitted and the
        method returns without error.

        Args:
            adapter_qualified_name (str): The `adapter.qualified_name` of the adapter
                to unload.
        """
        # Check if the backend knows about this adapter.
        adapter = self._loaded_adapters.get(adapter_qualified_name, None)
        if adapter is None:
            MelleaLogger.get_logger().info(
                f"could not unload adapter {adapter_qualified_name} for backend {self}: adapter is not loaded"
            )
            return

        self._model.delete_adapter(adapter.qualified_name)

        # Remove the adapter from the list of loaded adapters.
        del self._loaded_adapters[adapter.qualified_name]

    def remove_adapter(self, adapter_qualified_name: str) -> None:
        """Deregister a previously added adapter, freeing its qualified name for reuse.

        The inverse of `add_adapter()`: reverses all three of its mutations
        (registry entry, `.path`, `.backend`), not just the registry entry —
        otherwise the removed object would be left with `.backend` still
        naming this backend, silently blocking it from being re-added anywhere
        (`add_adapter()`'s `adapter.backend is self` guard treats that as an
        already-added no-op). If the adapter is not currently registered, a
        log message is emitted and the method returns without error.

        For a `LocalFileBinding`, call `binding.release()` instead: it runs
        this verb and also marks the binding released, so the binding's
        lifecycle state stays coherent with the registry. Calling this
        method directly on a registered binding clears only `.backend`/
        `.path` — the binding's `_loaded`/`_released` state is untouched, so
        a later `activate()`/`deactivate()` raises the `prepare()`-required
        error even though the binding was never released, and the real
        cause (an external `remove_adapter()`) is nowhere in the message.
        `prepare()` self-heals that state (re-registers via the staged
        backend and reloads), but `release()` is the intended way to free
        the name.

        Args:
            adapter_qualified_name (str): The `adapter.qualified_name` of the
                adapter to deregister.

        Raises:
            ValueError: The adapter is still loaded (`load_peft_adapter()` was
                called and `unload_peft_adapter()` was not). Freeing the name
                while it is still loaded would let a later `load_peft_adapter()`
                call for a *different* adapter hit PEFT's "already exists"
                error — silently swallowed by `load_peft_adapter()` — and keep
                running on this adapter's stale weights under the new one's
                identity. Call `unload_peft_adapter()` first.
        """
        if adapter_qualified_name in self._loaded_adapters:
            raise ValueError(
                f"cannot remove adapter {adapter_qualified_name} for backend {self}: "
                "it is still loaded; call unload_peft_adapter() first (release() "
                "does this for you)."
            )

        adapter = self._added_adapters.pop(adapter_qualified_name, None)
        if adapter is None:
            MelleaLogger.get_logger().info(
                f"could not remove adapter {adapter_qualified_name} for backend {self}: "
                "adapter was not registered"
            )
            return

        adapter.backend = None
        adapter.path = None

    def activate_peft_adapter(self, adapter_qualified_name: str) -> None:
        """Switch a previously loaded PEFT adapter on for subsequent generation.

        Must be called while holding `_generation_lock`.

        Args:
            adapter_qualified_name (str): The `adapter.qualified_name` of the adapter
                to activate.
        """
        self._model.set_adapter(adapter_qualified_name)

    def deactivate_peft_adapter(self, adapter_qualified_name: str) -> None:
        """Switch off any active PEFT adapter so generation uses the base model.

        Must be called while holding `_generation_lock`.

        Args:
            adapter_qualified_name (str): The `adapter.qualified_name` of the adapter
                to deactivate. Accepted for symmetry with `activate_peft_adapter`; the
                underlying primitive clears all active PEFT adapters regardless of
                name.

        Raises:
            ValueError: If the underlying PEFT model raises `ValueError` for a
                reason other than "no adapter loaded" (which is treated as a
                no-op, since deactivating is already a no-op in that case).
        """
        try:
            # `._model.disable_adapters()` doesn't seem to actually disable them or
            # remove them from the model's list of `.active_adapters()`.
            self._model.set_adapter([])
        except ValueError as e:
            # If no weights have been loaded, the model will raise a ValueError:
            # `ValueError("No adapter loaded. Please load an adapter first.")`
            if "No adapter loaded" not in str(e):
                raise e

    def _adapter_activation_lock(
        self,
    ) -> contextlib.AbstractContextManager[bool | None]:
        """Reuse `_generation_lock` for exclusivity around adapter activation.

        `activate_peft_adapter`/`deactivate_peft_adapter` document "must be called
        while holding `_generation_lock`" as a precondition, and they have three
        callers:

        1. `_generate_with_adapter_lock` (the standard, non-intrinsic
           generation path), which takes `_generation_lock` itself and calls
           only `deactivate_peft_adapter` — never the activation verbs,
           because standard generation always runs against the base model.
        2. `LocalFileBinding.activate()`/`.deactivate()` (driven by
           `AdapterMixin.adapter_scope()`), which holds no lock of its own.
        3. `_IntrinsicPeftBinding.activate()`/`.deactivate()` (driven by
           `_generate_intrinsic_with_adapter_scope` via
           `AdapterMixin.adapter_scope()`), the intrinsic-path counterpart of
           caller 2's `LocalFileBinding` — whose driver also holds
           `_generation_lock` itself for the whole prepare -> activate ->
           generate -> deactivate section (#1465).

        This method exists for callers 2 and 3 — so it is not a duplicate of
        the lock caller 1 (and, on the outer level, caller 3's driver) takes,
        it is the only thing satisfying the precondition on those paths.

        Caller 3's verb acquisitions therefore nest inside its driver's
        `_generation_lock` hold on the same thread (see
        `_generate_intrinsic_with_adapter_scope`'s docstring), which is
        exactly why `_generation_lock` is a `threading.RLock`: a plain
        `threading.Lock` would deadlock on that same-thread re-acquisition
        (this was #1465's known lock-reentrancy issue).
        """
        return self._generation_lock

    def list_adapters(self) -> list[str]:
        """List the qualified names of all adapters registered with this backend.

        Returns:
            list[str]: Qualified adapter names (i.e. `adapter.qualified_name`) for
                all adapters that have been registered via `add_adapter`, whether
                or not they have also been loaded via `load_peft_adapter`.
        """
        return list(self._added_adapters.keys())


def _assert_correct_adapters(expected_state: str, model: PreTrainedModel):
    """When generating with a Hugging Face model, this can be used to ensure the correct adapters are active.

    Args:
        expected_state: the current state of the lock
        model: the model underlying the LocalHFBackend; this is the model the adapters are activated on
    """
    try:
        active = model.active_adapters()

        if expected_state == "":
            assert len(active) == 0, (
                f'no adapters should be active if expected state is "", got "{active[0]}"'
            )
        else:
            assert len(active) == 1, (
                f'one adapter should be active if expected state is "{expected_state}"'
            )
            assert active[0] == expected_state, (
                f'the active adapter "{active[0]}" doesn\'t match the expected state: "{expected_state}"'
            )
    except ValueError as e:
        # If no weights have been loaded, the model will raise a ValueError:
        # `ValueError("No adapter loaded. Please load an adapter first.")`
        if "No adapter loaded" in str(e):
            assert expected_state == "", (
                f'got no adapters loaded but expected state is "{expected_state}"'
            )
        else:
            raise e


class _UnusedIOContract(IOContract):
    """Placeholder `IOContract` for `_generate_intrinsic_with_adapter_scope`'s `_AdapterCore`.

    `_generate_from_intrinsic` builds its own I/O handling directly (the
    rewriter/result-processor pipeline driven by `adapter.config`) and only
    needs `adapter_scope()` for its activate/deactivate lifecycle hooks —
    `adapter_scope()` never reads `Adapter.io_contract`, so this slot is filled
    only to satisfy the dataclass's required field.
    """

    def build_prompt(self, **kwargs: object) -> Component:
        raise NotImplementedError(
            "not used on the intrinsic generation path: "
            "_generate_from_intrinsic builds prompts via IntrinsicsRewriter."
        )

    def parse(self, raw: str) -> dict[str, object]:
        raise NotImplementedError(
            "not used on the intrinsic generation path: "
            "_generate_from_intrinsic parses output via IntrinsicsResultProcessor."
        )


class _IntrinsicPeftBinding(WeightsBinding):
    """Drives already-registered PEFT verbs for one intrinsic-generation call.

    `IntrinsicAdapter` (the legacy shim, see `mellea.backends.adapters.adapter`)
    carries an inert `_ShimWeightsBinding` that raises on every verb, so it
    can't be handed to `adapter_scope()` directly.
    `_generate_intrinsic_with_adapter_scope` builds one of these fresh per call
    instead, and it drives the backend's own `load_peft_adapter`/
    `activate_peft_adapter`/`deactivate_peft_adapter` verbs — the same verbs
    `_generate_with_adapter_lock` used to call directly before generation was
    routed through `adapter_scope`.

    `binding_type` is `"local_file"`, not a distinct value: this binding loads
    LoRA/aLoRA weights downloaded to local disk via PEFT — the same reality
    `LocalFileBinding` reports — so it must share that value rather than
    fragment the weights-reality metric dimension. A distinct class exists
    rather than reusing `LocalFileBinding` directly because `LocalFileBinding.prepare()`
    calls `add_adapter()`, which would collide with the `IntrinsicAdapter`
    already registered under this `qualified_name`.

    `prepare()` loads the weights (mirrors `LocalFileBinding.prepare()`'s
    phase boundary: the load belongs to `"prepare"`, not `"activate"`, so
    `phase="activate"` duration stays comparable across bindings).
    `release()` is a no-op: registration happened in `add_adapter()` when the
    intrinsic adapter was resolved (see `AdapterMixin.resolve_adapter`), and
    this binding doesn't own that lifecycle — it only exists for the duration
    of one generate call. `revision` — the value `adapter_scope()` reports to
    the `ADAPTER_FUNCTION_INVOCATION_COMPLETE` hook — is set explicitly rather
    than left for `adapter_scope()`'s `getattr(..., None)` fallback, because
    `IntrinsicAdapter`s are always pinned to a catalogue SHA; reporting
    `None` would mislabel every intrinsic invocation as unpinned.
    """

    binding_type: ClassVar[str] = "local_file"

    def __init__(
        self, backend: LocalHFBackend, qualified_name: str, revision: str | None
    ) -> None:
        self._backend = backend
        self._qualified_name = qualified_name
        self.revision = revision

    def prepare(self) -> None:
        with self._backend._adapter_activation_lock():
            self._backend.load_peft_adapter(self._qualified_name)

    def activate(self) -> None:
        with self._backend._adapter_activation_lock():
            self._backend.activate_peft_adapter(self._qualified_name)

    def deactivate(self) -> None:
        with self._backend._adapter_activation_lock():
            self._backend.deactivate_peft_adapter(self._qualified_name)

    def release(self) -> None:
        return
