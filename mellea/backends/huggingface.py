"""A backend that uses the Huggingface Transformers library.

The purpose of the Hugginface backend is to provide a setting for implementing experimental features. If you want a performance local backend, and do not need experimental features such as Span-based context or ALoras, consider using Ollama backends instead.
"""

from __future__ import annotations

import asyncio
import dataclasses
import datetime
import functools
import json
import threading
from collections.abc import Callable, Coroutine, Sequence
from typing import Any, overload

import granite_common
import llguidance
import llguidance.hf
import llguidance.torch
import peft
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache
from transformers.generation.logits_process import LogitsProcessorList
from transformers.generation.streamers import AsyncTextIteratorStreamer
from transformers.generation.utils import GenerateDecoderOnlyOutput
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.trainer_utils import set_seed

from ..backends import kv_block_helpers
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
from ..core.base import AbstractMelleaTool
from ..formatters import ChatFormatter, TemplateFormatter
from ..helpers import message_to_openai_message, messages_to_docs, send_to_queue
from ..stdlib.components import Intrinsic, Message
from ..stdlib.requirements import ALoraRequirement, LLMaJRequirement
from ..telemetry.backend_instrumentation import (
    instrument_generate_from_raw,
    start_generate_span,
)
from .adapters import (
    AdapterMixin,
    AdapterType,
    GraniteCommonAdapter,
    LocalHFAdapter,
    get_adapter_for_intrinsic,
)
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

"""A configuration type for the unhappy path: Tokenizer * Model * torch device string

Huggingface backends can initialize themselves from a model string if the transformers `Auto*` classes can be used. Therefore, a TransformersTorchConfig usually isn't required. However, sometimes a model needs special care to instantiate properly, or a custom device type needs to bse used. Instead of trying to do a lot of partial magic, we basically have two modaliites: either the constructor can figure out everything from the model_id, or the user has to provide an entire config.
"""
TransformersTorchConfig = tuple[PreTrainedTokenizer, PreTrainedModel, torch.device]

format: None = None  # typing this variable in order to shadow the global format function and ensure mypy checks for errors


@dataclasses.dataclass
class HFAloraCacheInfo:
    """A dataclass for holding some KV cache and associated information."""

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


# modified from VLLM v0.9.2 code base
# https://github.com/vllm-project/vllm/blob/v0.9.2/vllm/model_executor/guided_decoding/guidance_logits_processors.py
class _GuidanceLogitsProcessor:
    def __init__(self, grammar: str, ll_tokenizer: llguidance.LLTokenizer) -> None:
        self.grammar = grammar
        self.vocab_size: int = ll_tokenizer.vocab_size
        self.ll_tokenizer: llguidance.LLTokenizer = ll_tokenizer
        self.ll_matchers: list[llguidance.LLMatcher] = []
        self.bitmasks: list[torch.Tensor] = []
        self.new_sampling: bool = False
        self.batch_size: int = -1

    def __call__(
        self, batch_input_ids: torch.Tensor, batch_scores: torch.Tensor
    ) -> torch.Tensor:
        i_batch, _ = batch_input_ids.shape
        s_batch, _ = batch_scores.shape
        assert i_batch == s_batch

        # s_batch, s_vocab = batch_scores.shape
        # assert s_vocab == self.vocab_size
        #
        # NOTE: somehow, this does not hold. s_vocab is not same as either of
        # * self._tokenizer._tokenizer.get_vocab_size(with_added_tokens=True) == self.vocab_size == ll_tokenizer.vocab_size
        # * self._tokenizer._tokenizer.get_vocab_size(with_added_tokens=False)

        if self.batch_size != i_batch:
            self.batch_size = i_batch
            self.bitmasks = [
                llguidance.torch.allocate_token_bitmask(1, self.vocab_size)  # type: ignore[attr-defined]
                for _ in range(self.batch_size)
            ]

            self.ll_matchers = [
                llguidance.LLMatcher(self.ll_tokenizer, self.grammar)
                for _ in range(self.batch_size)
            ]

        for input_ids, scores, ll_matcher, bitmask in zip(
            batch_input_ids, batch_scores, self.ll_matchers, self.bitmasks
        ):
            if self.new_sampling and len(input_ids) > 0:
                ll_matcher.consume_token(  # type: ignore[attr-defined]
                    input_ids.tolist()[-1]
                )
                err = ll_matcher.get_error()  # type: ignore[attr-defined]
                if err:
                    FancyLogger.get_logger().warning("Error in LLMatcher: %s", err)

            llguidance.torch.fill_next_token_bitmask(ll_matcher, bitmask, 0)
            llguidance.torch.apply_token_bitmask_inplace(
                scores, bitmask.to(scores.device)
            )  # type: ignore[attr-defined]

        self.new_sampling = True

        return batch_scores


class LocalHFBackend(FormatterBackend, AdapterMixin):
    """The LocalHFBackend uses Huggingface's transformers library for inference, and uses a Formatter to convert `Component`s into prompts. This backend also supports Activated LoRAs (ALoras)](https://arxiv.org/pdf/2504.12397).

    This backend is designed for running an HF model for small-scale inference locally on your machine.

    This backend is NOT designed for inference scaling on CUDA-enabled hardware.
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
        model_options: dict | None = None,
    ):
        """Attempt to load model weights using the model_id by default, or using `custom_config` if provided.

        WARNING: initializing a `LocalHFBackend` will download and load the model on your *local* machine.

        Args:
            model_id (str | ModelIdentifier): Used to load the model *and tokenizer* via transformers Auto* classes, and then moves the model to the best available device (cuda > mps > cpu). If loading the model and/or tokenizer from a string will not work, or if you want to use a different device string, then you can use custom_config.
            formatter (Formatter): A mechanism for turning `stdlib` stuff into strings. Experimental Span-based models should use `mellea.backends.span.*` backends.
            use_caches (bool): If set to False, then caching will not be used even if a Cache is provided.
            cache (Optional[Cache]): The caching strategy to use. If None, `LRUCache(3)` will be used.
            custom_config (Optional[TransformersTorchConfig]): Overrides loading from the `model_id`. If set, then the specified tokenizer/model/device will be used instead of auto-loading from the model_id.
            default_to_constraint_checking_alora: If set to False then aloras will be deactivated. This is primarily for performance benchmarking and debugging.
            model_options (Optional[dict]): Default model options.
            return_scores (bool): If True, return output logits from model.generate(). Default False to save GPU memory.
        """
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
        }

        # A mapping of Mellea specific ModelOptions to the specific names for this backend.
        # These options should almost always be a subset of those specified in the `to_mellea_model_opts_map`.
        # Usually, values that are intentionally extracted while prepping for the backend generate call
        # will be omitted here so that they will be removed when model_options are processed
        # for the call to the model.
        self.from_mellea_model_opts_map = {ModelOption.MAX_NEW_TOKENS: "max_new_tokens"}

        self.default_to_constraint_checking_alora = default_to_constraint_checking_alora

        # Either use the custom config or load the model from its model_id
        match model_id:
            case str():
                self._hf_model_id = model_id
            case ModelIdentifier():
                assert model_id.hf_model_name is not None, (
                    "model_id is None. This can also happen if the ModelIdentifier has no hf_model_id name set."
                )
                self._hf_model_id = model_id.hf_model_name
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
                self._model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
                    self._hf_model_id
                ).to(self._device)  # type: ignore
                self._tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
                    self._hf_model_id
                )
            case _:
                self._tokenizer, self._model, self._device = custom_config

        self._llguidance_tokenizer: llguidance.LLTokenizer = (
            llguidance.hf.from_tokenizer(self._tokenizer)  # type:ignore
        )
        assert (
            self._llguidance_tokenizer.vocab_size
            == self._tokenizer._tokenizer.get_vocab_size(with_added_tokens=True)
        ), "vocab size mismatch between llguidance and huggingface tokenizers ... wtf?"

        self._use_caches = use_caches
        self._cache = (
            cache
            if cache is not None
            else SimpleLRUCache(0, on_evict=_cleanup_kv_cache)
        )

        # Adapters can be made known to the backend (added) and loaded.
        self._added_adapters: dict[str, LocalHFAdapter] = {}
        self._loaded_adapters: dict[str, LocalHFAdapter] = {}

        self._generation_lock = threading.Lock()
        """Used to force generation requests to be non-concurrent. Necessary for preventing issues with adapters."""

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

    async def generate_from_context(
        self,
        action: Component[C] | CBlock,
        ctx: Context,
        *,
        format: type[BaseModelSubclass] | None = None,
        model_options: dict | None = None,
        tool_calls: bool = False,
    ) -> tuple[ModelOutputThunk[C], Context]:
        """Generate using the huggingface model."""
        span = start_generate_span(
            backend=self, action=action, ctx=ctx, format=format, tool_calls=tool_calls
        )
        await self.do_generate_walk(action)

        # Upsert model options.
        model_opts = self._simplify_and_merge(model_options)

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

            # Check if a requirement_check (or AloraRequirement specified) adapter
            # exists.
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
                    alora_action, ctx, model_options=model_opts
                )
                # Store span for telemetry
                if span is not None:
                    mot._meta["_telemetry_span"] = span
                return mot, ctx.add(alora_action).add(mot)

        elif isinstance(action, Intrinsic):
            mot = await self._generate_from_intrinsic(
                action, ctx, model_options=model_opts
            )
            # Store span for telemetry
            if span is not None:
                mot._meta["_telemetry_span"] = span
            return mot, ctx.add(action).add(mot)

        mot = await self._generate_from_context_standard(
            action, ctx, _format=format, model_options=model_opts, tool_calls=tool_calls
        )

        # Store span in metadata for post_processing to record telemetry
        if span is not None:
            mot._meta["_telemetry_span"] = span

        return mot, ctx.add(action).add(mot)

    def _generate_with_adapter_lock(
        self, adapter_name: str, generate_func: Callable, *args, **kwargs
    ):
        """Helper function for ensuring exclusive generation when adapters are present. Necessary to prevent generating with incorrect weights."""
        with self._generation_lock:
            if adapter_name != "":
                self.load_adapter(adapter_name)
                self._model.set_adapter(adapter_name)
            else:
                try:
                    # `._model.disable_adapters()` doesn't seem to actually disable them or
                    # remove them from the model's list of `.active_adapters()`.
                    self._model.set_adapter([])
                except ValueError as e:
                    # If no weights have been loaded, the model will raise a ValueError:
                    # `ValueError("No adapter loaded. Please load an adapter first.")`
                    if "No adapter loaded" in str(e):
                        pass
                    else:
                        raise e

            _assert_correct_adapters(adapter_name, self._model)
            out = generate_func(*args, **kwargs)
            _assert_correct_adapters(adapter_name, self._model)
            return out

    async def _generate_from_intrinsic(
        self, action: Intrinsic, ctx: Context, *, model_options: dict[str, Any]
    ) -> ModelOutputThunk:
        if not ctx.is_chat_context:
            raise Exception("Does not yet support non-chat contexts.")

        if len(model_options.items()) > 0:
            FancyLogger.get_logger().info(
                "passing in model options when generating with an adapter; some model options may be overwritten / ignored"
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

        conversation: list[dict] = []
        system_prompt = model_options.get(ModelOption.SYSTEM_PROMPT, "")
        if system_prompt != "":
            conversation.append({"role": "system", "content": system_prompt})

        conversation.extend([message_to_openai_message(m) for m in ctx_as_message_list])

        docs = messages_to_docs(ctx_as_message_list)

        seed = model_options.get(ModelOption.SEED, None)
        if seed is not None:
            set_seed(seed)

        if model_options.get(ModelOption.STREAM, None) is not None:
            # Intrinsics don't support streaming because of their post-processing step.
            raise Exception("Intrinsics do not support streaming.")

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

        intrinsic_config = adapter.config
        assert intrinsic_config is not None

        rewriter = granite_common.IntrinsicsRewriter(
            config_dict=intrinsic_config, model_name=adapter.name
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

        # Convert other parameters from Mellea proprietary format to standard format.
        for model_option in model_options:
            if model_option == ModelOption.TEMPERATURE:
                request_json["temperature"] = model_options[model_option]

        rewritten = rewriter.transform(request_json, **action.intrinsic_kwargs)

        # TODO: Handle caching here. granite_common doesn't tell us what changed,
        #       so we will have to invalidate the cache on our side. This requires
        #       us having specific caching for each Component/Message.

        generate_input, other_input = (
            granite_common.util.chat_completion_request_to_transformers_inputs(  # type: ignore
                rewritten, self._tokenizer, self._model
            )
        )

        chat_response = asyncio.to_thread(
            self._generate_with_adapter_lock,
            adapter.qualified_name,
            granite_common.util.generate_with_transformers,  # type: ignore
            # Passed as args/kwargs to generate.
            self._tokenizer,
            self._model,
            generate_input,
            other_input,
        )

        output = ModelOutputThunk(None)
        output._context = ctx.view_for_generation()
        output._action = action
        output._model_options = model_options

        # Add another step to the processing function.
        async def granite_common_processing(
            mot: ModelOutputThunk,
            chunk: granite_common.ChatCompletionResponse,
            rewritten: granite_common.ChatCompletion,
            result_processor: granite_common.IntrinsicsResultProcessor,
            input_ids,
        ):
            try:
                res = result_processor.transform(chunk, rewritten)  # type: ignore
            except json.JSONDecodeError:
                raise Exception(f"Intrinsic did not return a JSON: {chunk}")

            # TODO: If we want to support caches, we need to get the GenerateDecoderOnlyOutput. This means we
            #       probably need to break out the pieces from `generate_with_transformers`.
            # processing expects a str or a GenerateDecoderOnlyOutput. Extract the str.
            return await self.processing(
                mot, res.choices[0].message.content, input_ids=input_ids
            )

        output._process = functools.partial(
            granite_common_processing,
            rewritten=rewritten,
            result_processor=result_processor,
            input_ids=generate_input["input_tokens"],
        )

        # TODO: Post-processing should release the lock for this generation.
        output._post_process = functools.partial(
            self.post_processing,
            conversation=conversation,
            input_ids=generate_input["input_tokens"],
            _format=None,
            tool_calls=False,
            tools={},
            seed=seed,
        )

        try:
            # To support lazy computation, will need to remove this create_task and store just the unexecuted coroutine.
            # We can also support synchronous calls by adding a flag and changing this ._generate function.

            # This function should always be called from a running event loop so we don't have to worry about
            # scheduling the task to a specific event loop here.
            output._generate = asyncio.create_task(
                send_to_queue(chat_response, output._async_queue)  # type: ignore
            )
            output._generate_type = GenerateType.ASYNC
        except RuntimeError as e:
            # Most likely cause is running this function without an event loop present.
            raise e

        return output

    # TODO make this async.
    def _make_merged_kv_cache(
        self,
        linearized_ctx: list[Component | CBlock | ModelOutputThunk],
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
                        FancyLogger.get_logger().info(
                            f"KV CACHE HIT for: {hash(c.value)} ({c.value[:3]}..{c.value[-3:]})"  # type: ignore
                        )
                    else:
                        FancyLogger.get_logger().debug(
                            f"HF backend is caching a CBlock with hashed contents: {hash(c.value)} ({c.value[:3]}..{c.value[-3:]})"
                        )
                        tokens = self._tokenizer(c.value, return_tensors="pt")
                        self._cached_blocks[c.value] = self._make_dc_cache(tokens)
                        cached_block_keys.append(c.value)
                case _:
                    continue

        # 3. apply the chat template WITHOUT tokenization.
        # Doing this without tokenization and then gluing together the tokens is necessary because
        # things that KV cache together must tokenize together.
        input_text = self._tokenizer.apply_chat_template(  # type: ignore
            ctx_as_conversation,
            tools=convert_tools_to_json(tools),  # type: ignore
            **self._make_backend_specific_and_remove(model_options),
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
                FancyLogger.get_logger().debug(
                    f"Doing a forward pass on uncached block which is prefix to a cached CBlock: {prefix[:3]}.{len(prefix)}.{prefix[-3:]}"
                )
                str_parts.append(prefix)
                tok_parts.append(self._tokenizer(prefix, return_tensors="pt"))
                dc_parts.append(self._make_dc_cache(tok_parts[-1]))
            # Add the cached CBlock to str+tok+dc parts.
            FancyLogger.get_logger().debug(
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
            FancyLogger.get_logger().debug(  # type: ignore
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
        merged_cache: DynamicCache = kv_block_helpers.merge_dynamic_caches(dc_parts)
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
        # Construct input.
        # If the Context is a ChatHistory then we will pretty-print each content as a message and then use apply_chat_template.
        # Otherwise, we will linearize the context and treat it as a raw input.
        if ctx.is_chat_context:
            system_prompt = model_options.get(ModelOption.SYSTEM_PROMPT, None)
            ctx_as_chat = to_chat(action, ctx, self.formatter, system_prompt)

            # Append tool call information if applicable.
            tools: dict[str, AbstractMelleaTool] = dict()
            if tool_calls:
                if _format:
                    FancyLogger.get_logger().warning(
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
                FancyLogger.get_logger().info(f"Tools for call: {tools.keys()}")

            seed = model_options.get(ModelOption.SEED, None)
            if seed is not None:
                set_seed(seed)

            format_kwargs = {}
            if _format:
                schema: dict[str, Any] = _format.model_json_schema()
                grammar: str = llguidance.LLMatcher.grammar_from_json_schema(
                    schema, defaults={"whitespace_flexible": False}
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
                    # HuggingFace uses a streaming interface that you pass to the generate call.
                    # Must be called from a running event loop. This should always be the case given the same
                    # requirement of the ._generate function below.
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

            # Filter out chat template-only options before passing to generate()
            generate_options = self._filter_chat_template_only_options(model_options)

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

            chat_response = asyncio.to_thread(
                self._generate_with_adapter_lock,
                "",  # Empty for no adapters.
                self._model.generate,  # type: ignore
                # Passed as args/kwargs to generate.
                input_ids.to(self._device),
                use_cache=True,
                past_key_values=merged_cache,
                attention_mask=attention_mask.to(self._device),
                return_dict_in_generate=True,
                output_scores=True,
                **self._make_backend_specific_and_remove(generate_options),
                **streaming_kwargs,  # type: ignore
                **format_kwargs,  # type: ignore
            )

            output = ModelOutputThunk(None)
            output._context = ctx.view_for_generation()
            output._action = action
            output._model_options = model_options

            # Processing functions only pass the ModelOutputThunk (and current chunk of response). Bind the other vars necessary for
            # each processing step.
            output._process = functools.partial(self.processing, input_ids=input_ids)
            output._post_process = functools.partial(
                self.post_processing,
                conversation=ctx_as_chat,
                input_ids=input_ids,
                _format=_format,
                tool_calls=tool_calls,
                tools=tools,
                seed=seed,
            )

            try:
                # To support lazy computation, will need to remove this create_task and store just the unexecuted coroutine.
                # We can also support synchronous calls by adding a flag and changing this ._generate function.

                response: AsyncTextIteratorStreamer | Coroutine = chat_response
                if stream and streamer is not None:
                    # For streaming, we want to pass the AsyncIterator to the function. Unlike other backends,
                    # this isn't returned by the chat_response coroutine. So we handle it here.
                    response = streamer

                    # Since the async iterator isn't returned by the chat_response coroutine, we have to create a separate
                    # task for it here so that it runs in the background. Attach it to the ModelOutputThunk.
                    output._generate_extra = asyncio.create_task(chat_response)

                # This function should always be called from a running event loop so we don't have to worry about
                # scheduling the task to a specific event loop here.
                output._generate = asyncio.create_task(
                    send_to_queue(response, output._async_queue)  # type: ignore
                )
                output._generate_type = GenerateType.ASYNC
            except RuntimeError as e:
                # Most likely cause is running this function without an event loop present.
                raise e

            return output

        else:
            raise Exception("Does not yet support non-chat contexts.")

    async def _generate_from_context_standard(
        self,
        action: Component | CBlock,
        ctx: Context,
        *,
        _format: type[BaseModelSubclass] | None = None,
        model_options: dict[str, Any],
        tool_calls: bool = False,
    ) -> ModelOutputThunk:
        # Construct input.
        # If the Context is a ChatHistory then we will pretty-print each content as a message and then use apply_chat_template.
        # Otherwise, we will linearize the context and treat it as a raw input.
        if ctx.is_chat_context:
            system_prompt = model_options.get(ModelOption.SYSTEM_PROMPT, None)
            ctx_as_chat = to_chat(action, ctx, self.formatter, system_prompt)

            # Append tool call information if applicable.
            tools: dict[str, AbstractMelleaTool] = dict()
            if tool_calls:
                if _format:
                    FancyLogger.get_logger().warning(
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
                FancyLogger.get_logger().info(f"Tools for call: {tools.keys()}")

            seed = model_options.get(ModelOption.SEED, None)
            if seed is not None:
                set_seed(seed)

            input_ids = self._tokenizer.apply_chat_template(  # type: ignore
                ctx_as_chat,
                tools=convert_tools_to_json(tools),  # type: ignore
                add_generation_prompt=True,  # If we change this, must modify huggingface granite guardian.
                return_tensors="pt",
                **self._make_backend_specific_and_remove(model_options),
            ).to(self._device)  # type: ignore

            format_kwargs = {}
            if _format:
                schema: dict[str, Any] = _format.model_json_schema()
                grammar: str = llguidance.LLMatcher.grammar_from_json_schema(
                    schema, defaults={"whitespace_flexible": False}
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
                    # HuggingFace uses a streaming interface that you pass to the generate call.
                    # Must be called from a running event loop. This should always be the case given the same
                    # requirement of the ._generate function below.
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

            # Filter out chat template-only options before passing to generate()
            generate_options = self._filter_chat_template_only_options(model_options)

            chat_response = asyncio.to_thread(
                self._generate_with_adapter_lock,
                "",  # Empty for no adapters.
                self._model.generate,  # type: ignore
                # Passed as args/kwargs to generate.
                input_ids,
                return_dict_in_generate=True,
                use_cache=self._use_caches,  # Only create KV cache if caching is enabled
                **self._make_backend_specific_and_remove(generate_options),
                **streaming_kwargs,  # type: ignore
                **format_kwargs,  # type: ignore
            )

            output = ModelOutputThunk(None)
            output._context = ctx.view_for_generation()
            output._action = action
            output._model_options = model_options

            # Processing functions only pass the ModelOutputThunk (and current chunk of response). Bind the other vars necessary for
            # each processing step.
            output._process = functools.partial(self.processing, input_ids=input_ids)
            output._post_process = functools.partial(
                self.post_processing,
                conversation=ctx_as_chat,
                input_ids=input_ids,
                _format=_format,
                tool_calls=tool_calls,
                tools=tools,
                seed=seed,
            )

            try:
                # To support lazy computation, will need to remove this create_task and store just the unexecuted coroutine.
                # We can also support synchronous calls by adding a flag and changing this ._generate function.

                response: AsyncTextIteratorStreamer | Coroutine = chat_response
                if stream and streamer is not None:
                    # For streaming, we want to pass the AsyncIterator to the function. Unlike other backends,
                    # this isn't returned by the chat_response coroutine. So we handle it here.
                    response = streamer

                    # Since the async iterator isn't returned by the chat_response coroutine, we have to create a separate
                    # task for it here so that it runs in the background. Attach it to the ModelOutputThunk.
                    output._generate_extra = asyncio.create_task(chat_response)

                # This function should always be called from a running event loop so we don't have to worry about
                # scheduling the task to a specific event loop here.
                output._generate = asyncio.create_task(
                    send_to_queue(response, output._async_queue)  # type: ignore
                )
                output._generate_type = GenerateType.ASYNC
            except RuntimeError as e:
                # Most likely cause is running this function without an event loop present.
                raise e

            return output

        else:
            raise Exception("Does not yet support non-chat contexts.")

    async def processing(
        self, mot: ModelOutputThunk, chunk: str | GenerateDecoderOnlyOutput, input_ids
    ):
        """Process the returned chunks or the complete response."""
        if mot._underlying_value is None:
            mot._underlying_value = ""

        # Because we use the AsyncTextIteratorStreamer, streaming responses are of type str;
        # and already decoded.
        if isinstance(chunk, str):
            mot._underlying_value += chunk
        elif isinstance(chunk, GenerateDecoderOnlyOutput):
            # Otherwise, it's a non-streaming request. Decode it here.
            mot._meta["hf_output"] = chunk
            mot._underlying_value += self._tokenizer.decode(
                chunk.sequences[0, input_ids.shape[1] :], skip_special_tokens=True
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
        """Called when generation is done."""
        if mot._meta.get("hf_output", None) is None:
            if mot._generate_extra is not None:
                full_output = await mot._generate_extra
                assert isinstance(full_output, GenerateDecoderOnlyOutput)
                mot._meta["hf_output"] = full_output

        # The ModelOutputThunk must be computed by this point.
        assert mot.value is not None

        # Store KV cache in LRU separately (not in mot._meta) to enable proper cleanup on eviction.
        # This prevents GPU memory from being held by ModelOutputThunk references.
        hf_output = mot._meta.get("hf_output", None)
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
                q_end=len(input_ids[0]),  # type: ignore
                scores=hf_output.scores,
            )

            cache_key = id(mot.value)
            self.cache_put(cache_key, cache_info)

            # Clear KV cache and scores from HF output - they're now owned by the LRU cache
            hf_output.past_key_values = None
            hf_output.scores = None

        # Only scan for tools if we are not doing structured output and tool calls were provided to the model.
        if _format is None and tool_calls:
            mot.tool_calls = to_tool_calls(tools, mot.value)

        assert mot._action is not None, (
            "ModelOutputThunks should have their action assigned during generation"
        )
        assert mot._model_options is not None, (
            "ModelOutputThunks should have their model_opts assigned during generation"
        )

        # Record telemetry if span is available
        span = mot._meta.get("_telemetry_span")
        if span is not None:
            from ..telemetry import end_backend_span
            from ..telemetry.backend_instrumentation import record_response_metadata

            # HuggingFace local models don't typically provide token counts
            # but we can record response metadata if available
            hf_output = mot._meta.get("hf_output")
            if isinstance(hf_output, GenerateDecoderOnlyOutput):
                record_response_metadata(span, hf_output)

            # Close the span now that async operation is complete
            end_backend_span(span)
            # Clean up span reference
            del mot._meta["_telemetry_span"]

        # When caching is disabled, clear hf_output from meta to free GPU memory.
        # The sequences tensor is on GPU and accumulates if not cleared.
        if not self._use_caches and isinstance(
            mot._meta.get("hf_output"), GenerateDecoderOnlyOutput
        ):
            import gc

            hf_out = mot._meta["hf_output"]
            if hasattr(hf_out, "sequences") and hf_out.sequences is not None:
                del hf_out.sequences
            if hasattr(hf_out, "scores") and hf_out.scores is not None:
                del hf_out.scores
            del mot._meta["hf_output"]

            # Force Python GC and return CUDA memory to device
            gc.collect()
            torch.cuda.empty_cache()

        # Generate the log for this ModelOutputThunk.
        generate_log = GenerateLog()
        generate_log.prompt = conversation
        generate_log.backend = f"hf::{self.model_id!s}"
        generate_log.model_options = mot._model_options
        generate_log.date = datetime.datetime.now()
        generate_log.model_output = mot.value
        generate_log.extra = {
            "format": _format,
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
        with instrument_generate_from_raw(
            backend=self, num_actions=len(actions), format=format, tool_calls=tool_calls
        ):
            await self.do_generate_walks(list(actions))

            if tool_calls:
                FancyLogger.get_logger().warning(
                    "The raw endpoint does not support tool calling at the moment."
                )

            if self._model.device.type == "mps":
                # TODO: Remove this when we are able to update the torch package.
                #       Test this by ensuring all outputs from this call are populated when running on mps.
                #       https://github.com/pytorch/pytorch/pull/157727
                FancyLogger.get_logger().warning(
                    "utilizing device mps with a `generate_from_raw` request; you may see issues when submitting batches of prompts to a huggingface backend; ensure all ModelOutputThunks have non-empty values."
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
                    schema, defaults={"whitespace_flexible": False}
                )
                logits_processor = _GuidanceLogitsProcessor(
                    grammar, self._llguidance_tokenizer
                )
                format_kwargs["logits_processor"] = LogitsProcessorList(
                    [logits_processor]
                )

            outputs = await asyncio.to_thread(
                self._generate_with_adapter_lock,
                "",  # Empty for no adapter.
                self._model.generate,  # type: ignore
                # Passed as args/kwargs to generate.
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                return_dict_in_generate=True,
                output_scores=True,
                **self._make_backend_specific_and_remove(model_opts),
                **format_kwargs,
            )

        sequences_to_decode = [
            sequence[inputs["input_ids"][i].size(0) :]  # type: ignore
            for i, sequence in enumerate(outputs.sequences)
        ]

        decoded_results = self._tokenizer.batch_decode(
            sequences_to_decode, skip_special_tokens=True
        )

        results = []
        for i, decoded_result in enumerate(decoded_results):
            n_prompt_tokens = inputs["input_ids"][i].size(0)  # type: ignore
            n_completion_tokens = len(sequences_to_decode[i])
            result = ModelOutputThunk(
                value=decoded_result,
                meta={
                    "usage": {
                        "prompt_tokens": n_prompt_tokens,  # type: ignore
                        "completion_tokens": n_completion_tokens,
                        "total_tokens": n_prompt_tokens + n_completion_tokens,
                    }
                },
            )
            action = actions[i]
            result.parsed_repr = (
                action.parse(result) if isinstance(action, Component) else result.value
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

        return results

    # region cache management
    def cache_get(self, id: str | int) -> HFAloraCacheInfo | None:
        """Retrieve from cache."""
        v = self._cache.get(id)
        assert v is None or type(v) is HFAloraCacheInfo
        return v

    def cache_put(self, id: str | int, v: HFAloraCacheInfo):
        """Put into cache."""
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

        Args:
            model_options: the model_options for this call

        Returns:
            a new dict
        """
        backend_specific = ModelOption.replace_keys(
            model_options, self.from_mellea_model_opts_map
        )
        return ModelOption.remove_special_keys(backend_specific)

    def _filter_chat_template_only_options(
        self, model_options: dict[str, Any]
    ) -> dict[str, Any]:
        """Remove options that are only for apply_chat_template, not for generate().

        Args:
            model_options: the model_options for this call

        Returns:
            a new dict without chat template-specific options
        """
        # Options that should only go to apply_chat_template, not generate()
        chat_template_only = {
            "guardian_config",
            "think",
            "add_generation_prompt",
            "documents",
        }
        return {k: v for k, v in model_options.items() if k not in chat_template_only}

    # region Adapter loading, unloading, and utility functions.
    @property
    def base_model_name(self):
        """Returns the base_model_id of the model used by the backend. For example, `granite-3.3-8b-instruct` for `ibm-granite/granite-3.3-8b-instruct`."""
        return self._hf_model_id.split("/")[1]

    def add_adapter(self, adapter: LocalHFAdapter):
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

        if self._added_adapters.get(adapter.qualified_name) is not None:
            FancyLogger.get_logger().warning(
                f"Client code attempted to add {adapter.name} with type {adapter.adapter_type} but {adapter.name} was already added to {self.__class__}. The backend is refusing to do this, because adapter loading is not idempotent."
            )
            return None

        adapter.path = adapter.get_local_hf_path(self.base_model_name)
        adapter.backend = self
        self._added_adapters[adapter.qualified_name] = adapter

    def load_adapter(self, adapter_qualified_name: str):
        """Loads the given adapter for the backend. Must have previously been added. Do not call when generation requests are happening."""
        adapter = self._added_adapters.get(adapter_qualified_name, None)
        if adapter is None:
            raise ValueError(
                f"could not load adapter {adapter_qualified_name} for backend {self}: adapter was not previously added"
            )

        try:
            adapter_kwargs = {}

            # Peft tries to stringify the device. If it's mps, it gets stringified as "mps:0" which causes
            # an error when loading with safetensors.torch.load_file. Force the device as a string "mps" to fix.
            if self._device == torch.device("mps"):
                adapter_kwargs["device"] = "mps"
            self._model.load_adapter(
                adapter.path, adapter.qualified_name, adapter_kwargs=adapter_kwargs
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

    def unload_adapter(self, adapter_qualified_name: str):
        """Unloads the given adapter from the backend."""
        # Check if the backend knows about this adapter.
        adapter = self._loaded_adapters.get(adapter_qualified_name, None)
        if adapter is None:
            FancyLogger.get_logger().info(
                f"could not unload adapter {adapter_qualified_name} for backend {self}: adapter is not loaded"
            )
            return

        self._model.delete_adapter(adapter.qualified_name)

        # Remove the adapter from the list of loaded adapters.
        del self._loaded_adapters[adapter.qualified_name]

    def list_adapters(self) -> list[str]:
        """Lists the adapters added via add_adapter().

        :returns: list of adapter names that are currently registered with this backend
        """
        return list(self._loaded_adapters.keys())


def _assert_correct_adapters(expected_state: str, model: PreTrainedModel):
    """When generating with a huggingface model, this can be used to ensure the correct adapters are active.

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
