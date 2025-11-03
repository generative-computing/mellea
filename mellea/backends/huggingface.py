"""A backend that uses the Huggingface Transformers library.

The purpose of the Hugginface backend is to provide a setting for implementing experimental features. If you want a performance local backend, and do not need experimental features such as Span-based context or ALoras, consider using Ollama backends instead.
"""

from __future__ import annotations

import abc
import asyncio
import dataclasses
import datetime
import functools
import inspect
import json
from collections.abc import Callable, Coroutine
from typing import TYPE_CHECKING, Any

import outlines
import outlines_core
import torch
from transformers import (
    AsyncTextIteratorStreamer,
    AutoModelForCausalLM,
    AutoTokenizer,
    DynamicCache,
    PreTrainedModel,
    PreTrainedTokenizer,
    set_seed,
)
from transformers.generation.utils import GenerateDecoderOnlyOutput

from mellea.backends import BaseModelSubclass
from mellea.backends._utils import to_chat, to_tool_calls, use_alora
from mellea.backends.aloras import Alora, AloraBackendMixin
from mellea.backends.cache import Cache, SimpleLRUCache
from mellea.backends.formatter import Formatter, FormatterBackend, TemplateFormatter
from mellea.backends.model_ids import ModelIdentifier
from mellea.backends.process_reward_models import PRM
from mellea.backends.tools import (
    add_tools_from_context_actions,
    add_tools_from_model_options,
    convert_tools_to_json,
)
from mellea.backends.types import ModelOption
from mellea.helpers.async_helpers import send_to_queue
from mellea.helpers.fancy_logger import FancyLogger
from mellea.stdlib.base import (
    CBlock,
    Component,
    Context,
    GenerateLog,
    GenerateType,
    ModelOutputThunk,
    ModelToolCall,
)
from mellea.stdlib.chat import Message
from mellea.stdlib.requirement import ALoraRequirement, LLMaJRequirement, Requirement

if TYPE_CHECKING:
    from alora.peft_model_alora import aLoRAPeftModelForCausalLM  # type: ignore

assert outlines, "outlines needs to be present to make outlines_core work"

"""A configuration type for the unhappy path: Tokenizer * Model * torch device string

Huggingface backends can initialize themselves from a model string if the transformers `Auto*` classes can be used. Therefore, a TransformersTorchConfig usually isn't required. However, sometimes a model needs special care to instantiate properly, or a custom device type needs to bse used. Instead of trying to do a lot of partial magic, we basically have two modaliites: either the constructor can figure out everything from the model_id, or the user has to provide an entire config.
"""
TransformersTorchConfig = tuple[PreTrainedTokenizer, PreTrainedModel, torch.device]

format: None = None  # typing this variable in order to shadow the global format function and ensure mypy checks for errors


@dataclasses.dataclass
class HFAloraCacheInfo:
    """A dataclass for holding some KV cache and associated information."""

    kv_cache: DynamicCache
    merged_token_ids: Any
    merged_attention: Any
    q_end: int = -1


class LocalHFBackend(FormatterBackend, AloraBackendMixin):
    """The LocalHFBackend uses Huggingface's transformers library for inference, and uses a Formatter to convert `Component`s into prompts. This backend also supports Activated LoRAs (ALoras)](https://arxiv.org/pdf/2504.12397).

    This backend is designed for running an HF model for small-scale inference locally on your machine.

    This backend is NOT designed for inference scaling on CUDA-enabled hardware.
    """

    def __init__(
        self,
        model_id: str | ModelIdentifier,
        formatter: Formatter | None = None,
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

        self._use_caches = use_caches
        self._cache = cache if cache is not None else SimpleLRUCache(3)

        # Used when running aLoRAs with this backend.
        self._alora_model: "aLoRAPeftModelForCausalLM | None" = None  # noqa: UP037
        # ALoras that have been loaded for this model.
        self._aloras: dict[str, HFAlora] = {}

    @property
    def alora_model(self) -> "aLoRAPeftModelForCausalLM | None":  # noqa: UP037
        """The ALora model."""
        return self._alora_model

    @alora_model.setter
    def alora_model(self, model: "aLoRAPeftModelForCausalLM | None"):  # noqa: UP037
        """Sets the ALora model. This should only happen once in a backend's lifetime."""
        assert self._alora_model is None
        self._alora_model = model

    def generate_from_context(
        self,
        action: Component | CBlock,
        ctx: Context,
        *,
        format: type[BaseModelSubclass] | None = None,
        model_options: dict | None = None,
        tool_calls: bool = False,
    ):
        """Generate using the huggingface model."""
        # Upsert model options.
        model_opts = self._simplify_and_merge(model_options)

        if use_alora(
            action,
            self.get_alora("constraint"),
            self.default_to_constraint_checking_alora,
        ):
            mot = self._generate_from_context_alora(
                action, ctx, _format=format, model_options=model_opts
            )
            return mot, ctx.add(mot)
        else:
            mot = self._generate_from_context_standard(
                action,
                ctx,
                _format=format,
                model_options=model_opts,
                tool_calls=tool_calls,
            )
            return mot, ctx.add(action).add(mot)

    def _generate_from_context_alora(
        self,
        action: Component | CBlock,
        ctx: Context,
        *,
        _format: type[BaseModelSubclass] | None = None,
        model_options: dict[str, Any],
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

        alora_output = alora_for_this_request.generate_using_strings(
            input=user_message,
            response=assistant_message,
            constraint=action.description,  # type: ignore
            stream=model_options.get(ModelOption.STREAM, False),
        )

        # The alora function doesn't set up all the fields.
        alora_output._context = linearized_ctx
        alora_output._action = action
        alora_output._model_options = model_options

        # TODO: Figure out what info we want to populate for aloras here.
        alora_output._generate_log = GenerateLog()

        return alora_output

    def _generate_from_context_standard(
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
            tools: dict[str, Callable] = dict()
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
                # outlines.generate.json always parses the resulting json into a python dict.
                # We however want to keep it as a json string for later storing it in ModelOutputThunk
                schema: dict[str, Any] = _format.model_json_schema()
                schema_json: str = json.dumps(schema)
                regex_str: str = outlines_core.fsm.json_schema.build_regex_from_schema(  # type: ignore
                    schema_json
                )

                from outlines.models.transformers import TransformerTokenizer
                from outlines.processors.structured import RegexLogitsProcessor
                from transformers import LogitsProcessorList

                format_kwargs["logits_processor"] = LogitsProcessorList(
                    [
                        RegexLogitsProcessor(
                            regex_str, tokenizer=TransformerTokenizer(self._tokenizer)
                        )
                    ]
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
                self._model.generate,  # type: ignore
                input_ids,
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
        else:
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
        tools: dict[str, Callable],
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

        # Add an entry to the cache for ALora reuse.
        if self._use_caches:
            output_complete = mot._meta["hf_output"].sequences[0]
            cache: DynamicCache = mot._meta["hf_output"].past_key_values  # type: ignore

            cache_info = HFAloraCacheInfo(
                kv_cache=cache,
                merged_token_ids=output_complete,
                merged_attention=torch.ones_like(output_complete).to(self._device),
                q_end=len(input_ids[0]),  # type: ignore
            )

            self.cache_put(mot.value, cache_info)

        # Only scan for tools if we are not doing structured output and tool calls were provided to the model.
        if _format is None and tool_calls:
            mot.tool_calls = to_tool_calls(tools, mot.value)

        assert mot._action is not None, (
            "ModelOutputThunks should have their action assigned during generation"
        )
        assert mot._model_options is not None, (
            "ModelOutputThunks should have their model_opts assigned during generation"
        )

        self.formatter.parse(mot._action, mot)

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
        if tool_calls:
            FancyLogger.get_logger().warning(
                "The raw endpoint does not support tool calling at the moment."
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

        if format is None:
            outputs = self._model.generate(  # type: ignore
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                return_dict_in_generate=True,
                output_scores=True,
                **self._make_backend_specific_and_remove(model_opts),
            )
        else:
            schema: dict[str, Any] = format.model_json_schema()
            schema_json: str = json.dumps(schema)
            regex_str: str = outlines_core.fsm.json_schema.build_regex_from_schema(
                schema_json
            )

            from outlines.models.transformers import TransformerTokenizer
            from outlines.processors import RegexLogitsProcessor
            from transformers import LogitsProcessorList

            outputs = self._model.generate(  # type: ignore
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                return_dict_in_generate=True,
                output_scores=True,
                logits_processor=LogitsProcessorList(
                    [
                        RegexLogitsProcessor(
                            regex_str, tokenizer=TransformerTokenizer(self._tokenizer)
                        )
                    ]
                ),
                **self._make_backend_specific_and_remove(model_opts),
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
            result = ModelOutputThunk(
                value=decoded_result,
                meta={
                    "usage": {
                        "prompt_tokens": inputs["input_ids"][i].size(0),  # type: ignore
                        "completion_tokens": len(sequences_to_decode[i]),
                    }
                },
            )

            self.formatter.parse(actions[i], result)

            generate_log = GenerateLog()
            generate_log.prompt = self.formatter.print(actions[i])
            generate_log.backend = f"hf::{self.model_id!s}"
            generate_log.model_options = model_opts
            generate_log.date = datetime.datetime.now()
            generate_log.model_output = decoded_result
            generate_log.extra = {"format": format, "seed": seed}
            generate_log.action = actions[i]

            result._generate_log = generate_log
            results.append(result)

        return results

    # region cache management
    def cache_get(self, id: str) -> HFAloraCacheInfo | None:
        """Retrieve from cache."""
        v = self._cache.get(id)
        assert v is None or type(v) is HFAloraCacheInfo
        return v

    def cache_put(self, id: str, v: HFAloraCacheInfo):
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

    # region ALora loading, unloading, and utility functions.
    def add_alora(self, alora: HFAlora):
        """Loads an ALora for this backend.

        Args:
            alora (str): identifier for the ALora adapter
        """
        from alora.peft_model_alora import aLoRAPeftModelForCausalLM  # type: ignore

        assert issubclass(alora.__class__, HFAlora), (
            f"cannot add an ALora of type {alora.__class__} to model; must inherit from {HFAlora.__class__}"
        )
        assert alora._backend == self, "Cannot load an ALora into the wrong backend."

        if self.get_alora(alora.name) is not None:
            FancyLogger.get_logger().warning(
                f"Client code attempted to add {alora.name} but {alora.name} was already added to {self.__class__}. The backend is refusing to do this, because ALora loading is not idempotent."
            )
            return None

        if self.alora_model is None:
            base_model = self._model
            self.alora_model = aLoRAPeftModelForCausalLM.from_pretrained(
                base_model, alora.path_or_model_id, alora.name
            )
        else:
            self.alora_model.load_adapter(alora.path_or_model_id, alora.name)

        self._aloras[alora.name] = alora

    def get_alora(self, alora_name: str) -> Alora | None:
        """Returns the ALora by name, or None if that ALora isn't loaded."""
        return self._aloras.get(alora_name)

    def get_aloras(self) -> list[Alora]:
        """Returns a list of all loaded ALora adapters."""
        return list(self._aloras.values())

    # endregion


class HFAlora(Alora, abc.ABC):
    """ALoras that work with the local huggingface backend."""

    def __init__(
        self,
        name: str,
        path_or_model_id: str,
        generation_prompt: str,
        backend: LocalHFBackend,
    ):
        """Initialize an ALora that should work with huggingface backends that support ALoras.

        Args:
            name (str): An arbitrary name/label to assign to an ALora. This is irrelevant from the alora's (huggingface) model id.
            path_or_model_id (str): A local path to ALora's weights or a Huggingface model_id to an ALora.
            generation_prompt (str): A prompt used to "activate" the Lora. This string goes between the pre-activation context and the aLora generate call. This needs to be provided by the entity that trained the ALora.
            backend (LocalHFBackend): Mained as a pointer to the backend to which this this ALora is attached.
        """
        super().__init__(name)
        self.path_or_model_id = path_or_model_id
        self._backend = backend
        self._generation_prompt = generation_prompt
        self._generation_prompt_tokens = self._backend._tokenizer(
            self._generation_prompt, return_tensors="pt"
        ).to(self._backend._device)


class HFProcessRewardModel(PRM, abc.ABC):
    """A Process Reward Model that works with a huggingface backend."""

    def __init__(
        self, model_name_or_path: str, score_token: str, device: str | None = None
    ):
        """Initialize an PRM that works with a huggingface backend. Currently supports and tested with IBM Process Reward Models.

        Args:
            model_name_or_path (str): A local path to PRM or a huggingface PRM
            score_token (str): token who's logits correspond to the PRM score. Can be a step demarker (for non-generative PRMs) or a correctness indicator (for generative PRMs)
            device (str): device: The computational device to use ("cuda" for GPU, "mps" for Apple Silicon, or "cpu"), defaults to None. If not specified, the best available device will be automatically selected.
        """
        super().__init__(model_name_or_path)

        # auto-device if not more specific
        self._device = device
        if device is None:
            device_name: str = (
                "cuda"
                if torch.cuda.is_available()
                else "mps"
                if torch.backends.mps.is_available()
                else "cpu"
            )
            assert device_name is not None
            self._device = torch.device(device_name)  # type: ignore

        self.model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
            self.model_name_or_path, torch_dtype=torch.bfloat16
        )
        self.model.to(self._device)  # type: ignore
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)

        self._score_token = score_token
        self._score_token_id = self.tokenizer.encode(
            self._score_token, add_special_tokens=False
        )[0]

    def stepify(self, content: str, step_separator: str) -> list[str]:
        """Splits the assistant response into steps to score.

        Args:
            content: assistant response to score
            step_separator: string on which to separate the content into steps
        """
        # convert assistant message into a list of steps
        list_of_steps = [
            step.strip() for step in content.split(step_separator) if step.strip != ""
        ]
        return list_of_steps
