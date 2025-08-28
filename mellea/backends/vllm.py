"""A backend that uses the Huggingface Transformers library.

The purpose of the Hugginface backend is to provide a setting for implementing experimental features. If you want a performance local backend, and do not need experimental features such as Span-based context or ALoras, consider using Ollama backends instead.
"""

from __future__ import annotations

import abc
import dataclasses
import datetime
import inspect
import json
import os
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Optional

import outlines
import outlines_core
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DynamicCache,
    PreTrainedModel,
    PreTrainedTokenizer,
    set_seed,
)

from mellea.backends import BaseModelSubclass
from mellea.backends.formatter import Formatter, FormatterBackend, TemplateFormatter
from mellea.backends.model_ids import ModelIdentifier
from mellea.backends.tools import (
    convert_tools_to_json,
    get_tools_from_action,
    parse_tools,
)
from mellea.backends.types import ModelOption
from mellea.helpers.fancy_logger import FancyLogger
from mellea.stdlib.base import (
    CBlock,
    Component,
    Context,
    GenerateLog,
    ModelOutputThunk,
    ModelToolCall,
    TemplateRepresentation,
)
from mellea.stdlib.chat import Message
from mellea.stdlib.requirement import LLMaJRequirement, Requirement

assert outlines, "outlines needs to be present to make outlines_core work"


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
        model_options: dict | None = None,
    ):
        """Attempt to load model weights using the model_id by default, or using `custom_config` if provided.

        WARNING: initializing a `LocalHFBackend` will download and load the model on your *local* machine.

        Args:
            model_id (str | ModelIdentifier): Used to load the model *and tokenizer* via transformers Auto* classes, and then moves the model to the best available device (cuda > mps > cpu). If loading the model and/or tokenizer from a string will not work, or if you want to use a different device string, then you can use custom_config.
            formatter (Formatter): A mechanism for turning `stdlib` stuff into strings. Experimental Span-based models should use `mellea.backends.span.*` backends.
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
        generate_logs: list[GenerateLog] | None = None,
        tool_calls: bool = False,
    ) -> ModelOutputThunk:
        """Generate using the huggingface model."""
        # Upsert model options.
        model_opts = self._simplify_and_merge(model_options)

        # TODO: insert the alora code here.

        return self._generate_from_context_standard(
            action,
            ctx,
            format=format,
            model_options=model_opts,
            generate_logs=generate_logs,
            tool_calls=tool_calls,
        )

    def _generate_from_context_standard(
        self,
        action: Component | CBlock,
        ctx: Context,
        *,
        format: type[BaseModelSubclass] | None = None,
        model_options: dict[str, Any],
        generate_logs: list[GenerateLog] | None = None,
        tool_calls: bool = False,
    ) -> ModelOutputThunk:
        # Construct input.
        # If the Context is a ChatHistory then we will pretty-print each content as a message and then use apply_chat_template.
        # Otherwise, we will linearize the context and treat it as a raw input.
        decoded_result: str | None = None
        if ctx.is_chat_context:
            linearized_ctx = ctx.render_for_generation()
            assert linearized_ctx is not None, (
                "If ctx.is_chat_context, then the context should be linearizable."
            )
            ctx_as_message_list: list[Message] = self.formatter.to_chat_messages(
                linearized_ctx
            )
            # add action
            ctx_as_message_list.extend(self.formatter.to_chat_messages([action]))
            ctx_as_conversation = [
                {"role": m.role, "content": m.content} for m in ctx_as_message_list
            ]

            # Check that we ddin't accidentally end up with CBlocks.
            for msg in ctx_as_conversation:
                for v in msg.values():
                    if "CBlock" in v:
                        FancyLogger.get_logger().error(
                            f"Found the string `CBlock` in what should've been a stringified context: {ctx_as_conversation}"
                        )

            # handle custom system prompts. It's important that we do this before the _parse_and_**clean**_model_options step.
            system_prompt = model_options.get(ModelOption.SYSTEM_PROMPT, None)
            if system_prompt is not None:
                system_msg: dict[str, str] = {
                    "role": "system",
                    "content": system_prompt,
                }
                ctx_as_conversation.insert(0, system_msg)

            # Append tool call information if applicable.
            tools: dict[str, Callable] = dict()
            if tool_calls:
                if format:
                    FancyLogger.get_logger().warning(
                        f"Tool calling typically uses constrained generation, but you have specified a `format` in your generate call. NB: tool calling is superseded by format; we will NOT call tools for your request: {action}"
                    )
                else:
                    if isinstance(action, Component) and isinstance(
                        action.format_for_llm(), TemplateRepresentation
                    ):
                        tools = get_tools_from_action(action)

                    model_options_tools = model_options.get(ModelOption.TOOLS, None)
                    if model_options_tools is not None:
                        assert isinstance(model_options_tools, dict)
                        for fn_name in model_options_tools:
                            # invariant re: relationship between the model_options set of tools and the TemplateRepresentation set of tools
                            assert fn_name not in tools.keys(), (
                                f"Cannot add tool {fn_name} because that tool was already defined in the TemplateRepresentation for the action."
                            )
                            # type checking because ModelOptions is an untyped dict and the calling convention for tools isn't clearly documented at our abstraction boundaries.
                            assert type(fn_name) is str, (
                                "When providing a `ModelOption.TOOLS` parameter to `model_options`, always used the type Dict[str, Callable] where `str` is the function name and the callable is the function."
                            )
                            assert callable(model_options_tools[fn_name]), (
                                "When providing a `ModelOption.TOOLS` parameter to `model_options`, always used the type Dict[str, Callable] where `str` is the function name and the callable is the function."
                            )
                            # Add the model_options tool to the existing set of tools.
                            tools[fn_name] = model_options_tools[fn_name]

            seed = model_options.get(ModelOption.SEED, None)
            if seed is not None:
                set_seed(seed)

            input_ids = self._tokenizer.apply_chat_template(  # type: ignore
                ctx_as_conversation,
                tools=convert_tools_to_json(tools),  # type: ignore
                return_tensors="pt",
                **self._make_backend_specific_and_remove(model_options),
            ).to(self._device)  # type: ignore

            if format is None:
                chat_output = self._model.generate(  # type: ignore
                    input_ids,
                    return_dict_in_generate=True,
                    output_scores=True,
                    **self._make_backend_specific_and_remove(model_options),
                )  # type: ignore

            else:
                # outlines.generate.json always parses the resulting json into a python dict.
                # We however want to keep it as a json string for later storing it in ModelOutputThunk
                schema: dict[str, Any] = format.model_json_schema()
                schema_json: str = json.dumps(schema)
                regex_str: str = outlines_core.fsm.json_schema.build_regex_from_schema(
                    schema_json
                )

                from outlines.models.transformers import TransformerTokenizer
                from outlines.processors import RegexLogitsProcessor
                from transformers import LogitsProcessorList

                chat_output = self._model.generate(  # type: ignore
                    input_ids,
                    return_dict_in_generate=True,
                    output_scores=True,
                    logits_processor=LogitsProcessorList(
                        [
                            RegexLogitsProcessor(
                                regex_str,
                                tokenizer=TransformerTokenizer(self._tokenizer),
                            )
                        ]
                    ),
                    **self._make_backend_specific_and_remove(model_options),
                )

            decoded_result = self._tokenizer.decode(
                chat_output.sequences[0, input_ids.shape[1] :], skip_special_tokens=True
            )


        else:
            raise Exception("Does not yet support non-chat contexts.")

        assert decoded_result is not None

        result = ModelOutputThunk(value=decoded_result)

        # Only scan for tools if we are not doing structured decoding and tool calls were provided to the model.
        if format is None and tool_calls:
            result.tool_calls = self._extract_model_tool_requests(tools, decoded_result)

        parsed_result = self.formatter.parse(action, result)
        if generate_logs is not None:
            assert isinstance(generate_logs, list)
            generate_log = GenerateLog()
            generate_log.prompt = ctx_as_conversation
            generate_log.backend = f"hf::{self.model_id!s}"
            generate_log.model_options = model_options
            generate_log.date = datetime.datetime.now()
            generate_log.model_output = decoded_result
            generate_log.extra = {
                "format": format,
                "tools_available": tools,
                "tools_called": result.tool_calls,
                "seed": seed,
            }
            generate_log.action = action
            generate_log.result = parsed_result
            generate_logs.append(generate_log)
        return parsed_result

    def _generate_from_raw(
        self,
        actions: list[Component | CBlock],
        *,
        format: type[BaseModelSubclass] | None = None,
        model_options: dict | None = None,
        generate_logs: list[GenerateLog] | None = None,
    ) -> list[ModelOutputThunk]:
        """Generate using the completions api. Gives the input provided to the model without templating."""
        model_opts = self._simplify_and_merge(model_options)
        seed = model_opts.get(ModelOption.SEED, None)
        if seed is not None:
            set_seed(seed)

        prompts = [self.formatter.print(action) for action in actions]

        # batch-encoding call is deprecated in favor of this
        inputs = self._tokenizer(prompts, return_tensors="pt").to(self._device)

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

        results = [
            ModelOutputThunk(value=decoded_result) for decoded_result in decoded_results
        ]

        for i, result in enumerate(results):
            self.formatter.parse(actions[i], result)

        if generate_logs is not None:
            assert isinstance(generate_logs, list)
            date = datetime.datetime.now()

            for i in range(len(prompts)):
                generate_log = GenerateLog()
                generate_log.prompt = prompts[i]
                generate_log.backend = f"hf::{self.model_id!s}"
                generate_log.model_options = model_opts
                generate_log.date = date
                generate_log.model_output = decoded_results
                generate_log.extra = {"format": format, "seed": seed}
                generate_log.action = actions[i]
                generate_log.result = results[i]
                generate_logs.append(generate_log)

        return results


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

    def _extract_model_tool_requests(
        self, tools: dict[str, Callable], decoded_result: str
    ) -> dict[str, ModelToolCall] | None:
        model_tool_calls: dict[str, ModelToolCall] = dict()
        for tool_name, tool_args in parse_tools(decoded_result):
            func = tools.get(tool_name)
            if func is None:
                FancyLogger.get_logger().warning(
                    f"model attempted to call a non-existing function: {tool_name}"
                )
                continue

            # Clean up the function args slightly. Some models seem to
            # hallucinate parameters when none are required.
            sig = inspect.signature(func)
            if len(sig.parameters) == 0:
                tool_args = {}

            model_tool_calls[tool_name] = ModelToolCall(tool_name, func, tool_args)

        if len(model_tool_calls) > 0:
            return model_tool_calls
        return None



