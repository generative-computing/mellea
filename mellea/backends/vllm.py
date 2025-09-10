"""A backend that uses a VLLM in the current process.

The purpose of the VLLM backend is to provide a locally running fast inference engine.
"""

from __future__ import annotations

import abc
import dataclasses
import datetime
import inspect
import json
import os
import shutil
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Optional

import msgspec
import outlines
import outlines_core
import torch
import vllm
from transformers import PreTrainedTokenizerBase

from mellea.backends import BaseModelSubclass
from mellea.backends.formatter import Formatter, FormatterBackend, TemplateFormatter
from mellea.backends.model_ids import ModelIdentifier
from mellea.backends.tools import (
    add_tools_from_context_actions,
    add_tools_from_model_options,
    convert_tools_to_json,
)
from mellea.backends.types import ModelOption
from mellea.backends.utils import extract_model_tool_requests, to_chat
from mellea.helpers.fancy_logger import FancyLogger
from mellea.stdlib.base import (
    CBlock,
    Component,
    Context,
    GenerateLog,
    ModelOutputThunk,
    TemplateRepresentation,
)
from mellea.stdlib.chat import Message
from mellea.stdlib.requirement import LLMaJRequirement, Requirement

assert outlines, "outlines needs to be present to make outlines_core work"


class LocalVLLMBackend(FormatterBackend):
    """The LocalVLLMBackend uses vLLM's python interface for inference, and uses a Formatter to convert `Component`s into prompts.

    The support for Activated LoRAs (ALoras)](https://arxiv.org/pdf/2504.12397) is planned.

    This backend is designed for running an HF model for small-scale inference locally on your machine.

    Its throughput is generally higher than that of LocalHFBackend.
    However, it takes longer to load the weights during the instantiation.
    Also, if you submit a request one by one, it can be slower.
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
            # "system": ModelOption.SYSTEM_PROMPT,
            "max_tokens": ModelOption.MAX_NEW_TOKENS,
            "seed": ModelOption.SEED,
            "temperature": ModelOption.TEMPERATURE,
        }

        # A mapping of Mellea specific ModelOptions to the specific names for this backend.
        # These options should almost always be a subset of those specified in the `to_mellea_model_opts_map`.
        # Usually, values that are intentionally extracted while prepping for the backend generate call
        # will be omitted here so that they will be removed when model_options are processed
        # for the call to the model.
        self.from_mellea_model_opts_map = {
            ModelOption.MAX_NEW_TOKENS: "max_tokens",
            ModelOption.SEED: "seed",
            ModelOption.TEMPERATURE: "temperature",
        }

        model_options = self._simplify_and_merge(model_options)
        model_options = self._make_backend_specific_and_remove(
            model_options, vllm.EngineArgs
        )

        # Either use the custom config or load the model from its model_id
        match model_id:
            case str():
                self._hf_model_id = model_id
            case ModelIdentifier():
                assert model_id.hf_model_name is not None, (
                    "model_id is None. This can also happen if the ModelIdentifier has no hf_model_id name set."
                )
                self._hf_model_id = model_id.hf_model_name

        # Get the model and tokenizer.
        # Getting vllm instantiated is tricky as it does not automatically detect some of these parameters.
        model_options["gpu_memory_utilization"] = model_options.get(
            "gpu_memory_utilization", 0.9
        )
        model_options["max_num_seqs"] = model_options.get("max_num_seqs", 16)
        model_options["max_model_len"] = model_options.get("max_model_len", 16384)
        print(
            f"Instantiating vllm with the following model parameters:\n"
            f"gpu_memory_utilization: {model_options['gpu_memory_utilization']}\n"
            f"max_model_len: {model_options['max_model_len']}\n"
            f"max_num_seqs: {model_options['max_num_seqs']}\n"
        )
        retry = 0
        while True:
            retry += 1
            try:
                self._model = vllm.LLM(
                    model=self._hf_model_id, task="generate", **model_options
                )
                break
            except torch._dynamo.exc.BackendCompilerFailed as e:
                # example:
                # torch._dynamo.exc.BackendCompilerFailed: backend='<vllm.compilation.backends.VllmBackend object at 0x7f6d3f341730>' raised:
                # RuntimeError: vLLM failed to compile the model. The most likely reason for this is that a previous compilation failed, leading to a corrupted compilation artifact. We recommend trying to remove ~/.cache/vllm/torch_compile_cache and try again to see the real issue.

                if "~/.cache/vllm/torch_compile_cache" in str(e.inner_exception):
                    print("removing ~/.cache/vllm/torch_compile_cache and retry")
                    shutil.rmtree("~/.cache/vllm/torch_compile_cache")
                    # then retry

            except Exception as e:
                print(e)
                if retry % 3 == 0:
                    model_options["max_model_len"] //= 2
                elif retry % 3 == 1:
                    model_options["max_num_seqs"] //= 2
                elif retry % 3 == 2:
                    model_options["gpu_memory_utilization"] *= 0.9
                if (
                    model_options["max_model_len"] == 0
                    or model_options["max_num_seqs"] == 0
                    or model_options["gpu_memory_utilization"] < 0.1
                ):
                    raise RuntimeError(
                        "no matter how I reduced max_model_len and max_num_seqs, there is not enough memory! \n"
                        "final values:\n"
                        f"gpu_memory_utilization: {model_options['gpu_memory_utilization']}\n"
                        f"max_model_len: {model_options['max_model_len']}\n"
                        f"max_num_seqs: {model_options['max_num_seqs']}\n"
                    )
                print(
                    f"Reducing vllm model parameters to make it fit in the GPU memory.\n"
                    "current values:\n"
                    f"gpu_memory_utilization: {model_options['gpu_memory_utilization']}\n"
                    f"max_model_len: {model_options['max_model_len']}\n"
                    f"max_num_seqs: {model_options['max_num_seqs']}\n"
                )

        print(
            f"vllm instantiated.\n"
            "final model parameters:\n"
            f"gpu_memory_utilization: {model_options['gpu_memory_utilization']}\n"
            f"max_model_len: {model_options['max_model_len']}\n"
            f"max_num_seqs: {model_options['max_num_seqs']}\n"
        )

        self._tokenizer: PreTrainedTokenizerBase = self._model.get_tokenizer()  # type:ignore

        # see notes in outlines.models.vllm.adapt_tokenizer
        self._tokenizer_for_outlines: PreTrainedTokenizerBase = outlines.models.VLLM(
            self._model
        ).tokenizer  # type:ignore

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
        model_options = self._simplify_and_merge(model_options)

        # TODO: insert the alora code here.

        return self._generate_from_context_standard(
            action,
            ctx,
            format=format,
            model_options=model_options,
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
            system_prompt = model_options.get(ModelOption.SYSTEM_PROMPT, None)
            ctx_as_chat = to_chat(action, ctx, self.formatter, system_prompt)

            # Append tool call information if applicable.
            tools: dict[str, Callable] = dict()
            if tool_calls:
                if format:
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

            input_str: str = self._tokenizer.apply_chat_template(  # type: ignore
                ctx_as_chat,
                tokenize=False,
                tools=convert_tools_to_json(tools),  # type: ignore
            )

            sampling_params = vllm.SamplingParams(
                **self._make_backend_specific_and_remove(
                    model_options, vllm.SamplingParams
                )
            )

            if format is not None:
                # outlines.generate.json always parses the resulting json into a python dict.
                # We however want to keep it as a json string for later storing it in ModelOutputThunk
                schema: dict[str, Any] = format.model_json_schema()
                schema_json: str = json.dumps(schema)
                regex_str: str = outlines_core.fsm.json_schema.build_regex_from_schema(
                    schema_json
                )

                from outlines.processors import RegexLogitsProcessor

                logits_processor = RegexLogitsProcessor(
                    regex_str,
                    tokenizer=self._tokenizer_for_outlines,  # type: ignore
                )
                sampling_params.logits_processors = (
                    [logits_processor] if logits_processor is not None else []
                )

            ros: list[vllm.RequestOutput] = self._model.generate(  # type: ignore
                [input_str], sampling_params=sampling_params
            )  # type: ignore

            decoded_results = [ro.outputs[0].text for ro in ros]

            decoded_result = decoded_results[0]

        else:
            raise Exception("Does not yet support non-chat contexts.")

        assert decoded_result is not None

        result = ModelOutputThunk(value=decoded_result)

        # Only scan for tools if we are not doing structured decoding and tool calls were provided to the model.
        if format is None and tool_calls:
            result.tool_calls = extract_model_tool_requests(tools, decoded_result)

        parsed_result = self.formatter.parse(action, result)
        if generate_logs is not None:
            assert isinstance(generate_logs, list)
            generate_log = GenerateLog()
            generate_log.prompt = ctx_as_chat
            generate_log.backend = f"hf::{self.model_id!s}"
            generate_log.model_options = model_options
            generate_log.date = datetime.datetime.now()
            generate_log.model_output = decoded_result
            generate_log.extra = {
                "format": format,
                "tools_available": tools,
                "tools_called": result.tool_calls,
                "seed": model_options.get(ModelOption.SEED, None),
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
        model_options = self._simplify_and_merge(model_options)

        prompts = [self.formatter.print(action) for action in actions]

        sampling_params = vllm.SamplingParams(
            **self._make_backend_specific_and_remove(model_options, vllm.SamplingParams)
        )

        if format is not None:
            schema: dict[str, Any] = format.model_json_schema()
            schema_json: str = json.dumps(schema)
            regex_str: str = outlines_core.fsm.json_schema.build_regex_from_schema(
                schema_json
            )

            from outlines.processors import RegexLogitsProcessor

            logits_processor = RegexLogitsProcessor(
                regex_str,
                tokenizer=self._tokenizer_for_outlines,  # type: ignore
            )
            sampling_params.logits_processors = (
                [logits_processor] if logits_processor is not None else []
            )

        ros: list[vllm.RequestOutput] = self._model.generate(  # type: ignore
            prompts, sampling_params=sampling_params
        )  # type: ignore

        decoded_results = [ro.outputs[0].text for ro in ros]

        results = [ModelOutputThunk(value=text) for text in decoded_results]

        for i, result in enumerate(results):
            self.formatter.parse(actions[i], result)

        if generate_logs is not None:
            assert isinstance(generate_logs, list)
            date = datetime.datetime.now()

            for i in range(len(prompts)):
                generate_log = GenerateLog()
                generate_log.prompt = prompts[i]
                generate_log.backend = f"hf::{self.model_id!s}"
                generate_log.model_options = model_options
                generate_log.date = date
                generate_log.model_output = decoded_results
                generate_log.extra = {
                    "format": format,
                    "seed": model_options.get(ModelOption.SEED, None),
                }
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
        self, model_options: dict[str, Any], cls: type[Any]
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
        backend_specific = ModelOption.remove_special_keys(backend_specific)
        try:
            # note: dataclasses.Field objects
            return {
                field.name: backend_specific[field.name]
                for field in dataclasses.fields(cls)
                if field.name in backend_specific
            }
        except TypeError:
            # note: msgspec.structs.FieldInfo objects
            return {
                field.name: backend_specific[field.name]
                for field in msgspec.structs.fields(cls)
                if field.name in backend_specific
            }
