"""A backend that uses a VLLM in the current process.

The purpose of the VLLM backend is to provide a locally running fast inference engine.
"""

from __future__ import annotations

import abc
import asyncio
import dataclasses
import datetime
import functools
import importlib
import inspect
import json
import os
import shutil
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Optional

import msgspec  # type:ignore
import outlines
import outlines_core
import torch
import vllm  # type:ignore
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from mellea.backends import BaseModelSubclass
from mellea.backends._utils import to_chat, to_tool_calls
from mellea.backends.formatter import Formatter, FormatterBackend, TemplateFormatter
from mellea.backends.model_ids import ModelIdentifier
from mellea.backends.tools import (
    add_tools_from_context_actions,
    add_tools_from_model_options,
    convert_tools_to_json,
)
from mellea.backends.types import ModelOption
from mellea.helpers.async_helpers import get_current_event_loop, send_to_queue
from mellea.helpers.event_loop_helper import _run_async_in_thread
from mellea.helpers.fancy_logger import FancyLogger
from mellea.stdlib.base import (
    CBlock,
    Component,
    Context,
    GenerateLog,
    GenerateType,
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
        if os.environ.get("VLLM_USE_V1", -1) != "0":
            FancyLogger.get_logger().error(
                "Mellea LocalVLLMBackend doesn't support VLLM V1. Must `export VLLM_USE_V1=0`."
            )
            raise ValueError(
                "Mellea LocalVLLMBackend doesn't support VLLM V1. Must `export VLLM_USE_V1=0`."
            )

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

        # Either use the custom config or load the model from its model_id
        match model_id:
            case str():
                self._hf_model_id = model_id
            case ModelIdentifier():
                assert model_id.hf_model_name is not None, (
                    "model_id is None. This can also happen if the ModelIdentifier has no hf_model_id name set."
                )
                self._hf_model_id = model_id.hf_model_name

        # vllm requires some model options during instantiation.
        engine_args = self._simplify_and_merge(model_options)
        engine_args = self._make_backend_specific_and_remove(
            engine_args, vllm.AsyncEngineArgs
        )

        logger = FancyLogger.get_logger()
        # Get the model and tokenizer.
        # Getting vllm instantiated is tricky as it does not automatically detect some of these parameters.
        engine_args["gpu_memory_utilization"] = engine_args.get(
            "gpu_memory_utilization", 0.9
        )
        engine_args["max_num_seqs"] = engine_args.get("max_num_seqs", 16)
        engine_args["max_model_len"] = engine_args.get("max_model_len", 16384)
        logger.info(
            f"Instantiating vllm with the following model parameters:\n"
            f"gpu_memory_utilization: {engine_args['gpu_memory_utilization']}\n"
            f"max_model_len: {engine_args['max_model_len']}\n"
            f"max_num_seqs: {engine_args['max_num_seqs']}\n"
        )
        retry = 0
        while True:
            retry += 1
            try:
                self._underlying_model = vllm.AsyncLLMEngine.from_engine_args(
                    vllm.AsyncEngineArgs(model=self._hf_model_id, **engine_args)
                )
                break
            except torch._dynamo.exc.BackendCompilerFailed as e:
                # example:
                # torch._dynamo.exc.BackendCompilerFailed: backend='<vllm.compilation.backends.VllmBackend object at 0x7f6d3f341730>' raised:
                # RuntimeError: vLLM failed to compile the model. The most likely reason for this is that a previous compilation failed, leading to a corrupted compilation artifact. We recommend trying to remove ~/.cache/vllm/torch_compile_cache and try again to see the real issue.

                if "~/.cache/vllm/torch_compile_cache" in str(e.inner_exception):
                    logger.warning(
                        "removing ~/.cache/vllm/torch_compile_cache and retry"
                    )
                    shutil.rmtree("~/.cache/vllm/torch_compile_cache")
                    # then retry

            except Exception as e:
                logger.info(e)
                if retry % 3 == 0:
                    engine_args["max_model_len"] //= 2
                elif retry % 3 == 1:
                    engine_args["max_num_seqs"] //= 2
                elif retry % 3 == 2:
                    engine_args["gpu_memory_utilization"] *= 0.9
                if (
                    engine_args["max_model_len"] == 0
                    or engine_args["max_num_seqs"] == 0
                    or engine_args["gpu_memory_utilization"] < 0.1
                ):
                    raise RuntimeError(
                        "no matter how I reduced max_model_len and max_num_seqs, there is not enough memory! \n"
                        "final values:\n"
                        f"gpu_memory_utilization: {engine_args['gpu_memory_utilization']}\n"
                        f"max_model_len: {engine_args['max_model_len']}\n"
                        f"max_num_seqs: {engine_args['max_num_seqs']}\n"
                    )
                logger.info(
                    f"Reducing vllm model parameters to make it fit in the GPU memory.\n"
                    "current values:\n"
                    f"gpu_memory_utilization: {engine_args['gpu_memory_utilization']}\n"
                    f"max_model_len: {engine_args['max_model_len']}\n"
                    f"max_num_seqs: {engine_args['max_num_seqs']}\n"
                )

        logger.info(
            f"vllm instantiated.\n"
            "final model parameters:\n"
            f"gpu_memory_utilization: {engine_args['gpu_memory_utilization']}\n"
            f"max_model_len: {engine_args['max_model_len']}\n"
            f"max_num_seqs: {engine_args['max_num_seqs']}\n"
        )

        # Keep track of the event loop the engine was instantiated in.
        self._event_loop = get_current_event_loop()

        self._tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
            self._hf_model_id
        )  # type:ignore

        # See the notes in outlines.models.vllm.adapt_tokenizer for why this is needed.
        # Note: there is a module named outlines.models.vllm and a function named outlines.models.vllm.vllm .
        # However, outlines.models import outlines.models.vllm.vllm as vllm,
        # thus the module outlines.models.vllm becomes inaccessible,
        # hence the use of importlib to get the module.
        self._tokenizer_for_outlines: PreTrainedTokenizerBase = importlib.import_module(
            "outlines.models.vllm"
        ).adapt_tokenizer(self._tokenizer)

    @property
    def _model(self) -> vllm.AsyncLLMEngine:
        """Use model when making generation requests."""
        el = get_current_event_loop()

        # vLLM attaches itself to the event loop that is running when instantiated /
        # the first generate request is made. Thankfully, they provide helpers to
        # reset that. We do that here if the event loop changes.

        # Most of the time, this should be a no-op. The event loop will only change
        # if switching between async and sync calls.
        if el != self._event_loop:
            self._underlying_model.shutdown_background_loop()
            self._underlying_model.start_background_loop()
            self._event_loop = el

        return self._underlying_model

    async def generate_from_context(
        self,
        action: Component | CBlock,
        ctx: Context,
        *,
        format: type[BaseModelSubclass] | None = None,
        model_options: dict | None = None,
        generate_logs: list[GenerateLog] | None = None,
        tool_calls: bool = False,
    ) -> tuple[ModelOutputThunk, Context]:
        """Generate using the huggingface model."""
        # Upsert model options.
        model_options = self._simplify_and_merge(model_options)

        # TODO: insert the alora code here.

        mot = await self._generate_from_context_standard(
            action,
            ctx,
            format=format,
            model_options=model_options,
            generate_logs=generate_logs,
            tool_calls=tool_calls,
        )
        return mot, ctx.add(action).add(mot)

    async def _generate_from_context_standard(
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
                ),
                output_kind=(
                    # returns results incrementally
                    vllm.sampling_params.RequestOutputKind.DELTA
                    if model_options.get(ModelOption.STREAM, False)
                    # returns only the final result
                    else vllm.sampling_params.RequestOutputKind.FINAL_ONLY
                ),
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

            # stream = model_options.get(ModelOption.STREAM, False)
            # if stream:

            output = ModelOutputThunk(None)

            generator = self._model.generate(  # type: ignore
                request_id=str(id(output)),
                prompt=input_str,
                sampling_params=sampling_params,
            )  # type: ignore

            output._context = ctx.view_for_generation()
            output._action = action
            output._model_options = model_options

            output._process = self.processing
            output._post_process = functools.partial(
                self.post_processing,
                conversation=ctx_as_chat,
                tool_calls=tool_calls,
                tools=tools,
                seed=model_options.get(ModelOption.SEED, None),
            )

            try:
                # This function should always be called from a running event loop so we don't have to worry about
                # scheduling the task to a specific event loop here.
                output._generate = asyncio.create_task(
                    send_to_queue(generator, output._async_queue)  # type: ignore
                )
                output._generate_type = GenerateType.ASYNC
            except RuntimeError as e:
                # Most likely cause is running this function without an event loop present.
                raise e

            return output

        else:
            raise Exception("Does not yet support non-chat contexts.")

    async def processing(self, mot: ModelOutputThunk, chunk: vllm.RequestOutput):
        """Process the returned chunks or the complete response."""
        if mot._underlying_value is None:
            mot._underlying_value = ""
        mot._underlying_value += chunk.outputs[0].text

    async def post_processing(
        self,
        mot: ModelOutputThunk,
        conversation: list[dict],
        tool_calls: bool,
        tools: dict[str, Callable],
        seed,
    ):
        """Called when generation is done."""
        # The ModelOutputThunk must be computed by this point.
        assert mot.value is not None

        # Only scan for tools if we are not doing structured output and tool calls were provided to the model.
        if format is None and tool_calls:
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
        generate_log.backend = f"vllm::{self.model_id!s}"
        generate_log.model_options = mot._model_options
        generate_log.date = datetime.datetime.now()
        generate_log.model_output = mot.value
        generate_log.extra = {
            "format": format,
            "tools_available": tools,
            "tools_called": mot.tool_calls,
            "seed": seed,
        }
        generate_log.action = mot._action
        generate_log.result = mot

        mot._generate_log = generate_log

    async def generate_from_raw(
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
                "The completion endpoint does not support tool calling at the moment."
            )

        model_options = self._simplify_and_merge(model_options)

        prompts = [self.formatter.print(action) for action in actions]

        sampling_params = vllm.SamplingParams(
            **self._make_backend_specific_and_remove(
                model_options, vllm.SamplingParams
            ),
            output_kind=vllm.sampling_params.RequestOutputKind.FINAL_ONLY,  # returns only the final results
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

        async def generate(prompt, request_id):
            async for result_output in self._model.generate(
                request_id=request_id, prompt=prompt, sampling_params=sampling_params
            ):
                assert result_output.finished
                return result_output.outputs[0].text

        tasks = [generate(p, f"{id(prompts)}-{i}") for i, p in enumerate(prompts)]
        decoded_results = await asyncio.gather(*tasks)

        results = [ModelOutputThunk(value=text) for text in decoded_results]

        for i, result in enumerate(results):
            self.formatter.parse(actions[i], result)
            date = datetime.datetime.now()

            generate_log = GenerateLog()
            generate_log.prompt = prompts[i]
            generate_log.backend = f"vllm::{self.model_id!s}"
            generate_log.model_options = model_options
            generate_log.date = date
            generate_log.model_output = decoded_results
            generate_log.extra = {
                "format": format,
                "seed": model_options.get(ModelOption.SEED, None),
            }
            generate_log.action = actions[i]
            generate_log.result = results[i]
            result._generate_log = generate_log

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
            cls: the target class. the returned dict contains the keys of this class

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
