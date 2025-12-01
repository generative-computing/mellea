"""A backend that uses a SGLang in the current process.

The purpose of the SGLang backend is to provide a locally running fast inference engine.
"""

from __future__ import annotations

import abc
import asyncio
import dataclasses
import datetime
import functools
import inspect
import json
import os
import shutil
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Optional

import nest_asyncio
import sglang as sgl  # type:ignore
from sglang.utils import async_stream_and_merge  # type:ignore
from transformers import AutoTokenizer, PreTrainedTokenizerBase

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
from mellea.helpers.async_helpers import send_to_queue
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

nest_asyncio.apply()


class LocalSGLangBackend(FormatterBackend):
    """The LocalSGLangBackend uses SGLang's python offline inference interface, and uses a Formatter to convert `Component`s into prompts.

    This backend is designed for running a HF model for small-scale inference locally on your machine.

    Its throughput is generally higher than that of LocalHFBackend and LocalVLLMBackend.
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
            "max_new_tokens": ModelOption.MAX_NEW_TOKENS,
            "seed": ModelOption.SEED,
            "temperature": ModelOption.TEMPERATURE,
        }

        # A mapping of Mellea specific ModelOptions to the specific names for this backend.
        # These options should almost always be a subset of those specified in the `to_mellea_model_opts_map`.
        # Usually, values that are intentionally extracted while prepping for the backend generate call
        # will be omitted here so that they will be removed when model_options are processed
        # for the call to the model.
        self.from_mellea_model_opts_map = {
            ModelOption.MAX_NEW_TOKENS: "max_new_tokens",
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

        self._model = sgl.Engine(model_path=self._hf_model_id)
        self._tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
            self._hf_model_id
        )

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

            sampling_params: dict[str, Any] = self._make_backend_specific_and_remove(
                model_options
            )

            if format is not None:
                sampling_params["json_schema"] = json.dumps(format.model_json_schema())

            # if model_options.get(ModelOption.STREAM, False):
            #     generator = async_stream_and_merge(
            #         self._model,  # type: ignore
            #         input_str,
            #         sampling_params=sampling_params,
            #     )  # type: ignore
            # else:
            #     generator = self._model.async_generate(  # type: ignore
            #         input_str, sampling_params=sampling_params
            #     )  # type: ignore

            generator = self._model.async_generate(  # type: ignore
                input_str, sampling_params=sampling_params
            )  # type: ignore

            output = ModelOutputThunk(None)
            output._context = ctx.render_for_generation()
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

    async def processing(self, mot: ModelOutputThunk, chunk: str | dict[str, Any]):
        """Process the returned chunks or the complete response."""

        if isinstance(chunk, str):  # via async_stream_and_merge
            if mot._underlying_value is None:
                mot._underlying_value = ""
            mot._underlying_value += chunk
        else:
            mot._underlying_value = chunk["text"]

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
            mot.tool_calls = extract_model_tool_requests(tools, mot.value)

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
        generate_log.backend = f"sglang::{self.model_id!s}"
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

        sampling_params: dict[str, Any] = self._make_backend_specific_and_remove(
            model_options
        )

        if format is not None:
            sampling_params["json_schema"] = json.dumps(format.model_json_schema())

        outputs: list[dict[str, Any]] = self._model.generate(  # type: ignore
            prompts, sampling_params=sampling_params
        )  # type: ignore

        decoded_results = [output["text"] for output in outputs]
        results = [ModelOutputThunk(value=output["text"]) for output in outputs]

        for i, result in enumerate(results):
            self.formatter.parse(actions[i], result)

        if generate_logs is not None:
            assert isinstance(generate_logs, list)
            date = datetime.datetime.now()

            for i in range(len(prompts)):
                generate_log = GenerateLog()
                generate_log.prompt = prompts[i]
                generate_log.backend = f"sglang::{self.model_id!s}"
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
