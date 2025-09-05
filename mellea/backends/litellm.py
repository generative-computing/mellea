"""A generic OpenAI compatible backend that wraps around the openai python sdk."""

import datetime
import json
from collections.abc import Callable
from typing import Any

import litellm
import litellm.litellm_core_utils
import litellm.litellm_core_utils.get_supported_openai_params

import mellea.backends.model_ids as model_ids
from mellea.backends import BaseModelSubclass
from mellea.backends.formatter import Formatter, FormatterBackend, TemplateFormatter
from mellea.backends.tools import (
    add_tools_from_context_actions,
    add_tools_from_model_options,
    convert_tools_to_json,
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
)
from mellea.stdlib.chat import Message
from mellea.stdlib.requirement import ALoraRequirement


class LiteLLMBackend(FormatterBackend):
    """A generic LiteLLM compatible backend."""

    def __init__(
        self,
        model_id: str = "ollama/" + str(model_ids.IBM_GRANITE_3_3_8B.ollama_name),
        formatter: Formatter | None = None,
        base_url: str | None = "http://localhost:11434",
        model_options: dict | None = None,
    ):
        """Initialize and OpenAI compatible backend. For any additional kwargs that you need to pass the the client, pass them as a part of **kwargs.

        Args:
            model_id : The LiteLLM model identifier. Make sure that all necessary credentials are in OS environment variables.
            formatter: A custom formatter based on backend.If None, defaults to TemplateFormatter
            base_url : Base url for LLM API. Defaults to None.
            model_options : Generation options to pass to the LLM. Defaults to None.
        """
        super().__init__(
            model_id=model_id,
            formatter=(
                formatter
                if formatter is not None
                else TemplateFormatter(model_id=model_id)
            ),
            model_options=model_options,
        )

        assert isinstance(model_id, str), "Model ID must be a string."
        self._model_id = model_id

        if base_url is None:
            self._base_url = "http://localhost:11434/v1"  # ollama
        else:
            self._base_url = base_url

        # A mapping of common options for this backend mapped to their Mellea ModelOptions equivalent.
        # These are usually values that must be extracted before hand or that are common among backend providers.
        # OpenAI has some deprecated parameters. Those map to the same mellea parameter, but
        # users should only be specifying a single one in their request.
        self.to_mellea_model_opts_map = {
            "system": ModelOption.SYSTEM_PROMPT,
            "reasoning_effort": ModelOption.THINKING,  # TODO: JAL; see which of these are actually extracted...
            "seed": ModelOption.SEED,
            "max_completion_tokens": ModelOption.MAX_NEW_TOKENS,
            "max_tokens": ModelOption.MAX_NEW_TOKENS,
            "tools": ModelOption.TOOLS,
            "functions": ModelOption.TOOLS,
        }

        # A mapping of Mellea specific ModelOptions to the specific names for this backend.
        # These options should almost always be a subset of those specified in the `to_mellea_model_opts_map`.
        # Usually, values that are intentionally extracted while prepping for the backend generate call
        # will be omitted here so that they will be removed when model_options are processed
        # for the call to the model.
        self.from_mellea_model_opts_map = {
            ModelOption.SEED: "seed",
            ModelOption.MAX_NEW_TOKENS: "max_completion_tokens",
        }

    def generate_from_context(
        self,
        action: Component | CBlock,
        ctx: Context,
        *,
        format: type[BaseModelSubclass] | None = None,
        model_options: dict | None = None,
        generate_logs: list[GenerateLog] | None = None,
        tool_calls: bool = False,
    ):
        """See `generate_from_chat_context`."""
        assert ctx.is_chat_context, NotImplementedError(
            "The Openai backend only supports chat-like contexts."
        )
        return self._generate_from_chat_context_standard(
            action,
            ctx,
            format=format,
            model_options=model_options,
            generate_logs=generate_logs,
            tool_calls=tool_calls,
        )

    def _simplify_and_merge(
        self, model_options: dict[str, Any] | None
    ) -> dict[str, Any]:
        """Simplifies model_options to use the Mellea specific ModelOption.Option and merges the backend's model_options with those passed into this call.

        Rules:
        - Within a model_options dict, existing keys take precedence. This means remapping to mellea specific keys will maintain the value of the mellea specific key if one already exists.
        - When merging, the keys/values from the dictionary passed into this function take precedence.

        Because this function simplifies and then merges, non-Mellea keys from the passed in model_options will replace
        Mellea specific keys from the backend's model_options.

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

        Additionally, logs any params unknown to litellm and any params that are openai specific but not supported by this model/provider.

        Args:
            model_options: the model_options for this call

        Returns:
            a new dict
        """
        backend_specific = ModelOption.replace_keys(
            model_options, self.from_mellea_model_opts_map
        )
        backend_specific = ModelOption.remove_special_keys(backend_specific)

        # We set `drop_params=True` which will drop non-supported openai params; check for non-openai
        # params that might cause errors and log which openai params aren't supported here.
        # See https://docs.litellm.ai/docs/completion/input.
        # standard_openai_subset = litellm.get_standard_openai_params(backend_specific)
        supported_params_list = litellm.litellm_core_utils.get_supported_openai_params.get_supported_openai_params(
            self._model_id
        )
        supported_params = (
            set(supported_params_list) if supported_params_list is not None else set()
        )

        # unknown_keys = []  # keys that are unknown to litellm
        unsupported_openai_params = []  # openai params that are known to litellm but not supported for this model/provider
        for key in backend_specific.keys():
            if key not in supported_params:
                unsupported_openai_params.append(key)

        # if len(unknown_keys) > 0:
        #     FancyLogger.get_logger().warning(
        #         f"litellm allows for unknown / non-openai input params; mellea won't validate the following params that may cause issues: {', '.join(unknown_keys)}"
        #     )

        if len(unsupported_openai_params) > 0:
            FancyLogger.get_logger().warning(
                f"litellm will automatically drop the following openai keys that aren't supported by the current model/provider: {', '.join(unsupported_openai_params)}"
            )
            for key in unsupported_openai_params:
                del backend_specific[key]

        return backend_specific

    def _generate_from_chat_context_standard(
        self,
        action: Component | CBlock,
        ctx: Context,
        *,
        format: type[BaseModelSubclass]
        | None = None,  # Type[BaseModelSubclass] is a class object of a subclass of BaseModel
        model_options: dict | None = None,
        generate_logs: list[GenerateLog] | None = None,
        tool_calls: bool = False,
    ) -> ModelOutputThunk:
        model_opts = self._simplify_and_merge(model_options)
        linearized_context = ctx.render_for_generation()
        assert linearized_context is not None, (
            "Cannot generate from a non-linear context in a FormatterBackend."
        )
        # Convert our linearized context into a sequence of chat messages. Template formatters have a standard way of doing this.
        messages: list[Message] = self.formatter.to_chat_messages(linearized_context)
        # Add the final message.
        match action:
            case ALoraRequirement():
                raise Exception("The LiteLLM backend does not support activated LoRAs.")
            case _:
                messages.extend(self.formatter.to_chat_messages([action]))

        conversation: list[dict] = []
        system_prompt = model_opts.get(ModelOption.SYSTEM_PROMPT, "")
        if system_prompt != "":
            conversation.append({"role": "system", "content": system_prompt})
        conversation.extend([{"role": m.role, "content": m.content} for m in messages])

        if format is not None:
            response_format = {
                "type": "json_schema",
                "json_schema": {
                    "name": format.__name__,
                    "schema": format.model_json_schema(),
                    "strict": True,
                },
            }
        else:
            response_format = {"type": "text"}

        thinking = model_opts.get(ModelOption.THINKING, None)
        if type(thinking) is bool and thinking:
            # OpenAI uses strings for its reasoning levels.
            thinking = "medium"

        # Append tool call information if applicable.
        tools = self._extract_tools(action, format, model_opts, tool_calls, ctx)
        formatted_tools = convert_tools_to_json(tools) if len(tools) > 0 else None

        model_specific_options = self._make_backend_specific_and_remove(model_opts)

        chat_response: litellm.ModelResponse = litellm.completion(
            model=self._model_id,
            messages=conversation,
            tools=formatted_tools,
            response_format=response_format,
            reasoning_effort=thinking,  # type: ignore
            drop_params=True,  # See note in `_make_backend_specific_and_remove`.
            **model_specific_options,
        )

        choice_0 = chat_response.choices[0]
        assert isinstance(choice_0, litellm.utils.Choices), (
            "Only works for non-streaming response for now"
        )
        result = ModelOutputThunk(
            value=choice_0.message.content,
            meta={
                "litellm_chat_response": chat_response.choices[0].model_dump()
            },  # NOTE: Using model dump here to comply with `TemplateFormatter`
            tool_calls=self._extract_model_tool_requests(tools, chat_response),
        )

        parsed_result = self.formatter.parse(source_component=action, result=result)

        if generate_logs is not None:
            assert isinstance(generate_logs, list)
            generate_log = GenerateLog()
            generate_log.prompt = conversation
            generate_log.backend = f"litellm::{self.model_id!s}"
            generate_log.model_options = model_specific_options
            generate_log.date = datetime.datetime.now()
            generate_log.model_output = chat_response
            generate_log.extra = {
                "format": format,
                "tools_available": tools,
                "tools_called": result.tool_calls,
                "seed": model_opts.get("seed", None),
            }
            generate_log.action = action
            generate_log.result = parsed_result
            generate_logs.append(generate_log)

        return parsed_result

    @staticmethod
    def _extract_tools(
        action, format, model_opts, tool_calls, ctx
    ) -> dict[str, Callable]:
        tools: dict[str, Callable] = dict()
        if tool_calls:
            if format:
                FancyLogger.get_logger().warning(
                    f"Tool calling typically uses constrained generation, but you have specified a `format` in your generate call. NB: tool calling is superseded by format; we will NOT call tools for your request: {action}"
                )
            else:
                add_tools_from_model_options(tools, model_opts)
                add_tools_from_context_actions(tools, ctx.actions_for_available_tools())

                # Add the tools from the action for this generation last so that
                # they overwrite conflicting names.
                add_tools_from_context_actions(tools, [action])
            FancyLogger.get_logger().info(f"Tools for call: {tools.keys()}")
        return tools

    def _generate_from_raw(
        self,
        actions: list[Component | CBlock],
        *,
        format: type[BaseModelSubclass] | None = None,
        model_options: dict | None = None,
        generate_logs: list[GenerateLog] | None = None,
    ) -> list[ModelOutputThunk]:
        """Generate using the completions api. Gives the input provided to the model without templating."""
        raise NotImplementedError("This method is not implemented yet.")

    def _extract_model_tool_requests(
        self, tools: dict[str, Callable], chat_response: litellm.ModelResponse
    ) -> dict[str, ModelToolCall] | None:
        model_tool_calls: dict[str, ModelToolCall] = {}
        choice_0 = chat_response.choices[0]
        assert isinstance(choice_0, litellm.utils.Choices), (
            "Only works for non-streaming response for now"
        )
        calls = choice_0.message.tool_calls
        if calls:
            for tool_call in calls:
                tool_name = str(tool_call.function.name)
                tool_args = tool_call.function.arguments

                func = tools.get(tool_name)
                if func is None:
                    FancyLogger.get_logger().warning(
                        f"model attempted to call a non-existing function: {tool_name}"
                    )
                    continue  # skip this function if we can't find it.

                # Returns the args as a string. Parse it here.
                args = json.loads(tool_args)
                model_tool_calls[tool_name] = ModelToolCall(tool_name, func, args)

        if len(model_tool_calls) > 0:
            return model_tool_calls
        return None
