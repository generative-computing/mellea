"""A generic WatsonX.ai compatible backend that wraps around the watson_machine_learning library."""

import datetime
import json
import os
from collections.abc import Callable
from typing import Any

from ibm_watsonx_ai import APIClient, Credentials
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.foundation_models.schema import TextChatParameters

from mellea.backends import BaseModelSubclass, model_ids
from mellea.backends.formatter import Formatter, FormatterBackend, TemplateFormatter
from mellea.backends.model_ids import ModelIdentifier
from mellea.backends.tools import (
    add_tools_from_model_options,
    convert_tools_to_json,
    get_tools_from_action,
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
from mellea.stdlib.requirement import ALoraRequirement  # type: ignore


class WatsonxAIBackend(FormatterBackend):
    """A generic backend class for watsonx SDK."""

    def __init__(
        self,
        model_id: str | ModelIdentifier = model_ids.IBM_GRANITE_3_3_8B,
        formatter: Formatter | None = None,
        base_url: str | None = None,
        model_options: dict | None = None,
        *,
        api_key: str | None = None,
        project_id: str | None = None,
        **kwargs,
    ):
        """A generic watsonx backend that wraps around the ibm_watsonx_ai sdk.

        Args:
            model_id  : Model id. Defaults to model_ids.IBM_GRANITE_3_3_8B.
            formatter : input formatter. Defaults to TemplateFormatter in __init__.
            base_url  : url for watson ML deployment. Defaults to env(WATSONX_URL).
            model_options : Global model options to pass to the model. Defaults to None.
            api_key : watsonx API key. Defaults to None.
            project_id : watsonx project ID. Defaults to None.
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
        self._model_id = model_id

        if base_url is None:
            base_url = f"{os.environ.get('WATSONX_URL')}"
        if api_key is None:
            api_key = os.environ.get("WATSONX_API_KEY")
        if project_id is None:
            project_id = os.environ.get("WATSONX_PROJECT_ID")

        _creds = Credentials(url=base_url, api_key=api_key)
        _client = APIClient(credentials=_creds)
        self._model = ModelInference(
            model_id=self._get_watsonx_model_id(),
            api_client=_client,
            credentials=_creds,
            project_id=project_id,
            params=model_options,
            **kwargs,
        )

        # A mapping of common options for this backend mapped to their Mellea ModelOptions equivalent.
        # These are usually values that must be extracted before hand or that are common among backend providers.
        self.to_mellea_model_opts_map_chats = {
            "system": ModelOption.SYSTEM_PROMPT,
            "max_tokens": ModelOption.MAX_NEW_TOKENS,
            "tools": ModelOption.TOOLS,
        }
        # A mapping of Mellea specific ModelOptions to the specific names for this backend.
        # These options should almost always be a subset of those specified in the `to_mellea_model_opts_map`.
        # Usually, values that are intentionally extracted while prepping for the backend generate call
        # will be omitted here so that they will be removed when model_options are processed
        # for the call to the model.
        self.from_mellea_model_opts_map_chats = {
            ModelOption.MAX_NEW_TOKENS: "max_tokens"
        }

        # See notes above.
        self.to_mellea_model_opts_map_completions = {
            "random_seed": ModelOption.SEED,
            "max_new_tokens": ModelOption.MAX_NEW_TOKENS,
        }
        # See notes above.
        self.from_mellea_model_opts_map_completions = {
            ModelOption.SEED: "random_seed",
            ModelOption.MAX_NEW_TOKENS: "max_new_tokens",
        }

    def _get_watsonx_model_id(self) -> str:
        """Gets the watsonx model id from the model_id that was provided in the constructor. Raises AssertionError if the ModelIdentifier does not provide a watsonx_name."""
        watsonx_model_id = (
            self.model_id.watsonx_name
            if isinstance(self.model_id, ModelIdentifier)
            else self.model_id
        )
        assert watsonx_model_id is not None, (
            "model_id is None. This can also happen if the ModelIdentifier has no watsonx name set or this model is not available in watsonx."
        )
        return watsonx_model_id

    def filter_chat_completions_kwargs(self, model_options: dict) -> dict:
        """Filter kwargs to only include valid watsonx chat.completions.create parameters."""
        chat_params = TextChatParameters.get_sample_params().keys()
        return {k: v for k, v in model_options.items() if k in chat_params}

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

        Returns:
            a new dict
        """
        remap_dict = self.to_mellea_model_opts_map_chats
        if not is_chat_context:
            remap_dict = self.to_mellea_model_opts_map_completions

        backend_model_opts = ModelOption.replace_keys(self.model_options, remap_dict)

        if model_options is None:
            return backend_model_opts

        generate_call_model_opts = ModelOption.replace_keys(model_options, remap_dict)
        return ModelOption.merge_model_options(
            backend_model_opts, generate_call_model_opts
        )

    def _make_backend_specific_and_remove(
        self, model_options: dict[str, Any], is_chat_context: bool
    ) -> dict[str, Any]:
        """Maps specified Mellea specific keys to their backend specific version and removes any remaining Mellea keys.

        Args:
            model_options: the model_options for this call

        Returns:
            a new dict
        """
        remap_dict = self.from_mellea_model_opts_map_chats
        if not is_chat_context:
            remap_dict = self.from_mellea_model_opts_map_completions

        backend_specific = ModelOption.replace_keys(model_options, remap_dict)

        if is_chat_context:
            model_opts = self.filter_chat_completions_kwargs(backend_specific)
        else:
            model_opts = ModelOption.remove_special_keys(backend_specific)

        return model_opts

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
            "The watsonx.ai backend only supports chat-like contexts."
        )
        return self.generate_from_chat_context(
            action,
            ctx,
            format=format,
            model_options=model_options,
            generate_logs=generate_logs,
            tool_calls=tool_calls,
        )

    def generate_from_chat_context(
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
        """Generates a new completion from the provided Context using this backend's `Formatter`."""
        model_opts = self._simplify_and_merge(
            model_options, is_chat_context=ctx.is_chat_context
        )

        linearized_context = ctx.render_for_generation()
        assert linearized_context is not None, (
            "Cannot generate from a non-linear context in a FormatterBackend."
        )
        # Convert our linearized context into a sequence of chat messages. Template formatters have a standard way of doing this.
        messages: list[Message] = self.formatter.to_chat_messages(linearized_context)
        # Add the final message.
        match action:
            case ALoraRequirement():
                raise Exception(
                    "The watsonx backend does not support currently support activated LoRAs."
                )
            case _:
                messages.extend(self.formatter.to_chat_messages([action]))

        conversation: list[dict] = []
        system_prompt = model_opts.get(ModelOption.SYSTEM_PROMPT, "")
        if system_prompt != "":
            conversation.append({"role": "system", "content": system_prompt})
        conversation.extend([{"role": m.role, "content": m.content} for m in messages])

        if format is not None:
            model_opts["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": format.__name__,
                    "schema": format.model_json_schema(),
                    "strict": True,
                },
            }
        else:
            model_opts["response_format"] = {"type": "text"}

        # Append tool call information if applicable.
        tools: dict[str, Callable] = {}
        if tool_calls:
            if format:
                FancyLogger.get_logger().warning(
                    f"tool calling is superseded by format; will not call tools for request: {action}"
                )
            else:
                tools = get_tools_from_action(action)
                add_tools_from_model_options(tools, model_opts)

            FancyLogger.get_logger().info(f"Tools for call: {tools.keys()}")

        formatted_tools = convert_tools_to_json(tools)
        chat_response = self._model.chat(
            messages=conversation,
            tools=formatted_tools,
            tool_choice_option=(
                "auto" if formatted_tools and len(formatted_tools) > 0 else "none"
            ),
            params=self._make_backend_specific_and_remove(
                model_opts, is_chat_context=ctx.is_chat_context
            ),
        )

        # If a tool is called, there might not be content in the message.
        response_message = chat_response["choices"][0]["message"].get("content", "")
        result = ModelOutputThunk(
            value=response_message,
            meta={"oai_chat_response": chat_response["choices"][0]},
            tool_calls=self._extract_model_tool_requests(tools, chat_response),
        )

        parsed_result = self.formatter.parse(source_component=action, result=result)

        if generate_logs is not None:
            assert isinstance(generate_logs, list)
            generate_log = GenerateLog()
            generate_log.prompt = conversation
            generate_log.backend = f"watsonx::{self.model_id!s}"
            generate_log.model_options = model_opts
            generate_log.date = datetime.datetime.now()
            generate_log.model_output = chat_response
            generate_log.extra = {
                "format": format,
                # "thinking": thinking,
                "tools_available": tools,
                "tools_called": result.tool_calls,
                "seed": model_opts.get(ModelOption.SEED, None),
            }
            generate_log.result = parsed_result
            generate_log.action = action
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
        """Generates a completion text. Gives the input provided to the model without templating."""
        if format is not None:
            FancyLogger.get_logger().warning(
                "WatsonxAI completion api does not accept response format, ignoring it for this request."
            )

        model_opts = self._simplify_and_merge(model_options, is_chat_context=False)

        prompts = [self.formatter.print(action) for action in actions]

        responses = self._model.generate(
            prompt=prompts,
            params=self._make_backend_specific_and_remove(
                model_opts, is_chat_context=False
            ),
        )

        results = [
            ModelOutputThunk(
                value=response["results"][0]["generated_text"],
                meta={"oai_completion_response": response["results"][0]},
            )
            for response in responses
        ]

        for i, result in enumerate(results):
            self.formatter.parse(actions[i], result)

        if generate_logs is not None:
            assert isinstance(generate_logs, list)
            date = datetime.datetime.now()

            for i in range(len(prompts)):
                generate_log = GenerateLog()
                generate_log.prompt = prompts[i]
                generate_log.backend = f"watsonx::{self.model_id!s}"
                generate_log.model_options = model_opts
                generate_log.date = date
                generate_log.model_output = responses
                generate_log.extra = {
                    "format": format,
                    "seed": model_opts.get(ModelOption.SEED, None),
                }
                generate_log.action = actions[i]
                generate_log.result = results[i]
                generate_logs.append(generate_log)

        return results

    def _extract_model_tool_requests(
        self, tools: dict[str, Callable], chat_response: dict
    ) -> dict[str, ModelToolCall] | None:
        model_tool_calls: dict[str, ModelToolCall] = {}
        for tool_call in chat_response["choices"][0]["message"].get("tool_calls", []):
            tool_name = tool_call["function"]["name"]
            tool_args = tool_call["function"]["arguments"]

            func = tools.get(tool_name)
            if func is None:
                FancyLogger.get_logger().warning(
                    f"model attempted to call a non-existing function: {tool_name}"
                )
                continue  # skip this function if we can't find it.

            # Watsonx returns the args as a string. Parse it here.
            args = json.loads(tool_args)
            model_tool_calls[tool_name] = ModelToolCall(tool_name, func, args)

        if len(model_tool_calls) > 0:
            return model_tool_calls
        return None
