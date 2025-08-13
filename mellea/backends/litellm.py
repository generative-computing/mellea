"""A generic OpenAI compatible backend that wraps around the openai python sdk."""

import datetime
import json
from collections.abc import Callable

import litellm

import mellea.backends.model_ids as model_ids
from mellea.backends import BaseModelSubclass
from mellea.backends.formatter import Formatter, FormatterBackend, TemplateFormatter
from mellea.backends.tools import convert_tools_to_json, get_tools_from_action
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
from mellea.stdlib.requirement import ALoraRequirement, LLMaJRequirement, Requirement


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

    def _simplify_and_merge(self, mo: dict) -> dict:
        mo_safe = {} if mo is None else mo.copy()
        mo_merged = ModelOption.merge_model_options(self.model_options, mo_safe)

        # map to valid litellm names
        mo_mapping = {
            ModelOption.TOOLS: "tools",
            ModelOption.MAX_NEW_TOKENS: "max_completion_tokens",
            ModelOption.SEED: "seed",
            ModelOption.THINKING: "thinking",
        }
        mo_res = ModelOption.replace_keys(mo_merged, mo_mapping)
        mo_res = ModelOption.remove_special_keys(mo_res)

        supported_params = litellm.get_supported_openai_params(self._model_id)
        assert supported_params is not None
        for k in list(mo_res.keys()):
            if k not in supported_params:
                del mo_res[k]
                FancyLogger.get_logger().warn(
                    f"Skipping '{k}' -- Model-Option not supported by {self.model_id}."
                )

        return mo_res

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
        model_options = {} if model_options is None else model_options
        model_opts = self._simplify_and_merge(model_options)
        linearized_context = ctx.linearize()
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
        system_prompt = model_options.get(ModelOption.SYSTEM_PROMPT, "")
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

        # Append tool call information if applicable.
        tools = self._extract_tools(action, format, model_opts, tool_calls)
        formatted_tools = convert_tools_to_json(tools) if len(tools) > 0 else None

        chat_response: litellm.ModelResponse = litellm.completion(
            model=self._model_id,
            messages=conversation,
            tools=formatted_tools,
            response_format=response_format,
            **model_opts,
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
            generate_log.model_options = model_opts
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
    def _extract_tools(action, format, model_opts, tool_calls):
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

                model_options_tools = model_opts.get(ModelOption.TOOLS, None)
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
        # extra_body = {}
        # if format is not None:
        #     FancyLogger.get_logger().warning(
        #         "The official OpenAI completion api does not accept response format / structured decoding; "
        #         "it will be passed as an extra arg."
        #     )
        #
        #     # Some versions (like vllm's version) of the OpenAI API support structured decoding for completions requests.
        #     extra_body["guided_json"] = format.model_json_schema()
        #
        # model_opts = self._simplify_and_merge(model_options, is_chat_context=False)
        #
        # prompts = [self.formatter.print(action) for action in actions]
        #
        # try:
        #     completion_response: Completion = self._client.completions.create(
        #         model=self._hf_model_id,
        #         prompt=prompts,
        #         extra_body=extra_body,
        #         **self._make_backend_specific_and_remove(
        #             model_opts, is_chat_context=False
        #         ),
        #     )  # type: ignore
        # except openai.BadRequestError as e:
        #     if openai_ollama_batching_error in e.message:
        #         FancyLogger.get_logger().error(
        #             "If you are trying to call `OpenAIBackend._generate_from_raw while targeting an ollama server, "
        #             "your requests will fail since ollama doesn't support batching requests."
        #         )
        #     raise e
        #
        # # Necessary for type checker.
        # assert isinstance(completion_response, Completion)
        #
        # results = [
        #     ModelOutputThunk(
        #         value=response.text,
        #         meta={"oai_completion_response": response.model_dump()},
        #     )
        #     for response in completion_response.choices
        # ]
        #
        # for i, result in enumerate(results):
        #     self.formatter.parse(actions[i], result)
        #
        # if generate_logs is not None:
        #     assert isinstance(generate_logs, list)
        #     date = datetime.datetime.now()
        #
        #     for i in range(len(prompts)):
        #         generate_log = GenerateLog()
        #         generate_log.prompt = prompts[i]
        #         generate_log.backend = f"openai::{self.model_id!s}"
        #         generate_log.model_options = model_opts
        #         generate_log.date = date
        #         generate_log.model_output = completion_response
        #         generate_log.extra = {"seed": model_opts.get("seed", None)}
        #         generate_log.action = actions[i]
        #         generate_log.result = results[i]
        #         generate_logs.append(generate_log)
        #
        # return results

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
