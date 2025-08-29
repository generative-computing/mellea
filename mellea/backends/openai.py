"""A generic OpenAI compatible backend that wraps around the openai python sdk."""

import abc
import datetime
import inspect
import json
import re
from collections.abc import Callable
from enum import Enum
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

import openai
import requests
from huggingface_hub import snapshot_download
from openai.types.chat import ChatCompletion
from openai.types.completion import Completion, CompletionUsage

import mellea.backends.model_ids as model_ids
from mellea.backends import BaseModelSubclass
from mellea.backends.aloras import Alora, AloraBackendMixin
from mellea.backends.formatter import Formatter, FormatterBackend, TemplateFormatter
from mellea.backends.model_ids import ModelIdentifier
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
from mellea.stdlib.requirement import ALoraRequirement, LLMaJRequirement, Requirement

if TYPE_CHECKING:
    from transformers.tokenization_utils import PreTrainedTokenizer

openai_ollama_batching_error = "json: cannot unmarshal array into Go struct field CompletionRequest.prompt of type string"


class _ServerType(Enum):
    LOCALHOST = 1
    OPENAI = 2


def _server_type(url: str) -> _ServerType | None:
    try:
        parsed = urlparse(url)
        hostname = parsed.hostname
        if hostname in ("localhost", "127.0.0.1", "::1"):
            return _ServerType.LOCALHOST
        elif hostname == "api.openai.com":
            return _ServerType.OPENAI
    except Exception as e:
        print(f"Error parsing URL: {e}")
    return None


class OpenAIBackend(FormatterBackend, AloraBackendMixin):
    """A generic OpenAI compatible backend."""

    def __init__(
        self,
        model_id: str | ModelIdentifier = model_ids.IBM_GRANITE_3_3_8B,
        formatter: Formatter | None = None,
        base_url: str | None = None,
        model_options: dict | None = None,
        *,
        default_to_constraint_checking_alora: bool = True,
        api_key: str | None = None,
        **kwargs,
    ):
        """Initialize and OpenAI compatible backend. For any additional kwargs that you need to pass the the client, pass them as a part of **kwargs.

        Args:
            model_id : A generic model identifier or OpenAI compatible string. Defaults to model_ids.IBM_GRANITE_3_3_8B.
            formatter: A custom formatter based on backend.If None, defaults to TemplateFormatter
            base_url : Base url for LLM API. Defaults to None.
            model_options : Generation options to pass to the LLM. Defaults to None.
            default_to_constraint_checking_alora: If set to False then aloras will be deactivated. This is primarily for performance benchmarking and debugging.
            api_key : API key for generation. Defaults to None.
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

        # A mapping of common options for this backend mapped to their Mellea ModelOptions equivalent.
        # These are usually values that must be extracted before hand or that are common among backend providers.
        # OpenAI has some deprecated parameters. Those map to the same mellea parameter, but
        # users should only be specifying a single one in their request.
        self.to_mellea_model_opts_map_chats = {
            "system": ModelOption.SYSTEM_PROMPT,
            "reasoning_effort": ModelOption.THINKING,
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
        self.from_mellea_model_opts_map_chats = {
            ModelOption.SEED: "seed",
            ModelOption.MAX_NEW_TOKENS: "max_completion_tokens",
            ModelOption.THINKING: "reasoning_effort",
        }

        # See notes above.
        self.to_mellea_model_opts_map_completions = {
            "seed": ModelOption.SEED,
            "max_tokens": ModelOption.MAX_NEW_TOKENS,
        }
        # See notes above.
        self.from_mellea_model_opts_map_completions = {
            ModelOption.SEED: "seed",
            ModelOption.MAX_NEW_TOKENS: "max_tokens",
        }

        self.default_to_constraint_checking_alora = default_to_constraint_checking_alora

        self._model_id = model_id
        match model_id:
            case str():
                self._hf_model_id = model_id
            case ModelIdentifier():
                assert model_id.hf_model_name is not None, (
                    "model_id is None. This can also happen if the ModelIdentifier has no hf_model_id name set."
                )
                self._hf_model_id = model_id.hf_model_name

        if base_url is None:
            self._base_url = "http://localhost:11434/v1"  # ollama
        else:
            self._base_url = base_url
        if api_key is None:
            self._api_key = "ollama"
        else:
            self._api_key = api_key

        openai_client_kwargs = self.filter_openai_client_kwargs(**kwargs)

        self._client = openai.OpenAI(  # type: ignore
            api_key=self._api_key, base_url=self._base_url, **openai_client_kwargs
        )
        # ALoras that have been loaded for this model.
        self._aloras: dict[str, OpenAIAlora] = {}

    @staticmethod
    def filter_openai_client_kwargs(**kwargs) -> dict:
        """Filter kwargs to only include valid OpenAI client parameters."""
        openai_params = set(inspect.signature(openai.OpenAI.__init__).parameters.keys())  # type: ignore
        openai_params.discard("self")  # Remove 'self' parameter
        return {k: v for k, v in kwargs.items() if k in openai_params}

    def filter_chat_completions_kwargs(self, model_options: dict) -> dict:
        """Filter kwargs to only include valid OpenAI chat.completions.create parameters.

        https://platform.openai.com/docs/api-reference/chat/create
        """
        from openai.resources.chat.completions import Completions

        chat_params = set(inspect.signature(Completions.create).parameters.keys())
        chat_params.discard("self")
        return {k: v for k, v in model_options.items() if k in chat_params}

    def filter_completions_kwargs(self, model_options: dict) -> dict:
        """Filter kwargs to only include valid OpenAI completions.create parameters.

        https://platform.openai.com/docs/api-reference/completions
        """
        from openai.resources.completions import Completions

        completions_params = set(
            inspect.signature(Completions.create).parameters.keys()
        )
        completions_params.discard("self")  # Remove 'self' parameter
        return {k: v for k, v in model_options.items() if k in completions_params}

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

        # OpenAI Backend has specific filtering functionality.
        if is_chat_context:
            model_opts = self.filter_chat_completions_kwargs(backend_specific)
        else:
            model_opts = self.filter_completions_kwargs(backend_specific)

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
            "The Openai backend only supports chat-like contexts."
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
        if issubclass(type(action), Requirement):
            # The general rule is that we reroute to the alora if it exists.
            reroute_to_alora = self.get_alora("constraint") is not None
            # However, there are some exceptions:
            if not self.default_to_constraint_checking_alora:
                reroute_to_alora = False
            if issubclass(type(action), LLMaJRequirement):
                reroute_to_alora = False
            if issubclass(type(action), ALoraRequirement):
                reroute_to_alora = True
            if reroute_to_alora:
                return self._generate_from_chat_context_alora(
                    action, ctx, format=format, model_options=model_options
                )

        return self._generate_from_chat_context_standard(
            action,
            ctx,
            format=format,
            model_options=model_options,
            generate_logs=generate_logs,
            tool_calls=tool_calls,
        )

    def _generate_from_chat_context_alora(
        self,
        action: Component | CBlock,
        ctx: Context,
        *,
        format: type[BaseModelSubclass]
        | None = None,  # Type[BaseModelSubclass] is a class object of a subclass of BaseModel
        model_options: dict | None = None,
        generate_logs: list[GenerateLog] | None = None,
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
        linearized_ctx = ctx.render_for_generation()
        assert linearized_ctx is not None and len(linearized_ctx) > 1
        msgs = self.formatter.to_chat_messages(linearized_ctx)
        user_message, assistant_message = msgs[-2].content, msgs[-1].content
        assert alora_for_this_request is not None
        assert type(user_message) is str
        assert type(assistant_message) is str
        assert format is None, "Structured outputs are not supported by ALoRAs."
        alora_output = alora_for_this_request.generate_using_strings(
            input=user_message,
            response=assistant_message,
            constraint=action.description,  # type: ignore
        )
        return self.formatter.parse(
            action,
            ModelOutputThunk(
                alora_output, meta={"alora_name": alora_for_this_request.name}
            ),
        )

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
        # NOTE: Currently, the `thinking` param is going to be set to "medium" if `thinking` is True, else it is None.
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
                    "The OpenAI backend does not support currently support activated LoRAs."
                )
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

        # Append tool call information if applicable.
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

        thinking = model_opts.get(ModelOption.THINKING, None)
        if type(thinking) is bool and thinking:
            # OpenAI uses strings for its reasoning levels.
            thinking = "medium"

        formatted_tools = convert_tools_to_json(tools)
        chat_response: ChatCompletion = self._client.chat.completions.create(
            model=self._hf_model_id,
            messages=conversation,  # type: ignore
            reasoning_effort=thinking,  # type: ignore
            response_format=response_format,  # type: ignore
            tool_choice=(
                "auto" if formatted_tools and len(formatted_tools) > 0 else "none"
            ),
            tools=formatted_tools,  # type: ignore
            # parallel_tool_calls=False, # We only support calling one tool per turn. But we do the choosing on our side so we leave this False.
            **self._make_backend_specific_and_remove(
                model_opts, is_chat_context=ctx.is_chat_context
            ),
        )  # type: ignore

        result = ModelOutputThunk(
            value=chat_response.choices[0].message.content,
            meta={
                "oai_chat_response": chat_response.choices[0].model_dump()
            },  # NOTE: Using model dump here to comply with `TemplateFormatter`
            tool_calls=self._extract_model_tool_requests(tools, chat_response),
        )

        parsed_result = self.formatter.parse(source_component=action, result=result)

        if generate_logs is not None:
            assert isinstance(generate_logs, list)
            generate_log = GenerateLog()
            generate_log.prompt = conversation
            generate_log.backend = f"openai::{self.model_id!s}"
            generate_log.model_options = model_opts
            generate_log.date = datetime.datetime.now()
            generate_log.model_output = chat_response
            generate_log.extra = {
                "format": format,
                "thinking": thinking,
                "tools_available": tools,
                "tools_called": result.tool_calls,
                "seed": model_opts.get("seed", None),
            }
            generate_log.action = action
            generate_log.result = parsed_result
            generate_logs.append(generate_log)

        return parsed_result

    def generate_with_budget_forcing(
        self,
        action: CBlock,
        *,
        think_max_tokens: int = 4096,
        answer_max_tokens: int | None = None,
        start_think_token: str = "<think>",
        end_think_token: str = "</think>",
        begin_response_token: str = "",
        end_response_token: str = "",
        think_wait_suffix: str = "",
        answer_suffix: str = "The final answer is:",
        answer_regex: str = "boxed",
        model_options: dict | None = None,
    ) -> tuple[str, int]:
        """Generate with budget forcing using the completions APIs. This relies on raw autocompletion and assumes the model's output is structured in the following form: '<think> ... </think> summary answer'
        The budget forcing method is proposed in the paper: https://arxiv.org/abs/2501.19393
        This implementation tries to follow the key outlines in the paper while ensuring stable and fail-safe operation.
        This is performed via multi-step generation. The model will be called multiple times until requirements are met, in other words, the response will be assembled conditionally.

        Args:
            think_max_tokens: Budget in number of tokens allocated for the think block
            answer_max_tokens: Budget in number of tokens allocated for the summary and answer block, None indicates generating till EoS
            start_think_token: String indicating start of think block, default <think>
            end_think_token: String indicating end of think block, default </think>
            begin_response_token: Used by certain models, string indicating start of response block, e.g. "<response>", default None
            end_response_token: Used by certain models, string indicating end of response block, e.g. "</response>", default None
            think_wait_suffix: String to append to force continued thinking, e.g. "\nWait" if set to None we will not force additional thinking. Use None for upper-bound budget case
            answer_suffix: String to append to force a final answer
            answer_regex: Answer regex which indicates an answer is generated

        Assumptions:
            -  The chat template is applied on prompt, with think mode enabled
            -  Model is think mode activated
            -  enabling prefix-caching improves performance

        Limitations:
            -  Does not support batching
        """

        model_opts = self._simplify_and_merge(model_options, is_chat_context=False)

        responses = []
        prompt = self.formatter.print(action)
        if start_think_token:
            prompt += start_think_token
            responses.append(start_think_token)

        backend_opts = self._make_backend_specific_and_remove(
            model_opts, is_chat_context=False
        )
        # Generate thinking portion
        # backend_opts["echo"] = True
        # backend_opts["logprobs"] = 1
        backend_opts["n"] = 1
        rem_toks = think_max_tokens
        gen_tok_count = 0
        curr_prompt = prompt
        min_step_len = 10  # minimum character length of step to be considered valid

        # think block indefinite multi-step operation to satisfy user's budget
        while True:
            if rem_toks <= 0:  # zero-think case
                break

            if rem_toks <= min_step_len:  # minimum step length reached
                break

            backend_opts["max_tokens"] = rem_toks
            try:
                completion_response = self._client.completions.create(
                    model=self._hf_model_id, prompt=curr_prompt, **backend_opts
                )  # type: ignore
            except openai.BadRequestError as e:
                if openai_ollama_batching_error in e.message:
                    FancyLogger.get_logger().error(
                        "If you are trying to call `OpenAIBackend.generate_with_budget_forcing while targeting an ollama server, "
                        "your requests will fail since ollama doesn't support batching requests."
                    )
                raise e

            # Necessary for type checker.
            assert isinstance(completion_response.usage, CompletionUsage)
            gen_tok_count += completion_response.usage.completion_tokens
            rem_toks = think_max_tokens - gen_tok_count
            response = completion_response.choices[0].text

            if think_wait_suffix == "":
                # non-strict budget form
                responses.append(response)
                break

            if rem_toks <= 0:
                responses.append(response)
                break

            else:
                if end_think_token:
                    step = response.split(end_think_token)[0]
                # model fails to produce thoughts, let's exit
                if len(step.strip()) <= min_step_len:
                    responses.append(response)
                    break

                # request more steps
                step = f"{step} {think_wait_suffix}"
                responses.append(step)
                curr_prompt += step

        response = "".join(responses)
        if answer_regex is None or answer_suffix is None:
            return response, gen_tok_count

        # Now get a final answer if we need to
        # TODO: Here we check if a final answer exists, technically we should check for an answer outside
        # The think block, but we will use relaxed requirement of finding any answer in the model's response.
        # Consider a strict structural approach in the future.
        # e.g.
        # answer_blk = response.split(end_think_token)[-1]

        # Check if answer in response
        matches = re.findall(answer_regex, response, re.DOTALL)
        if len(matches) > 0:
            return response, gen_tok_count

        # Answer is not in response, let's force an answer
        if end_think_token and end_think_token not in response:
            response += f" {end_think_token}"

        if begin_response_token and begin_response_token not in response:
            response += f" {begin_response_token}"

        if answer_suffix:
            response += f" {answer_suffix}"

        # update original prompt with assembled response
        prompt += response
        if answer_max_tokens is not None:
            backend_opts["max_tokens"] = answer_max_tokens

        else:
            backend_opts.pop("max_tokens", None)  # generate unconditionally

        try:
            completion_response = self._client.completions.create(
                model=self._hf_model_id, prompt=prompt, **backend_opts
            )  # type: ignore
        except openai.BadRequestError as e:
            if openai_ollama_batching_error in e.message:
                FancyLogger.get_logger().error(
                    "If you are trying to call `OpenAIBackend.generate_with_budget_forcing while targeting an ollama server, "
                    "your requests will fail since ollama doesn't support batching requests."
                )
            raise e

        # Necessary for type checker.
        assert isinstance(completion_response.usage, CompletionUsage)
        response += completion_response.choices[0].text
        gen_tok_count += completion_response.usage.completion_tokens
        return response, gen_tok_count

    def _generate_from_raw(
        self,
        actions: list[Component | CBlock],
        *,
        format: type[BaseModelSubclass] | None = None,
        model_options: dict | None = None,
        generate_logs: list[GenerateLog] | None = None,
    ) -> list[ModelOutputThunk]:
        """Generate using the completions api. Gives the input provided to the model without templating."""
        extra_body = {}
        if format is not None:
            FancyLogger.get_logger().warning(
                "The official OpenAI completion api does not accept response format / structured decoding; "
                "it will be passed as an extra arg."
            )

            # Some versions (like vllm's version) of the OpenAI API support structured decoding for completions requests.
            extra_body["guided_json"] = format.model_json_schema()

        model_opts = self._simplify_and_merge(model_options, is_chat_context=False)

        prompts = [self.formatter.print(action) for action in actions]

        try:
            completion_response: Completion = self._client.completions.create(
                model=self._hf_model_id,
                prompt=prompts,
                extra_body=extra_body,
                **self._make_backend_specific_and_remove(
                    model_opts, is_chat_context=False
                ),
            )  # type: ignore
        except openai.BadRequestError as e:
            if openai_ollama_batching_error in e.message:
                FancyLogger.get_logger().error(
                    "If you are trying to call `OpenAIBackend._generate_from_raw while targeting an ollama server, "
                    "your requests will fail since ollama doesn't support batching requests."
                )
            raise e

        # Necessary for type checker.
        assert isinstance(completion_response, Completion)

        results = [
            ModelOutputThunk(
                value=response.text,
                meta={"oai_completion_response": response.model_dump()},
            )
            for response in completion_response.choices
        ]

        for i, result in enumerate(results):
            self.formatter.parse(actions[i], result)

        if generate_logs is not None:
            assert isinstance(generate_logs, list)
            date = datetime.datetime.now()

            for i in range(len(prompts)):
                generate_log = GenerateLog()
                generate_log.prompt = prompts[i]
                generate_log.backend = f"openai::{self.model_id!s}"
                generate_log.model_options = model_opts
                generate_log.date = date
                generate_log.model_output = completion_response
                generate_log.extra = {"seed": model_opts.get("seed", None)}
                generate_log.action = actions[i]
                generate_log.result = results[i]
                generate_logs.append(generate_log)

        return results

    def _extract_model_tool_requests(
        self, tools: dict[str, Callable], chat_response: ChatCompletion
    ) -> dict[str, ModelToolCall] | None:
        model_tool_calls: dict[str, ModelToolCall] = {}
        calls = chat_response.choices[0].message.tool_calls
        if calls:
            for tool_call in calls:
                tool_name = tool_call.function.name  # type: ignore
                tool_args = tool_call.function.arguments  # type: ignore

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

    def add_alora(self, alora: "OpenAIAlora"):
        """Loads an ALora for this backend.

        Args:
            alora (str): identifier for the ALora adapter
        """
        assert issubclass(alora.__class__, OpenAIAlora), (
            f"cannot add an ALora of type {alora.__class__} to model; must inherit from {OpenAIAlora.__class__}"
        )
        assert alora._backend == self, "Cannot load an ALora into the wrong backend."

        if self.get_alora(alora.name) is not None:
            FancyLogger.get_logger().warning(
                f"Client code attempted to add {alora.name} but {alora.name} was already added to {self.__class__}. The backend is refusing to do this, because ALora loading is not idempotent."
            )
            return None

        assert _server_type(self._base_url) == _ServerType.LOCALHOST, (
            "alora is supported only for locally running vllm instances"
        )

        snapshot_path = snapshot_download(alora.path)

        # https://docs.vllm.ai/en/stable/features/lora.html#using-api-endpoints
        # curl -X POST http://localhost:8000/v1/load_lora_adapter \
        #     -H "Content-Type: application/json" \
        #     -d '{
        #     "lora_name": "sql_adapter",
        #     "lora_path": "/path/to/sql-lora-adapter"
        #     }'

        url = f"{self._base_url}/load_lora_adapter"
        response = requests.post(
            url,
            json={"lora_name": alora.name, "lora_path": snapshot_path},
            headers={"Content-Type": "application/json"},
        )

        match response.status_code:
            case 200:
                FancyLogger.get_logger().info(
                    f"{url}: status {response.status_code} {response.text}"
                )
                self._aloras[alora.name] = alora
            case _:
                FancyLogger.get_logger().error(
                    f"{url}: status {response.status_code} {response.text}"
                )

        self._aloras[alora.name] = alora

        return None

    def get_alora(self, alora_name: str) -> Alora | None:
        """Returns the ALora by name, or None if that ALora isn't loaded."""
        return self._aloras.get(alora_name)

    def get_aloras(self) -> list[Alora]:
        """Returns a list of all loaded ALora adapters."""
        return list(self._aloras.values())

    def apply_chat_template(self, chat: list[dict[str, str]]):
        """Apply the chat template for the model, if such a model is available (e.g., when it can deduce the huggingface model id)."""
        from transformers import AutoTokenizer

        if not hasattr(self, "_tokenizer"):
            match _server_type(self._base_url):
                case _ServerType.LOCALHOST:
                    self._tokenizer: "PreTrainedTokenizer" = (  # noqa: UP037
                        AutoTokenizer.from_pretrained(self._hf_model_id)
                    )
                case _ServerType.OPENAI:
                    raise Exception(
                        "apply_chat_template is called while targeting a server at openai.com. "
                        "This is not supported --- openai.com does not support Activated Lora. "
                        "Use a locally served vllm instance. "
                    )

        return self._tokenizer.apply_chat_template(chat, tokenize=False)


class OpenAIAlora(Alora, abc.ABC):
    """ALoras that work with OpenAI backend."""

    def __init__(
        self, name: str, path: str, generation_prompt: str, backend: OpenAIBackend
    ):
        """Initialize an ALora that should work with OpenAI backends that support ALoras.

        Args:
            name (str): An arbitrary name/label to assign to an ALora. This is irrelevant from the alora's (huggingface) model id.
            path (str): A local path to ALora's weights or a Huggingface model_id to an ALora.
            generation_prompt (str): A prompt used to "activate" the Lora. This string goes between the pre-activation context and the aLora generate call. This needs to be provided by the entity that trained the ALora.
            backend (OpenAIBackend): Mained as a pointer to the backend to which this this ALora is attached.
        """
        super().__init__(name)
        self.path = path
        self._backend = backend
        self._generation_prompt = generation_prompt
