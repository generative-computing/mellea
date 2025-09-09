"""A model backend wrapping the Ollama Python SDK."""

import asyncio
import datetime
from collections.abc import Callable
from typing import Any

import ollama
from tqdm import tqdm

import mellea.backends.model_ids as model_ids
from mellea.backends import BaseModelSubclass
from mellea.backends.formatter import Formatter, FormatterBackend, TemplateFormatter
from mellea.backends.model_ids import ModelIdentifier
from mellea.backends.tools import (
    add_tools_from_context_actions,
    add_tools_from_model_options,
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
from mellea.stdlib.requirement import ALoraRequirement


class OllamaModelBackend(FormatterBackend):
    """A model that uses the Ollama Python SDK for local inference."""

    def __init__(
        self,
        model_id: str | ModelIdentifier = model_ids.IBM_GRANITE_3_3_8B,
        formatter: Formatter | None = None,
        base_url: str | None = None,
        model_options: dict | None = None,
    ):
        """Initializes an ollama backend for local models.

        WARNING: may use up a lot of your machine's memory.

        Args:
            model_id (str | ModelIdentifier): Ollama model ID. If ModelIdentifier, then an `ollama_name` must be provided by that ModelIdentifier.
            base_url (str): Endpoint that is serving the model API; defaults to env(OLLAMA_HOST) or `http://localhost:11434`
            model_options (dict): Ollama model options
            formatter (Formatter): formatter for creating input
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
        # Run the ollama model id accessor early, so that an Assertion fails immediately if we cannot find an ollama model id for the provided ModelIdentifier.
        self._get_ollama_model_id()

        # Setup the client and ensure that we have the model available.
        self._client = ollama.Client(base_url)
        self._async_client = ollama.AsyncClient(base_url)
        if not self._check_ollama_server():
            err = f"could not create OllamaModelBackend: ollama server not running at {base_url}"
            FancyLogger.get_logger().error(err)
            raise Exception(err)
        if not self._pull_ollama_model():
            err = f"could not create OllamaModelBackend: {self._get_ollama_model_id()} could not be pulled from ollama library"
            FancyLogger.get_logger().error(err)
            raise Exception(err)

        # A mapping of common options for this backend mapped to their Mellea ModelOptions equivalent.
        # These are usually values that must be extracted before hand or that are common among backend providers.
        self.to_mellea_model_opts_map = {
            "system": ModelOption.SYSTEM_PROMPT,
            "think": ModelOption.THINKING,
            "num_ctx": ModelOption.CONTEXT_WINDOW,
            "num_predict": ModelOption.MAX_NEW_TOKENS,
            "seed": ModelOption.SEED,
            "tools": ModelOption.TOOLS,
        }

        # A mapping of Mellea specific ModelOptions to the specific names for this backend.
        # These options should almost always be a subset of those specified in the `to_mellea_model_opts_map`.
        # Usually, values that are intentionally extracted while prepping for the backend generate call
        # will be omitted here so that they will be removed when model_options are processed
        # for the call to the model.
        self.from_mellea_model_opts_map = {
            ModelOption.CONTEXT_WINDOW: "num_ctx",
            ModelOption.MAX_NEW_TOKENS: "num_predict",
            ModelOption.SEED: "seed",
        }

    def _get_ollama_model_id(self) -> str:
        """Gets the ollama model id from the model_id that was provided in the constructor. Raises AssertionError is the ModelIdentifier does not provide an ollama_name."""
        ollama_model_id = (
            self.model_id.ollama_name
            if isinstance(self.model_id, ModelIdentifier)
            else self.model_id
        )
        assert ollama_model_id is not None, (
            "model_id is None. This can also happen if the ModelIdentifier has no ollama name set or this model is not available in ollama."
        )
        return ollama_model_id

    def _check_ollama_server(self) -> bool:
        """Requests generic info about the Ollama server to ensure it's running."""
        try:
            self._client.ps()
        except ConnectionError:
            return False
        return True

    def is_model_available(self, model_name):
        """Checks if a specific Ollama model is available locally.

        Args:
          model_name: The name of the model to check for (e.g., "llama2").

        Returns:
          True if the model is available, False otherwise.
        """
        try:
            models = self._client.list()
            for model in models["models"]:
                if model.model.startswith(model_name):
                    return True
            return False
        except Exception as e:
            print(f"An error occurred: {e}")
            return False

    def _pull_ollama_model(self) -> bool:
        """Either gets the cached ollama model or else attempts to pull the provided model from Ollama. Raises an exception of the model cannot be pulled.

        This code was generated by ChatGPT.
        """
        # shortcut --  if model is in list-- don't try to pull
        if self.is_model_available(self._get_ollama_model_id()):
            return True

        try:
            FancyLogger.get_logger().debug(
                f"Loading/Pulling model from Ollama: {self._get_ollama_model_id()}"
            )
            stream = self._client.pull(self._get_ollama_model_id(), stream=True)
            progress_bars = {}
            for update in stream:
                status = update.status
                digest = update.digest
                completed = update.completed or 0
                total = update.total or 0
                # Only track digests with a known total
                if digest and total > 0:
                    if digest not in progress_bars:
                        progress_bars[digest] = tqdm(
                            total=total,
                            desc=f"{status} {digest[:12]}",
                            unit="B",
                            unit_scale=True,
                            leave=False,
                        )
                    pbar = progress_bars[digest]
                    delta = completed - pbar.n
                    if delta > 0:
                        pbar.update(delta)
            # Close all progress bars
            for pbar in progress_bars.values():
                pbar.close()
            return True
        except ollama.ResponseError:
            return False

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

        Args:
            model_options: the model_options for this call

        Returns:
            a new dict
        """
        backend_specific = ModelOption.replace_keys(
            model_options, self.from_mellea_model_opts_map
        )
        return ModelOption.remove_special_keys(backend_specific)

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
        assert ctx.is_chat_context, (
            "The ollama backend only supports chat-like contexts."
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
        format: type[BaseModelSubclass] | None = None,
        model_options: dict | None = None,
        generate_logs: list[GenerateLog] | None = None,
        tool_calls: bool = False,
    ) -> ModelOutputThunk:
        """Generates a new completion from the provided Context using this backend's `Formatter`.

        This implementation treats the `Context` as a chat history, and uses the  `ollama.Client.chat()` interface to generate a completion.
        This will not always work, because sometimes we want to use non-chat models.
        """
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
                raise Exception(
                    "The ollama backend does not support currently support activated LoRAs."
                )
            case _:
                messages.extend(self.formatter.to_chat_messages([action]))
        # construct the conversation from our messages, adding a system prompt at the first message if one was provided.
        conversation: list[dict] = []
        # We use system prompt None/empty-string semantics in a way that is consistent with huggingface and other libraries.
        # If the system prompt is None, the the default system prompt gets used.
        system_prompt = model_opts.get(ModelOption.SYSTEM_PROMPT, "")
        if system_prompt != "":
            conversation.append({"role": "system", "content": system_prompt})

        conversation.extend(
            [
                {"role": m.role, "content": m.content, "images": m.images}
                for m in messages
            ]
        )

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

        # Generate a chat response from ollama, using the chat messages.
        chat_response: ollama.ChatResponse = self._client.chat(
            model=self._get_ollama_model_id(),
            messages=conversation,
            tools=list(tools.values()),
            think=model_opts.get(ModelOption.THINKING, None),
            options=self._make_backend_specific_and_remove(model_opts),
            stream=False,
            format=format.model_json_schema() if format is not None else None,
        )  # type: ignore

        result = ModelOutputThunk(
            value=chat_response.message.content,  # For an ollama tool call, content will be an empty string.
            meta={"chat_response": chat_response},
            tool_calls=self._extract_model_tool_requests(tools, chat_response),
        )

        formatted_result = self.formatter.parse(action, result)

        if generate_logs is not None:
            # noinspection DuplicatedCode
            assert isinstance(generate_logs, list)
            generate_log = GenerateLog()
            generate_log.prompt = conversation
            generate_log.backend = f"ollama::{self.model_id!s}"
            generate_log.model_options = model_opts
            generate_log.date = datetime.datetime.now()
            generate_log.model_output = chat_response
            generate_log.extra = {
                "format": format,
                "thinking": model_opts.get(ModelOption.THINKING, None),
                "tools_available": tools,
                "tools_called": result.tool_calls,
                "seed": model_opts.get(ModelOption.SEED, None),
            }
            generate_log.action = action
            generate_log.result = formatted_result
            generate_logs.append(generate_log)

        return formatted_result

    def _generate_from_raw(
        self,
        actions: list[Component | CBlock],
        *,
        format: type[BaseModelSubclass] | None = None,
        model_options: dict | None = None,
        generate_logs: list[GenerateLog] | None = None,
    ) -> list[ModelOutputThunk]:
        """Generate using the generate api. Gives the input provided to the model without templating."""
        if len(actions) > 1:
            FancyLogger.get_logger().info(
                "Ollama doesn't support batching; will attempt to process concurrently."
            )

        model_opts = self._simplify_and_merge(model_options)

        # Ollama doesn't support "batching". There's some ability for concurrency. Use that here.
        # See https://github.com/ollama/ollama/blob/main/docs/faq.md#how-does-ollama-handle-concurrent-requests.
        prompts = [self.formatter.print(action) for action in actions]

        async def get_response(coroutines):
            responses = await asyncio.gather(*coroutines, return_exceptions=True)
            return responses

        # Run async so that we can make use of Ollama's concurrency.
        coroutines = []
        for prompt in prompts:
            co = self._async_client.generate(
                model=self._get_ollama_model_id(),
                prompt=prompt,
                raw=True,
                think=model_opts.get(ModelOption.THINKING, None),
                format=format.model_json_schema() if format is not None else None,
                options=self._make_backend_specific_and_remove(model_opts),
            )
            coroutines.append(co)

        # Revisit this once we start using async elsewhere. Only one asyncio event
        # loop can be running in a given thread.
        responses: list[ollama.GenerateResponse | BaseException] = asyncio.run(
            get_response(coroutines)
        )

        results = []
        date = datetime.datetime.now()
        for i, response in enumerate(responses):
            result = None
            error = None
            if isinstance(response, BaseException):
                result = ModelOutputThunk(value="")
                error = response
            else:
                result = ModelOutputThunk(
                    value=response.response,
                    meta={"generate_response": response.model_dump()},
                )

            self.formatter.parse(actions[i], result)
            results.append(result)

            if generate_logs is not None:
                # noinspection DuplicatedCode
                assert isinstance(generate_logs, list)
                generate_log = GenerateLog()
                generate_log.prompt = prompts[i]
                generate_log.backend = f"ollama::{self.model_id!s}"
                generate_log.date = date
                generate_log.model_options = model_opts
                generate_log.model_output = result.value
                generate_log.extra = {
                    "format": format,
                    "thinking": model_opts.get(ModelOption.THINKING, None),
                    "seed": model_opts.get(ModelOption.SEED, None),
                }
                generate_log.action = actions[i]
                generate_log.result = result

                if error:
                    generate_log.extra["error"] = error
                generate_logs.append(generate_log)

        return results

    def _extract_model_tool_requests(
        self, tools: dict[str, Callable], chat_response: ollama.ChatResponse
    ) -> dict[str, ModelToolCall] | None:
        model_tool_calls: dict[str, ModelToolCall] = {}

        if chat_response.message.tool_calls:
            for tool in chat_response.message.tool_calls:
                func = tools.get(tool.function.name)
                if func is None:
                    FancyLogger.get_logger().warning(
                        f"model attempted to call a non-existing function: {tool.function.name}"
                    )
                    continue  # skip this function if we can't find it.

                args = tool.function.arguments
                model_tool_calls[tool.function.name] = ModelToolCall(
                    tool.function.name, func, args
                )

        if len(model_tool_calls) > 0:
            return model_tool_calls
        return None
