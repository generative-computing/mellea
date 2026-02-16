"""A model backend wrapping the Ollama Python SDK."""

import asyncio
import datetime
import functools
from collections.abc import AsyncIterator, Callable, Coroutine, Sequence
from typing import Any, overload

import ollama
from tqdm import tqdm

from ..backends import ModelIdentifier, model_ids
from ..core import (
    BaseModelSubclass,
    C,
    CBlock,
    Component,
    Context,
    FancyLogger,
    GenerateLog,
    GenerateType,
    ModelOutputThunk,
    ModelToolCall,
)
from ..core.base import AbstractMelleaTool
from ..formatters import ChatFormatter, TemplateFormatter
from ..helpers import ClientCache, get_current_event_loop, send_to_queue
from ..stdlib.components import Message
from ..stdlib.requirements import ALoraRequirement
from ..telemetry.backend_instrumentation import (
    instrument_generate_from_context,
    instrument_generate_from_raw,
)
from .backend import FormatterBackend
from .model_options import ModelOption
from .tools import add_tools_from_context_actions, add_tools_from_model_options

format: None = None  # typing this variable in order to shadow the global format function and ensure mypy checks for errors


class OllamaModelBackend(FormatterBackend):
    """A model that uses the Ollama Python SDK for local inference."""

    def __init__(
        self,
        model_id: str | ModelIdentifier = model_ids.IBM_GRANITE_4_MICRO_3B,
        formatter: ChatFormatter | None = None,
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
        self._base_url = base_url
        self._client = ollama.Client(base_url)

        self._client_cache = ClientCache(2)

        # Call once to set up an async client and prepopulate the cache.
        _ = self._async_client

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
            "stream": ModelOption.STREAM,
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

    @property
    def _async_client(self) -> ollama.AsyncClient:
        """Ollama's client gets tied to a specific event loop. Reset it if needed here."""
        key = id(get_current_event_loop())

        _async_client = self._client_cache.get(key)
        if _async_client is None:
            _async_client = ollama.AsyncClient(self._base_url)
            self._client_cache.put(key, _async_client)
        return _async_client

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

    async def generate_from_context(
        self,
        action: Component[C] | CBlock,
        ctx: Context,
        *,
        format: type[BaseModelSubclass] | None = None,
        model_options: dict | None = None,
        tool_calls: bool = False,
    ) -> tuple[ModelOutputThunk[C], Context]:
        """See `generate_from_chat_context`."""
        from ..telemetry.backend_instrumentation import start_generate_span

        # Start span without auto-closing (will be closed in post_processing)
        span = start_generate_span(self, action, ctx, format, tool_calls)

        assert ctx.is_chat_context, (
            "The ollama backend only supports chat-like contexts."
        )
        mot = await self.generate_from_chat_context(
            action,
            ctx,
            _format=format,
            model_options=model_options,
            tool_calls=tool_calls,
        )

        # Store span for telemetry recording and closing in post_processing
        if span is not None:
            mot._meta["_telemetry_span"] = span

        return mot, ctx.add(action).add(mot)

    async def generate_from_chat_context(
        self,
        action: Component[C] | CBlock,
        ctx: Context,
        *,
        _format: type[BaseModelSubclass] | None = None,
        model_options: dict | None = None,
        tool_calls: bool = False,
    ) -> ModelOutputThunk[C]:
        """Generates a ModelOutputThunk. The final value for this object can be awaited.

        The new completion is generated from the provided Context using this backend's `Formatter`.

        This implementation treats the `Context` as a chat history, and uses the  `ollama.Client.chat()` interface to generate a completion.
        This will not always work, because sometimes we want to use non-chat models.

        Raises:
            RuntimeError: If not called from a thread with a running event loop.
        """
        # Start by awaiting any necessary computation.
        await self.do_generate_walk(action)

        model_opts = self._simplify_and_merge(model_options)

        linearized_context = ctx.view_for_generation()
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
        tools: dict[str, AbstractMelleaTool] = dict()
        if tool_calls:
            if _format:
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

        # Generate a chat response from ollama, using the chat messages. Can be either type since stream is passed as a model option.
        chat_response: Coroutine[
            Any, Any, AsyncIterator[ollama.ChatResponse] | ollama.ChatResponse
        ] = self._async_client.chat(
            model=self._get_ollama_model_id(),
            messages=conversation,
            tools=[t.as_json_tool for t in tools.values()],
            think=model_opts.get(ModelOption.THINKING, None),
            stream=model_opts.get(ModelOption.STREAM, False),
            options=self._make_backend_specific_and_remove(model_opts),
            format=_format.model_json_schema() if _format is not None else None,  # type: ignore
        )  # type: ignore

        output = ModelOutputThunk(None)
        output._context = linearized_context
        output._action = action
        output._model_options = model_opts

        # Processing functions only pass the ModelOutputThunk (and current chunk of response). Bind the other vars necessary for
        # each processing step.
        output._process = functools.partial(self.processing, tools=tools)
        output._post_process = functools.partial(
            self.post_processing,
            conversation=conversation,
            tools=tools,
            _format=_format,
        )

        try:
            # To support lazy computation, will need to remove this create_task and store just the unexecuted coroutine.
            # We can also support synchronous calls by adding a flag and changing this ._generate function.

            # This function should always be called from a running event loop so we don't have to worry about
            # scheduling the task to a specific event loop here.

            # Use `create_task` so that we don't have to specifically await this task before it starts executing.
            output._generate = asyncio.create_task(
                send_to_queue(chat_response, output._async_queue)
            )
            output._generate_type = GenerateType.ASYNC
        except RuntimeError as e:
            # Most likely cause is running this function without an event loop present
            raise e

        return output

    @overload
    async def generate_from_raw(
        self,
        actions: list[Component[C]],
        ctx: Context,
        *,
        format: type[BaseModelSubclass] | None = None,
        model_options: dict | None = None,
        tool_calls: bool = False,
    ) -> list[ModelOutputThunk[C]]: ...

    @overload
    async def generate_from_raw(
        self,
        actions: list[Component[C] | CBlock],
        ctx: Context,
        *,
        format: type[BaseModelSubclass] | None = None,
        model_options: dict | None = None,
        tool_calls: bool = False,
    ) -> list[ModelOutputThunk[C | str]]: ...

    async def generate_from_raw(
        self,
        actions: Sequence[Component[C] | CBlock],
        ctx: Context,
        *,
        format: type[BaseModelSubclass] | None = None,
        model_options: dict | None = None,
        tool_calls: bool = False,
    ) -> list[ModelOutputThunk]:
        """Generate using the generate api. Gives the input provided to the model without templating."""
        if len(actions) > 1:
            FancyLogger.get_logger().info(
                "Ollama doesn't support batching; will attempt to process concurrently."
            )
        if tool_calls:
            FancyLogger.get_logger().warning(
                "The completion endpoint does not support tool calling at the moment."
            )

        model_opts = self._simplify_and_merge(model_options)

        await self.do_generate_walks(list(actions))
        prompts = [self.formatter.print(action) for action in actions]

        # Ollama doesn't support "batching". There's some ability for concurrency. Use that here.
        # See https://github.com/ollama/ollama/blob/main/docs/faq.md#how-does-ollama-handle-concurrent-requests.

        with instrument_generate_from_raw(
            backend=self, num_actions=len(actions), format=format, tool_calls=tool_calls
        ):
            # Run async so that we can make use of Ollama's concurrency.
            coroutines: list[Coroutine[Any, Any, ollama.GenerateResponse]] = []
            for prompt in prompts:
                co = self._async_client.generate(
                    model=self._get_ollama_model_id(),
                    prompt=prompt,
                    raw=True,
                    think=model_opts.get(ModelOption.THINKING, None),
                    format=format.model_json_schema() if format is not None else None,  # type: ignore
                    options=self._make_backend_specific_and_remove(model_opts),
                )
                coroutines.append(co)

            responses = await asyncio.gather(*coroutines, return_exceptions=True)

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
                    meta={
                        "generate_response": response.model_dump(),
                        "usage": {
                            "completion_tokens": response.eval_count,
                            "prompt_tokens": response.prompt_eval_count,
                            "total_tokens": (
                                response.prompt_eval_count + response.eval_count
                                if response.prompt_eval_count is not None
                                and response.eval_count is not None
                                else None
                            ),
                        },
                    },
                )
            action = actions[i]
            result.parsed_repr = (
                action.parse(result) if isinstance(action, Component) else result.value
            )

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
            generate_log.action = action

            if error:
                generate_log.extra["error"] = error
            result._generate_log = generate_log

            results.append(result)

        return results

    def _extract_model_tool_requests(
        self, tools: dict[str, AbstractMelleaTool], chat_response: ollama.ChatResponse
    ) -> dict[str, ModelToolCall] | None:
        from .tools import validate_tool_arguments

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

                # Validate and coerce argument types
                validated_args = validate_tool_arguments(func, args, strict=False)
                model_tool_calls[tool.function.name] = ModelToolCall(
                    tool.function.name, func, validated_args
                )

        if len(model_tool_calls) > 0:
            return model_tool_calls
        return None

    async def processing(
        self,
        mot: ModelOutputThunk,
        chunk: ollama.ChatResponse,
        tools: dict[str, AbstractMelleaTool],
    ):
        """Called during generation to add information from a single ChatResponse to the ModelOutputThunk."""
        if mot._thinking is None:
            mot._thinking = ""
        thinking_chunk = chunk.message.thinking
        if thinking_chunk is not None:
            mot._thinking += thinking_chunk

        if mot._underlying_value is None:
            mot._underlying_value = ""
        content_chunk = chunk.message.content
        if content_chunk is not None:
            mot._underlying_value += content_chunk

        tool_chunk = self._extract_model_tool_requests(tools, chunk)
        if tool_chunk is not None:
            # Only set tool_calls if there is one.
            if mot.tool_calls is None:
                mot.tool_calls = {}

            # Merge the tool_chunk dict.
            for key, val in tool_chunk.items():
                mot.tool_calls[key] = val

        # Ollama responses are mostly self-contained. Merge chunks immediately.
        chat_response_delta_merge(mot, chunk)

    async def post_processing(
        self,
        mot: ModelOutputThunk,
        conversation: list[dict],
        tools: dict[str, AbstractMelleaTool],
        _format,
    ):
        """Called when generation is done."""
        assert mot._action is not None, (
            "ModelOutputThunks should have their action assigned during generation"
        )
        assert mot._model_options is not None, (
            "ModelOutputThunks should have their model_opts assigned during generation"
        )

        # Generate the log for this ModelOutputThunk.
        generate_log = GenerateLog()
        generate_log.prompt = conversation
        generate_log.backend = f"ollama::{self._get_ollama_model_id()}"
        generate_log.model_options = mot._model_options
        generate_log.date = datetime.datetime.now()
        generate_log.model_output = mot._meta.get("chat_response")
        generate_log.extra = {
            "format": _format,
            "thinking": mot._model_options.get(ModelOption.THINKING, None),
            "tools_available": tools,
            "tools_called": mot.tool_calls,
            "seed": mot._model_options.get(ModelOption.SEED, None),
        }
        generate_log.action = mot._action
        generate_log.result = mot

        mot._generate_log = generate_log
        mot._generate = None

        # Record telemetry and close span now that response is available
        span = mot._meta.get("_telemetry_span")
        if span is not None:
            from ..telemetry import end_backend_span
            from ..telemetry.backend_instrumentation import (
                record_response_metadata,
                record_token_usage,
            )

            response = mot._meta.get("chat_response")
            if response:
                # Ollama responses may have usage information
                usage = (
                    response.get("usage")
                    if isinstance(response, dict)
                    else getattr(response, "usage", None)
                )
                if usage:
                    record_token_usage(span, usage)
                record_response_metadata(span, response)

            # Close the span now that telemetry is recorded
            end_backend_span(span)

            # Clean up the span reference
            del mot._meta["_telemetry_span"]


def chat_response_delta_merge(mot: ModelOutputThunk, delta: ollama.ChatResponse):
    """Merges the individual ChatResponse chunks from a streaming response into a single ChatResponse.

    Args:
        mot: the ModelOutputThunk that the deltas are being used to populated.
        delta: the most recent ollama ChatResponse.
    """
    if mot._meta.get("chat_response", None) is None:
        mot._meta["chat_response"] = delta
        return  # Return early, no need to merge.

    merged: ollama.ChatResponse = mot._meta["chat_response"]
    if not merged.done:
        merged.done = delta.done
    if merged.done_reason is None:
        merged.done_reason = delta.done_reason
    if merged.total_duration is None:
        merged.total_duration = delta.total_duration
    if merged.load_duration is None:
        merged.load_duration = delta.load_duration
    if merged.prompt_eval_count is None:
        merged.prompt_eval_count = delta.prompt_eval_count
    if merged.prompt_eval_duration is None:
        merged.prompt_eval_duration = delta.prompt_eval_duration
    if merged.eval_count is None:
        merged.eval_count = delta.eval_count

    if merged.message.role == "":
        merged.message.role = delta.message.role

    if merged.message.content is None:
        merged.message.content = delta.message.content
    elif delta.message.content is not None:
        merged.message.content += delta.message.content

    if merged.message.thinking is None:
        merged.message.thinking = delta.message.thinking
    elif delta.message.thinking is not None:
        merged.message.thinking += delta.message.thinking

    if merged.message.tool_calls is None:
        merged.message.tool_calls = delta.message.tool_calls
    elif delta.message.tool_calls is not None:
        merged.message.tool_calls = [
            *merged.message.tool_calls,
            *delta.message.tool_calls,
        ]
