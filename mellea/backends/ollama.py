"""A model backend wrapping the Ollama Python SDK."""

import asyncio
import datetime
import functools
from collections.abc import AsyncIterator, Coroutine, Sequence
from typing import Any

import httpx
import ollama
from tqdm import tqdm

from ..backends import ModelIdentifier, model_ids
from ..core import (
    BaseModelSubclass,
    C,
    CBlock,
    Component,
    Context,
    GenerateLog,
    GenerateType,
    ImageUrlBlock,
    MelleaLogger,
    ModelOutputThunk,
    ModelToolCall,
    RawProviderResponse,
)
from ..core.base import AbstractMelleaTool
from ..formatters import ChatFormatter, TemplateFormatter
from ..helpers import (
    DEFAULT_CHUNK_TIMEOUT,
    ClientCache,
    get_current_event_loop,
    send_to_queue,
)
from ..stdlib.components import Message
from ..stdlib.requirements import ALoraRequirement
from ..telemetry.context import generate_request_id, with_context
from .backend import FormatterBackend
from .model_options import ModelOption
from .tools import add_tools_from_context_actions, add_tools_from_model_options

format: None = None  # typing this variable in order to shadow the global format function and ensure mypy checks for errors


def _strip_data_uri_prefix(images: list[str]) -> list[str]:
    """Strip data URI prefix from base64 image strings for Ollama.

    Ollama expects raw base64 strings without the 'data:image/...;base64,' prefix.
    This function removes the prefix if present, leaving just the base64 data.

    Args:
        images: List of base64 image strings, potentially with data URI prefixes.

    Returns:
        List of base64 strings with data URI prefixes removed.
    """
    stripped = []
    for img in images:
        # Check if the string has a data URI prefix and remove it
        if "data:" in img and "base64," in img:
            img = img.split("base64,")[1]
        stripped.append(img)
    return stripped


class OllamaModelBackend(FormatterBackend):
    """A model that uses the Ollama Python SDK for local inference.

    Args:
        model_id (str | ModelIdentifier): Ollama model ID. If a
            ``ModelIdentifier`` is passed, its ``ollama_name`` attribute must
            be set.
        formatter (ChatFormatter | None): Formatter for rendering components.
            Defaults to ``TemplateFormatter``.
        base_url (str | None): Ollama server endpoint; defaults to
            ``env(OLLAMA_HOST)`` or ``http://localhost:11434``.
        model_options (dict | None): Default model options for generation requests.
        timeout (float | None): Per-operation HTTP timeout in seconds (connect,
            read, write, pool). Defaults to 300 s. For streaming requests this
            bounds the wait between consecutive chunks; for non-streaming requests
            it bounds total time-to-response. Pass ``None`` to use the upstream
            ``ollama`` SDK default (no timeout).

    Attributes:
        to_mellea_model_opts_map (dict): Mapping from Ollama-specific option names
            to Mellea ``ModelOption`` sentinel keys.
        from_mellea_model_opts_map (dict): Mapping from Mellea ``ModelOption``
            sentinel keys to Ollama-specific option names.

    Raises:
        ValueError: If ``model_id`` is a ``ModelIdentifier`` with no ``ollama_name`` set.
        ConnectionError: If the Ollama server is not running at ``base_url``.
        OSError: If the model cannot be pulled from the Ollama library.
    """

    def __init__(
        self,
        model_id: str | ModelIdentifier = model_ids.IBM_GRANITE_4_1_3B,
        formatter: ChatFormatter | None = None,
        base_url: str | None = None,
        model_options: dict | None = None,
        timeout: float | None = 300.0,
    ):
        """Initialize an Ollama backend, connecting to the server and pulling the model if needed."""
        super().__init__(
            model_id=model_id,
            formatter=(
                formatter
                if formatter is not None
                else TemplateFormatter(model_id=model_id)
            ),
            model_options=model_options,
        )
        # Resolve to a concrete ollama model name; raises ValueError if no ollama_name is set.
        ollama_model_id = (
            model_id.ollama_name if isinstance(model_id, ModelIdentifier) else model_id
        )
        if ollama_model_id is None or ollama_model_id == "":
            raise ValueError(
                "Cannot create OllamaModelBackend: the ModelIdentifier has no ollama_name set. "
                "Check mellea/backends/model_ids.py and ensure the constant you are using "
                "has an ollama_name value, or pass the Ollama model tag as a plain string."
            )
        self._model_id: str = ollama_model_id
        self._provider: str = "ollama"

        # Setup the client and ensure that we have the model available.
        self._base_url = base_url
        self._timeout = timeout
        client_kwargs: dict[str, Any] = {}
        if timeout is not None:
            client_kwargs["timeout"] = timeout
        self._client_kwargs = client_kwargs
        self._client = ollama.Client(base_url, **client_kwargs)

        self._client_cache = ClientCache(2)

        # Call once to set up an async client and prepopulate the cache.
        _ = self._async_client

        if not self._check_ollama_server():
            err = f"could not create OllamaModelBackend: ollama server not running at {base_url}"
            MelleaLogger.get_logger().error(err)
            raise ConnectionError(err)
        if not self._pull_ollama_model():
            err = (
                f"Model '{self._model_id}' could not be pulled from the Ollama library. "
                f"Check that the model name is correct (run 'ollama list' to see locally "
                f"available models, or 'ollama pull {self._model_id}' to fetch it manually)."
            )
            MelleaLogger.get_logger().error(err)
            raise OSError(err)

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
            "stop": ModelOption.STOP_SEQUENCES,
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
            ModelOption.STOP_SEQUENCES: "stop",
        }

    def _check_ollama_server(self) -> bool:
        """Requests generic info about the Ollama server to ensure it's running."""
        try:
            self._client.ps()
        except (ConnectionError, httpx.TimeoutException, httpx.ConnectError):
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
        if self.is_model_available(self._model_id):
            return True

        try:
            MelleaLogger.get_logger().debug(
                f"Loading/Pulling model from Ollama: {self._model_id}"
            )
            stream = self._client.pull(self._model_id, stream=True)
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
        except (ollama.ResponseError, httpx.TimeoutException, httpx.ConnectError):
            return False

    @property
    def _async_client(self) -> ollama.AsyncClient:
        """Ollama's client gets tied to a specific event loop. Reset it if needed here."""
        key = id(get_current_event_loop())

        _async_client = self._client_cache.get(key)
        if _async_client is None:
            _async_client = ollama.AsyncClient(self._base_url, **self._client_kwargs)
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
        merged = ModelOption.merge_model_options(
            backend_model_opts, generate_call_model_opts
        )
        return merged

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

    async def _generate_from_context(
        self,
        action: Component[C] | CBlock,
        ctx: Context,
        *,
        format: type[BaseModelSubclass] | None = None,
        model_options: dict | None = None,
        tool_calls: bool = False,
    ) -> tuple[ModelOutputThunk[C], Context]:
        """Generate a completion for ``action`` given ``ctx`` via the Ollama chat API.

        Delegates to ``generate_from_chat_context``. Only chat contexts are supported.

        Args:
            action (Component[C] | CBlock): The component or content block to generate
                a completion for.
            ctx (Context): The current generation context (must be a chat context).
            format (type[BaseModelSubclass] | None): Optional Pydantic model class for
                structured/constrained output decoding.
            model_options (dict | None): Per-call model options that override the
                backend's defaults.
            tool_calls (bool): If ``True``, expose available tools to the model and
                parse tool-call responses.

        Returns:
            tuple[ModelOutputThunk[C], Context]: A thunk holding the (lazy) model output
                and an updated context that includes ``action`` and the new output.
        """
        assert ctx.is_chat_context, (
            "The ollama backend only supports chat-like contexts."
        )

        _model_id_str = str(getattr(self, "model_id", "unknown"))
        with with_context(request_id=generate_request_id(), model_id=_model_id_str):
            mot = await self.generate_from_chat_context(
                action,
                ctx,
                _format=format,
                model_options=model_options,
                tool_calls=tool_calls,
            )

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
        """Generate a new completion from the provided context using this backend's formatter.

        Treats the ``Context`` as a chat history and uses the ``ollama.Client.chat()``
        interface to generate a completion. Returns a thunk that lazily resolves
        the model output.

        Args:
            action (Component[C] | CBlock): The component or content block to generate
                a completion for.
            ctx (Context): The current generation context (must be a chat context).
            _format (type[BaseModelSubclass] | None): Optional Pydantic model class for
                structured output decoding.
            model_options (dict | None): Per-call model options.
            tool_calls (bool): If ``True``, expose available tools and parse responses.

        Returns:
            ModelOutputThunk[C]: A thunk holding the (lazy) model output.

        Raises:
            RuntimeError: If not called from a thread with a running event loop.
            ValueError: If a message contains an ``ImageUrlBlock``; Ollama requires
                base64-encoded images — convert to an ``ImageBlock`` first.
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
                    "The ollama backend does not currently support aLoRA adapters."
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

        # NOTE: `self.formatter.to_chat_messages` explicitly skips `Message` objects. However, we need
        # to print `Message`s to correctly serialize any documents with the message. Do the printing here.
        for m in messages:
            if m.images is not None:
                for img in m.images:
                    if isinstance(img, ImageUrlBlock):
                        raise ValueError(
                            "OllamaModelBackend does not support URL images (ImageUrlBlock). "
                            "Convert the image to a base64-encoded ImageBlock before passing it to Ollama."
                        )
            conversation.append(
                {
                    "role": m.role,
                    "content": self.formatter.print(m),
                    "images": (
                        _strip_data_uri_prefix([str(img.value) for img in m.images])
                        if m.images
                        else None
                    ),
                }
            )

        # Append tool call information if applicable.
        tools: dict[str, AbstractMelleaTool] = dict()
        if tool_calls:
            if _format:
                MelleaLogger.get_logger().warning(
                    f"Tool calling typically uses constrained generation, but you have specified a `format` in your generate call. NB: tool calling is superseded by format; we will NOT call tools for your request: {action}"
                )
            else:
                add_tools_from_model_options(tools, model_opts)
                add_tools_from_context_actions(tools, ctx.actions_for_available_tools())

                # Add the tools from the action for this generation last so that
                # they overwrite conflicting names.
                add_tools_from_context_actions(tools, [action])
            MelleaLogger.get_logger().info(f"Tools for call: {tools.keys()}")

        # Generate a chat response from ollama, using the chat messages. Can be either type since stream is passed as a model option.
        chat_response: Coroutine[
            Any, Any, AsyncIterator[ollama.ChatResponse] | ollama.ChatResponse
        ] = self._async_client.chat(
            model=self._model_id,
            messages=conversation,
            tools=[t.as_json_tool for t in tools.values()],
            think=model_opts.get(ModelOption.THINKING, None),
            stream=model_opts.get(ModelOption.STREAM, False),
            options=self._make_backend_specific_and_remove(model_opts),
            format=_format.model_json_schema() if _format is not None else None,  # type: ignore
        )  # type: ignore

        output = ModelOutputThunk(None)
        output._start = datetime.datetime.now()
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

        # Set model/provider early so they are available in the error path
        output.generation.model = self._model_id
        output.generation.provider = self._provider

        try:
            # To support lazy computation, will need to remove this create_task and store just the unexecuted coroutine.
            # We can also support synchronous calls by adding a flag and changing this ._generate function.

            # This function should always be called from a running event loop so we don't have to worry about
            # scheduling the task to a specific event loop here.

            # Use `create_task` so that we don't have to specifically await this task before it starts executing.
            output._generate = asyncio.create_task(
                send_to_queue(
                    chat_response,
                    output._async_queue,
                    chunk_timeout=model_opts.get(
                        ModelOption.STREAM_TIMEOUT, DEFAULT_CHUNK_TIMEOUT
                    ),
                )
            )
            output._generate_type = GenerateType.ASYNC
        except RuntimeError as e:
            # Most likely cause is running this function without an event loop present
            raise e

        return output

    async def _generate_from_raw(
        self,
        actions: Sequence[Component[C] | CBlock],
        ctx: Context,
        *,
        format: type[BaseModelSubclass] | None = None,
        model_options: dict | None = None,
        tool_calls: bool = False,
    ) -> tuple[list[ModelOutputThunk], dict[str, Any] | None]:
        """Generate completions for multiple actions without chat templating via Ollama.

        Passes formatted prompt strings directly to Ollama's generate endpoint.
        Requests are submitted concurrently to make use of Ollama's concurrency support.

        Args:
            actions (Sequence[Component[C] | CBlock]): Actions to generate completions for.
            ctx (Context): The current generation context.
            format (type[BaseModelSubclass] | None): Optional Pydantic model for
                structured output decoding.
            model_options (dict | None): Per-call model options.
            tool_calls (bool): Ignored; tool calling is not supported on this endpoint.

        Returns:
            tuple[list[ModelOutputThunk], dict | None]: `(results, usage)` where
                `results` is a list of model output thunks, one per action, and
                `usage` is the aggregate token-usage dict for the batch (or `None`
                when no request reported usage).

                If Ollama returns an empty done response (``response=""``,
                ``done=True``, no thinking content) for an action, that thunk
                soft-fails: it has ``value=""`` and ``thunk.error`` carries the
                ``RuntimeError`` describing the cause. Other actions in the
                batch are unaffected.

        Note:
            Requests are awaited with ``asyncio.gather`` (all-or-nothing): if any
            request raises (e.g. ``ollama.ResponseError`` or a connection error),
            that exception propagates to the caller and no list is returned, even
            for requests that completed successfully.
        """
        if len(actions) > 1:
            MelleaLogger.get_logger().info(
                "Ollama doesn't support batching; will attempt to process concurrently."
            )
        if tool_calls:
            MelleaLogger.get_logger().warning(
                "The completion endpoint does not support tool calling at the moment."
            )

        model_opts = self._simplify_and_merge(model_options)

        await self.do_generate_walks(list(actions))
        prompts = [self.formatter.print(action) for action in actions]

        # Ollama doesn't support "batching". There's some ability for concurrency. Use that here.
        # See https://github.com/ollama/ollama/blob/main/docs/faq.md#how-does-ollama-handle-concurrent-requests.

        # Run async so that we can make use of Ollama's concurrency.
        coroutines: list[Coroutine[Any, Any, ollama.GenerateResponse]] = []
        for prompt in prompts:
            co = self._async_client.generate(
                model=self._model_id,
                prompt=prompt,
                raw=True,
                think=model_opts.get(ModelOption.THINKING, None),
                format=format.model_json_schema() if format is not None else None,  # type: ignore
                options=self._make_backend_specific_and_remove(model_opts),
            )
            coroutines.append(co)

        # All-or-nothing: first failure raises; remaining in-flight requests
        # complete but their results are discarded.
        responses = await asyncio.gather(*coroutines)

        results = []
        date = datetime.datetime.now()
        agg_prompt = 0
        agg_completion = 0
        for i, response in enumerate(responses):
            result = None
            per_mot_usage: dict[str, Any] | None = None
            if response.done and not response.response and not response.thinking:
                # Empty done response with no thinking content. Commonly caused by the
                # Ollama model-load race (#599) but can also occur on an early stop or
                # stop-sequence hit.
                empty_err = RuntimeError(
                    f"generate_from_raw: request {i} returned an empty response from Ollama "
                    "(response='', done=True). This commonly occurs when the model is still "
                    "loading, but can also indicate an early stop or stop-sequence hit. "
                    "See https://github.com/generative-computing/mellea/issues/599 "
                    "and https://github.com/ollama/ollama/issues/16326"
                )
                MelleaLogger.get_logger().warning(str(empty_err))
                result = ModelOutputThunk(value="")
                result._error = empty_err
            else:
                n_in = response.prompt_eval_count
                n_out = response.eval_count
                if n_in is not None and n_out is not None:
                    agg_prompt += n_in
                    agg_completion += n_out
                    per_mot_usage = {
                        "prompt_tokens": n_in,
                        "completion_tokens": n_out,
                        "total_tokens": n_in + n_out,
                    }
                result = ModelOutputThunk(value=response.response)
                result.raw = RawProviderResponse(
                    provider=self._provider, response=response.model_dump()
                )
            result.generation.usage = per_mot_usage
            result.generation.model = self._model_id
            result.generation.provider = self._provider

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
            result._generate_log = generate_log

            results.append(result)

        usage: dict[str, Any] | None = (
            {
                "prompt_tokens": agg_prompt,
                "completion_tokens": agg_completion,
                "total_tokens": agg_prompt + agg_completion,
            }
            if (agg_prompt or agg_completion)
            else None
        )
        return results, usage

    def _extract_model_tool_requests(
        self, tools: dict[str, AbstractMelleaTool], chat_response: ollama.ChatResponse
    ) -> dict[str, ModelToolCall] | None:
        from .tools import validate_tool_arguments

        model_tool_calls: dict[str, ModelToolCall] = {}

        if chat_response.message.tool_calls:
            for tool in chat_response.message.tool_calls:
                func = tools.get(tool.function.name)
                if func is None:
                    MelleaLogger.get_logger().warning(
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
        """Accumulate text and tool calls from a single Ollama ChatResponse chunk.

        Called for each streaming or non-streaming `ollama.ChatResponse`. Also
        extracts tool call requests inline and merges the chunk into the running
        aggregated response stored in `mot.raw.response`.

        Args:
            mot (ModelOutputThunk): The output thunk being populated.
            chunk (ollama.ChatResponse): A single chat response object from Ollama.
            tools (dict[str, AbstractMelleaTool]): Available tools, keyed by name,
                used for extracting tool call requests from the response.
        """
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
        """Finalize the output thunk after Ollama generation completes.

        Attaches the generate log, records token usage metrics, emits telemetry,
        and cleans up the span reference.

        Args:
            mot (ModelOutputThunk): The output thunk to finalize.
            conversation (list[dict]): The chat conversation sent to the model,
                used for logging.
            tools (dict[str, AbstractMelleaTool]): Available tools, keyed by name.
            _format: The structured output format class used during generation, if any.
        """
        assert mot._action is not None, (
            "ModelOutputThunks should have their action assigned during generation"
        )
        assert mot._model_options is not None, (
            "ModelOutputThunks should have their model_opts assigned during generation"
        )

        # Generate the log for this ModelOutputThunk.
        generate_log = GenerateLog()
        generate_log.prompt = conversation
        generate_log.backend = f"ollama::{self._model_id}"
        generate_log.model_options = mot._model_options
        generate_log.date = datetime.datetime.now()
        generate_log.model_output = mot.raw.response
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

        # Extract token counts from response
        response = mot.raw.response
        prompt_tokens = (
            getattr(response, "prompt_eval_count", None) if response else None
        )
        completion_tokens = getattr(response, "eval_count", None) if response else None

        # Populate standardized usage field (convert to OpenAI format)
        if prompt_tokens is not None and completion_tokens is not None:
            mot.generation.usage = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            }

        # Populate model and provider metadata
        mot.generation.model = self._model_id
        mot.generation.provider = self._provider
        mot.raw.provider = self._provider

        # Populate response-side metadata for telemetry
        if response is not None:
            mot.generation.response_model = getattr(response, "model", None)
            if done_reason := getattr(response, "done_reason", None):
                mot.generation.finish_reasons = [done_reason]


def chat_response_delta_merge(mot: ModelOutputThunk, delta: ollama.ChatResponse):
    """Merges the individual ChatResponse chunks from a streaming response into a single ChatResponse.

    Args:
        mot: the ModelOutputThunk that the deltas are being used to populated.
        delta: the most recent ollama ChatResponse.
    """
    if mot.raw.response is None:
        mot.raw.response = delta
        return  # Return early, no need to merge.

    merged: ollama.ChatResponse = mot.raw.response
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
