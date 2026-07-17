"""`MelleaSession`: the primary entry point for running generative programs.

`MelleaSession` wraps a `Backend` and a `Context` and exposes high-level methods
(`act`, `instruct`, `sample`) that drive the generate-validate-repair loop. It
also manages a global context variable (accessible via `get_session()`) so that
nested components can reach the current session without explicit threading. Use
`start_session(...)` as a context manager to create and automatically clean up a
session.
"""

from __future__ import annotations

import collections.abc
import contextlib
import contextvars
import inspect
from copy import copy
from typing import Any, Literal, overload

from PIL import Image as PILImage

from ..backends.model_ids import (
    IBM_GRANITE_4_1_3B,
    IBM_GRANITE_4_HYBRID_SMALL,
    ModelIdentifier,
)
from ..core import (
    AudioBlock,
    AudioUrlBlock,
    Backend,
    BaseModelSubclass,
    CBlock,
    Component,
    ComputedModelOutputThunk,
    Context,
    GenerateLog,
    ImageBlock,
    ImageUrlBlock,
    MelleaLogger,
    ModelOutputThunk,
    Requirement,
    S,
    SamplingResult,
    SamplingStrategy,
    ValidationResult,
)
from ..core.utils import _log_context
from ..helpers import _run_async_in_thread
from ..plugins.manager import has_plugins, invoke_hook
from ..plugins.types import HookType
from ..stdlib import functional as mfuncs
from ..telemetry.context import with_context
from ..telemetry.tracing import (
    finish_session_span,
    finish_session_startup_span,
    start_session_span,
    start_session_startup_span,
)
from .components import Document, Message
from .context import ChatContext, SimpleContext
from .sampling import RejectionSamplingStrategy
from .start_backend import (
    _resolve_context,
    _resolve_model_id_str,
    backend_name_to_class,
)

# Global context variable for the context session
_context_session: contextvars.ContextVar[MelleaSession | None] = contextvars.ContextVar(
    "context_session", default=None
)


def get_session() -> MelleaSession:
    """Get the current session from context.

    Returns:
        The currently active `MelleaSession`.

    Raises:
        RuntimeError: If no session is currently active.
    """
    session = _context_session.get()
    if session is None:
        raise RuntimeError(
            "No active session found. Use 'with start_session(...):' to create one."
        )
    return session


def start_session(
    backend_name: Literal["ollama", "hf", "openai", "watsonx", "litellm"] = "ollama",
    model_id: str | ModelIdentifier = IBM_GRANITE_4_1_3B,
    ctx: Context | None = None,
    *,
    context_type: Literal["simple", "chat"] | None = None,
    model_options: dict | None = None,
    plugins: list[Any] | None = None,
    **backend_kwargs: Any,
) -> MelleaSession:
    """Start a new Mellea session. Can be used as a context manager or called directly.

    This function creates and configures a new Mellea session with the specified backend
    and model. When used as a context manager (with `with` statement), it automatically
    sets the session as the current active session for use with convenience functions
    like `instruct()`, `chat()`, `query()`, and `transform()`. When called directly,
    it returns a session object that can be used directly.

    Args:
        backend_name: The backend to use. Options are:
            - "ollama": Use Ollama backend for local models
            - "hf" or "huggingface": Use Hugging Face transformers backend
            - "openai": Use OpenAI API backend
            - "watsonx": Use IBM WatsonX backend, WARNING: this defaults to the IBM_GRANITE_4_HYBRID_SMALL model for now.
            - "litellm": Use the LiteLLM backend
        model_id: Model identifier or name. Can be a `ModelIdentifier` from
            mellea.backends.model_ids or a string model name.
        ctx: Context instance for conversation history. Defaults to
            `SimpleContext()`. Mutually exclusive with `context_type`.
        context_type: Shorthand for creating a context — `"simple"` for
            `SimpleContext`, `"chat"` for `ChatContext`. Mutually
            exclusive with `ctx`.
        model_options: Additional model configuration options that will be passed
            to the backend (e.g., temperature, max_tokens, etc.).
        plugins: Optional list of plugins scoped to this session. Accepts
            `@hook`-decorated functions, `@plugin`-decorated class instances,
            `MelleaPlugin` instances, or `PluginSet` instances.
        **backend_kwargs: Additional keyword arguments passed to the backend constructor.

    Returns:
        MelleaSession: A session object that can be used as a context manager
        or called directly with session methods.

    Raises:
        ValueError: If both `ctx` and `context_type` are provided.
        Exception: If `backend_name` is not one of the recognised backend
            identifiers.
        ImportError: If the requested backend requires optional dependencies
            that are not installed.

    Examples:
        ```python
        # Basic usage with default settings
        with start_session() as session:
            response = session.instruct("Explain quantum computing")

        # Using OpenAI with custom model options
        with start_session("openai", "gpt-4", model_options={"temperature": 0.7}):
            response = session.chat("Write a poem")

        # Using context_type shorthand for chat conversations
        with start_session("ollama", context_type="chat") as session:
            session.chat("Hello!")
            session.chat("How are you?")  # Remembers previous message

        # Direct usage.
        session = start_session()
        response = session.instruct("Explain quantum computing")
        session.cleanup()
        ```
    """
    import uuid

    logger = MelleaLogger.get_logger()

    # Validate args.
    resolved_ctx = _resolve_context(ctx, context_type)
    model_id_str = _resolve_model_id_str(model_id, backend_name)
    backend_class = backend_name_to_class(backend_name)
    if backend_class is None:
        raise Exception(
            f"Backend name {backend_name} unknown. Valid options are: "
            "`ollama`, `hf`, `openai`, `watsonx`, `litellm`."
        )

    # Pre-generate so the startup span and session_pre_init payload share an id.
    session_id = str(uuid.uuid4())

    # Called directly, not via hook: OTel Token attach/detach is task-affine,
    # and each hook firing runs in a separate _run_async_in_thread Task.
    start_session_startup_span(
        session_id,
        backend=backend_name,
        model_id=model_id_str,
        context_type=resolved_ctx.__class__.__name__,
    )
    try:
        # --- session_pre_init hook ---
        if has_plugins(HookType.SESSION_PRE_INIT):
            from ..plugins.hooks.session import SessionPreInitPayload

            pre_payload = SessionPreInitPayload(
                session_id=session_id,
                backend_name=backend_name,
                model_id=model_id_str,
                model_options=model_options,
                context_type=resolved_ctx.__class__.__name__,
            )
            _, pre_payload = _run_async_in_thread(
                invoke_hook(HookType.SESSION_PRE_INIT, pre_payload)
            )
            # Apply writable field modifications
            model_id_str = pre_payload.model_id
            model_options = pre_payload.model_options

        backend_class = backend_name_to_class(backend_name)
        if backend_class is None:
            raise Exception(
                f"Backend name {backend_name} unknown. Please see the docstring for `mellea.stdlib.session.start_session` for a list of options."
            )
        assert backend_class is not None
        if "watsonx" in backend_name:
            # Temp hack for watsonx for granite 4.1
            backend = backend_class(
                IBM_GRANITE_4_HYBRID_SMALL.watsonx_name,
                model_options=model_options,
                **backend_kwargs,
            )
        else:
            backend = backend_class(
                model_id_str, model_options=model_options, **backend_kwargs
            )

        logger.info(
            f"Starting Mellea session: backend={backend_name}, model={model_id_str}, "
            f"context={resolved_ctx.__class__.__name__}"
            + (f", model_options={model_options}" if model_options else "")
        )

        session = MelleaSession(backend, resolved_ctx, session_id=session_id)

        # Register session-scoped plugins
        if plugins:
            from ..plugins.registry import register as register_plugins

            register_plugins(plugins, session_id=session.id)

        # --- session_post_init hook ---
        if has_plugins(HookType.SESSION_POST_INIT):
            from ..plugins.hooks.session import SessionPostInitPayload

            post_payload = SessionPostInitPayload(
                session_id=session.id, model_id=model_id_str, context=session.ctx
            )
            _run_async_in_thread(
                invoke_hook(HookType.SESSION_POST_INIT, post_payload, backend=backend)
            )
    except BaseException as exc:
        finish_session_startup_span(session_id, exception=exc)
        raise
    else:
        finish_session_startup_span(session_id)

    return session


class MelleaSession:
    """Mellea sessions are a THIN wrapper around `m` convenience functions with NO special semantics.

    Using a Mellea session is not required, but it does represent the "happy path" of Mellea programming. Some nice things about ussing a `MelleaSession`:
    1. In most cases you want to keep a Context together with the Backend from which it came.
    2. You can directly run an instruction or a send a chat, instead of first creating the `Instruction` or `Chat` object and then later calling backend.generate on the object.
    3. The context is "threaded-through" for you, which allows you to issue a sequence of commands instead of first calling backend.generate on something and then appending it to your context.

    These are all relatively simple code hygiene and state management benefits, but they add up over time.
    If you are doing complicating programming (e.g., non-trivial inference scaling) then you might be better off forgoing `MelleaSession`s and managing your Context and Backend directly.

    Note: we put the `instruct`, `validate`, and other convenience functions here instead of in `Context` or `Backend` to avoid import resolution issues.

    Args:
        backend (Backend): The backend to use for all model inference in this
            session.
        ctx (Context | None): The conversation context. Defaults to a new
            `SimpleContext` if `None`.

    Attributes:
        ctx (Context): The active conversation context; never `None` (defaults
            to a fresh `SimpleContext` when `None` is passed). Updated after
            every call that produces model output.
        id (str): Unique session UUID assigned at construction.
    """

    # ``ctx`` is exposed as a property below; backing field is ``_ctx``.

    def __init__(
        self,
        backend: Backend,
        ctx: Context | None = None,
        *,
        session_id: str | None = None,
    ):
        """Initialize MelleaSession with a backend and optional conversation context.

        Note:
            When *ctx* is a bare (unbound, empty) `ChatContext` and *backend*
            exposes a `model_id`, the constructor replaces `self.ctx` with a
            re-bound copy that carries the model identifier.  After construction,
            `session.ctx is ctx` will therefore be `False`.
        """
        import uuid

        self._init_fields(
            session_id if session_id is not None else str(uuid.uuid4()),
            backend,
            ctx if ctx is not None else SimpleContext(),
        )
        self._auto_bind_model()

    def _init_fields(self, session_id: str, backend: Backend, ctx: Context) -> None:
        """Set all instance fields. Shared by __init__ and __copy__ so neither diverges."""
        self.id = session_id
        self.backend = backend
        # Bypass the ctx setter so this initial assignment doesn't count as an
        # interaction.
        self._ctx: Context = ctx
        self._interaction_count: int = 0
        self._session_logger = MelleaLogger.get_logger()
        self._context_token = None
        self._log_context_token = None
        self._exit_stack: contextlib.ExitStack | None = None

    @property
    def ctx(self) -> Context:
        """The session's current conversation context."""
        return self._ctx

    @ctx.setter
    def ctx(self, value: Context) -> None:
        """Replace the context and count this as one interaction.

        Every model-interaction code path in this class assigns to ``self.ctx``
        with the post-interaction context, so each setter call is exactly one
        interaction. Lifecycle paths that swap the context wholesale (``reset``,
        ``_auto_bind_model``) write to ``self._ctx`` directly to bypass this
        counter.
        """
        self._ctx = value
        self._interaction_count += 1

    def _auto_bind_model(self) -> None:
        """Bind the backend's model_id to a bare ChatContext, enabling auto context-window sizing."""
        if (
            isinstance(self.ctx, ChatContext)
            and self.ctx.model_id is None
            and self.ctx.is_root_node
            and getattr(self.backend, "model_id", None) is not None
        ):
            # Bypass the setter — binding at construction is not an interaction.
            self._ctx = self.ctx._bind_model(getattr(self.backend, "model_id"))

    def __enter__(self):
        """Enter context manager and set this session as the current global session."""
        # Called directly, not via hook: OTel Token attach/detach is task-affine,
        # and each hook firing runs in a separate _run_async_in_thread Task.
        start_session_span(
            self.id,
            context_type=self.ctx.__class__.__name__,
            backend=self.backend._provider,
        )
        self._context_token = _context_session.set(self)
        # TODO: Migrate telemetry fields from _log_context to with_context() system.
        # Currently session_id and model_id are duplicated in both systems. The
        # 'backend' field only exists in _log_context and would need to be added to
        # the new telemetry context system (mellea/telemetry/context.py) before this
        # _log_context.set() call can be removed. Once 'backend' is added to
        # _CONTEXT_VARS, remove this block and set all three fields via with_context().
        self._log_context_token = _log_context.set(
            {
                **_log_context.get(),
                "session_id": self.id,
                "backend": self.backend.__class__.__name__,
                "model_id": str(getattr(self.backend, "model_id", "unknown")),
            }
        )
        # Set session_id for the full session lifetime. model_id is intentionally
        # omitted here — backends own it and set it per-call via with_context().
        self._exit_stack = contextlib.ExitStack()
        self._exit_stack.enter_context(with_context(session_id=self.id))
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager and cleanup session."""
        self.cleanup(exception=exc_val)
        if self._log_context_token is not None:
            _log_context.reset(self._log_context_token)
            self._log_context_token = None
        if self._context_token is not None:
            _context_session.reset(self._context_token)
            self._context_token = None
        if self._exit_stack is not None:
            self._exit_stack.__exit__(exc_type, exc_val, exc_tb)
            self._exit_stack = None
        finish_session_span(self.id, exception=exc_val)

    def __copy__(self):
        """Use self.clone. Copies the current session but keeps references to the backend and context."""
        import uuid

        new = object.__new__(MelleaSession)
        new._init_fields(str(uuid.uuid4()), self.backend, self.ctx)
        # Do not call _auto_bind_model: the session state is already settled.
        return new

    def clone(self) -> MelleaSession:
        """Useful for running multiple generation requests while keeping the context at a given point in time.

        Returns:
            a copy of the current session. Keeps the context, backend, and session logger.

        Examples:
            ```python
            >>> from mellea import start_session
            >>> m = start_session()
            >>> m.instruct("What is 2x2?")
            >>>
            >>> m1 = m.clone()
            >>> out = m1.instruct("Multiply that by 2")
            >>> print(out)
            ... 8
            >>>
            >>> m2 = m.clone()
            >>> out = m2.instruct("Multiply that by 3")
            >>> print(out)
            ... 12
            ```
        """
        return copy(self)

    def reset(self) -> None:
        """Reset the context state to a fresh, empty context of the same type.

        Fires the `SESSION_RESET` plugin hook if any plugins are registered, then
        replaces `self.ctx` with a fresh empty context, discarding all accumulated
        conversation history. For `ChatContext`, uses `new_instance()` so the
        `model_id` and `window_size` bindings are preserved; for all other context
        types, uses `reset_to_new()`.
        """
        if has_plugins(HookType.SESSION_RESET):
            from ..plugins.hooks.session import SessionResetPayload

            payload = SessionResetPayload(previous_context=self.ctx)
            _run_async_in_thread(
                invoke_hook(HookType.SESSION_RESET, payload, backend=self.backend)
            )
        # Bypass the setter — a reset is a lifecycle event, not an interaction.
        # new_instance() preserves ChatContext config (model_id, compactor); the
        # base Context implementation falls back to reset_to_new().
        self._ctx = self._ctx.new_instance()
        self._interaction_count = 0

    def cleanup(self, *, exception: BaseException | None = None) -> None:
        """Clean up session resources and deregister session-scoped plugins.

        Args:
            exception: Optional exception that triggered cleanup. Forwarded
                to plugins via `SessionCleanupPayload.exception`.
        """
        if has_plugins(HookType.SESSION_CLEANUP):
            from ..plugins.hooks.session import SessionCleanupPayload

            payload = SessionCleanupPayload(
                session_id=self.id,
                context=self.ctx,
                interaction_count=self._interaction_count,
                exception=exception,
            )
            _run_async_in_thread(
                invoke_hook(HookType.SESSION_CLEANUP, payload, backend=self.backend)
            )

        # Deregister session-scoped plugins — must run whenever plugins are
        # enabled, regardless of whether any plugin subscribes to SESSION_CLEANUP.
        if has_plugins():
            from ..plugins.manager import deregister_session_plugins

            deregister_session_plugins(self.id)

    @overload
    def act(
        self,
        action: Component[S],
        *,
        requirements: list[Requirement] | None = None,
        strategy: SamplingStrategy | None = RejectionSamplingStrategy(loop_budget=2),
        return_sampling_results: Literal[False] = False,
        format: type[BaseModelSubclass] | None = None,
        model_options: dict | None = None,
        tool_calls: bool = False,
    ) -> ComputedModelOutputThunk[S]: ...

    @overload
    def act(
        self,
        action: Component[S],
        *,
        requirements: list[Requirement] | None = None,
        strategy: SamplingStrategy | None = RejectionSamplingStrategy(loop_budget=2),
        return_sampling_results: Literal[True],
        format: type[BaseModelSubclass] | None = None,
        model_options: dict | None = None,
        tool_calls: bool = False,
    ) -> SamplingResult[S]: ...

    def act(
        self,
        action: Component[S],
        *,
        requirements: list[Requirement] | None = None,
        strategy: SamplingStrategy | None = RejectionSamplingStrategy(loop_budget=2),
        return_sampling_results: bool = False,
        format: type[BaseModelSubclass] | None = None,
        model_options: dict | None = None,
        tool_calls: bool = False,
    ) -> ModelOutputThunk[S] | SamplingResult:
        """Runs a generic action, and adds both the action and the result to the context.

        Args:
            action: the Component from which to generate.
            requirements: used as additional requirements when a sampling strategy is provided
            strategy: a SamplingStrategy that describes the strategy for validating and repairing/retrying for the instruct-validate-repair pattern. None means that no particular sampling strategy is used.
            return_sampling_results: attach the (successful and failed) sampling attempts to the results.
            format: Constrains generation to JSON matching this Pydantic
                schema. The result's `.value` is always a JSON string — not a
                parsed model instance. Parse with
                `MyModel.model_validate_json(str(result))` to get a typed instance.
            model_options: additional model options, which will upsert into the model/backend's defaults.
            tool_calls: if true, tool calling is enabled.

        Returns:
            A ModelOutputThunk if `return_sampling_results` is `False`, else returns a `SamplingResult`.
        """
        r = mfuncs.act(
            action,
            context=self.ctx,
            backend=self.backend,
            requirements=requirements,
            strategy=strategy,
            return_sampling_results=return_sampling_results,
            format=format,
            model_options=model_options,
            tool_calls=tool_calls,
        )  # type: ignore

        if isinstance(r, SamplingResult):
            self.ctx = r.result_ctx
            return r
        else:
            result, context = r
            self.ctx = context
            return result

    @overload
    def instruct(
        self,
        description: str,
        *,
        images: list[ImageBlock | ImageUrlBlock] | list[PILImage.Image] | None = None,
        audio: list[AudioBlock | AudioUrlBlock] | None = None,
        requirements: list[Requirement | str] | None = None,
        icl_examples: list[str | CBlock] | None = None,
        grounding_context: dict[str, str | CBlock | ModelOutputThunk | Component]
        | None = None,
        user_variables: dict[str, str] | None = None,
        prefix: str | CBlock | None = None,
        output_prefix: str | CBlock | None = None,
        strategy: SamplingStrategy | None = RejectionSamplingStrategy(loop_budget=2),
        return_sampling_results: Literal[False] = False,
        format: type[BaseModelSubclass] | None = None,
        model_options: dict | None = None,
        tool_calls: bool = False,
    ) -> ComputedModelOutputThunk[str]: ...

    @overload
    def instruct(
        self,
        description: str,
        *,
        images: list[ImageBlock | ImageUrlBlock] | list[PILImage.Image] | None = None,
        audio: list[AudioBlock | AudioUrlBlock] | None = None,
        requirements: list[Requirement | str] | None = None,
        icl_examples: list[str | CBlock] | None = None,
        grounding_context: dict[str, str | CBlock | ModelOutputThunk | Component]
        | None = None,
        user_variables: dict[str, str] | None = None,
        prefix: str | CBlock | None = None,
        output_prefix: str | CBlock | None = None,
        strategy: SamplingStrategy | None = RejectionSamplingStrategy(loop_budget=2),
        return_sampling_results: Literal[True],
        format: type[BaseModelSubclass] | None = None,
        model_options: dict | None = None,
        tool_calls: bool = False,
    ) -> SamplingResult[str]: ...

    def instruct(
        self,
        description: str,
        *,
        images: list[ImageBlock | ImageUrlBlock] | list[PILImage.Image] | None = None,
        audio: list[AudioBlock | AudioUrlBlock] | None = None,
        requirements: list[Requirement | str] | None = None,
        icl_examples: list[str | CBlock] | None = None,
        grounding_context: dict[str, str | CBlock | ModelOutputThunk | Component]
        | None = None,
        user_variables: dict[str, str] | None = None,
        prefix: str | CBlock | None = None,
        output_prefix: str | CBlock | None = None,
        strategy: SamplingStrategy | None = RejectionSamplingStrategy(loop_budget=2),
        return_sampling_results: bool = False,
        format: type[BaseModelSubclass] | None = None,
        model_options: dict | None = None,
        tool_calls: bool = False,
    ) -> ModelOutputThunk[str] | SamplingResult:
        """Generates from an instruction.

        Args:
            description: The description of the instruction.
            requirements: A list of requirements that the instruction can be validated against.
            icl_examples: A list of in-context-learning examples that the instruction can be validated against.
            grounding_context: A list of grounding contexts that the instruction can use. They can bind as variables using a (key: str, value: str | CBlock | Component) tuple.
            user_variables: A dict of user-defined variables used to fill in Jinja placeholders in other parameters. This requires that all other provided parameters are provided as strings.
            prefix: A prefix string or ContentBlock to use when generating the instruction.
            output_prefix: A string or ContentBlock that defines a prefix for the output generation. Usually you do not need this.
            strategy: A SamplingStrategy that describes the strategy for validating and repairing/retrying for the instruct-validate-repair pattern. None means that no particular sampling strategy is used.
            return_sampling_results: attach the (successful and failed) sampling attempts to the results.
            format: Constrains generation to JSON matching this Pydantic
                schema. The result's `.value` is always a JSON string — not a
                parsed model instance. Parse with
                `MyModel.model_validate_json(str(result))` to get a typed instance.
            model_options: Additional model options, which will upsert into the model/backend's defaults.
            tool_calls: If true, tool calling is enabled.
            images: A list of images to be used in the instruction or None if none.
            audio: A list of audio blocks to be used in the instruction or None if none.

        Returns:
            A `ModelOutputThunk` if `return_sampling_results` is `False`,
            else a `SamplingResult`.
        """
        r = mfuncs.instruct(
            description,
            context=self.ctx,
            backend=self.backend,
            images=images,
            audio=audio,
            requirements=requirements,
            icl_examples=icl_examples,
            grounding_context=grounding_context,
            user_variables=user_variables,
            prefix=prefix,
            output_prefix=output_prefix,
            strategy=strategy,
            return_sampling_results=return_sampling_results,  # type: ignore
            format=format,
            model_options=model_options,
            tool_calls=tool_calls,
        )

        if isinstance(r, SamplingResult):
            self.ctx = r.result_ctx
            return r
        else:
            # It's a tuple[ModelOutputThunk, Context].
            result, context = r
            self.ctx = context
            return result

    def chat(
        self,
        content: str,
        role: Message.Role = "user",
        *,
        images: list[ImageBlock | ImageUrlBlock] | list[PILImage.Image] | None = None,
        audio: list[AudioBlock | AudioUrlBlock] | None = None,
        documents: collections.abc.Iterable[str | Document] | None = None,
        user_variables: dict[str, str] | None = None,
        format: type[BaseModelSubclass] | None = None,
        model_options: dict | None = None,
        tool_calls: bool = False,
    ) -> Message:
        """Sends a simple chat message and returns the response. Adds both messages to the Context.

        Args:
            content: The message text to send.
            role: The role for the outgoing message (default `"user"`).
            images: Optional list of images to include in the message.
            audio: Optional list of audio blocks to include in the message.
            documents: Optional documents to attach to the message. Each element
                may be a string or a `Document` object.
            user_variables: Optional Jinja variable substitutions applied to `content`.
            format: Optional Pydantic model for constrained decoding of the response.
            model_options: Additional model options to merge with backend defaults.
            tool_calls: If true, tool calling is enabled.

        Returns:
            The assistant `Message` response.
        """
        result, context = mfuncs.chat(
            content=content,
            context=self.ctx,
            backend=self.backend,
            role=role,
            images=images,
            audio=audio,
            documents=documents,
            user_variables=user_variables,
            format=format,
            model_options=model_options,
            tool_calls=tool_calls,
        )

        self.ctx = context
        return result

    def validate(
        self,
        reqs: Requirement | list[Requirement],
        *,
        output: CBlock | ModelOutputThunk | None = None,
        format: type[BaseModelSubclass] | None = None,
        model_options: dict | None = None,
        generate_logs: list[GenerateLog] | None = None,
        input: CBlock | ModelOutputThunk | None = None,
    ) -> list[ValidationResult]:
        """Validates a set of requirements over the output (if provided) or the current context (if the output is not provided).

        Args:
            reqs: A single `Requirement` or a list of them to validate.
            output: Optional model output to validate against instead of the context.
            format: Optional Pydantic model for constrained decoding.
            model_options: Additional model options to merge with backend defaults.
            generate_logs: Optional list to append generation logs to.
            input: Optional input to include alongside `output` when validating.

        Returns:
            List of `ValidationResult` objects, one per requirement.
        """
        return mfuncs.validate(
            reqs=reqs,
            context=self.ctx,
            backend=self.backend,
            output=output,
            format=format,
            model_options=model_options,
            generate_logs=generate_logs,
            input=input,
        )

    def query(
        self,
        obj: Any,
        query: str,
        *,
        format: type[BaseModelSubclass] | None = None,
        model_options: dict | None = None,
        tool_calls: bool = False,
    ) -> ComputedModelOutputThunk:
        """Query method for retrieving information from an object.

        Args:
            obj : The object to be queried. It should be an instance of MObject or can be converted to one if necessary.
            query:  The string representing the query to be executed against the object.
            format:  format for output parsing.
            model_options: Model options to pass to the backend.
            tool_calls: If true, the model may make tool calls. Defaults to False.

        Returns:
            ComputedModelOutputThunk: The result of the query as processed by the backend.
        """
        result, context = mfuncs.query(
            obj=obj,
            query=query,
            context=self.ctx,
            backend=self.backend,
            format=format,
            model_options=model_options,
            tool_calls=tool_calls,
        )
        self.ctx = context
        return result

    def transform(
        self,
        obj: Any,
        transformation: str,
        *,
        format: type[BaseModelSubclass] | None = None,
        model_options: dict | None = None,
    ) -> ModelOutputThunk | Any:
        """Transform method for creating a new object with the transformation applied.

        Args:
            obj : The object to be queried. It should be an instance of MObject or can be converted to one if necessary.
            transformation:  The string representing the query to be executed against the object.
            format: format for output parsing; usually not needed with transform.
            model_options: Model options to pass to the backend.

        Returns:
            ModelOutputThunk|Any: The result of the transformation as processed by the backend. If no tools were called,
            the return type will be always be ModelOutputThunk. If a tool was called, the return type will be the return type
            of the function called, usually the type of the object passed in.
        """
        result, context = mfuncs.transform(
            obj=obj,
            transformation=transformation,
            context=self.ctx,
            backend=self.backend,
            format=format,
            model_options=model_options,
        )
        self.ctx = context
        return result

    @overload
    async def aact(
        self,
        action: Component[S],
        *,
        requirements: list[Requirement] | None = None,
        strategy: None = None,
        return_sampling_results: Literal[False] = False,
        format: type[BaseModelSubclass] | None = None,
        model_options: dict | None = None,
        tool_calls: bool = False,
        await_result: Literal[True],
    ) -> ComputedModelOutputThunk[S]: ...

    @overload
    async def aact(
        self,
        action: Component[S],
        *,
        requirements: list[Requirement] | None = None,
        strategy: SamplingStrategy,
        return_sampling_results: Literal[False] = False,
        format: type[BaseModelSubclass] | None = None,
        model_options: dict | None = None,
        tool_calls: bool = False,
        await_result: bool = False,
    ) -> ComputedModelOutputThunk[S]: ...

    @overload
    async def aact(
        self,
        action: Component[S],
        *,
        requirements: list[Requirement] | None = None,
        strategy: None = None,
        return_sampling_results: Literal[False] = False,
        format: type[BaseModelSubclass] | None = None,
        model_options: dict | None = None,
        tool_calls: bool = False,
        await_result: Literal[False] = False,
    ) -> ModelOutputThunk[S]: ...

    @overload
    async def aact(
        self,
        action: Component[S],
        *,
        requirements: list[Requirement] | None = None,
        strategy: SamplingStrategy | None = RejectionSamplingStrategy(loop_budget=2),
        return_sampling_results: Literal[True],
        format: type[BaseModelSubclass] | None = None,
        model_options: dict | None = None,
        tool_calls: bool = False,
        await_result: bool = False,
    ) -> SamplingResult[S]: ...

    async def aact(
        self,
        action: Component[S],
        *,
        requirements: list[Requirement] | None = None,
        strategy: SamplingStrategy | None = RejectionSamplingStrategy(loop_budget=2),
        return_sampling_results: bool = False,
        format: type[BaseModelSubclass] | None = None,
        model_options: dict | None = None,
        tool_calls: bool = False,
        await_result: bool = False,
    ) -> ModelOutputThunk[S] | SamplingResult:
        """Runs a generic action, and adds both the action and the result to the context.

        Args:
            action: the Component from which to generate.
            requirements: used as additional requirements when a sampling strategy is provided
            strategy: a SamplingStrategy that describes the strategy for validating and repairing/retrying for the instruct-validate-repair pattern. None means that no particular sampling strategy is used.
            return_sampling_results: attach the (successful and failed) sampling attempts to the results.
            format: Constrains generation to JSON matching this Pydantic
                schema. The result's `.value` is always a JSON string — not a
                parsed model instance. Parse with
                `MyModel.model_validate_json(str(result))` to get a typed instance.
            model_options: additional model options, which will upsert into the model/backend's defaults.
            tool_calls: if true, tool calling is enabled.
            await_result: if False and strategy is None, returns uncomputed ModelOutputThunk for streaming. Default is False.

        Returns:
            A ModelOutputThunk if `return_sampling_results` is `False`, else returns a `SamplingResult`.
            When await_result=False and strategy=None, returns uncomputed ModelOutputThunk that can be streamed.
        """
        r = await mfuncs.aact(
            action,
            context=self.ctx,
            backend=self.backend,
            requirements=requirements,
            strategy=strategy,
            return_sampling_results=return_sampling_results,
            format=format,
            model_options=model_options,
            tool_calls=tool_calls,
            await_result=await_result,
        )  # type: ignore

        if isinstance(r, SamplingResult):
            self.ctx = r.result_ctx
            return r
        else:
            result, context = r
            self.ctx = context
            return result

    @overload
    async def ainstruct(
        self,
        description: str,
        *,
        images: list[ImageBlock | ImageUrlBlock] | list[PILImage.Image] | None = None,
        audio: list[AudioBlock | AudioUrlBlock] | None = None,
        requirements: list[Requirement | str] | None = None,
        icl_examples: list[str | CBlock] | None = None,
        grounding_context: dict[str, str | CBlock | ModelOutputThunk | Component]
        | None = None,
        user_variables: dict[str, str] | None = None,
        prefix: str | CBlock | None = None,
        output_prefix: str | CBlock | None = None,
        strategy: None = None,
        return_sampling_results: Literal[False] = False,
        format: type[BaseModelSubclass] | None = None,
        model_options: dict | None = None,
        tool_calls: bool = False,
        await_result: Literal[True],
    ) -> ComputedModelOutputThunk[str]: ...

    @overload
    async def ainstruct(
        self,
        description: str,
        *,
        images: list[ImageBlock | ImageUrlBlock] | list[PILImage.Image] | None = None,
        audio: list[AudioBlock | AudioUrlBlock] | None = None,
        requirements: list[Requirement | str] | None = None,
        icl_examples: list[str | CBlock] | None = None,
        grounding_context: dict[str, str | CBlock | ModelOutputThunk | Component]
        | None = None,
        user_variables: dict[str, str] | None = None,
        prefix: str | CBlock | None = None,
        output_prefix: str | CBlock | None = None,
        strategy: SamplingStrategy,
        return_sampling_results: Literal[False] = False,
        format: type[BaseModelSubclass] | None = None,
        model_options: dict | None = None,
        tool_calls: bool = False,
        await_result: bool = False,
    ) -> ComputedModelOutputThunk[str]: ...

    @overload
    async def ainstruct(
        self,
        description: str,
        *,
        images: list[ImageBlock | ImageUrlBlock] | list[PILImage.Image] | None = None,
        audio: list[AudioBlock | AudioUrlBlock] | None = None,
        requirements: list[Requirement | str] | None = None,
        icl_examples: list[str | CBlock] | None = None,
        grounding_context: dict[str, str | CBlock | ModelOutputThunk | Component]
        | None = None,
        user_variables: dict[str, str] | None = None,
        prefix: str | CBlock | None = None,
        output_prefix: str | CBlock | None = None,
        strategy: None = None,
        return_sampling_results: Literal[False] = False,
        format: type[BaseModelSubclass] | None = None,
        model_options: dict | None = None,
        tool_calls: bool = False,
        await_result: Literal[False] = False,
    ) -> ModelOutputThunk[str]: ...

    @overload
    async def ainstruct(
        self,
        description: str,
        *,
        images: list[ImageBlock | ImageUrlBlock] | list[PILImage.Image] | None = None,
        audio: list[AudioBlock | AudioUrlBlock] | None = None,
        requirements: list[Requirement | str] | None = None,
        icl_examples: list[str | CBlock] | None = None,
        grounding_context: dict[str, str | CBlock | ModelOutputThunk | Component]
        | None = None,
        user_variables: dict[str, str] | None = None,
        prefix: str | CBlock | None = None,
        output_prefix: str | CBlock | None = None,
        strategy: SamplingStrategy | None = RejectionSamplingStrategy(loop_budget=2),
        return_sampling_results: Literal[True],
        format: type[BaseModelSubclass] | None = None,
        model_options: dict | None = None,
        tool_calls: bool = False,
        await_result: bool = False,
    ) -> SamplingResult[str]: ...

    async def ainstruct(
        self,
        description: str,
        *,
        images: list[ImageBlock | ImageUrlBlock] | list[PILImage.Image] | None = None,
        audio: list[AudioBlock | AudioUrlBlock] | None = None,
        requirements: list[Requirement | str] | None = None,
        icl_examples: list[str | CBlock] | None = None,
        grounding_context: dict[str, str | CBlock | ModelOutputThunk | Component]
        | None = None,
        user_variables: dict[str, str] | None = None,
        prefix: str | CBlock | None = None,
        output_prefix: str | CBlock | None = None,
        strategy: SamplingStrategy | None = RejectionSamplingStrategy(loop_budget=2),
        return_sampling_results: bool = False,
        format: type[BaseModelSubclass] | None = None,
        model_options: dict | None = None,
        tool_calls: bool = False,
        await_result: bool = False,
    ) -> ModelOutputThunk[str] | SamplingResult[str]:
        """Generates from an instruction.

        Args:
            description: The description of the instruction.
            requirements: A list of requirements that the instruction can be validated against.
            icl_examples: A list of in-context-learning examples that the instruction can be validated against.
            grounding_context: A list of grounding contexts that the instruction can use. They can bind as variables using a (key: str, value: str | ContentBlock) tuple.
            user_variables: A dict of user-defined variables used to fill in Jinja placeholders in other parameters. This requires that all other provided parameters are provided as strings.
            prefix: A prefix string or ContentBlock to use when generating the instruction.
            output_prefix: A string or ContentBlock that defines a prefix for the output generation. Usually you do not need this.
            strategy: A SamplingStrategy that describes the strategy for validating and repairing/retrying for the instruct-validate-repair pattern. None means that no particular sampling strategy is used.
            return_sampling_results: attach the (successful and failed) sampling attempts to the results.
            format: Constrains generation to JSON matching this Pydantic
                schema. The result's `.value` is always a JSON string — not a
                parsed model instance. Parse with
                `MyModel.model_validate_json(str(result))` to get a typed instance.
            model_options: Additional model options, which will upsert into the model/backend's defaults.
            tool_calls: If true, tool calling is enabled.
            images: A list of images to be used in the instruction or None if none.
            audio: A list of audio blocks to be used in the instruction or None if none.
            await_result: if False and strategy is None, returns uncomputed ModelOutputThunk for streaming. Default is False.

        Returns:
            A `ComputedModelOutputThunk` if `strategy` is `None` and `await_results` is `False`,
            else returns a `ModelOutputThunk` if `return_sampling_results` is `False`,
            else a `SamplingResult`.
        """
        r = await mfuncs.ainstruct(
            description,
            context=self.ctx,
            backend=self.backend,
            images=images,
            audio=audio,
            requirements=requirements,
            icl_examples=icl_examples,
            grounding_context=grounding_context,
            user_variables=user_variables,
            prefix=prefix,
            output_prefix=output_prefix,
            strategy=strategy,
            return_sampling_results=return_sampling_results,  # type: ignore
            format=format,
            model_options=model_options,
            tool_calls=tool_calls,
            await_result=await_result,
        )

        if isinstance(r, SamplingResult):
            self.ctx = r.result_ctx
            return r
        else:
            # It's a tuple[ModelOutputThunk, Context].
            result, context = r
            self.ctx = context
            return result

    async def achat(
        self,
        content: str,
        role: Message.Role = "user",
        *,
        images: list[ImageBlock | ImageUrlBlock] | list[PILImage.Image] | None = None,
        audio: list[AudioBlock | AudioUrlBlock] | None = None,
        documents: collections.abc.Iterable[str | Document] | None = None,
        user_variables: dict[str, str] | None = None,
        format: type[BaseModelSubclass] | None = None,
        model_options: dict | None = None,
        tool_calls: bool = False,
    ) -> Message:
        """Sends a simple chat message and returns the response. Adds both messages to the Context.

        Args:
            content: The message text to send.
            role: The role for the outgoing message (default `"user"`).
            images: Optional list of images to include in the message.
            audio: Optional list of audio blocks to include in the message.
            documents: Optional documents to attach to the message. Each element
                may be a string or a `Document` object.
            user_variables: Optional Jinja variable substitutions applied to `content`.
            format: Optional Pydantic model for constrained decoding of the response.
            model_options: Additional model options to merge with backend defaults.
            tool_calls: If true, tool calling is enabled.

        Returns:
            The assistant `Message` response.
        """
        result, context = await mfuncs.achat(
            content=content,
            context=self.ctx,
            backend=self.backend,
            role=role,
            images=images,
            audio=audio,
            documents=documents,
            user_variables=user_variables,
            format=format,
            model_options=model_options,
            tool_calls=tool_calls,
        )

        self.ctx = context
        return result

    async def avalidate(
        self,
        reqs: Requirement | list[Requirement],
        *,
        output: CBlock | ModelOutputThunk | None = None,
        format: type[BaseModelSubclass] | None = None,
        model_options: dict | None = None,
        generate_logs: list[GenerateLog] | None = None,
        input: CBlock | ModelOutputThunk | None = None,
    ) -> list[ValidationResult]:
        """Validates a set of requirements over the output (if provided) or the current context (if the output is not provided).

        Args:
            reqs: A single `Requirement` or a list of them to validate.
            output: Optional model output to validate against instead of the context.
            format: Optional Pydantic model for constrained decoding.
            model_options: Additional model options to merge with backend defaults.
            generate_logs: Optional list to append generation logs to.
            input: Optional input to include alongside `output` when validating.

        Returns:
            List of `ValidationResult` objects, one per requirement.
        """
        return await mfuncs.avalidate(
            reqs=reqs,
            context=self.ctx,
            backend=self.backend,
            output=output,
            format=format,
            model_options=model_options,
            generate_logs=generate_logs,
            input=input,
        )

    @overload
    async def aquery(
        self,
        obj: Any,
        query: str,
        *,
        format: type[BaseModelSubclass] | None = None,
        model_options: dict | None = None,
        tool_calls: bool = False,
        await_result: Literal[True],
    ) -> ComputedModelOutputThunk: ...

    @overload
    async def aquery(
        self,
        obj: Any,
        query: str,
        *,
        format: type[BaseModelSubclass] | None = None,
        model_options: dict | None = None,
        tool_calls: bool = False,
        await_result: Literal[False] = False,
    ) -> ModelOutputThunk: ...

    async def aquery(
        self,
        obj: Any,
        query: str,
        *,
        format: type[BaseModelSubclass] | None = None,
        model_options: dict | None = None,
        tool_calls: bool = False,
        await_result: bool = False,
    ) -> ModelOutputThunk:
        """Query method for retrieving information from an object.

        Args:
            obj : The object to be queried. It should be an instance of MObject or can be converted to one if necessary.
            query:  The string representing the query to be executed against the object.
            format:  format for output parsing.
            model_options: Model options to pass to the backend.
            tool_calls: If true, the model may make tool calls. Defaults to False.
            await_result: if False (default), returns uncomputed ModelOutputThunk. If True, awaits and returns ComputedModelOutputThunk.

        Returns:
            ModelOutputThunk: The result of the query as processed by the backend.
        """
        result, context = await mfuncs.aquery(
            obj=obj,
            query=query,
            context=self.ctx,
            backend=self.backend,
            format=format,
            model_options=model_options,
            tool_calls=tool_calls,
            await_result=await_result,  # type: ignore[call-overload]
        )
        self.ctx = context
        return result

    async def atransform(
        self,
        obj: Any,
        transformation: str,
        *,
        format: type[BaseModelSubclass] | None = None,
        model_options: dict | None = None,
    ) -> ModelOutputThunk | Any:
        """Transform method for creating a new object with the transformation applied.

        Args:
            obj: The object to be queried. It should be an instance of MObject or can be converted to one if necessary.
            transformation:  The string representing the query to be executed against the object.
            format: format for output parsing; usually not needed with transform.
            model_options: Model options to pass to the backend.

        Returns:
            ModelOutputThunk|Any: The result of the transformation as processed by the backend. If no tools were called,
            the return type will be always be ModelOutputThunk. If a tool was called, the return type will be the return type
            of the function called, usually the type of the object passed in.
        """
        result, context = await mfuncs.atransform(
            obj=obj,
            transformation=transformation,
            context=self.ctx,
            backend=self.backend,
            format=format,
            model_options=model_options,
        )
        self.ctx = context
        return result

    @classmethod
    def powerup(cls, powerup_cls: type) -> None:
        """Appends methods in a class object `powerup_cls` to MelleaSession.

        Iterates over all functions defined on `powerup_cls` and attaches each
        one as a method on the `MelleaSession` class, effectively extending
        the session with domain-specific helpers at runtime.

        Args:
            powerup_cls (type): A class whose functions should be added to
                `MelleaSession` as instance methods.
        """
        for name, fn in inspect.getmembers(powerup_cls, predicate=inspect.isfunction):
            setattr(cls, name, fn)

    # ###############################
    #  Convenience functions
    # ###############################
    def last_prompt(self) -> str | list[dict] | None:
        """Returns the last prompt that has been called from the session context.

        Returns:
            A string if the last prompt was a raw call to the model OR a list of messages (as role-msg-dicts). Is None if none could be found.
        """
        op = self.ctx.last_output()
        if op is None:
            return None
        log = op._generate_log
        if isinstance(log, GenerateLog):
            return log.prompt
        elif isinstance(log, list):
            last_el = log[-1]
            if isinstance(last_el, GenerateLog):
                return last_el.prompt
        return None
