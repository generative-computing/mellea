"""Abstract `Backend` interface and generation-walk utilities.

Defines the `Backend` abstract base class. Backend implementers override
`_generate_from_context` (context-aware single-action generation) and
`_generate_from_raw` (context-free batch generation); the public
`generate_from_context` and `generate_from_raw` wrappers on `Backend` own
hook firing and delegate to the backend implementations. Also provides
`generate_walk`, which traverses a `Component` tree to find un-computed
`ModelOutputThunk` leaves that need to be resolved before rendering.
"""

import abc
import asyncio
import functools
import itertools
import time
import uuid
from collections.abc import Sequence
from typing import Any, final, overload

import pydantic
import typing_extensions

from ..plugins.manager import has_plugins, invoke_hook
from ..plugins.types import HookType
from .base import C, CBlock, Component, Context, ModelOutputThunk
from .utils import MelleaLogger

# Necessary to define a type variable that has a default value.
# This is because VSCode's pyright static type checker instantiates
# all type parameters, order them in a specific order, and checks
# that a type parameter with a default is not followed by a type parameter without a default.
# This was originally specified in PEP 696 (Type Defaults for Type Parameters) and
# is now part of official spec https://typing.python.org/en/latest/spec/generics.html#type-parameter-defaults .
# For example, in mellea/stdlib/functional.py ,
#    def act(action: Component[S], ... format: type[BaseModelSubclass] | None = None, ...)
# gets instantiated as act[S, BaseModelSubclass].
# S is TypeVar("S", default=Any, covariant=True) which has a default.
#
BaseModelSubclass = typing_extensions.TypeVar(
    "BaseModelSubclass", bound=pydantic.BaseModel, default=pydantic.BaseModel
)  # must be a subclass of BaseModel


class Backend(abc.ABC):
    """Abstract base class for all inference backends.

    Backend implementers override `_generate_from_context` (context-aware
    single-action generation) and `_generate_from_raw` (context-free batch
    generation). The public `generate_from_context` and `generate_from_raw`
    methods own hook firing and delegate to the implementations. The
    `do_generate_walk` / `do_generate_walks` helpers can be used to
    pre-compute any unresolved `ModelOutputThunk` leaves before rendering.
    """

    _model_id: str
    """Resolved model identifier string. Must be set by every backend implementation."""

    _provider: str
    """Provider name (e.g. 'openai', 'ollama'). Must be set by every backend implementation."""

    @final
    async def generate_from_context(
        self,
        action: Component[C] | CBlock,
        ctx: Context,
        *,
        format: type[BaseModelSubclass] | None = None,
        model_options: dict | None = None,
        tool_calls: bool = False,
    ) -> tuple[ModelOutputThunk[C], Context]:
        """Generates a model output from a context. May not mutate the context. This must be called from a running event loop as it creates a task to run the generation request.

        Args:
            action: The last item of the context should be passed in as an `action` instead of as part of the `ctx`. See `docs/dev/generate_signature_decisions.md`.
            ctx: The rest of the context.
            format: A response format to used for structured outputs / constrained decoding.
            model_options: Any model options to upsert into the defaults for this call.
            tool_calls: If `True`, then tool calls are extracts from the `action` `Component`. Assumption: if tool_calls is enabled, then the action `Component` has a TemplateRepresentation

        Returns:
            a tuple of (ModelOutputThunk, Context) where the Context is the new context after the generation has been completed.
        """
        generation_id = str(uuid.uuid4())

        # --- generation_pre_call hook ---
        if has_plugins(HookType.GENERATION_PRE_CALL):
            from ..plugins.hooks.generation import GenerationPreCallPayload

            pre_payload = GenerationPreCallPayload(
                action=action,
                context=ctx,
                format=format,
                model_options=model_options or {},
                tool_calls=tool_calls,
                generation_id=generation_id,
            )
            _, pre_payload = await invoke_hook(
                HookType.GENERATION_PRE_CALL, pre_payload, backend=self
            )
            model_options = pre_payload.model_options
            format = pre_payload.format
            tool_calls = pre_payload.tool_calls

        try:
            mot, new_ctx = await self._generate_from_context(
                action,
                ctx,
                format=format,
                model_options=model_options,
                tool_calls=tool_calls,
            )
        except BaseException as e:
            # No MOT to fire the per-MOT error hook in astream from; emit here.
            if has_plugins(HookType.GENERATION_ERROR):
                from ..plugins.hooks.generation import GenerationErrorPayload

                await invoke_hook(
                    HookType.GENERATION_ERROR,
                    GenerationErrorPayload(
                        exception=e, model_output=None, generation_id=generation_id
                    ),
                    backend=self,
                )
            raise
        # Save the ID for the post_call / error hooks.
        mot._call.generation_id = generation_id
        return mot, new_ctx

    @abc.abstractmethod
    async def _generate_from_context(
        self,
        action: Component[C] | CBlock,
        ctx: Context,
        *,
        format: type[BaseModelSubclass] | None = None,
        model_options: dict | None = None,
        tool_calls: bool = False,
    ) -> tuple[ModelOutputThunk[C], Context]:
        """Backend implementers should override this method to generate the actual response.

        Args:
            action: The last item of the context should be passed in as an `action` instead of as part of the `ctx`. See `docs/dev/generate_signature_decisions.md`.
            ctx: The rest of the context.
            format: A response format to used for structured outputs / constrained decoding.
            model_options: Any model options to upsert into the defaults for this call.
            tool_calls: If `True`, then tool calls are extracts from the `action` `Component`. Assumption: if tool_calls is enabled, then the action `Component` has a TemplateRepresentation

        Returns:
            a tuple of (ModelOutputThunk, Context) where the Context is the new context after the generation has been completed.
        """
        ...

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

    @final
    async def generate_from_raw(
        self,
        actions: Sequence[Component[C] | CBlock],
        ctx: Context,
        *,
        format: type[BaseModelSubclass] | None = None,
        model_options: dict | None = None,
        tool_calls: bool = False,
    ) -> list[ModelOutputThunk]:
        """Generates a model output from the provided input. Does not use context or templates.

        Args:
            actions: list of actions to generate responses for. Each action is separate.
            ctx: context passed to generation. Currently not used in generate_from_raw
            format: A response format to used for structured outputs / constrained decoding. Note: some backends do not support this parameter. They will log warnings and continue to generate.
            model_options: Any model options to upsert into the defaults for this call.
            tool_calls: Always set to false unless supported by backend.

        Returns:
            list[ModelOutputThunk]: A list of output thunks, one per action, in the same order as `actions`.
        """
        gen_id = str(uuid.uuid4())

        # --- generation_batch_pre_call hook ---
        if has_plugins(HookType.GENERATION_BATCH_PRE_CALL):
            from ..plugins.hooks.generation import GenerationBatchPreCallPayload

            pre_payload = GenerationBatchPreCallPayload(
                actions=tuple(actions),
                generation_id=gen_id,
                model_options=model_options or {},
                format=format,
                tool_calls=tool_calls,
                num_actions=len(actions),
                model=self._model_id,
                provider=self._provider,
            )
            _, pre_payload = await invoke_hook(
                HookType.GENERATION_BATCH_PRE_CALL, pre_payload, backend=self
            )
            model_options = pre_payload.model_options
            format = pre_payload.format
            tool_calls = pre_payload.tool_calls

        _start = time.perf_counter()
        try:
            results, usage = await self._generate_from_raw(
                actions,
                ctx,
                format=format,
                model_options=model_options,
                tool_calls=tool_calls,
            )
        except BaseException as e:
            if has_plugins(HookType.GENERATION_BATCH_ERROR):
                from ..plugins.hooks.generation import GenerationBatchErrorPayload

                await invoke_hook(
                    HookType.GENERATION_BATCH_ERROR,
                    GenerationBatchErrorPayload(
                        generation_id=gen_id,
                        exception=e,
                        model=self._model_id,
                        provider=self._provider,
                        latency_ms=(time.perf_counter() - _start) * 1000,
                    ),
                )
            raise
        latency_ms = (time.perf_counter() - _start) * 1000

        # --- generation_batch_post_call hook ---
        if has_plugins(HookType.GENERATION_BATCH_POST_CALL):
            from ..plugins.hooks.generation import GenerationBatchPostCallPayload

            await invoke_hook(
                HookType.GENERATION_BATCH_POST_CALL,
                GenerationBatchPostCallPayload(
                    generation_id=gen_id,
                    model_outputs=results,
                    usage=usage,
                    model=self._model_id,
                    provider=self._provider,
                    latency_ms=latency_ms,
                ),
            )

        return results

    @abc.abstractmethod
    async def _generate_from_raw(
        self,
        actions: Sequence[Component[C] | CBlock],
        ctx: Context,
        *,
        format: type[BaseModelSubclass] | None = None,
        model_options: dict | None = None,
        tool_calls: bool = False,
    ) -> tuple[list[ModelOutputThunk], dict[str, Any] | None]:
        """Backend implementers should override this method to generate the actual responses.

        Backends should populate `mot.generation.usage` on each returned MOT
        when the underlying API exposes per-MOT token counts; otherwise leave
        it as `None` and report only the whole-batch aggregate via the
        returned tuple's second element.

        Args:
            actions: list of actions to generate responses for. Each action is separate.
            ctx: context passed to generation. Currently not used in `_generate_from_raw`.
            format: A response format to use for structured outputs / constrained decoding. Some backends do not support this parameter and will log warnings.
            model_options: Any model options to upsert into the defaults for this call.
            tool_calls: Always set to false unless supported by backend.

        Returns:
            tuple[list[ModelOutputThunk], dict | None]: `(results, usage)` where
                `results` is a list of model output thunks, one per action, and
                `usage` is the aggregate token-usage dict for the whole batch
                (OpenAI shape) or `None`.
        """

    async def do_generate_walk(
        self, action: CBlock | Component | ModelOutputThunk
    ) -> None:
        """Awaits all uncomputed `ModelOutputThunk` leaves reachable from `action`.

        Traverses the component tree rooted at `action` via `generate_walk`, collects
        any uncomputed `ModelOutputThunk` nodes, and concurrently awaits them all.

        Args:
            action (CBlock | Component | ModelOutputThunk): The root node to traverse.
        """
        _to_compute = list(generate_walk(action))
        coroutines = [x.avalue() for x in _to_compute]
        # The following log message might get noisy. Feel free to remove if so.
        if len(_to_compute) > 0:
            MelleaLogger.get_logger().info(
                f"generate_from_chat_context awaited on {len(_to_compute)} uncomputed mots."
            )
        await asyncio.gather(*coroutines)

    async def do_generate_walks(
        self, actions: list[CBlock | Component | ModelOutputThunk]
    ) -> None:
        """Awaits all uncomputed `ModelOutputThunk` leaves reachable from each action in `actions`.

        Traverses the component tree of every action in the list via `generate_walk`, collects
        all uncomputed `ModelOutputThunk` nodes across all actions, and concurrently awaits them.

        Args:
            actions (list[CBlock | Component | ModelOutputThunk]): The list of root nodes to traverse.
        """
        _to_compute = []
        for action in actions:
            _to_compute.extend(list(generate_walk(action)))
        coroutines = [x.avalue() for x in _to_compute]
        # The following log message might get noisy. Feel free to remove if so.
        if len(_to_compute) > 0:
            MelleaLogger.get_logger().info(
                f"generate_from_chat_context awaited on {len(_to_compute)} uncomputed mots."
            )
        await asyncio.gather(*coroutines)


def generate_walk(c: CBlock | Component | ModelOutputThunk) -> list[ModelOutputThunk]:
    """Return all uncomputed `ModelOutputThunk` leaves reachable from `c`.

    Args:
        c: A `CBlock`, `Component`, or `ModelOutputThunk` to traverse.

    Returns:
        A flat list of uncomputed `ModelOutputThunk` instances in the order
        they need to be resolved (depth-first over `Component.parts()`).

    Raises:
        ValueError: If any element encountered during traversal is not a `CBlock`,
            `Component`, or `ModelOutputThunk`.
    """
    match c:
        case ModelOutputThunk() if not c.is_computed():
            return [c]
        case CBlock():
            return []
        case Component():
            parts_walk = [generate_walk(p) for p in c.parts()]
            return list(itertools.chain.from_iterable(parts_walk))  # aka flatten
        case _:
            raise ValueError(
                f"parts should only contain CBlocks, Components, or ModelOutputThunks; found `{c!s:.10}{'...' if len(str(c)) > 10 else ''}` (type: {type(c)})"
            )
