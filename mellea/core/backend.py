"""Interfaces for Backends and Generation."""

import abc
import asyncio
import functools
import itertools
import time
from collections.abc import Sequence
from typing import overload

import pydantic
import typing_extensions

from ..plugins.manager import has_plugins, invoke_hook
from ..plugins.types import HookType
from .base import C, CBlock, Component, Context, ModelOutputThunk
from .utils import FancyLogger

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
    """An abstract `Backend`."""

    @abc.abstractmethod
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

    @abc.abstractmethod
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
        """

    async def do_generate_walk(
        self, action: CBlock | Component | ModelOutputThunk
    ) -> None:
        """Does the generation walk."""
        _to_compute = list(generate_walk(action))
        coroutines = [x.avalue() for x in _to_compute]
        # The following log message might get noisy. Feel free to remove if so.
        if len(_to_compute) > 0:
            FancyLogger.get_logger().info(
                f"generate_from_chat_context awaited on {len(_to_compute)} uncomputed mots."
            )
        await asyncio.gather(*coroutines)

    async def do_generate_walks(
        self, actions: list[CBlock | Component | ModelOutputThunk]
    ) -> None:
        """Does the generation walk."""
        _to_compute = []
        for action in actions:
            _to_compute.extend(list(generate_walk(action)))
        coroutines = [x.avalue() for x in _to_compute]
        # The following log message might get noisy. Feel free to remove if so.
        if len(_to_compute) > 0:
            FancyLogger.get_logger().info(
                f"generate_from_chat_context awaited on {len(_to_compute)} uncomputed mots."
            )
        await asyncio.gather(*coroutines)

    def __init_subclass__(cls):
        """Injects generation hooks in concrete backends.

        .. note::
            The ``generation_post_call`` hook fires via ``_on_computed``, a
            callback set on the returned ``ModelOutputThunk``:

            - **Lazy MOTs** (normal path): ``_on_computed`` is called inside
              ``ModelOutputThunk.astream`` after ``post_process`` completes and
              the value is fully materialized. ``latency_ms`` reflects the full
              time from the ``generate_from_context`` call to value availability.
              ``model_output`` replacement is supported — the original MOT's
              output fields are updated in-place via ``_copy_from``.
            - **Already-computed MOTs** (e.g. cached responses): ``astream``
              returns early and never invokes ``_on_computed``, so the hook is
              fired inline here before returning. ``model_output`` replacement is
              supported in this path.

            Both paths support ``model_output`` replacement by plugins.

        """
        if "generate_from_context" in cls.__dict__:
            original = cls.__dict__["generate_from_context"]

            @functools.wraps(original)
            async def wrapped(
                self,
                action: Component[C] | CBlock,
                ctx: Context,
                *,
                format: type[BaseModelSubclass] | None = None,
                model_options: dict | None = None,
                tool_calls: bool = False,
            ):
                if has_plugins(HookType.GENERATION_PRE_CALL):
                    from ..plugins.hooks.generation import GenerationPreCallPayload

                    pre_payload = GenerationPreCallPayload(
                        action=action,
                        context=ctx,
                        format=format,
                        model_options=model_options or {},
                        tool_calls=tool_calls,
                    )
                    _, pre_payload = await invoke_hook(
                        HookType.GENERATION_PRE_CALL,
                        pre_payload,
                        backend=self,
                        context=ctx,
                    )
                    model_options = pre_payload.model_options
                    format = pre_payload.format
                    tool_calls = pre_payload.tool_calls
                start_time = time.monotonic()
                out_result, new_ctx = await original(
                    self,
                    action,
                    ctx,
                    format=format,
                    model_options=model_options,
                    tool_calls=tool_calls,
                )

                _backend_ref = self
                _ctx_ref = new_ctx

                async def _fire_post_call(mot: ModelOutputThunk) -> ModelOutputThunk:
                    """Fires GENERATION_POST_CALL and returns the (possibly replaced) MOT."""
                    mot._on_computed = None  # prevent double-firing
                    if not has_plugins(HookType.GENERATION_POST_CALL):
                        return mot
                    from ..plugins.hooks.generation import GenerationPostCallPayload

                    latency_ms = (time.monotonic() - start_time) * 1000
                    glog = getattr(mot, "_generate_log", None)
                    post_payload = GenerationPostCallPayload(
                        prompt=glog.prompt if glog else "",
                        model_output=mot,
                        latency_ms=latency_ms,
                    )
                    _, post_payload = await invoke_hook(
                        HookType.GENERATION_POST_CALL,
                        post_payload,
                        backend=_backend_ref,
                        context=_ctx_ref,
                    )
                    if (
                        post_payload.model_output is not None
                        and post_payload.model_output is not mot
                    ):
                        return post_payload.model_output
                    return mot

                out_result._on_computed = _fire_post_call

                # For already-computed MOTs (e.g. cached responses or test mocks),
                # astream() returns early so _on_computed never fires. Fire here
                # and use the return value to support model_output replacement.
                if getattr(out_result, "is_computed", lambda: False)():
                    out_result = await _fire_post_call(out_result)

                return out_result, new_ctx

            setattr(cls, "generate_from_context", wrapped)


def generate_walk(c: CBlock | Component | ModelOutputThunk) -> list[ModelOutputThunk]:
    """Returns the generation walk ordering for a Span."""
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
