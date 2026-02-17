"""Interfaces for Backends and Generation."""

import abc
import asyncio
import itertools
from collections.abc import Sequence
from typing import overload

import pydantic
import typing_extensions

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

    async def generate_from_context_with_hooks(
        self,
        action: Component[C] | CBlock,
        ctx: Context,
        *,
        format: type[BaseModelSubclass] | None = None,
        model_options: dict | None = None,
        tool_calls: bool = False,
    ) -> tuple[ModelOutputThunk[C], Context]:
        """Wraps ``generate_from_context`` with generation_pre_call / generation_post_call hooks.

        Falls through to ``generate_from_context`` directly when no plugins are configured.
        """
        import time

        from mellea.plugins.manager import has_plugins, invoke_hook
        from mellea.plugins.types import MelleaHookType

        if has_plugins():
            from mellea.plugins.hooks.generation import (
                GenerationPostCallPayload,
                GenerationPreCallPayload,
            )

            pre_payload = GenerationPreCallPayload(
                action=action,
                context=ctx,
                formatted_prompt="",
                model_options=model_options or {},
                format=format,
                tools=None,
            )
            _, pre_payload = await invoke_hook(
                MelleaHookType.GENERATION_PRE_CALL,
                pre_payload,
                backend=self,
                context=ctx,
            )
            model_options = pre_payload.model_options
            format = pre_payload.format

        t0 = time.monotonic()
        out_result, new_ctx = await self.generate_from_context(
            action,
            ctx,
            format=format,
            model_options=model_options,
            tool_calls=tool_calls,
        )

        if has_plugins():
            from mellea.plugins.hooks.generation import GenerationPostCallPayload

            post_payload = GenerationPostCallPayload(
                model_output=out_result, latency_ms=int((time.monotonic() - t0) * 1000)
            )
            await invoke_hook(
                MelleaHookType.GENERATION_POST_CALL,
                post_payload,
                backend=self,
                context=new_ctx,
            )

        return out_result, new_ctx

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
