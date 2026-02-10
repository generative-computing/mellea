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

BaseModelSubclass = typing_extensions.TypeVar(
    "BaseModelSubclass", bound=pydantic.BaseModel
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
