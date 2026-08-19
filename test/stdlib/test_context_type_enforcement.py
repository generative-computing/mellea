# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for context-type enforcement in the functional layer (issue #1522).

Mellea functions must return the same `Context` subtype they were given. These
tests use `DummyBackend` (no LLM) to verify the runtime guard: the type is
preserved on the happy path, a mismatch raises `ContextTypeMismatchError`, and
the `allow_context_type_change` escape hatch permits a deliberate change.
"""

import pytest

from mellea.backends.dummy import DummyBackend
from mellea.core import (
    BaseModelSubclass,
    C,
    CBlock,
    Component,
    Context,
    ContextTypeMismatchError,
    ModelOutputThunk,
)
from mellea.stdlib.components import Message
from mellea.stdlib.context import ChatContext, SimpleContext
from mellea.stdlib.functional import aact


class _TypeChangingBackend(DummyBackend):
    """Backend that always returns a `SimpleContext`, ignoring the input type.

    Used to force the input/output context-type mismatch the guard must catch.
    """

    async def _generate_from_context(
        self,
        action: Component[C] | CBlock | ModelOutputThunk,
        ctx: Context,
        *,
        format: type[BaseModelSubclass] | None = None,
        model_options: dict | None = None,
        tool_calls: bool = False,
    ) -> tuple[ModelOutputThunk[C], Context]:
        """Return an output whose context is a fresh `SimpleContext`."""
        mot: ModelOutputThunk = ModelOutputThunk(value="dummy")
        new_ctx = SimpleContext().add(action).add(mot)
        return mot, new_ctx  # type: ignore[return-value]


def _action() -> Message:
    return Message(role="user", content="hello")


async def test_chat_context_type_preserved():
    """A `ChatContext` in yields a `ChatContext` out; no error is raised."""
    backend = DummyBackend(responses=None)
    ctx_in = ChatContext()

    _, ctx_out = await aact(
        _action(), ctx_in, backend, silence_context_type_warning=True
    )

    assert type(ctx_out) is type(ctx_in)
    assert isinstance(ctx_out, ChatContext)


async def test_simple_context_type_preserved():
    """A `SimpleContext` in yields a `SimpleContext` out; no error is raised."""
    backend = DummyBackend(responses=None)
    ctx_in = SimpleContext()

    _, ctx_out = await aact(_action(), ctx_in, backend)

    assert type(ctx_out) is type(ctx_in)
    assert isinstance(ctx_out, SimpleContext)


async def test_type_mismatch_raises():
    """A backend that changes the context type trips the guard by default."""
    backend = _TypeChangingBackend(responses=None)
    ctx_in = ChatContext()

    with pytest.raises(ContextTypeMismatchError):
        await aact(_action(), ctx_in, backend, silence_context_type_warning=True)


async def test_escape_hatch_allows_type_change():
    """`allow_context_type_change=True` permits a deliberate type change."""
    backend = _TypeChangingBackend(responses=None)
    ctx_in = ChatContext()

    _, ctx_out = await aact(
        _action(),
        ctx_in,
        backend,
        silence_context_type_warning=True,
        allow_context_type_change=True,
    )

    assert isinstance(ctx_out, SimpleContext)
