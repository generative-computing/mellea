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
    Span,
)
from mellea.stdlib.components import Message
from mellea.stdlib.context import ChatContext, SimpleContext
from mellea.stdlib.functional import aact


class _MinimalContext(Context):
    """A minimal `Context` subclass that overrides only the two abstract methods.

    Used to prove that the inherited, `Self`-typed helpers (`new_instance`,
    `reset_to_new`) return the *subclass* type rather than the base `Context`,
    and that a straightforward `add` override returns its own type too.
    """

    def add(self, c: Span) -> "_MinimalContext":
        """Return a new `_MinimalContext` node with `c` appended."""
        return _MinimalContext.from_previous(self, c)

    def view_for_generation(self) -> list[Span] | None:
        """Return the full linear history."""
        return self.as_list()


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


# --- self-returning Context functions preserve the concrete subtype (#1522) ---
#
# The runtime guard in `functional.py` only holds if the context helpers that
# claim to return `Self` (`add`) or the subclass type (`new_instance`,
# `reset_to_new`) actually do so when subclassed. These tests pin that
# invariant directly on the context types.

_CONTEXT_TYPES = [ChatContext, SimpleContext, _MinimalContext]


@pytest.mark.parametrize("ctx_cls", _CONTEXT_TYPES)
def test_add_returns_same_subtype(ctx_cls):
    """`ctx.add(...)` returns a context of the same subtype it was called on."""
    ctx = ctx_cls()
    added = ctx.add(_action())
    assert type(added) is ctx_cls


@pytest.mark.parametrize("ctx_cls", _CONTEXT_TYPES)
def test_new_instance_returns_same_subtype(ctx_cls):
    """`ctx.new_instance()` returns a fresh root context of the same subtype."""
    ctx = ctx_cls().add(_action())
    fresh = ctx.new_instance()
    assert type(fresh) is ctx_cls
    assert fresh.is_root_node


@pytest.mark.parametrize("ctx_cls", _CONTEXT_TYPES)
def test_reset_to_new_returns_same_subtype(ctx_cls):
    """The `reset_to_new()` classmethod returns an instance of the class it is called on."""
    fresh = ctx_cls.reset_to_new()
    assert type(fresh) is ctx_cls
    assert fresh.is_root_node


# --- session-level allow_context_type_change flag (#1522) ---


async def test_session_enforces_context_type_by_default():
    """A session trips the guard when a backend changes the context type."""
    from mellea import MelleaSession

    session = MelleaSession(_TypeChangingBackend(responses=None), ChatContext())
    assert session.allow_context_type_change is False

    with pytest.raises(ContextTypeMismatchError):
        await session.aact(_action())


async def test_session_flag_allows_context_type_change():
    """`allow_context_type_change=True` lets a session switch context types."""
    from mellea import MelleaSession

    session = MelleaSession(
        _TypeChangingBackend(responses=None),
        ChatContext(),
        allow_context_type_change=True,
    )
    assert session.allow_context_type_change is True

    await session.aact(_action())
    assert isinstance(session.ctx, SimpleContext)


def test_session_flag_preserved_across_clone():
    """`clone()` carries the `allow_context_type_change` flag to the copy."""
    from mellea import MelleaSession

    session = MelleaSession(
        DummyBackend(responses=None), ChatContext(), allow_context_type_change=True
    )
    assert session.clone().allow_context_type_change is True
