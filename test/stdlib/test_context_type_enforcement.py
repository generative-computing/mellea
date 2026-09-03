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
    ComputedModelOutputThunk,
    Context,
    ContextTypeMismatchError,
    GenerateLog,
    ModelOutputThunk,
    SamplingResult,
    SamplingStrategy,
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


class _ChatSubclass(ChatContext):
    """A bare subclass of `ChatContext` that adds no behaviour of its own.

    Inherits `add` / `new_instance` unchanged. If those helpers construct the
    hard-coded `ChatContext` type instead of `type(self)`, an instance of this
    class silently degrades to `ChatContext` on the first `add` — the exact
    regression these tests guard against.
    """


class _SimpleSubclass(SimpleContext):
    """A bare subclass of `SimpleContext`; see `_ChatSubclass` for the rationale."""


class _RequiredArgChatSubclass(ChatContext):
    """A `ChatContext` subclass whose constructor takes a required argument.

    Registers `label` in `_propagated_fields` so it travels across `add()`,
    `new_instance()`, and compaction. Its presence proves the built-via-`__new__`
    construction path never re-runs `__init__` (which would raise `TypeError`
    for the missing `label`), yet the subclass state still propagates.
    """

    _propagated_fields = (*ChatContext._propagated_fields, "label")

    def __init__(self, label: str, **kwargs) -> None:
        """Store the required `label` after delegating to `ChatContext.__init__`."""
        super().__init__(**kwargs)
        self.label = label


class _StatefulSimpleSubclass(SimpleContext):
    """A `SimpleContext` subclass that stores state set in its `__init__`.

    `SimpleContext` has no config of its own, so this exercises the *base*
    `Context._propagated_fields` mechanism: `SimpleContext.add` inherits
    `Context.from_previous`, which builds via `__new__` and skips `__init__`.
    Registering `tag` in `_propagated_fields` is what keeps it alive across
    `add()`; without the base-level copy the next attribute access would raise
    `AttributeError`.
    """

    _propagated_fields = (*SimpleContext._propagated_fields, "tag")

    def __init__(self, tag: str = "default") -> None:
        """Store `tag` after delegating to `SimpleContext.__init__`."""
        super().__init__()
        self.tag = tag


# The stdlib contexts, a direct `Context` subclass, and bare subclasses of the
# stdlib contexts. The subclasses are what catch the "named constructor instead
# of `type(self)`" bug: they inherit `add` verbatim, so they only stay their own
# type if the inherited helper builds `type(self)`.
_CONTEXT_TYPES = [
    ChatContext,
    SimpleContext,
    _MinimalContext,
    _ChatSubclass,
    _SimpleSubclass,
]


@pytest.mark.parametrize("ctx_cls", _CONTEXT_TYPES)
def test_add_returns_same_subtype(ctx_cls):
    """`ctx.add(...)` returns a context of the same subtype it was called on."""
    ctx = ctx_cls()
    added = ctx.add(_action())
    assert type(added) is ctx_cls


@pytest.mark.parametrize("ctx_cls", _CONTEXT_TYPES)
def test_add_twice_preserves_subtype(ctx_cls):
    """Chaining `add` keeps the subtype, so history nodes never demote."""
    ctx = ctx_cls().add(_action()).add(Message(role="assistant", content="hi"))
    assert type(ctx) is ctx_cls


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


def test_chat_subclass_survives_compaction():
    """A `ChatContext` subclass stays its own type after the compactor fires.

    Compaction rebuilds the linked list via `_rebuild_chat_context`, which must
    reconstruct the concrete subtype (`type(ctx)`) rather than a plain
    `ChatContext`. A `window_size` of 1 forces the `WindowCompactor` to run on
    the third `add`.
    """
    ctx = _ChatSubclass(window_size=1)
    ctx = (
        ctx.add(_action()).add(Message(role="assistant", content="one")).add(_action())
    )
    assert type(ctx) is _ChatSubclass


def test_chat_subclass_propagated_field_survives_compaction():
    """A subclass-owned `_propagated_fields` value survives the compactor rebuild.

    `_rebuild_chat_context` must copy *every* attribute named in the concrete
    class's `_propagated_fields` — not just the three built-in `ChatContext`
    fields — onto each rebuilt node. If it copied only the built-ins, the extra
    field would vanish here and the next `add` would raise `AttributeError`.
    """
    ctx = _RequiredArgChatSubclass("keep-me", window_size=1)
    ctx = (
        ctx.add(_action()).add(Message(role="assistant", content="one")).add(_action())
    )
    assert type(ctx) is _RequiredArgChatSubclass
    # The compactor fired on the third add; the subclass field must still be here.
    assert ctx.label == "keep-me"


def test_custom_compactor_cannot_demote_subtype():
    """A custom `InlineCompactor` that returns a plain `ChatContext` cannot demote the caller.

    `ChatContext.add()` promises `Self`, but the `InlineCompactor` extension
    contract lets a custom `compact()` return any `ChatContext` — including a
    demoted one. `add()` must detect the demotion and rebuild the compacted
    history back into `type(self)`, so a `ChatContext` subtype paired with a
    demoting compactor still gets its own type back (issue #1522).
    """
    from mellea.stdlib.context.chat import _rebuild_chat_context
    from mellea.stdlib.context.compactor import InlineCompactor

    class _DemotingCompactor(InlineCompactor):
        """Discards the input's subtype by rebuilding as a bare `ChatContext`."""

        def compact(self, ctx: ChatContext, *, backend=None) -> ChatContext:
            return _rebuild_chat_context(ctx.as_list(), source=ctx, cls=ChatContext)

    ctx = _RequiredArgChatSubclass("keep-me", compactor=_DemotingCompactor())
    added = ctx.add(_action())
    # The rebuild must restore the subtype and its propagated state, despite the
    # compactor having demoted the compacted node to a plain `ChatContext`.
    assert type(added) is _RequiredArgChatSubclass
    assert added.label == "keep-me"


def test_required_arg_subclass_add_preserves_state():
    """`add()` on a required-arg subclass keeps its type and state without re-running __init__.

    The construction path builds via `type(self).__new__(...)`, so a subclass
    whose `__init__` demands a required argument does not raise `TypeError` on
    `add()`, and its registered `_propagated_fields` value carries forward.
    """
    ctx = _RequiredArgChatSubclass("hello")
    added = ctx.add(_action())
    assert type(added) is _RequiredArgChatSubclass
    assert added.label == "hello"


def test_required_arg_subclass_new_instance_preserves_state():
    """`new_instance()` on a required-arg subclass keeps its type and state.

    Like `add()`, `new_instance()` builds via `__new__`, so the missing required
    constructor argument never trips `TypeError`, and the subclass field is
    carried onto the fresh root.
    """
    ctx = _RequiredArgChatSubclass("hello").add(_action())
    fresh = ctx.new_instance()
    assert type(fresh) is _RequiredArgChatSubclass
    assert fresh.is_root_node
    assert fresh.label == "hello"


def test_simple_subclass_propagated_field_survives_add():
    """A `SimpleContext` subclass field survives `add()` via the base mechanism.

    `SimpleContext.add` inherits `Context.from_previous`, which builds via
    `__new__` and skips `__init__`. The base `Context._propagated_fields` copy
    is the only thing keeping `tag` alive here; without it the assertion below
    would raise `AttributeError`. This is the regression the base-level lift
    fixes for non-`ChatContext` extension points.
    """
    ctx = _StatefulSimpleSubclass("keep-me")
    added = ctx.add(_action())
    assert type(added) is _StatefulSimpleSubclass
    assert added.tag == "keep-me"


def test_simple_subclass_propagated_field_survives_repeated_add():
    """Chained `add()` on a stateful `SimpleContext` subclass keeps its field.

    `SimpleContext` retains no history, so each `add` builds a fresh node from
    the previous one; the propagated `tag` must ride along every hop.
    """
    ctx = (
        _StatefulSimpleSubclass("keep-me")
        .add(_action())
        .add(Message(role="assistant", content="hi"))
    )
    assert type(ctx) is _StatefulSimpleSubclass
    assert ctx.tag == "keep-me"


async def test_chat_subclass_passes_functional_guard():
    """A subclassed `ChatContext` round-trips through `aact` without tripping the guard.

    This is the end-to-end payoff of the `type(self)` fix: because `add`
    preserves the subtype, input type == output type and
    `_enforce_context_type` is satisfied with no escape hatch.
    """
    backend = DummyBackend(responses=None)
    ctx_in = _ChatSubclass()

    _, ctx_out = await aact(
        _action(), ctx_in, backend, silence_context_type_warning=True
    )

    assert type(ctx_out) is _ChatSubclass


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


# --- context-type enforcement over ALL sample contexts (#1522) ---
#
# `aact(..., return_sampling_results=True)` returns a `SamplingResult` whose
# type is not statically tracked, so the input==output convention is enforced at
# runtime against *every* `sample_contexts` entry — not just the chosen one — so
# a strategy that produces a mismatched context for any attempt is caught rather
# than silently returned. These tests drive that loop with a stub strategy that
# lets each test dictate the exact contexts attached to the result.


def _computed(value: str) -> ComputedModelOutputThunk:
    """Build a computed thunk with a final-marked `GenerateLog`, as `aact` requires."""
    mot: ModelOutputThunk = ModelOutputThunk(value=value)
    mot._generate_log = GenerateLog(is_final_result=True)
    return ComputedModelOutputThunk(mot)


class _StubStrategy(SamplingStrategy):
    """A sampling strategy that returns a caller-supplied list of sample contexts.

    Bypasses real generation entirely: `sample` ignores the backend and returns
    a `SamplingResult` whose `sample_contexts` are exactly `self._contexts`, with
    one computed generation per context. This lets a test attach a deliberately
    mismatched context to any slot and assert whether the functional guard fires.
    """

    def __init__(self, contexts: list[Context]) -> None:
        """Store the contexts to attach, one per generation, to the result."""
        self._contexts = contexts

    async def sample(
        self,
        action,
        context,
        backend,
        requirements,
        *,
        validation_ctx=None,
        format=None,
        model_options=None,
        tool_calls=False,
    ) -> SamplingResult:
        """Return a `SamplingResult` wrapping `self._contexts`; the last is chosen."""
        gens = [_computed(f"gen{i}") for i in range(len(self._contexts))]
        return SamplingResult(
            result_index=len(gens) - 1,
            success=True,
            sample_generations=gens,
            sample_contexts=list(self._contexts),
        )


async def test_sampling_result_all_contexts_type_checked():
    """A mismatched context in a *non-chosen* sample slot trips the guard.

    The chosen (last) context matches the input type, but an earlier attempt is a
    `SimpleContext`. The guard must still catch it — enforcement covers the whole
    `sample_contexts` list, not just `result_ctx`.
    """
    backend = DummyBackend(responses=None)
    ctx_in = ChatContext()
    strategy = _StubStrategy([SimpleContext(), ChatContext()])

    with pytest.raises(ContextTypeMismatchError):
        await aact(
            _action(),
            ctx_in,
            backend,
            strategy=strategy,
            return_sampling_results=True,
            silence_context_type_warning=True,
        )


async def test_sampling_result_matching_contexts_pass():
    """When every sample context matches the input type, the result is returned."""
    backend = DummyBackend(responses=None)
    ctx_in = ChatContext()
    strategy = _StubStrategy([ChatContext(), ChatContext()])

    result = await aact(
        _action(),
        ctx_in,
        backend,
        strategy=strategy,
        return_sampling_results=True,
        silence_context_type_warning=True,
    )

    assert isinstance(result, SamplingResult)
    assert all(type(c) is ChatContext for c in result.sample_contexts)


async def test_sampling_result_escape_hatch_allows_all_mismatches():
    """`allow_context_type_change=True` skips the per-sample check entirely."""
    backend = DummyBackend(responses=None)
    ctx_in = ChatContext()
    strategy = _StubStrategy([SimpleContext(), SimpleContext()])

    result = await aact(
        _action(),
        ctx_in,
        backend,
        strategy=strategy,
        return_sampling_results=True,
        silence_context_type_warning=True,
        allow_context_type_change=True,
    )

    assert isinstance(result, SamplingResult)
