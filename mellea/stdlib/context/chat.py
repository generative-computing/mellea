# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Chat-style context with pluggable compaction."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Self, cast

from mellea.backends.context_lengths import get_context_length
from mellea.backends.model_ids import ModelIdentifier
from mellea.core import Context, Span
from mellea.core.utils import MelleaLogger

if TYPE_CHECKING:
    from mellea.stdlib.context.compactor import InlineCompactor

logger = MelleaLogger.get_logger()


class ChatContext(Context):
    """Chat context that accumulates turns and optionally compacts on each `add`.

    By default the context performs **no compaction** — the full history is
    retained. Compaction is opt-in: pass `compactor=` for a custom
    strategy, or `window_size=` as sugar for `WindowCompactor(size=...)`.

    Independently of compaction, a token budget can cap what
    `view_for_generation` forwards to the model: pass
    `token_context_length_limit=` for an explicit cap, or `model_id=` to
    derive the cap from the model's known context length. These operate at
    view time (newest-first, dropping oldest) and compose on top of any
    compaction already applied at `add` time.

    Note:
        Compaction is applied at `add()` time and persists in the linked
        list, so `as_list()` and `view_for_generation()` both reflect the
        post-compaction history. Callers that use `len(ctx.as_list())` as a
        session-wide interaction count will silently undercount once the
        compactor fires — track turn counts out-of-band (e.g. on the
        session) if you need them.

    Args:
        compactor (InlineCompactor | None): The compactor invoked on every
            `add`. `None` (the default) means no compaction; full history
            is kept.
        window_size (int | None): Sugar that constructs a
            `WindowCompactor(size=window_size)`, whose default
            `pin_predicate` is `pin_system`. Mutually exclusive with
            `compactor`. `None` (the default) means no windowing.

            **Behavior change (deliberate):** before the compaction refactor,
            `window_size=N` was a raw last-N view (`as_list(N)`) with no
            pinning — a system message could age out and exactly `N` items
            were returned. It now pins the system message (which therefore
            survives) and the `size` limit counts only the non-pinned body,
            so the returned view can exceed `N` items. To recover the old
            drop-in semantics, pass
            `compactor=WindowCompactor(size=N, pin_predicate=pin_nothing)`.
        token_context_length_limit (int | None): Explicit token budget cap
            for `view_for_generation`. Overrides the model-derived limit when
            set. `None` (the default) means no token cap.
        model_id (str | ModelIdentifier | None): Optional model identifier
            used for automatic context-window sizing. When set and
            `token_context_length_limit` is not provided,
            `view_for_generation` looks up the model's known context length
            and uses it as the token budget.
        retain_token_ids (bool): Opt into id-preserving history. When `True`, a
            backend that supports it sends the exact ids already sent plus only the
            new turn's, instead of re-rendering from text -- which would drop an
            earlier turn's adapter control tokens (a Granite Switch control token
            substitutes for the role marker) and cannot reproduce non-canonical BPE
            splits, both of which invalidate the server's prefix cache. Defaults to
            `False`. Two caveats: generation becomes EAGER (the id path awaits the
            completion and returns a computed thunk, so `aact` fan-out loses
            concurrency), and it needs vLLM 0.10.2+ for `return_token_ids` (without
            it nothing is retained and each turn warns).
        sent_token_ids (tuple[int, ...]): Ids the server has already seen, verbatim.
            Empty until a backend records a turn. A tuple so callers cannot mutate it.
        sent_model_id (str | None): Model those ids were produced by. Ids are not
            portable across vocabularies, so a backend can refuse a mismatched prefix.
        sent_message_count (int): How many chat messages `sent_token_ids` covers, so
            a backend can re-render exactly the already-sent side to subtract against.
        sent_template_kwargs (dict[str, Any]): Chat-template variables the retained
            ids were rendered under, so a backend re-renders the already-sent side the
            way it was actually sent. A kwarg introduced mid-conversation is refused
            rather than silently cancelling out of the subtraction.
        sent_prompt_digest (tuple[str, ...]): Per-message fingerprint of the messages
            `sent_token_ids` covers. Count says WHICH messages; the digest proves they
            are still the SAME messages, so any path (chat or intrinsic) can reuse a
            genuinely-unchanged prefix and refuse a changed one.

    A backend does not mutate this state directly: it stashes ids on the thunk's
    `_meta`, and `generate_from_chat_context` moves them onto the new node.

    Class Attributes:
        _propagated_fields: Instance-attribute names copied into every descendant
            node built via `__new__` (by `Context.from_previous`, `_make_root()`,
            and `_rebuild_chat_context()`). Extends the base `Context` tuple; add
            new `ChatContext` fields here so they propagate automatically.
    """

    _propagated_fields: tuple[str, ...] = (
        *Context._propagated_fields,
        "_compactor",
        "_token_context_length_limit",
        "_model_id",
        "_retain_token_ids",
        "_sent_token_ids",
        "_sent_model_id",
        "_sent_message_count",
        "_sent_prompt_digest",
        "_sent_template_kwargs",
    )

    # Class-level defaults: `_rebuild_chat_context` builds nodes via `__new__`
    # (skipping `__init__`), so these keep `_propagated_fields` iteration from
    # raising AttributeError on the first `add()` afterwards.
    _retain_token_ids: bool = False
    _sent_token_ids: tuple[int, ...] = ()
    _sent_model_id: str | None = None
    _sent_message_count: int = 0
    _sent_prompt_digest: tuple[str, ...] = ()
    _sent_template_kwargs: dict[str, Any] = {}

    def __init__(
        self,
        *,
        compactor: InlineCompactor | None = None,
        window_size: int | None = None,
        token_context_length_limit: int | None = None,
        model_id: str | ModelIdentifier | None = None,
        retain_token_ids: bool = False,
    ) -> None:
        """Initialize a ChatContext with an optional compactor, token budget, model binding, and id-retention policy."""
        if compactor is not None and window_size is not None:
            raise ValueError(
                "ChatContext: pass either `compactor` or `window_size`, not both."
            )
        if compactor is not None:
            from mellea.stdlib.context.compactor import InlineCompactor

            if not isinstance(compactor, InlineCompactor):
                raise TypeError(
                    f"ChatContext requires an InlineCompactor; got "
                    f"{type(compactor).__name__}. Wrap it in ThresholdCompactor, "
                    "use via react(compactor=...), or call compact(ctx, ...) "
                    "manually instead."
                )
        super().__init__()
        if compactor is None and window_size is not None:
            from mellea.stdlib.context.compactor import WindowCompactor

            self._compactor: InlineCompactor | None = WindowCompactor(size=window_size)
        else:
            self._compactor = compactor
        self._token_context_length_limit = token_context_length_limit
        self._model_id: str | ModelIdentifier | None = model_id
        # Policy (propagates and survives a root reset) vs. state (propagates but
        # is per-conversation, so `_make_root` clears it below).
        self._retain_token_ids: bool = retain_token_ids
        self._sent_token_ids: tuple[int, ...] = ()
        self._sent_model_id: str | None = None
        self._sent_message_count: int = 0
        self._sent_prompt_digest: tuple[str, ...] = ()
        self._sent_template_kwargs: dict[str, Any] = {}

    @property
    def model_id(self) -> str | ModelIdentifier | None:
        """The model identifier bound to this context, or `None` if unbound."""
        return self._model_id

    @property
    def retains_token_ids(self) -> bool:
        """Whether this context asked for id-preserving history."""
        return self._retain_token_ids

    @property
    def sent_token_ids(self) -> tuple[int, ...]:
        """Ids the server has already seen, verbatim. Empty if none recorded."""
        return self._sent_token_ids

    @property
    def sent_model_id(self) -> str | None:
        """Model the retained ids were produced by, or `None` if none are held."""
        return self._sent_model_id

    @property
    def sent_message_count(self) -> int:
        """How many chat messages `sent_token_ids` covers. `0` if none are held."""
        return self._sent_message_count

    @property
    def sent_prompt_digest(self) -> tuple[str, ...]:
        """Per-message fingerprint of the model-input prefix `sent_token_ids` covers.

        One opaque string per message, in order; empty if no ids are held. Before
        splicing, a backend re-fingerprints the leading messages of the conversation
        it is about to send and refuses reuse on a mismatch -- so the prefix is
        VERIFIED unchanged, not merely the first `sent_message_count` messages (which
        a rewritten historical message or dropped oldest turns would leave intact).
        """
        return self._sent_prompt_digest

    @property
    def sent_template_kwargs(self) -> dict[str, Any]:
        """Chat-template variables the retained ids were rendered under (a copy).

        Empty if no ids are held. `adapter_name` is never included: it applies to the
        turn being generated, not to the prefix, so keeping it would make every
        adapter turn look like drift against the next one.

        A backend re-renders the already-sent side under THESE kwargs, not the current
        turn's. Using the current turn's puts a newly-introduced kwarg on both sides of
        the subtraction, where it cancels out -- so a turn that first supplies
        `documents=[...]` would send a prefix rendered without them and a delta that
        does not contain them either, activating a RAG adapter against an empty context
        with every other guard still passing.
        """
        return dict(self._sent_template_kwargs)

    def with_sent_token_ids(
        self,
        ids: list[int],
        model_id: str | None = None,
        message_count: int = 0,
        prompt_digest: tuple[str, ...] = (),
        template_kwargs: dict[str, Any] | None = None,
    ) -> Self:
        """Return a copy of this context at the same position, holding retained `ids`.

        A `Context` is immutable, so recording what the server saw produces a new
        node rather than mutating this one. Built by hand rather than via
        `Context.from_previous`, which asserts `data is not None` and so rejects a
        root node (a root must stay a root).

        Constructed with `type(self).__new__(...)` rather than `type(self)()`, the
        same way `_make_root` and `Context.from_previous` build theirs: calling the
        initializer would raise `TypeError` on a subclass with required constructor
        arguments, and a backend records ids on every retained turn, so that would
        make id retention unusable for such a subclass. Configuration travels via
        `_propagated_fields` instead.

        Args:
            ids (list[int]): Full id sequence the server has now seen (prompt sent
                plus answer produced).
            model_id (str | None): Model that produced them, so a later turn against
                a different model can be refused.
            message_count (int): How many chat messages these ids cover.
            prompt_digest (tuple[str, ...]): Per-message fingerprint of those
                messages; see `sent_prompt_digest`.
            template_kwargs (dict[str, Any] | None): Chat-template variables the ids
                were rendered under. `adapter_name` is stripped; see
                `sent_template_kwargs`.

        Returns:
            Self: A new context of the same concrete subtype at the same position;
            this one is unchanged.

        Raises:
            TypeError: If an entry of `ids` is not convertible to `int` (e.g. `None`).
            ValueError: If an entry is a string that does not name an integer. Note
                that a float is silently truncated rather than refused; the backend
                that produces these ids validates them on the way in
                (`_tokenize_chat`), so this is a backstop, not the guard.
        """
        cls = type(self)
        new = cls.__new__(cls)
        Context.__init__(new)
        for field in self._propagated_fields:
            setattr(new, field, getattr(self, field))
        new._previous = self._previous
        new._data = self._data
        new._is_root = self._is_root
        new._is_chat_context = self._is_chat_context
        new._sent_token_ids = tuple(int(i) for i in ids)
        new._sent_model_id = model_id
        new._sent_message_count = message_count
        new._sent_prompt_digest = tuple(prompt_digest)
        new._sent_template_kwargs = {
            k: v for k, v in (template_kwargs or {}).items() if k != "adapter_name"
        }
        return new

    def _make_root(self, model_id: str | ModelIdentifier | None) -> ChatContext:
        """Return a new empty root `ChatContext`, propagating all `_propagated_fields` then binding `model_id`.

        Builds via `type(self).__new__(...)` (not `type(self)()`), so a subclass
        gets an instance of itself back rather than being demoted to
        `ChatContext`, and a subclass with required constructor arguments does not
        raise `TypeError` here (the subclass `__init__` is deliberately not
        re-run). Subclass configuration travels via `_propagated_fields`.
        """
        cls = type(self)
        new = cls.__new__(cls)
        Context.__init__(new)
        for field in self._propagated_fields:
            setattr(new, field, getattr(self, field))
        # Override whatever _propagated_fields copied for _model_id: the caller
        # explicitly supplies the model_id to bind (e.g. _bind_model changes it).
        new._model_id = model_id
        # Keep the `_retain_token_ids` policy, but clear the per-conversation ids:
        # a fresh root has sent nothing, so a stale prefix would subtract ids the
        # server never saw for this conversation.
        new._sent_token_ids = ()
        new._sent_model_id = None
        new._sent_message_count = 0
        new._sent_prompt_digest = ()
        new._sent_template_kwargs = {}
        return new

    def _bind_model(self, model_id: str | ModelIdentifier) -> ChatContext:
        """Return a new root `ChatContext` with the given model bound, preserving compactor and token budget.

        Internal use only — called by `MelleaSession` to wire the backend's
        model identifier into the context at session construction and after
        `reset()`. To bind a model at construction time, pass `model_id=`
        directly to `ChatContext()`.

        Args:
            model_id: The model identifier to associate with this context.

        Returns:
            ChatContext: A new root `ChatContext` with `_model_id` set.

        Raises:
            ValueError: If called on a non-root (non-empty) context. History
                would be silently discarded, so this is disallowed. Create a
                new `ChatContext(model_id=...)` instead.
        """
        if not self.is_root_node:
            raise ValueError(
                "_bind_model() must be called on a root (empty) ChatContext. "
                "To bind a model on a context that already has history, create a "
                "new ChatContext(model_id=...) before adding items."
            )
        return self._make_root(model_id)

    def new_instance(self) -> Self:
        """Return a new empty root `ChatContext`, preserving compactor, token budget, and `model_id`.

        Use this instead of `reset_to_new()` when you need to preserve the
        configuration of an existing instance. `reset_to_new()` is a classmethod
        that returns a bare `ChatContext()` with no configuration.

        Narrows the base `Context.new_instance()` return (`Context`) to `Self`, so
        a subclass gets an instance of its own type back. `_make_root` builds via
        `type(self).__new__(...)`, so the runtime guarantee matches this static
        type.

        Returns:
            Self: A fresh root context of the same concrete subtype with the same
            compactor, `token_context_length_limit`, and `model_id` as this
            instance, but no history.
        """
        return cast("Self", self._make_root(self._model_id))

    def add(self, c: Span) -> Self:
        """Append `c` and run the compactor; return the resulting context.

        Args:
            c (Span): The component, content
                block, or model output to append.

        Returns:
            Self: A new context of the same concrete subtype carrying the same
            configuration. Returning `Self` (not the hard-coded `ChatContext`)
            keeps a subclass statically its own type, matching the runtime
            `type(self)` construction below.
        """
        # `type(self)`, not `ChatContext`, so a subclass gets an instance of
        # itself back rather than being silently demoted to `ChatContext`.
        # Typed as `ChatContext` (not `Self`) because `compact()` below returns
        # `ChatContext`; the final `cast` re-narrows to `Self` for the return.
        # `from_previous` already copies `_propagated_fields` from `self` onto the
        # new node, so no explicit copy is needed here.
        new: ChatContext = type(self).from_previous(self, c)
        if self._compactor is not None:
            new = self._compactor.compact(new)
            # The built-in compactors rebuild via `type(ctx)`, so they preserve
            # the concrete subtype. A *custom* `InlineCompactor` need not — its
            # `compact()` may return a plain `ChatContext`, which would make the
            # `Self` cast below unsound (issue #1522). If the returned type was
            # demoted, rebuild the compacted history back into `type(self)`,
            # re-copying `_propagated_fields` so subclass-owned state survives.
            if type(new) is not type(self):
                new = _rebuild_chat_context(new.as_list(), source=self)
            # Reapplied even when the type survived, because a compactor builds the
            # context it returns however it likes -- `_rebuild_chat_context` is a
            # convenience, not a contract, and a hand-rolled one that copies nothing
            # would otherwise hand back a context with every propagated field at its
            # class default. That is silent: the compactor would switch off a
            # token-budget view or id retention with nothing to indicate it. Enforcing
            # the guarantee here, where `add` makes it, means no compactor has to
            # remember. (A no-op after the rebuild above, which copied the same fields
            # from the same source.)
            for field in self._propagated_fields:
                setattr(new, field, getattr(self, field))
        # `new` is now guaranteed to be a `type(self)` instance (either the
        # compactor preserved it, or the rebuild above restored it), so the cast
        # informs the checker of what the runtime guarantees.
        return cast("Self", new)

    def view_for_generation(self) -> list[Span] | None:
        """Return the components to forward to the model.

        Compaction is applied at `add` time (Pattern 1), so the stored history
        is already post-compaction. A token budget, if configured, is applied
        here on top of that:

        1. Explicit `token_context_length_limit` — token cap, overrides model table.
        2. Model-derived context length looked up via `model_id`.
        3. No token limit — return the full (post-compaction) history.

        `None` is returned when the underlying history is non-linear.

        Returns:
            list[Span] | None: Ordered list of
            context entries, or `None` if the history is non-linear.
        """
        if self._token_context_length_limit is not None:
            return self._as_list_token_budget(self._token_context_length_limit)

        if self._model_id is not None:
            token_budget = get_context_length(self._model_id)
            if token_budget is not None:
                return self._as_list_token_budget(token_budget)

        return self.as_list()

    def _as_list_token_budget(self, token_budget: int) -> list[Span]:
        """Return history items that fit within *token_budget*, dropping oldest first.

        Walks the linked list from newest to oldest, accumulating items until
        adding the next item would exceed the budget. The returned list is in
        chronological order (oldest-first), matching `as_list` behaviour.

        Per-item token count is estimated as `len(rendered) // 4` (1 token ≈
        4 characters) where `rendered` is the string produced by
        `TemplateFormatter` for the bound model — the same renderer used at
        generation time. A 0.75 headroom factor (retaining 75 % of the rated
        context length) reserves capacity for the system prompt, injected tool
        schemas, the current action, and the model's response. Use
        `window_size` / a compactor for precise item-count control.
        """
        from mellea.formatters import TemplateFormatter  # deferred: circular import

        formatter = (
            TemplateFormatter(self._model_id) if self._model_id is not None else None
        )
        effective_budget = int(token_budget * 0.75)
        collected: list[Span] = []
        spent = 0
        chain_length = 0
        node: Context = self
        while not node.is_root_node:
            chain_length += 1
            node = node.previous_node  # type: ignore[assignment]
        current: Context = self
        while not current.is_root_node:
            item = current.node_data
            if item is None:  # pragma: no cover
                raise RuntimeError(
                    "Malformed context chain: node_data is None at a non-root node"
                )
            rendered = formatter.print(item) if formatter is not None else str(item)
            cost = max(1, len(rendered) // 4)
            if spent + cost > effective_budget:
                break
            collected.append(item)
            spent += cost
            prev = current.previous_node
            if prev is None:  # pragma: no cover
                raise RuntimeError(
                    "Malformed context chain: previous_node is None at a non-root node"
                )
            current = prev
        dropped = chain_length - len(collected)
        if dropped:
            logger.debug(
                "Context truncated: dropped %d item(s) to stay within %d-token budget "
                "(effective budget after 0.75 headroom: %d tokens, used: %d tokens).",
                dropped,
                token_budget,
                effective_budget,
                spent,
            )
        collected.reverse()
        return collected


def _rebuild_chat_context(
    components: list[Span],
    *,
    source: ChatContext,
    compactor: InlineCompactor | None = None,
    token_context_length_limit: int | None = None,
    model_id: str | ModelIdentifier | None = None,
    cls: type[ChatContext] | None = None,
) -> ChatContext:
    """Build a fresh `ChatContext` linked-list without triggering compaction.

    Manual node construction sidesteps `ChatContext.add` so compactors don't
    recurse into their own compactor while rebuilding history. Every node is
    given the same configuration so the rebuilt context behaves identically to
    its source (e.g. token-budget views still apply).

    Subclass state is preserved: every attribute named in the concrete class's
    `_propagated_fields` is copied from `source` onto each rebuilt node — not
    just the three built-in `ChatContext` fields — so a subclass that registers
    its own field there keeps it across compaction rather than losing it (and
    then raising `AttributeError` on the next `add`). The three built-in fields
    can be overridden via the explicit `compactor` / `token_context_length_limit`
    / `model_id` arguments; any that is left `None` falls back to `source`'s
    value.

    The retained token ids are the one exception to that copy. `_retain_token_ids`
    is POLICY and is preserved, so compaction does not silently downgrade a
    retaining conversation to full chat renders; the ids, their count, digest, and
    template kwargs are per-conversation STATE naming exactly which messages the
    server has already seen, and compaction just dropped some of them, so they are
    reset to the class defaults on every rebuilt node. This is the same policy/state
    split `_make_root` makes.

    Note:
        Nodes are constructed via `cls.__new__(cls)` and configured by copying
        fields, so the subclass initializer is deliberately not re-run. A
        subclass whose invariants live only in `__init__` (rather than in
        `_propagated_fields`) will not have them re-established here; register
        such state in `_propagated_fields` so it propagates.

    Migration:
        `source` is now a **required** keyword argument (it was absent before the
        subclass-preservation change). Custom compactors that call this helper —
        including the documented `custom_compactor` recipe — must pass
        `source=ctx` so the rebuilt nodes inherit `type(ctx)` and its
        `_propagated_fields`. It is required rather than optional because a
        missing `source` would silently discard subclass identity and state (the
        exact regression this argument fixes). The `compactor` /
        `token_context_length_limit` / `model_id` arguments still default to
        `None`; a `None` now inherits the corresponding value from `source`
        rather than clearing the field.

    Args:
        components: Components to materialise as the new context, in order.
        source: The context being rebuilt. Its `_propagated_fields` values are
            copied onto every node so subclass-owned state survives -- except the
            retained token ids, which are reset (see the note above).
        compactor: Compactor to attach to every node; when `None`, `source`'s
            compactor is used.
        token_context_length_limit: Token budget to attach to every node; when
            `None`, `source`'s value is used.
        model_id: Model identifier to attach to every node; when `None`,
            `source`'s value is used.
        cls: The concrete `ChatContext` subtype to construct. Defaults to
            `type(source)` so a subclassed context is rebuilt as its own type
            rather than being demoted to `ChatContext`.

    Returns:
        A new context of type `cls` whose linear history is exactly `components`.
    """
    target_cls = cls if cls is not None else type(source)
    overrides = {
        "_compactor": compactor,
        "_token_context_length_limit": token_context_length_limit,
        "_model_id": model_id,
    }

    def _configure(node: ChatContext) -> None:
        # Copy every propagated field from the source so subclass-owned state
        # survives compaction; explicit non-None overrides take precedence.
        for field in source._propagated_fields:
            setattr(node, field, getattr(source, field))
        for field, value in overrides.items():
            if value is not None:
                setattr(node, field, value)
        # `_retain_token_ids` rides along with that copy, which is what compaction
        # must preserve: the POLICY, so a retaining conversation is not silently
        # downgraded to full chat renders. The retained ids are the opposite -- they
        # are per-conversation STATE naming exactly which messages the server has
        # seen, and compaction just dropped some of those, so ids covering them
        # describe a conversation that no longer exists. Reset them to the class
        # defaults, the same policy/state split `_make_root` makes. A fresh dict per
        # node, never the shared class-level default.
        node._sent_token_ids = ()
        node._sent_model_id = None
        node._sent_message_count = 0
        node._sent_prompt_digest = ()
        node._sent_template_kwargs = {}

    ctx: ChatContext = target_cls.__new__(target_cls)
    Context.__init__(ctx)
    _configure(ctx)
    for c in components:
        new: ChatContext = target_cls.__new__(target_cls)
        new._previous = ctx
        new._data = c
        new._is_root = False
        new._is_chat_context = ctx._is_chat_context
        _configure(new)
        ctx = new
    return ctx
