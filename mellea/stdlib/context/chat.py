# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Chat-style context with pluggable compaction."""

from __future__ import annotations

from typing import TYPE_CHECKING

from mellea.backends.context_lengths import get_context_length
from mellea.backends.model_ids import ModelIdentifier
from mellea.core import CBlock, Component, Context, ModelOutputThunk
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

    Class Attributes:
        _propagated_fields: Instance-attribute names copied by `add()` and
            `_make_root()` into every descendant node. Add new `ChatContext`
            fields here so they propagate automatically.
    """

    _propagated_fields: tuple[str, ...] = (
        "_compactor",
        "_token_context_length_limit",
        "_model_id",
    )

    def __init__(
        self,
        *,
        compactor: InlineCompactor | None = None,
        window_size: int | None = None,
        token_context_length_limit: int | None = None,
        model_id: str | ModelIdentifier | None = None,
    ) -> None:
        """Initialize a ChatContext with an optional compactor, token budget, and model binding."""
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

    @property
    def model_id(self) -> str | ModelIdentifier | None:
        """The model identifier bound to this context, or `None` if unbound."""
        return self._model_id

    def _make_root(self, model_id: str | ModelIdentifier | None) -> ChatContext:
        """Return a new empty root `ChatContext`, propagating all `_propagated_fields` then binding `model_id`."""
        new = ChatContext()
        for field in self._propagated_fields:
            setattr(new, field, getattr(self, field))
        # Override whatever _propagated_fields copied for _model_id: the caller
        # explicitly supplies the model_id to bind (e.g. _bind_model changes it).
        new._model_id = model_id
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

    def new_instance(self) -> ChatContext:
        """Return a new empty root `ChatContext`, preserving compactor, token budget, and `model_id`.

        Use this instead of `reset_to_new()` when you need to preserve the
        configuration of an existing instance. `reset_to_new()` is a classmethod
        that returns a bare `ChatContext()` with no configuration.

        Returns:
            ChatContext: A fresh root context with the same compactor,
            `token_context_length_limit`, and `model_id` as this instance, but
            no history.
        """
        return self._make_root(self._model_id)

    def add(self, c: Component | CBlock | ModelOutputThunk) -> ChatContext:
        """Append `c` and run the compactor; return the resulting context.

        Args:
            c (Component | CBlock | ModelOutputThunk): The component, content
                block, or model output to append.

        Returns:
            ChatContext: A new `ChatContext` carrying the same configuration.
        """
        new = ChatContext.from_previous(self, c)
        for field in self._propagated_fields:
            setattr(new, field, getattr(self, field))
        if self._compactor is not None:
            new = self._compactor.compact(new)
        return new

    def view_for_generation(self) -> list[Component | CBlock | ModelOutputThunk] | None:
        """Return the components to forward to the model.

        Compaction is applied at `add` time (Pattern 1), so the stored history
        is already post-compaction. A token budget, if configured, is applied
        here on top of that:

        1. Explicit `token_context_length_limit` — token cap, overrides model table.
        2. Model-derived context length looked up via `model_id`.
        3. No token limit — return the full (post-compaction) history.

        `None` is returned when the underlying history is non-linear.

        Returns:
            list[Component | CBlock | ModelOutputThunk] | None: Ordered list of
            context entries, or `None` if the history is non-linear.
        """
        if self._token_context_length_limit is not None:
            return self._as_list_token_budget(self._token_context_length_limit)

        if self._model_id is not None:
            token_budget = get_context_length(self._model_id)
            if token_budget is not None:
                return self._as_list_token_budget(token_budget)

        return self.as_list()

    def _as_list_token_budget(
        self, token_budget: int
    ) -> list[Component | CBlock | ModelOutputThunk]:
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
        collected: list[Component | CBlock | ModelOutputThunk] = []
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
    components: list[Component | CBlock | ModelOutputThunk],
    *,
    compactor: InlineCompactor | None = None,
    token_context_length_limit: int | None = None,
    model_id: str | ModelIdentifier | None = None,
) -> ChatContext:
    """Build a fresh `ChatContext` linked-list without triggering compaction.

    Manual node construction sidesteps `ChatContext.add` so compactors don't
    recurse into their own compactor while rebuilding history. Every node is
    given the same configuration so the rebuilt context behaves identically to
    its source (e.g. token-budget views still apply).

    Args:
        components: Components to materialise as the new context, in order.
        compactor: Compactor to attach to every node of the rebuilt context.
        token_context_length_limit: Token budget to attach to every node.
        model_id: Model identifier to attach to every node.

    Returns:
        A new `ChatContext` whose linear history is exactly `components`.
    """

    def _configure(node: ChatContext) -> None:
        node._compactor = compactor
        node._token_context_length_limit = token_context_length_limit
        node._model_id = model_id

    ctx: ChatContext = ChatContext.__new__(ChatContext)
    Context.__init__(ctx)
    _configure(ctx)
    for c in components:
        new: ChatContext = ChatContext.__new__(ChatContext)
        new._previous = ctx
        new._data = c
        new._is_root = False
        new._is_chat_context = ctx._is_chat_context
        _configure(new)
        ctx = new
    return ctx
