"""Concrete `Context` implementations for common conversation patterns.

Provides `ChatContext`, which accumulates all turns in a sliding-window chat history
(configurable via `window_size`), and `SimpleContext`, in which each interaction
is treated as a stateless single-turn exchange (no prior history is passed to the
model). Import `ChatContext` for multi-turn conversations and `SimpleContext` when
you want each call to the model to be independent.
"""

from __future__ import annotations

from ..backends.context_lengths import get_context_length
from ..backends.model_ids import ModelIdentifier

# Leave unused `ContextTurn` import for import ergonomics.
from ..core import CBlock, Component, Context, ContextTurn, MelleaLogger

logger = MelleaLogger.get_logger()


class ChatContext(Context):
    """Initializes a chat context with unbounded window_size and is_chat=True by default.

    Args:
        window_size (int | None): Maximum number of context turns to include when
            calling `view_for_generation`. `None` (the default) means the full
            history is returned, unless a `model_id` is bound and has a known
            context length — in that case the model's context length (in tokens)
            is used as an upper bound on the number of items returned.
        model_id (str | ModelIdentifier | None): Optional model identifier used
            for automatic context-window sizing. When set and `window_size` is
            `None`, `view_for_generation` looks up the model's known context
            length and uses it as the window size. Explicit `window_size` always
            takes priority.

    Class Attributes:
        _propagated_fields: Instance-attribute names copied by `add()` and
            `_make_root()` into every descendant node.  Add new `ChatContext`
            fields here so they propagate automatically without touching both
            methods.

    Note:
        `context_length` is measured in tokens; `window_size` counts context
        items (CBlocks / Components). When both `window_size` and `model_id`
        are set, `window_size` always takes priority — the token-budget path is
        skipped entirely. When only `model_id` is bound, `view_for_generation`
        estimates per-item token counts (via ``len(rendered) // 4``) and walks
        history newest-first, dropping the oldest items until the running sum
        fits within `context_length`. Set `window_size` explicitly to enforce
        an item-count limit instead of a token budget.

        Per-item token count is estimated as ``len(rendered) // 4`` where
        ``rendered`` is the string produced by `TemplateFormatter` (the same
        renderer used at generation time), so the estimate reflects actual prompt
        content.  A 0.75 headroom factor (retaining 75 % of the model's rated
        context length) is applied to leave capacity for the system prompt,
        injected tool schemas, the current action, and the model's response —
        none of which are tracked here.  Use ``window_size`` for precise
        item-count control.
    """

    _propagated_fields: tuple[str, ...] = ("_window_size", "_model_id")

    def __init__(
        self,
        *,
        window_size: int | None = None,
        model_id: str | ModelIdentifier | None = None,
    ):
        """Initialize ChatContext with an optional sliding-window size and model binding."""
        super().__init__()
        self._window_size = window_size
        self._model_id: str | ModelIdentifier | None = model_id

    @property
    def model_id(self) -> str | ModelIdentifier | None:
        """The model identifier bound to this context, or `None` if unbound."""
        return self._model_id

    def _make_root(self, model_id: str | ModelIdentifier | None) -> ChatContext:
        """Return a new empty root `ChatContext` with the given `model_id`, propagating all `_propagated_fields`."""
        new = ChatContext()
        for field in self._propagated_fields:
            setattr(new, field, getattr(self, field))
        # Override whatever _propagated_fields copied for _model_id: the caller
        # explicitly supplies the model_id to bind (e.g. _bind_model changes it).
        new._model_id = model_id
        return new

    def _bind_model(self, model_id: str | ModelIdentifier) -> ChatContext:
        """Return a new root `ChatContext` with the given model bound, preserving `window_size`.

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
                new `ChatContext(model_id=...)` instead. Note: `__init__` is
                the unrestricted path — it produces the same end state without
                this root check.
        """
        if not self.is_root_node:
            raise ValueError(
                "_bind_model() must be called on a root (empty) ChatContext. "
                "To bind a model on a context that already has history, create a "
                "new ChatContext(model_id=...) before adding items."
            )
        return self._make_root(model_id)

    def new_instance(self) -> ChatContext:
        """Return a new empty root `ChatContext`, preserving `window_size` and `model_id`.

        Use this instead of `reset_to_new()` when you need to preserve the
        model binding and window size from an existing instance. `reset_to_new()`
        is a classmethod that returns a bare `ChatContext()` with no configuration.

        Returns:
            ChatContext: A fresh root context with the same `window_size`
            and `model_id` as this instance, but no history.
        """
        return self._make_root(self._model_id)

    def add(self, c: Component | CBlock) -> ChatContext:
        """Add a new component or CBlock to the context and return the updated context.

        Args:
            c (Component | CBlock): The component or content block to append.

        Returns:
            ChatContext: A new `ChatContext` with the added entry, preserving
            both `window_size` and any `model_id` binding.
        """
        new = ChatContext.from_previous(self, c)
        for field in self._propagated_fields:
            setattr(new, field, getattr(self, field))
        return new

    def view_for_generation(self) -> list[Component | CBlock] | None:
        """Return the context entries to pass to the model, respecting the configured window.

        Window size resolution priority:

        1. Explicit `window_size` passed at construction — item-count limit, always wins.
        2. Model-derived context length looked up via `model_id` — token-budget truncation.
           Items are added newest-first until adding the next item would exceed the budget.
           Token count is estimated as ``len(rendered) // 4`` via `TemplateFormatter`.
        3. No limit — return the full history.

        Returns:
            list[Component | CBlock] | None: Ordered list of context entries up to
            the effective window, or `None` if the history is non-linear.
        """
        if self._window_size is not None:
            return self.as_list(self._window_size)

        if self._model_id is not None:
            token_budget = get_context_length(self._model_id)
            if token_budget is not None:
                return self._as_list_token_budget(token_budget)

        return self.as_list(None)

    def _as_list_token_budget(self, token_budget: int) -> list[Component | CBlock]:
        """Return history items that fit within *token_budget*, dropping oldest first.

        Walks the linked list from newest to oldest, accumulating items until
        adding the next item would exceed the budget.  The returned list is in
        chronological order (oldest-first), matching `as_list` behaviour.

        Per-item token count is estimated as ``len(rendered) // 4`` where
        ``rendered`` is the string produced by `TemplateFormatter` for the
        bound model — the same renderer used at generation time.  A 0.75
        headroom factor (retaining 75 % of the rated context length) reserves
        capacity for the system prompt, injected tool schemas, the current
        action, and the model's response.  Use ``window_size`` for precise
        item-count control.
        """
        assert self._model_id is not None
        from ..formatters import TemplateFormatter  # deferred to avoid circular import

        formatter = TemplateFormatter(self._model_id)
        effective_budget = int(token_budget * 0.75)
        collected: list[Component | CBlock] = []
        spent = 0
        total = 0
        current: Context = self
        while not current.is_root_node:
            item = current.node_data
            assert item is not None
            cost = max(1, len(formatter.print(item)) // 4)
            total += 1
            if spent + cost > effective_budget:
                break
            collected.append(item)
            spent += cost
            prev = current.previous_node
            assert prev is not None
            current = prev
        dropped = total - len(collected)
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


class SimpleContext(Context):
    """A `SimpleContext` is a context in which each interaction is a separate and independent turn. The history of all previous turns is NOT saved.."""

    def add(self, c: Component | CBlock) -> SimpleContext:
        """Add a new component or CBlock to the context and return the updated context.

        Args:
            c (Component | CBlock): The component or content block to record.

        Returns:
            SimpleContext: A new `SimpleContext` containing only the added entry;
            prior history is not retained.
        """
        return SimpleContext.from_previous(self, c)

    def view_for_generation(self) -> list[Component | CBlock] | None:
        """Return an empty list, since `SimpleContext` does not pass history to the model.

        Each call to the model is treated as a stateless, independent exchange.
        No prior turns are forwarded.

        Returns:
            list[Component | CBlock] | None: Always an empty list.
        """
        return []
