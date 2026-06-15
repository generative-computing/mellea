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
from ..core import CBlock, Component, Context, ContextTurn


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

    Note:
        `context_length` is measured in tokens; `window_size` counts context
        items (CBlocks / Components). When only a `model_id` is bound (no
        explicit `window_size`), the model's context length in tokens is used
        as an absolute upper bound on the number of items passed to the model.
        In practice this ceiling is never reached — real conversations do not
        accumulate hundreds of thousands of items — so the full history is
        returned for typical sessions. Set `window_size` explicitly to enforce
        a tighter item-count limit.
    """

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
        """The model identifier bound to this context, or ``None`` if unbound."""
        return self._model_id

    def _make_root(self, model_id: str | ModelIdentifier | None) -> ChatContext:
        """Return a new empty root ``ChatContext`` with the given ``model_id``, preserving ``window_size``."""
        return ChatContext(window_size=self._window_size, model_id=model_id)

    def _bind_model(self, model_id: str | ModelIdentifier) -> ChatContext:
        """Return a new root `ChatContext` with the given model bound, preserving `window_size`.

        Internal use only — called by ``MelleaSession`` to wire the backend's
        model identifier into the context at session construction and after
        ``reset()``. To bind a model at construction time, pass ``model_id=``
        directly to ``ChatContext()``.

        Args:
            model_id: The model identifier to associate with this context.

        Returns:
            ChatContext: A new root `ChatContext` with ``_model_id`` set.

        Raises:
            ValueError: If called on a non-root (non-empty) context. History
                would be silently discarded, so this is disallowed. Create a
                new ``ChatContext(model_id=...)`` instead.
        """
        if not self.is_root_node:
            raise ValueError(
                "_bind_model() must be called on a root (empty) ChatContext. "
                "To bind a model on a context that already has history, create a "
                "new ChatContext(model_id=...) before adding items."
            )
        return self._make_root(model_id)

    def reset_to_new(self) -> ChatContext:  # type: ignore[override]
        """Return a new empty root ``ChatContext``, preserving ``window_size`` and ``model_id``.

        Overrides ``Context.reset_to_new()`` so that the model binding and
        window size set at construction are not lost when a session is reset
        or when user code calls this method directly.

        Returns:
            ChatContext: A fresh root context with the same ``window_size``
            and ``model_id`` as this context, but no history.
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
        new._window_size = self._window_size
        new._model_id = self._model_id
        return new

    def view_for_generation(self) -> list[Component | CBlock] | None:
        """Return the context entries to pass to the model, respecting the configured window.

        Window size resolution priority:

        1. Explicit ``window_size`` passed at construction — always wins.
        2. Model-derived context length looked up via ``model_id`` — used when
           ``window_size`` is ``None`` and a model binding exists.
        3. ``None`` — return the full history (unbounded).

        Returns:
            list[Component | CBlock] | None: Ordered list of context entries up to
            the effective window size, or `None` if the history is non-linear.
        """
        effective_window = self._window_size
        if effective_window is None and self._model_id is not None:
            effective_window = get_context_length(self._model_id)
        return self.as_list(effective_window)


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
