"""Concrete `Context` implementations for common conversation patterns.

Provides `ChatContext`, which accumulates all turns in a sliding-window chat history
(configurable via `window_size`), and `SimpleContext`, in which each interaction
is treated as a stateless single-turn exchange (no prior history is passed to the
model). Import `ChatContext` for multi-turn conversations and `SimpleContext` when
you want each call to the model to be independent.
"""

from __future__ import annotations

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
        items (CBlocks / Components). When deriving the window from a model's
        context length, the token count is used as a maximum item count — a
        conservative proxy that is correct in practice because the number of
        context items is always far smaller than the token budget.
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

    def bind_model(self, model_id: str | ModelIdentifier) -> ChatContext:
        """Return a new root `ChatContext` with the given model bound, preserving `window_size`.

        Creates a new root node (not a linked continuation) with the same
        `window_size` and the provided `model_id`. Intended to be called on a
        freshly-created root context before any items are added; subsequent
        `add()` calls propagate the binding automatically.

        Args:
            model_id: The model identifier to associate with this context.

        Returns:
            ChatContext: A new root `ChatContext` with `_model_id` set.
        """
        return ChatContext(window_size=self._window_size, model_id=model_id)

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
            from ..backends.context_lengths import get_context_length

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
