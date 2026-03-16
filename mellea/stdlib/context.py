"""Concrete `Context` implementations for common conversation patterns.

Provides `ChatContext`, which accumulates all turns in a sliding-window chat history
(configurable via `window_size`), and `SimpleContext`, in which each interaction
is treated as a stateless single-turn exchange (no prior history is passed to the
model). Import `ChatContext` for multi-turn conversations and `SimpleContext` when
you want each call to the model to be independent.
"""

from __future__ import annotations

# Leave unused `ContextTurn` import for import ergonomics.
from ..core import CBlock, Component, Context, ContextTurn


class ChatContext(Context):
    """Initializes a chat context with unbounded window_size and is_chat=True by default.

    Args:
        window_size (int | None): Maximum number of context turns to include when
            calling `view_for_generation`. `None` (the default) means the full
            history is always returned.
    """

    def __init__(self, *, window_size: int | None = None):
        """Initialize ChatContext with an optional sliding-window size."""
        super().__init__()
        self._window_size = window_size

    def add(self, c: Component | CBlock) -> ChatContext:
        """Add a new component or CBlock to the context and return the updated context.

        Args:
            c (Component | CBlock): The component or content block to append.

        Returns:
            ChatContext: A new `ChatContext` with the added entry, preserving the
            current `window_size` setting.
        """
        new = ChatContext.from_previous(self, c)
        new._window_size = self._window_size
        return new

    def view_for_generation(self) -> list[Component | CBlock] | None:
        """Return the context entries to pass to the model, respecting the configured window.

        Uses the `window_size` set during initialisation to limit how many past
        turns are included.  `None` is returned when the underlying history is
        non-linear.

        Returns:
            list[Component | CBlock] | None: Ordered list of context entries up to
            `window_size` turns, or `None` if the history is non-linear.
        """
        return self.as_list(self._window_size)


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
