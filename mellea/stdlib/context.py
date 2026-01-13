"""Basic Contexts."""

from __future__ import annotations

# Leave unused `ContextTurn` import for import ergonomics.
from ..core import CBlock, Component, Context, ContextTurn


class ChatContext(Context):
    """Initializes a chat context with unbounded window_size and is_chat=True by default."""

    def __init__(self, *, window_size: int | None = None):
        """Constructs a new chat context."""
        super().__init__()
        self._window_size = window_size

    def add(self, c: Component | CBlock) -> ChatContext:
        """Add a new component/cblock to the context. Returns the new context."""
        new = ChatContext.from_previous(self, c)
        new._window_size = self._window_size
        return new

    def view_for_generation(self) -> list[Component | CBlock] | None:
        """Returns the context in a linearized form. Uses the window_size set during initialization."""
        return self.as_list(self._window_size)


class SimpleContext(Context):
    """A `SimpleContext` is a context in which each interaction is a separate and independent turn. The history of all previous turns is NOT saved.."""

    def add(self, c: Component | CBlock) -> SimpleContext:
        """Add a new component/cblock to the context. Returns the new context."""
        return SimpleContext.from_previous(self, c)

    def view_for_generation(self) -> list[Component | CBlock] | None:
        """Returns an empty list."""
        return []
