"""Stateless single-turn context (no history is forwarded to the model)."""

from __future__ import annotations

from mellea.core import CBlock, Component, Context


class SimpleContext(Context):
    """A `SimpleContext` is a context in which each interaction is a separate and independent turn. The history of all previous turns is NOT saved.."""

    def add(self, c: Component | CBlock) -> SimpleContext:
        """Add a new component or CBlock to the context and return the updated context.

        Args:
            c (Component | CBlock): The component or content block to record.

        Returns:
            SimpleContext: A new ``SimpleContext`` containing only the added entry;
            prior history is not retained.
        """
        return SimpleContext.from_previous(self, c)

    def view_for_generation(self) -> list[Component | CBlock] | None:
        """Return an empty list, since ``SimpleContext`` does not pass history to the model.

        Each call to the model is treated as a stateless, independent exchange.
        No prior turns are forwarded.

        Returns:
            list[Component | CBlock] | None: Always an empty list.
        """
        return []
