# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stateless single-turn context (no history is forwarded to the model)."""

from __future__ import annotations

from mellea.core import Context, NodeData


class SimpleContext(Context):
    """A `SimpleContext` is a context in which each interaction is a separate and independent turn. The history of all previous turns is NOT saved.."""

    def add(self, c: NodeData) -> SimpleContext:
        """Add a new component or CBlock to the context and return the updated context.

        Args:
            c (NodeData): The component, content
                block, or model output to record.

        Returns:
            SimpleContext: A new `SimpleContext` containing only the added entry;
            prior history is not retained.
        """
        return SimpleContext.from_previous(self, c)

    def view_for_generation(self) -> list[NodeData] | None:
        """Return an empty list, since `SimpleContext` does not pass history to the model.

        Each call to the model is treated as a stateless, independent exchange.
        No prior turns are forwarded.

        Returns:
            list[NodeData] | None: Always an empty list.
        """
        return []
