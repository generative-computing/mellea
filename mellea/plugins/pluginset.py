"""PluginSet — composable groups of hooks and plugins."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


class PluginSet:
    """A named, composable group of hook functions and plugin instances.

    PluginSets are inert containers — they do not register anything themselves.
    Registration happens when they are passed to ``register()`` or
    ``start_session(plugins=[...])``.

    PluginSets can be nested: a PluginSet can contain other PluginSets.
    """

    def __init__(
        self,
        name: str,
        items: list[Callable | Any | PluginSet],
        *,
        priority: int | None = None,
    ):
        self.name = name
        self.items = items
        self.priority = priority

    def flatten(self) -> list[tuple[Callable | Any, int | None]]:
        """Recursively flatten nested PluginSets into ``(item, priority_override)`` pairs."""
        result: list[tuple[Callable | Any, int | None]] = []
        for item in self.items:
            if isinstance(item, PluginSet):
                result.extend(item.flatten())
            else:
                result.append((item, self.priority))
        return result

    def __repr__(self) -> str:
        return f"PluginSet({self.name!r}, {len(self.items)} items)"
