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

    PluginSets also support the context manager protocol for temporary activation::

        with PluginSet("observability", [log_hook, audit_hook]):
            result, ctx = instruct("Summarize this", ctx, backend)

        # or async
        async with PluginSet("guards", [safety_hook]):
            result, ctx = await ainstruct("Generate code", ctx, backend)
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
        self._scope_id: str | None = None

    def flatten(self) -> list[tuple[Callable | Any, int | None]]:
        """Recursively flatten nested PluginSets into ``(item, priority_override)`` pairs."""
        result: list[tuple[Callable | Any, int | None]] = []
        for item in self.items:
            if isinstance(item, PluginSet):
                result.extend(item.flatten())
            else:
                result.append((item, self.priority))
        return result

    def __enter__(self) -> PluginSet:
        """Register all plugins in this set for the duration of the ``with`` block."""
        if self._scope_id is not None:
            raise RuntimeError(
                f"PluginSet {self.name!r} is already active as a context manager. "
                "Concurrent or nested reuse of the same instance is not supported; "
                "create a new PluginSet instance instead."
            )
        import uuid

        from mellea.plugins.registry import register

        self._scope_id = str(uuid.uuid4())
        register(self, session_id=self._scope_id)
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Deregister all plugins registered by this context manager entry."""
        if self._scope_id is not None:
            from mellea.plugins.manager import deregister_session_plugins

            deregister_session_plugins(self._scope_id)
            self._scope_id = None

    async def __aenter__(self) -> PluginSet:
        """Async variant — delegates to the synchronous ``__enter__``."""
        return self.__enter__()

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async variant — delegates to the synchronous ``__exit__``."""
        self.__exit__(exc_type, exc_val, exc_tb)

    def __repr__(self) -> str:
        return f"PluginSet({self.name!r}, {len(self.items)} items)"
