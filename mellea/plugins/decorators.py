"""Mellea hook decorator."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from mellea.plugins.types import PluginMode


@dataclass(frozen=True)
class HookMeta:
    """Metadata attached by the @hook decorator."""

    hook_type: str
    mode: PluginMode = PluginMode.SEQUENTIAL
    priority: int | None = None


def hook(
    hook_type: str,
    *,
    mode: PluginMode = PluginMode.SEQUENTIAL,
    priority: int | None = None,
) -> Callable:
    """Register an async function or method as a hook handler.

    Args:
        hook_type: The hook point name (e.g., `"generation_pre_call"`).
        mode: Execution mode — `PluginMode.SEQUENTIAL` (default), `PluginMode.CONCURRENT`,
              `PluginMode.AUDIT`, or `PluginMode.FIRE_AND_FORGET`.
        priority: Lower numbers execute first. For methods on a `Plugin` subclass, falls back
                  to the class-level priority, then 50. For standalone functions, defaults to 50.
    """

    def decorator(fn: Callable) -> Callable:
        import inspect

        if not inspect.iscoroutinefunction(fn):
            raise TypeError(
                f"@hook-decorated function {fn.__qualname__!r} must be async "
                f"(defined with 'async def'), got a regular function."
            )
        fn._mellea_hook_meta = HookMeta(  # type: ignore[attr-defined]
            hook_type=hook_type, mode=mode, priority=priority
        )
        return fn

    return decorator
