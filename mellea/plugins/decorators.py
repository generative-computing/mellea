"""Mellea hook and plugin decorators."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal


@dataclass(frozen=True)
class HookMeta:
    """Metadata attached by the @hook decorator."""

    hook_type: str
    mode: Literal["enforce", "permissive", "fire_and_forget"] = "enforce"
    priority: int = 50


def hook(
    hook_type: str,
    *,
    mode: Literal["enforce", "permissive", "fire_and_forget"] = "enforce",
    priority: int = 50,
) -> Callable:
    """Register an async function or method as a hook handler.

    Args:
        hook_type: The hook point name (e.g., ``"generation_pre_call"``).
        mode: Execution mode — ``"enforce"`` (default), ``"permissive"``,
              or ``"fire_and_forget"``.
        priority: Lower numbers execute first (default: 50).
    """

    def decorator(fn: Callable) -> Callable:
        fn._mellea_hook_meta = HookMeta(  # type: ignore[attr-defined]
            hook_type=hook_type, mode=mode, priority=priority
        )
        return fn

    return decorator


@dataclass(frozen=True)
class PluginMeta:
    """Metadata attached by the @plugin decorator."""

    name: str
    priority: int = 50


def plugin(name: str, *, priority: int = 50) -> Callable:
    """Mark a class as a Mellea plugin.

    Args:
        name: Plugin name (required).
        priority: Default priority for all hooks in this plugin (default: 50).
              Individual ``@hook`` decorators on methods can override.
    """

    def decorator(cls: Any) -> Any:
        cls._mellea_plugin_meta = PluginMeta(  # type: ignore[attr-defined]
            name=name, priority=priority
        )
        return cls

    return decorator
