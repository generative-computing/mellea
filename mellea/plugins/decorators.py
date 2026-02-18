"""Mellea hook and plugin decorators."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from mellea.plugins.types import PluginMode


# ---------------------------------------------------------------------------
# Context-manager helpers injected into @plugin-decorated classes
# ---------------------------------------------------------------------------


def _plugin_cm_enter(self: Any) -> Any:
    if getattr(self, "_scope_id", None) is not None:
        meta = getattr(type(self), "_mellea_plugin_meta", None)
        plugin_name = meta.name if meta else type(self).__name__
        raise RuntimeError(
            f"Plugin {plugin_name!r} is already active as a context manager. "
            "Concurrent or nested reuse of the same instance is not supported; "
            "create a new instance instead."
        )
    import uuid

    from mellea.plugins.registry import register

    self._scope_id = str(uuid.uuid4())
    register(self, session_id=self._scope_id)
    return self


def _plugin_cm_exit(self: Any, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
    scope_id = getattr(self, "_scope_id", None)
    if scope_id is not None:
        from mellea.plugins.manager import deregister_session_plugins

        deregister_session_plugins(scope_id)
        self._scope_id = None


async def _plugin_cm_aenter(self: Any) -> Any:
    return self.__enter__()


async def _plugin_cm_aexit(self: Any, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
    self.__exit__(exc_type, exc_val, exc_tb)


@dataclass(frozen=True)
class HookMeta:
    """Metadata attached by the @hook decorator."""

    hook_type: str
    mode: PluginMode = PluginMode.ENFORCE
    priority: int = 50


def hook(
    hook_type: str, *, mode: PluginMode = PluginMode.ENFORCE, priority: int = 50
) -> Callable:
    """Register an async function or method as a hook handler.

    Args:
        hook_type: The hook point name (e.g., ``"generation_pre_call"``).
        mode: Execution mode — ``PluginMode.ENFORCE`` (default), ``PluginMode.PERMISSIVE``,
              or ``PluginMode.FIRE_AND_FORGET``.
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
        cls.__enter__ = _plugin_cm_enter
        cls.__exit__ = _plugin_cm_exit
        cls.__aenter__ = _plugin_cm_aenter
        cls.__aexit__ = _plugin_cm_aexit
        return cls

    return decorator
