"""Plugin registration and helpers."""

from __future__ import annotations

import logging
import uuid
from collections.abc import Callable
from typing import Any

from mellea.plugins.decorators import HookMeta, PluginMeta
from mellea.plugins.pluginset import PluginSet
from mellea.plugins.types import PluginMode

try:
    from mcpgateway.plugins.framework.base import Plugin
    from mcpgateway.plugins.framework.models import (
        PluginConfig,
        PluginMode as _CFPluginMode,
        PluginResult,
        PluginViolation,
    )

    _HAS_PLUGIN_FRAMEWORK = True
except ImportError:
    _HAS_PLUGIN_FRAMEWORK = False

logger = logging.getLogger(__name__)

_MODE_MAP: dict[PluginMode, Any] = {}
if _HAS_PLUGIN_FRAMEWORK:
    _MODE_MAP = {
        PluginMode.ENFORCE: _CFPluginMode.ENFORCE,
        PluginMode.PERMISSIVE: _CFPluginMode.PERMISSIVE,
        # fire_and_forget deferred — store as enforce for now
        PluginMode.FIRE_AND_FORGET: _CFPluginMode.ENFORCE,
    }


def _map_mode(mode: PluginMode) -> Any:
    """Map Mellea PluginMode to ContextForge PluginMode."""
    return _MODE_MAP.get(mode, _MODE_MAP.get(PluginMode.ENFORCE))


def block(
    reason: str,
    *,
    code: str = "",
    description: str = "",
    details: dict[str, Any] | None = None,
) -> Any:
    """Convenience helper for returning a blocking ``PluginResult``.

    Args:
        reason: Short reason for the violation.
        code: Machine-readable violation code.
        description: Longer description (defaults to ``reason``).
        details: Additional structured details.
    """
    if not _HAS_PLUGIN_FRAMEWORK:
        raise ImportError(
            "block() requires the ContextForge plugin framework. "
            "Install it with: pip install 'mellea[contextforge]'"
        )
    return PluginResult(
        continue_processing=False,
        violation=PluginViolation(
            reason=reason,
            description=description or reason,
            code=code,
            details=details or {},
        ),
    )


def register(
    items: Callable | Any | PluginSet | list[Callable | Any | PluginSet],
    *,
    session_id: str | None = None,
) -> None:
    """Register plugins globally or for a specific session.

    When ``session_id`` is ``None``, plugins are global (fire for all invocations).
    When ``session_id`` is provided, plugins fire only within that session.

    Accepts standalone ``@hook`` functions, ``@plugin``-decorated class instances,
    ``MelleaPlugin`` instances, ``PluginSet`` instances, or lists thereof.
    """
    if not _HAS_PLUGIN_FRAMEWORK:
        raise ImportError(
            "register() requires the ContextForge plugin framework. "
            "Install it with: pip install 'mellea[contextforge]'"
        )

    from mellea.plugins.manager import _ensure_plugin_manager

    pm = _ensure_plugin_manager()

    if not isinstance(items, list):
        items = [items]

    for item in items:
        if isinstance(item, PluginSet):
            for flattened_item, priority_override in item.flatten():
                _register_single(pm, flattened_item, session_id, priority_override)
        else:
            _register_single(pm, item, session_id, None)


def _register_single(
    pm: Any, item: Callable | Any, session_id: str | None, priority_override: int | None
) -> None:
    """Register a single hook function or plugin instance.

    - Standalone functions with ``_mellea_hook_meta``: wrapped in ``_FunctionHookAdapter``
    - ``@plugin``-decorated class instances: methods with ``_mellea_hook_meta`` discovered
    - ``MelleaPlugin`` instances: registered directly
    """
    meta: HookMeta | None = getattr(item, "_mellea_hook_meta", None)
    plugin_meta: PluginMeta | None = getattr(type(item), "_mellea_plugin_meta", None)

    if meta is not None:
        # Standalone @hook function
        adapter = _FunctionHookAdapter(
            item, session_id=session_id, priority_override=priority_override
        )
        pm._registry.register(adapter)
        if session_id:
            from mellea.plugins.manager import _track_session_plugin

            _track_session_plugin(session_id, adapter.name)
        logger.debug(
            "Registered standalone hook: %s for %s", item.__qualname__, meta.hook_type
        )

    elif plugin_meta is not None:
        # @plugin-decorated class instance
        adapter = _ClassPluginAdapter(
            item,
            plugin_meta,
            session_id=session_id,
            priority_override=priority_override,
        )
        pm._registry.register(adapter)
        if session_id:
            from mellea.plugins.manager import _track_session_plugin

            _track_session_plugin(session_id, adapter.name)
        logger.debug("Registered class plugin: %s", plugin_meta.name)

    elif isinstance(item, Plugin):
        # MelleaPlugin / ContextForge Plugin instance
        pm._registry.register(item)
        if session_id:
            from mellea.plugins.manager import _track_session_plugin

            _track_session_plugin(session_id, item.name)
        logger.debug("Registered MelleaPlugin: %s", item.name)

    else:
        raise TypeError(
            f"Cannot register {item!r}: expected a @hook-decorated function, "
            f"a @plugin-decorated class instance, or a MelleaPlugin instance."
        )


class _FunctionHookAdapter(Plugin):
    """Adapts a standalone ``@hook``-decorated function into a ContextForge Plugin."""

    def __init__(
        self,
        fn: Callable,
        session_id: str | None = None,
        priority_override: int | None = None,
    ):
        meta: HookMeta = fn._mellea_hook_meta  # type: ignore[attr-defined]
        priority = priority_override if priority_override is not None else meta.priority
        config = PluginConfig(
            name=fn.__qualname__,
            kind=f"{fn.__module__}.{fn.__qualname__}",
            hooks=[meta.hook_type],
            mode=_map_mode(meta.mode),
            priority=priority,
        )
        super().__init__(config)
        self._fn = fn
        self._session_id = session_id

    async def initialize(self) -> None:
        pass

    async def shutdown(self) -> None:
        pass

    # The hook method is discovered by convention: method name == hook_type.
    # We dynamically add it so ContextForge's HookRef can find it.
    def __getattr__(self, name: str) -> Any:
        meta: HookMeta | None = getattr(self._fn, "_mellea_hook_meta", None)
        if meta and name == meta.hook_type:
            return self._invoke
        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{name}'"
        )

    async def _invoke(self, payload: Any, context: Any) -> Any:
        result = await self._fn(payload, context)
        if result is None:
            return PluginResult(continue_processing=True, modified_payload=payload)
        return result


class _ClassPluginAdapter(Plugin):
    """Adapts a ``@plugin``-decorated class instance into a ContextForge Plugin."""

    def __init__(
        self,
        instance: Any,
        plugin_meta: PluginMeta,
        session_id: str | None = None,
        priority_override: int | None = None,
    ):
        # Discover all @hook-decorated methods
        hook_methods: dict[str, tuple[Callable, HookMeta]] = {}
        for attr_name in dir(instance):
            if attr_name.startswith("_"):
                continue
            attr = getattr(instance, attr_name, None)
            if attr is None:
                continue
            hook_meta: HookMeta | None = getattr(attr, "_mellea_hook_meta", None)
            if hook_meta is not None:
                hook_methods[hook_meta.hook_type] = (attr, hook_meta)

        priority = (
            priority_override if priority_override is not None else plugin_meta.priority
        )
        config = PluginConfig(
            name=plugin_meta.name,
            kind=f"{type(instance).__module__}.{type(instance).__qualname__}",
            hooks=list(hook_methods.keys()),
            mode=PluginMode.ENFORCE,
            priority=priority,
        )
        super().__init__(config)
        self._instance = instance
        self._hook_methods = hook_methods
        self._session_id = session_id

    async def initialize(self) -> None:
        init = getattr(self._instance, "initialize", None)
        if init and callable(init):
            await init()

    async def shutdown(self) -> None:
        shut = getattr(self._instance, "shutdown", None)
        if shut and callable(shut):
            await shut()

    def __getattr__(self, name: str) -> Any:
        if name in self._hook_methods:
            bound_method = self._hook_methods[name][0]

            async def _wrapped(payload: Any, context: Any) -> Any:
                result = await bound_method(payload, context)
                if result is None:
                    return PluginResult(
                        continue_processing=True, modified_payload=payload
                    )
                return result

            return _wrapped
        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{name}'"
        )


class _PluginScope:
    """Context manager returned by :func:`plugin_scope`.

    Supports both synchronous and asynchronous ``with`` statements.
    """

    def __init__(self, items: list[Callable | Any | PluginSet]) -> None:
        self._items = items
        self._scope_id: str | None = None

    def _activate(self) -> None:
        self._scope_id = str(uuid.uuid4())
        register(self._items, session_id=self._scope_id)

    def _deactivate(self) -> None:
        if self._scope_id is not None:
            from mellea.plugins.manager import deregister_session_plugins

            deregister_session_plugins(self._scope_id)
            self._scope_id = None

    def __enter__(self) -> _PluginScope:
        self._activate()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self._deactivate()

    async def __aenter__(self) -> _PluginScope:
        self._activate()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self._deactivate()


def plugin_scope(*items: Callable | Any | PluginSet) -> _PluginScope:
    """Return a context manager that temporarily registers plugins for a block of code.

    Accepts the same items as :func:`register`: standalone ``@hook``-decorated
    functions, ``@plugin``-decorated class instances, ``MelleaPlugin`` instances,
    and :class:`~mellea.plugins.PluginSet` instances — or any mix thereof.

    Supports both synchronous and asynchronous ``with`` statements::

        # Sync functional API
        with plugin_scope(log_hook, audit_plugin):
            result, ctx = instruct("Summarize this", ctx, backend)

        # Async functional API
        async with plugin_scope(safety_hook, rate_limit_plugin):
            result, ctx = await ainstruct("Generate code", ctx, backend)

    Args:
        *items: One or more plugins to register for the duration of the block.

    Returns:
        A context manager that registers the given plugins on entry and
        deregisters them on exit.
    """
    return _PluginScope(list(items))
