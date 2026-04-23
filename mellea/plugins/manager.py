"""Singleton plugin manager wrapper with session-tag filtering."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Literal, TypeVar, cast

from mellea.plugins.base import MelleaBasePayload, PluginViolationError
from mellea.plugins.context import build_global_context
from mellea.plugins.policies import MELLEA_HOOK_PAYLOAD_POLICIES
from mellea.plugins.types import HookType, register_mellea_hooks

try:
    from cpex.framework.manager import PluginManager

    _HAS_PLUGIN_FRAMEWORK = True
except ImportError:
    _HAS_PLUGIN_FRAMEWORK = False

if TYPE_CHECKING:
    from mellea.core.backend import Backend

logger = logging.getLogger(__name__)

# Module-level singleton state
_plugin_manager: Any | None = None
_plugins_enabled: bool = False
_session_tags: dict[str, set[str]] = {}  # session_id -> set of plugin names
_pending_background_results: list[Any] = []

DEFAULT_PLUGIN_TIMEOUT: int = 5  # seconds
DEFAULT_HOOK_POLICY: Literal["allow"] | Literal["deny"] = "deny"


async def drain_background_tasks() -> None:
    """Await all accumulated FIRE_AND_FORGET tasks and clear the pending list.

    Call this in tests after any operation that may have triggered FAF plugins,
    to ensure side effects (metrics recording, etc.) complete before assertions.
    """
    global _pending_background_results
    pending, _pending_background_results = _pending_background_results, []
    for result in pending:
        await result.wait_for_background_tasks()


def has_plugins(hook_type: HookType | None = None) -> bool:
    """Fast check: are plugins configured and available for the given hook type.

    When ``hook_type`` is provided, also checks whether any plugin has
    registered a handler for that specific hook, enabling callers to skip
    payload construction entirely when no plugin subscribes.

    Args:
        hook_type: Optional hook type to check for registered handlers.

    Returns:
        ``True`` if plugins are enabled and (when ``hook_type`` is given)
        at least one plugin subscribes to that hook.
    """
    if not _plugins_enabled or _plugin_manager is None:
        return False
    if hook_type is not None:
        return _plugin_manager.has_hooks_for(hook_type.value)
    return True


def get_plugin_manager() -> Any | None:
    """Return the initialized PluginManager, or ``None`` if plugins are not configured.

    Returns:
        The singleton ``PluginManager`` instance, or ``None``.
    """
    return _plugin_manager


def ensure_plugin_manager() -> Any:
    """Lazily initialize the PluginManager if not already created.

    Returns:
        The singleton ``PluginManager`` instance.

    Raises:
        ImportError: If the ContextForge plugin framework is not installed.
    """
    global _plugin_manager, _plugins_enabled

    if not _HAS_PLUGIN_FRAMEWORK:
        raise ImportError(
            "Plugin system requires the ContextForge plugin framework. "
            "Install it with: pip install 'mellea[hooks]'"
        )

    if _plugin_manager is None:
        register_mellea_hooks()
        # Reset PluginManager singleton state to ensure clean init
        PluginManager.reset()
        pm = PluginManager(
            timeout=DEFAULT_PLUGIN_TIMEOUT,
            hook_policies=MELLEA_HOOK_PAYLOAD_POLICIES,  # type: ignore[arg-type]
            default_hook_policy=DEFAULT_HOOK_POLICY,
        )
        from mellea.helpers import _run_async_in_thread

        _run_async_in_thread(pm.initialize())
        _plugin_manager = pm
        _plugins_enabled = True
    return _plugin_manager


async def initialize_plugins(
    config_path: str | None = None,
    *,
    timeout: int = DEFAULT_PLUGIN_TIMEOUT,  # noqa: ASYNC109
) -> Any:
    """Initialize the PluginManager with Mellea hook registrations and optional YAML config.

    Args:
        config_path: Optional path to a YAML plugin configuration file.
        timeout: Maximum execution time per plugin in seconds.

    Returns:
        The initialized ``PluginManager`` instance.

    Raises:
        ImportError: If the ContextForge plugin framework is not installed.
    """
    global _plugin_manager, _plugins_enabled

    if not _HAS_PLUGIN_FRAMEWORK:
        raise ImportError(
            "Plugin system requires the ContextForge plugin framework. "
            "Install it with: pip install 'mellea[hooks]'"
        )

    register_mellea_hooks()
    PluginManager.reset()
    pm = PluginManager(
        config_path or "",
        timeout=timeout,
        hook_policies=MELLEA_HOOK_PAYLOAD_POLICIES,  # type: ignore[arg-type]
        default_hook_policy=DEFAULT_HOOK_POLICY,
    )
    await pm.initialize()
    _plugin_manager = pm
    _plugins_enabled = True
    return pm


async def shutdown_plugins() -> None:
    """Shut down the PluginManager and reset all state."""
    global _plugin_manager, _plugins_enabled, _session_tags

    if _plugin_manager is not None:
        await _plugin_manager.shutdown()
    _plugin_manager = None
    _plugins_enabled = False
    _session_tags.clear()


def track_session_plugin(session_id: str, plugin_name: str) -> None:
    """Track a plugin as belonging to a session for later deregistration.

    Args:
        session_id: Identifier for the session that owns the plugin.
        plugin_name: Registered name of the plugin.
    """
    _session_tags.setdefault(session_id, set()).add(plugin_name)


def deregister_session_plugins(session_id: str) -> None:
    """Deregister all plugins scoped to the given session.

    Args:
        session_id: Identifier for the session whose plugins should be removed.
    """
    if not _plugins_enabled or _plugin_manager is None:
        return

    plugin_names = _session_tags.pop(session_id, set())
    for name in plugin_names:
        try:
            _plugin_manager._registry.unregister(name)
            logger.debug(
                "Deregistered session plugin: %s (session=%s)", name, session_id
            )
        except Exception:
            logger.debug("Plugin %s already unregistered", name, exc_info=True)


# Hooks return the same payload they received. Use this to accurately reflect that typing.
_MelleaBasePayload = TypeVar("_MelleaBasePayload", bound=MelleaBasePayload)


async def invoke_hook(
    hook_type: HookType,
    payload: _MelleaBasePayload,
    *,
    backend: Backend | None = None,
    **context_fields: Any,
) -> tuple[Any | None, _MelleaBasePayload]:
    """Invoke a hook if plugins are configured.

    Returns ``(result, possibly-modified-payload)``.
    If plugins are not configured, returns ``(None, original_payload)`` immediately.

    Three layers of no-op guards ensure zero overhead when plugins are not configured:
    1. ``_plugins_enabled`` boolean — single pointer dereference
    2. ``has_hooks_for(hook_type)`` — skips when no plugin subscribes
    3. Returns immediately when either guard fails

    Args:
        hook_type: The hook point to invoke.
        payload: The immutable payload to pass to plugin handlers.
        backend: Optional backend for building the global context.
        **context_fields: Additional fields passed to ``build_global_context``.

    Returns:
        A ``(result, payload)`` tuple where *result* is the ``PluginResult``
        (or ``None`` when no plugins ran) and *payload* is the
        possibly-modified payload.

    Raises:
        PluginViolationError: If a plugin blocks execution.
    """
    if not _plugins_enabled or _plugin_manager is None:
        return None, payload

    if not _plugin_manager.has_hooks_for(hook_type.value):
        return None, payload

    # Payloads are frozen — use model_copy to set dispatch-time fields
    updates: dict[str, Any] = {"hook": hook_type.value}
    payload = payload.model_copy(update=updates)

    global_ctx = build_global_context(backend=backend, **context_fields)

    result, _ = await _plugin_manager.invoke_hook(
        hook_type=hook_type.value,
        payload=payload,
        global_context=global_ctx,
        violations_as_exceptions=False,
    )

    if result and result.background_tasks:
        _pending_background_results.append(result)

    if result and not result.continue_processing and result.violation:
        v = result.violation
        logger.warning(
            "Plugin violation on %s: [%s] %s (plugin=%s)",
            hook_type.value,
            v.code,
            v.reason,
            v.plugin_name or "unknown",
        )
        raise PluginViolationError(
            hook_type=hook_type.value,
            reason=v.reason,
            code=v.code,
            plugin_name=v.plugin_name or "",
        )

    # `result` doesn't type the returned payload correctly.
    # If the modified payload exists, cast it as the correct type here,
    # else return the original payload.
    modified: _MelleaBasePayload = (
        cast(_MelleaBasePayload, result.modified_payload)
        if result and result.modified_payload
        else payload
    )
    return result, modified
