"""Singleton plugin manager wrapper with session-tag filtering."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from mellea.plugins.base import MelleaBasePayload, PluginViolationError
from mellea.plugins.context import build_global_context
from mellea.plugins.policies import MELLEA_HOOK_PAYLOAD_POLICIES
from mellea.plugins.types import HookType, _register_mellea_hooks

try:
    from mcpgateway.plugins.framework.manager import PluginManager

    _HAS_PLUGIN_FRAMEWORK = True
except ImportError:
    _HAS_PLUGIN_FRAMEWORK = False

if TYPE_CHECKING:
    from mellea.core.backend import Backend
    from mellea.core.base import Context
    from mellea.stdlib.session import MelleaSession

logger = logging.getLogger(__name__)

# Module-level singleton state
_plugin_manager: Any | None = None
_plugins_enabled: bool = False
_session_tags: dict[str, set[str]] = {}  # session_id -> set of plugin names


def has_plugins() -> bool:
    """Fast check: are plugins configured and available?"""
    return _plugins_enabled


def get_plugin_manager() -> Any | None:
    """Returns the initialized PluginManager, or ``None`` if plugins are not configured."""
    return _plugin_manager


def _ensure_plugin_manager() -> Any:
    """Lazily initialize the PluginManager if not already created."""
    global _plugin_manager, _plugins_enabled

    if not _HAS_PLUGIN_FRAMEWORK:
        raise ImportError(
            "Plugin system requires the ContextForge plugin framework. "
            "Install it with: pip install 'mellea[contextforge]'"
        )

    if _plugin_manager is None:
        _register_mellea_hooks()
        # Reset PluginManager singleton state to ensure clean init
        PluginManager.reset()
        pm = PluginManager(
            "",
            timeout=5,
            hook_policies=MELLEA_HOOK_PAYLOAD_POLICIES,  # type: ignore[arg-type]
        )
        from mellea.helpers import _run_async_in_thread

        _run_async_in_thread(pm.initialize())
        _plugin_manager = pm
        _plugins_enabled = True
    return _plugin_manager


async def initialize_plugins(
    config_path: str | None = None, *, timeout: float = 5.0
) -> Any:
    """Initialize the PluginManager with Mellea hook registrations and optional YAML config.

    Args:
        config_path: Optional path to a YAML plugin configuration file.
        timeout: Maximum execution time per plugin in seconds.
    """
    global _plugin_manager, _plugins_enabled

    if not _HAS_PLUGIN_FRAMEWORK:
        raise ImportError(
            "Plugin system requires the ContextForge plugin framework. "
            "Install it with: pip install 'mellea[contextforge]'"
        )

    _register_mellea_hooks()
    PluginManager.reset()
    pm = PluginManager(
        config_path or "",
        timeout=int(timeout),
        hook_policies=MELLEA_HOOK_PAYLOAD_POLICIES,  # type: ignore[arg-type]
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


def _track_session_plugin(session_id: str, plugin_name: str) -> None:
    """Track a plugin as belonging to a session for later deregistration."""
    _session_tags.setdefault(session_id, set()).add(plugin_name)


def deregister_session_plugins(session_id: str) -> None:
    """Deregister all plugins scoped to the given session."""
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


async def invoke_hook(
    hook_type: HookType,
    payload: MelleaBasePayload,
    *,
    session_id: str | None = None,
    session: MelleaSession | None = None,
    backend: Backend | None = None,
    context: Context | None = None,
    request_id: str = "",
    **context_fields: Any,
) -> tuple[Any | None, MelleaBasePayload]:
    """Invoke a hook if plugins are configured.

    Returns ``(result, possibly-modified-payload)``.
    If plugins are not configured, returns ``(None, original_payload)`` immediately.

    Three layers of no-op guards ensure zero overhead when plugins are not configured:
    1. ``_plugins_enabled`` boolean — single pointer dereference
    2. ``has_hooks_for(hook_type)`` — skips when no plugin subscribes
    3. Returns immediately when either guard fails
    """
    if not _plugins_enabled or _plugin_manager is None:
        return None, payload

    if not _plugin_manager.has_hooks_for(hook_type.value):
        return None, payload

    # Payloads are frozen — use model_copy to set dispatch-time fields
    updates: dict[str, Any] = {"hook": hook_type.value}
    if session_id is not None:
        updates["session_id"] = session_id
    if not payload.request_id:
        updates["request_id"] = request_id
    payload = payload.model_copy(update=updates)

    global_ctx = build_global_context(
        session=session,
        backend=backend,
        context=context,
        request_id=request_id,
        session_id=session_id,
        **context_fields,
    )

    result, _ = await _plugin_manager.invoke_hook(
        hook_type=hook_type.value,
        payload=payload,
        global_context=global_ctx,
        violations_as_exceptions=False,
    )

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

    modified = (
        result.modified_payload if result and result.modified_payload else payload
    )
    return result, modified
