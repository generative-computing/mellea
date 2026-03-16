"""Plugin context factory — maps Mellea domain objects to ContextForge GlobalContext."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

try:
    from cpex.framework.models import GlobalContext

    _HAS_PLUGIN_FRAMEWORK = True
except ImportError:
    _HAS_PLUGIN_FRAMEWORK = False

if TYPE_CHECKING:
    from mellea.core.backend import Backend


def build_global_context(*, backend: Backend | None = None, **extra_fields: Any) -> Any:
    """Build a ContextForge `GlobalContext` from Mellea domain objects.

    The global context carries lightweight, cross-cutting ambient metadata
    (e.g. `backend_name`) that is useful to every hook regardless of type.
    Hook-specific data (context, session, action, etc.) belongs on the typed
    payload, not here.

    Returns `None` if ContextForge is not installed.
    """
    if not _HAS_PLUGIN_FRAMEWORK:
        return None

    state: dict[str, Any] = {}
    if backend is not None:
        state["backend_name"] = getattr(backend, "model_id", "unknown")
    state.update(extra_fields)

    return GlobalContext(request_id="", state=state)
