"""Plugin context factory — maps Mellea domain objects to ContextForge GlobalContext."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

try:
    from mcpgateway.plugins.framework.models import GlobalContext

    _HAS_PLUGIN_FRAMEWORK = True
except ImportError:
    _HAS_PLUGIN_FRAMEWORK = False

if TYPE_CHECKING:
    from mellea.core.backend import Backend
    from mellea.core.base import Context
    from mellea.stdlib.session import MelleaSession


def build_global_context(
    *,
    session: MelleaSession | None = None,
    backend: Backend | None = None,
    context: Context | None = None,
    request_id: str = "",
    **extra_fields: Any,
) -> Any:
    """Build a ContextForge ``GlobalContext`` from Mellea domain objects.

    Returns ``None`` if ContextForge is not installed.
    """
    if not _HAS_PLUGIN_FRAMEWORK:
        return None

    state: dict[str, Any] = {}
    if session is not None:
        state["session"] = session
    if backend is not None:
        state["backend"] = backend
        state["backend_name"] = getattr(backend, "model_id", "unknown")
    if context is not None:
        state["context"] = context
    state.update(extra_fields)

    return GlobalContext(request_id=request_id, state=state)
