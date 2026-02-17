"""Base types for the Mellea plugin system."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field

try:
    from mcpgateway.plugins.framework.base import Plugin
    from mcpgateway.plugins.framework.models import PluginContext

    _HAS_PLUGIN_FRAMEWORK = True
except ImportError:
    _HAS_PLUGIN_FRAMEWORK = False

if TYPE_CHECKING:
    from mellea.core.backend import Backend
    from mellea.core.base import Context
    from mellea.stdlib.session import MelleaSession


class MelleaBasePayload(BaseModel):
    """Frozen base — all payloads are immutable by design.

    Plugins must use ``model_copy(update={...})`` to propose modifications
    and return the copy via ``PluginResult.modified_payload``.  The plugin
    manager applies the hook's ``HookPayloadPolicy`` to filter changes to
    writable fields only.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    session_id: str | None = None
    request_id: str = ""
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    hook: str = ""
    user_metadata: dict[str, Any] = Field(default_factory=dict)


if _HAS_PLUGIN_FRAMEWORK:

    class MelleaPlugin(Plugin):
        """Base class for Mellea plugins with lifecycle hooks and typed accessors.

        Use this when you need lifecycle hooks (``initialize``/``shutdown``)
        or typed context accessors.  For simpler plugins, prefer ``@hook``
        on standalone functions or ``@plugin`` on plain classes.
        """

        def get_backend(self, context: PluginContext) -> Backend | None:
            """Get the Backend from the plugin context."""
            return context.global_context.state.get("backend")

        def get_mellea_context(self, context: PluginContext) -> Context | None:
            """Get the Mellea Context from the plugin context."""
            return context.global_context.state.get("context")

        def get_session(self, context: PluginContext) -> MelleaSession | None:
            """Get the MelleaSession from the plugin context."""
            return context.global_context.state.get("session")

        @property
        def plugin_config(self) -> dict[str, Any]:
            """Plugin-specific configuration from PluginConfig.config."""
            return self._config.config or {}

else:
    # Provide a stub when the plugin framework is not installed.
    class MelleaPlugin:  # type: ignore[no-redef]
        """Stub — install ``mcp-contextforge-gateway`` for full plugin support."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError(
                "MelleaPlugin requires the ContextForge plugin framework. "
                "Install it with: pip install 'mellea[contextforge]'"
            )
