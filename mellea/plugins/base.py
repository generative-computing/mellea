"""Base types for the Mellea plugin system."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, TypeAlias

from pydantic import Field

try:
    from mcpgateway.plugins.framework.base import Plugin
    from mcpgateway.plugins.framework.models import (
        PluginContext,
        PluginPayload,
        PluginResult as _CFPluginResult,
    )

    _HAS_PLUGIN_FRAMEWORK = True
except ImportError:
    _HAS_PLUGIN_FRAMEWORK = False

if TYPE_CHECKING:
    from mellea.core.backend import Backend
    from mellea.core.base import Context
    from mellea.stdlib.session import MelleaSession


class PluginViolationError(Exception):
    """Raised when a plugin blocks execution in enforce mode."""

    def __init__(  # noqa: D107
        self, hook_type: str, reason: str, code: str = "", plugin_name: str = ""
    ):
        self.hook_type = hook_type
        self.reason = reason
        self.code = code
        self.plugin_name = plugin_name
        detail = f"[{code}] " if code else ""
        super().__init__(f"Plugin blocked {hook_type}: {detail}{reason}")


class MelleaBasePayload(PluginPayload):
    """Frozen base — all payloads are immutable by design.

    Plugins must use ``model_copy(update={...})`` to propose modifications
    and return the copy via ``PluginResult.modified_payload``.  The plugin
    manager applies the hook's ``HookPayloadPolicy`` to filter changes to
    writable fields only.
    """

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

        Instances support the context manager protocol for temporary activation::

            class MyPlugin(MelleaPlugin):
                def __init__(self):
                    super().__init__(PluginConfig(name="my-plugin", hooks=[...]))

                async def some_hook(self, payload, ctx):
                    ...

            with MyPlugin() as p:
                result, ctx = instruct("...", ctx, backend)

            # or async
            async with MyPlugin() as p:
                result, ctx = await ainstruct("...", ctx, backend)
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

        def __enter__(self) -> MelleaPlugin:
            """Register this plugin for the duration of a ``with`` block."""
            if getattr(self, "_scope_id", None) is not None:
                raise RuntimeError(
                    f"MelleaPlugin {self.name!r} is already active as a context manager. "
                    "Concurrent or nested reuse of the same instance is not supported; "
                    "create a new instance instead."
                )
            import uuid

            from mellea.plugins.registry import register

            self._scope_id = str(uuid.uuid4())
            register(self, session_id=self._scope_id)
            return self

        def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
            """Deregister this plugin on exit."""
            scope_id = getattr(self, "_scope_id", None)
            if scope_id is not None:
                from mellea.plugins.manager import deregister_session_plugins

                deregister_session_plugins(scope_id)
                self._scope_id = None

        async def __aenter__(self) -> MelleaPlugin:
            """Async variant — delegates to the synchronous ``__enter__``."""
            return self.__enter__()

        async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
            """Async variant — delegates to the synchronous ``__exit__``."""
            self.__exit__(exc_type, exc_val, exc_tb)

    PluginResult: TypeAlias = _CFPluginResult

else:
    # Provide a stub when the plugin framework is not installed.
    class MelleaPlugin:  # type: ignore[no-redef]
        """Stub — install ``mcp-contextforge-gateway`` for full plugin support."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError(
                "MelleaPlugin requires the ContextForge plugin framework. "
                "Install it with: pip install 'mellea[contextforge]'"
            )

    # Provide an alias when the plugin framework is not installed.
    PluginResult: TypeAlias = Any
