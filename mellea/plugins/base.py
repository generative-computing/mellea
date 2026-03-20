"""Base types for the Mellea plugin system."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timezone
from types import TracebackType
from typing import TYPE_CHECKING, Any, TypeAlias

from pydantic import Field

# ---------------------------------------------------------------------------
# Plugin base class (no cpex dependency)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PluginMeta:
    """Metadata attached to Plugin subclasses.

    Args:
        name: Unique identifier for the plugin.
        priority: Execution priority — lower numbers execute first.
    """

    name: str
    priority: int = 50


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


class Plugin:
    """Base class for multi-hook Mellea plugins.

    Subclasses get automatic context-manager support and plugin metadata::

        class PIIRedactor(Plugin, name="pii-redactor", priority=5):
            @hook(HookType.COMPONENT_PRE_EXECUTE, mode=PluginMode.SEQUENTIAL)
            async def redact_input(self, payload, ctx):
                ...

        with PIIRedactor():
            result = session.instruct("...")
    """

    def __init_subclass__(
        cls, *, name: str = "", priority: int = 50, **kwargs: Any
    ) -> None:
        """Set plugin metadata on subclasses that provide a ``name``."""
        super().__init_subclass__(**kwargs)
        if name:
            cls._mellea_plugin_meta = PluginMeta(name=name, priority=priority)  # type: ignore[attr-defined]

    def __enter__(self) -> Any:
        """Register this plugin for the duration of a ``with`` block."""
        return _plugin_cm_enter(self)

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Deregister this plugin on exit."""
        _plugin_cm_exit(self, exc_type, exc_val, exc_tb)

    async def __aenter__(self) -> Any:
        """Async variant — delegates to ``__enter__``."""
        return self.__enter__()

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Async variant — delegates to ``__exit__``."""
        self.__exit__(exc_type, exc_val, exc_tb)


# ---------------------------------------------------------------------------
# cpex plugin framework (optional dependency)
# ---------------------------------------------------------------------------

try:
    from cpex.framework.base import Plugin as _CpexPlugin
    from cpex.framework.models import (
        PluginContext,
        PluginPayload,
        PluginResult as _CFPluginResult,
    )

    _HAS_PLUGIN_FRAMEWORK = True
except ImportError:
    _HAS_PLUGIN_FRAMEWORK = False

if TYPE_CHECKING:
    from cpex.framework.base import Plugin as _CpexPlugin
    from cpex.framework.models import (
        PluginContext,
        PluginPayload,
        PluginResult as _CFPluginResult,
    )

    from mellea.core.backend import Backend
    from mellea.core.base import Context
    from mellea.stdlib.session import MelleaSession


class PluginViolationError(Exception):
    """Raised when a plugin blocks execution in enforce mode.

    Args:
        hook_type: The hook point where the violation occurred.
        reason: Human-readable reason for the violation.
        code: Machine-readable violation code.
        plugin_name: Name of the plugin that raised the violation.
    """

    def __init__(  # noqa: D107
        self, hook_type: str, reason: str, code: str = "", plugin_name: str = ""
    ):
        self.hook_type = hook_type
        self.reason = reason
        self.code = code
        self.plugin_name = plugin_name
        detail = f"[{code}] " if code else ""
        super().__init__(f"Plugin blocked {hook_type}: {detail}{reason}")


# ---------------------------------------------------------------------------
# Dynamic base classes — a single ClassDef node per class ensures Griffe's
# static AST parser always picks up the authoritative docstring.  See #667.
# ---------------------------------------------------------------------------

if _HAS_PLUGIN_FRAMEWORK or TYPE_CHECKING:
    _PayloadBase = PluginPayload
    _PluginBase = _CpexPlugin
else:
    _PayloadBase = object
    _PluginBase = object


class MelleaBasePayload(_PayloadBase):  # type: ignore[misc,valid-type]
    """Frozen base — all payloads are immutable by design.

    Plugins must use ``model_copy(update={...})`` to propose modifications
    and return the copy via ``PluginResult.modified_payload``.  The plugin
    manager applies the hook's ``HookPayloadPolicy`` to filter changes to
    writable fields only.
    """

    session_id: str | None = None
    request_id: str = ""
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    hook: str = ""
    user_metadata: dict[str, Any] = Field(default_factory=dict)


class MelleaPlugin(_PluginBase):  # type: ignore[misc,valid-type]
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
        """Get the Backend from the plugin context.

        Args:
            context: The plugin context provided by the hook framework.

        Returns:
            The active Backend, or ``None`` if unavailable.
        """
        return context.global_context.state.get("backend")

    def get_mellea_context(self, context: PluginContext) -> Context | None:
        """Get the Mellea Context from the plugin context.

        Args:
            context: The plugin context provided by the hook framework.

        Returns:
            The active Mellea Context, or ``None`` if unavailable.
        """
        return context.global_context.state.get("context")

    def get_session(self, context: PluginContext) -> MelleaSession | None:
        """Get the MelleaSession from the plugin context.

        Args:
            context: The plugin context provided by the hook framework.

        Returns:
            The active MelleaSession, or ``None`` if unavailable.
        """
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
            self._scope_id = None  # type: ignore[assignment]

    async def __aenter__(self) -> MelleaPlugin:
        """Async variant — delegates to the synchronous ``__enter__``."""
        return self.__enter__()

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async variant — delegates to the synchronous ``__exit__``."""
        self.__exit__(exc_type, exc_val, exc_tb)


# When cpex is not installed, override __init__ to provide a clear error
# message.  This is done post-hoc so there is only one ClassDef node in the
# AST for each class (the docstring-carrying definition above).
if not _HAS_PLUGIN_FRAMEWORK:

    def _stub_init(_self: Any, *_args: Any, **_kwargs: Any) -> None:
        raise ImportError(
            "This class requires the ContextForge plugin framework. "
            "Install it with: pip install 'mellea[hooks]'"
        )

    MelleaBasePayload.__init__ = _stub_init  # type: ignore[assignment,misc]
    MelleaPlugin.__init__ = _stub_init  # type: ignore[assignment,misc]

if _HAS_PLUGIN_FRAMEWORK:
    PluginResult: TypeAlias = _CFPluginResult  # type: ignore[misc]
else:
    PluginResult: TypeAlias = Any  # type: ignore[no-redef,misc]
