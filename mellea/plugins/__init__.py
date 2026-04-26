"""Mellea Plugin System — extension points for policy enforcement, observability, and customization.

Public API::

    from mellea.plugins import Plugin, hook, block, modify, PluginSet, register, unregister
"""

from __future__ import annotations

from .base import Plugin, PluginResult, PluginViolationError
from .decorators import hook
from .manager import (
    is_internal_tool,
    set_skip_hooks_for_internal_tools,
    skip_hooks_for_internal_tools,
)
from .pluginset import PluginSet
from .registry import block, modify, plugin_scope, register, unregister
from .types import HookType, PluginMode

__all__ = [
    "HookType",
    "Plugin",
    "PluginMode",
    "PluginResult",
    "PluginSet",
    "PluginViolationError",
    "block",
    "hook",
    "is_internal_tool",
    "modify",
    "plugin_scope",
    "register",
    "set_skip_hooks_for_internal_tools",
    "skip_hooks_for_internal_tools",
    "unregister",
]
