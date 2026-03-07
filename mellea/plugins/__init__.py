"""Mellea Plugin System — extension points for policy enforcement, observability, and customization.

Public API::

    from mellea.plugins import Plugin, hook, block, modify, PluginSet, register, unregister
"""

from __future__ import annotations

from mellea.plugins.base import Plugin, PluginResult, PluginViolationError
from mellea.plugins.decorators import hook
from mellea.plugins.pluginset import PluginSet
from mellea.plugins.registry import block, modify, plugin_scope, register, unregister
from mellea.plugins.types import HookType, PluginMode

__all__ = [
    "HookType",
    "Plugin",
    "PluginMode",
    "PluginResult",
    "PluginSet",
    "PluginViolationError",
    "block",
    "hook",
    "modify",
    "plugin_scope",
    "register",
    "unregister",
]
