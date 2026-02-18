"""Mellea Plugin System — extension points for policy enforcement, observability, and customization.

Public API::

    from mellea.plugins import hook, plugin, block, PluginSet, register, MelleaPlugin
"""

from __future__ import annotations

import logging

logging.getLogger("mcpgateway.config").setLevel(logging.ERROR)
logging.getLogger("mcpgateway.observability").setLevel(logging.ERROR)

from mellea.plugins.base import MelleaPlugin, PluginResult, PluginViolationError
from mellea.plugins.decorators import hook, plugin
from mellea.plugins.pluginset import PluginSet
from mellea.plugins.registry import block, plugin_scope, register
from mellea.plugins.types import HookType, PluginMode

__all__ = [
    "HookType",
    "MelleaPlugin",
    "PluginMode",
    "PluginResult",
    "PluginSet",
    "PluginViolationError",
    "block",
    "hook",
    "plugin",
    "plugin_scope",
    "register",
]
