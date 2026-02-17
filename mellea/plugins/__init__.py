"""Mellea Plugin System — extension points for policy enforcement, observability, and customization.

Public API::

    from mellea.plugins import hook, plugin, block, PluginSet, register, MelleaPlugin
"""

from __future__ import annotations

from mellea.plugins.base import MelleaPlugin, PluginViolationError
from mellea.plugins.decorators import hook, plugin
from mellea.plugins.pluginset import PluginSet
from mellea.plugins.registry import block, register

__all__ = [
    "MelleaPlugin",
    "PluginSet",
    "PluginViolationError",
    "block",
    "hook",
    "plugin",
    "register",
]
