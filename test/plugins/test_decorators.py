"""Tests for the @hook and @plugin decorators."""

import pytest

from mellea.plugins.decorators import HookMeta, PluginMeta, hook, plugin
from mellea.plugins.types import PluginMode


class TestHookDecorator:
    def test_hook_attaches_metadata(self):
        @hook("generation_pre_call")
        async def my_hook(payload, ctx):
            pass

        assert hasattr(my_hook, "_mellea_hook_meta")
        meta = my_hook._mellea_hook_meta
        assert isinstance(meta, HookMeta)
        assert meta.hook_type == "generation_pre_call"
        assert meta.mode == PluginMode.ENFORCE
        assert meta.priority == 50

    def test_hook_custom_mode_and_priority(self):
        @hook("component_post_success", mode=PluginMode.PERMISSIVE, priority=10)
        async def my_hook(payload, ctx):
            pass

        meta = my_hook._mellea_hook_meta
        assert meta.mode == PluginMode.PERMISSIVE
        assert meta.priority == 10

    def test_hook_fire_and_forget_mode(self):
        @hook("component_post_success", mode=PluginMode.FIRE_AND_FORGET)
        async def my_hook(payload, ctx):
            pass

        meta = my_hook._mellea_hook_meta
        assert meta.mode == PluginMode.FIRE_AND_FORGET

    def test_hook_preserves_function(self):
        @hook("generation_pre_call")
        async def my_hook(payload, ctx):
            return "result"

        assert my_hook.__name__ == "my_hook"


class TestPluginDecorator:
    def test_plugin_attaches_metadata(self):
        @plugin("my-plugin")
        class MyPlugin:
            pass

        assert hasattr(MyPlugin, "_mellea_plugin_meta")
        meta = MyPlugin._mellea_plugin_meta
        assert isinstance(meta, PluginMeta)
        assert meta.name == "my-plugin"
        assert meta.priority == 50

    def test_plugin_custom_priority(self):
        @plugin("my-plugin", priority=5)
        class MyPlugin:
            pass

        assert MyPlugin._mellea_plugin_meta.priority == 5

    def test_plugin_preserves_class(self):
        @plugin("my-plugin")
        class MyPlugin:
            def __init__(self):
                self.value = 42

        instance = MyPlugin()
        assert instance.value == 42
