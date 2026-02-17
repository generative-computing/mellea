"""Tests for PluginSet."""

from mellea.plugins.pluginset import PluginSet


class TestPluginSet:
    def test_basic_creation(self):
        ps = PluginSet("test", [1, 2, 3])
        assert ps.name == "test"
        assert len(ps.items) == 3
        assert ps.priority is None

    def test_with_priority(self):
        ps = PluginSet("test", [1], priority=10)
        assert ps.priority == 10

    def test_flatten_simple(self):
        ps = PluginSet("test", ["a", "b", "c"], priority=5)
        result = ps.flatten()
        assert result == [("a", 5), ("b", 5), ("c", 5)]

    def test_flatten_no_priority(self):
        ps = PluginSet("test", ["a", "b"])
        result = ps.flatten()
        assert result == [("a", None), ("b", None)]

    def test_flatten_nested(self):
        inner = PluginSet("inner", ["x", "y"], priority=10)
        outer = PluginSet("outer", [inner, "z"], priority=20)
        result = outer.flatten()
        # inner items keep inner's priority, outer item gets outer's priority
        assert result == [("x", 10), ("y", 10), ("z", 20)]

    def test_flatten_deeply_nested(self):
        deep = PluginSet("deep", ["a"], priority=1)
        mid = PluginSet("mid", [deep, "b"], priority=2)
        top = PluginSet("top", [mid, "c"], priority=3)
        result = top.flatten()
        assert result == [("a", 1), ("b", 2), ("c", 3)]

    def test_repr(self):
        ps = PluginSet("security", [1, 2])
        assert repr(ps) == "PluginSet('security', 2 items)"
