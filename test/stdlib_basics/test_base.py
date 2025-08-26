import pytest
from mellea.stdlib.base import CBlock, Component, LinearContext


def test_cblock():
    cb = CBlock(value="This is some text")
    str(cb)
    repr(cb)
    assert str(cb) == "This is some text"


def test_cblpock_meta():
    cb = CBlock("asdf", meta={"x": "y"})
    assert str(cb) == "asdf"
    assert cb._meta["x"] == "y"


def test_component():
    class _ClosuredComponent(Component):
        def parts(self):
            return []

        def format_for_llm(self) -> str:
            return ""

    c = _ClosuredComponent()
    assert len(c.parts()) == 0


def test_context():
    ctx = LinearContext(window_size=3)
    ctx.insert(CBlock("a"))
    ctx.insert(CBlock("b"))
    ctx.insert(CBlock("c"))
    ctx.insert(CBlock("d"))


def test_actions_for_available_tools():
    ctx = LinearContext(window_size=3)
    ctx.insert(CBlock("a"))
    ctx.insert(CBlock("b"))
    for_generation = ctx.render_for_generation()
    assert for_generation is not None

    actions = ctx.actions_for_available_tools()
    assert actions is not None

    assert len(for_generation) == len(actions)
    for i in range(len(actions)):
        assert actions[i] == for_generation[i]

if __name__ == "__main__":
    pytest.main([__file__])
