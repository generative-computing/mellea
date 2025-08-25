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
