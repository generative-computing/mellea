import pytest

from mellea.stdlib.base import Context, CBlock, SimpleContext, ChatContext


def context_construction(cls: type[Context]):
    tree0 = cls()
    tree1 = tree0.add(CBlock("abc"))
    assert tree1.previous == tree0

    tree1a = tree0.add(CBlock("def"))
    assert tree1a.previous == tree0


def test_context_construction():
    context_construction(SimpleContext)
    context_construction(ChatContext)


def large_context_construction(cls: type[Context]):
    root = cls()

    full_graph: Context = root
    for i in range(1000):
        full_graph = full_graph.add(CBlock(f"abc{i}"))

    all_data = full_graph.full_data_as_list()
    assert len(all_data) == 1000


def test_large_context_construction():
    large_context_construction(SimpleContext)
    large_context_construction(ChatContext)


def test_render_view_for_simple_context():
    ctx = SimpleContext()
    for i in range(5):
        ctx = ctx.add(CBlock(f"a {i}"))
    assert len(ctx.full_data_as_list()) == 5, "Adding 5 items to context should result in 5 items"
    assert len(ctx.render_for_generation()) == 0, "Render size should be 0 -- NO HISTORY for SimpleContext"


def test_render_view_for_chat_context():
    ctx = ChatContext(window_size=3)
    for i in range(5):
        ctx = ctx.add(CBlock(f"a {i}"))
    assert len(ctx.full_data_as_list()) == 5, "Adding 5 items to context should result in 5 items"
    assert len(ctx.render_for_generation()) == 3, "Render size should be 3"


def test_actions_for_available_tools():
    ctx = ChatContext(window_size=3)
    ctx = ctx.add(CBlock("a"))
    ctx = ctx.add(CBlock("b"))

    for_generation = ctx.render_for_generation()
    assert for_generation is not None

    actions = ctx.actions_for_available_tools()
    assert actions is not None

    assert len(for_generation) == len(actions)
    for i in range(len(actions)):
        assert actions[i] == for_generation[i]


if __name__ == "__main__":
    pytest.main([__file__])
