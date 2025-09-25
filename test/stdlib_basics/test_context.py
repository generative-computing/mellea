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
