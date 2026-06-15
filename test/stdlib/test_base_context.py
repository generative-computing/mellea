from unittest.mock import MagicMock

import pytest

from mellea.backends.context_lengths import get_context_length
from mellea.backends.model_ids import (
    IBM_GRANITE_4_1_8B,
    META_LLAMA_3_3_70B,
    META_LLAMA_4_SCOUT_17B_16E_INSTRUCT,
    MISTRALAI_MISTRAL_0_3_7B,
    ModelIdentifier,
)
from mellea.core import CBlock, Context
from mellea.stdlib.context import ChatContext, SimpleContext


def context_construction(cls: type[Context]):
    tree0 = cls()
    tree1 = tree0.add(CBlock("abc"))
    assert tree1.previous_node == tree0

    tree1a = tree0.add(CBlock("def"))
    assert tree1a.previous_node == tree0


def test_context_construction():
    context_construction(SimpleContext)
    context_construction(ChatContext)


def large_context_construction(cls: type[Context]):
    root = cls()

    full_graph: Context = root
    for i in range(1000):
        full_graph = full_graph.add(CBlock(f"abc{i}"))

    all_data = full_graph.as_list()
    assert len(all_data) == 1000


def test_large_context_construction():
    large_context_construction(SimpleContext)
    large_context_construction(ChatContext)


def test_render_view_for_simple_context():
    ctx = SimpleContext()
    for i in range(5):
        ctx = ctx.add(CBlock(f"a {i}"))
    assert len(ctx.as_list()) == 5, "Adding 5 items to context should result in 5 items"
    assert len(ctx.view_for_generation()) == 0, (  # type: ignore
        "Render size should be 0 -- NO HISTORY for SimpleContext"
    )


def test_render_view_for_chat_context():
    ctx = ChatContext(window_size=3)
    for i in range(5):
        ctx = ctx.add(CBlock(f"a {i}"))
    assert len(ctx.as_list()) == 5, "Adding 5 items to context should result in 5 items"
    assert len(ctx.view_for_generation()) == 3, "Render size should be 3"  # type: ignore


def test_actions_for_available_tools():
    ctx = ChatContext(window_size=3)
    ctx = ctx.add(CBlock("a"))
    ctx = ctx.add(CBlock("b"))

    for_generation = ctx.view_for_generation()
    assert for_generation is not None

    actions = ctx.actions_for_available_tools()
    assert actions is not None

    assert len(for_generation) == len(actions)
    for i in range(len(actions)):
        assert actions[i] == for_generation[i]


# ---------------------------------------------------------------------------
# get_context_length unit tests
# ---------------------------------------------------------------------------


def test_get_context_length_from_model_identifier_field():
    assert get_context_length(IBM_GRANITE_4_1_8B) == 131072


def test_get_context_length_returns_none_for_unknown_model_identifier():
    unknown = ModelIdentifier(hf_model_name="org/totally-unknown-model")
    assert get_context_length(unknown) is None


def test_get_context_length_from_raw_hf_string():
    assert get_context_length("mistralai/Mistral-7B-Instruct-v0.3") == 32768


def test_get_context_length_from_raw_ollama_string():
    assert get_context_length("mistral:7b") == 32768


def test_get_context_length_unknown_string():
    assert get_context_length("not-a-real-model") is None


def test_get_context_length_llama4_scout():
    assert get_context_length(META_LLAMA_4_SCOUT_17B_16E_INSTRUCT) == 10485760


# ---------------------------------------------------------------------------
# ChatContext constructor with model_id
# ---------------------------------------------------------------------------


def test_chat_context_model_id_constructor():
    ctx = ChatContext(model_id=IBM_GRANITE_4_1_8B)
    assert ctx._model_id is IBM_GRANITE_4_1_8B


def test_chat_context_existing_window_size_unchanged():
    ctx = ChatContext(window_size=3)
    for i in range(5):
        ctx = ctx.add(CBlock(f"item {i}"))
    assert len(ctx.view_for_generation()) == 3


def test_chat_context_default_no_model_id():
    ctx = ChatContext()
    assert ctx._model_id is None
    assert ctx._window_size is None


# ---------------------------------------------------------------------------
# bind_model
# ---------------------------------------------------------------------------


def test_bind_model_returns_new_root_context():
    ctx = ChatContext()
    bound = ctx.bind_model(IBM_GRANITE_4_1_8B)
    assert bound._model_id is IBM_GRANITE_4_1_8B
    assert bound.is_root_node


def test_bind_model_does_not_mutate_original():
    ctx = ChatContext()
    ctx.bind_model(IBM_GRANITE_4_1_8B)
    assert ctx._model_id is None


def test_bind_model_preserves_window_size():
    ctx = ChatContext(window_size=5)
    bound = ctx.bind_model(IBM_GRANITE_4_1_8B)
    assert bound._window_size == 5


# ---------------------------------------------------------------------------
# model_id propagates through add()
# ---------------------------------------------------------------------------


def test_model_id_propagates_through_add():
    ctx = ChatContext(model_id=IBM_GRANITE_4_1_8B)
    ctx = ctx.add(CBlock("hello"))
    ctx = ctx.add(CBlock("world"))
    assert ctx._model_id is IBM_GRANITE_4_1_8B


def test_model_id_propagates_through_bind_then_add():
    ctx = ChatContext().bind_model(META_LLAMA_3_3_70B)
    for i in range(3):
        ctx = ctx.add(CBlock(f"msg {i}"))
    assert ctx._model_id is META_LLAMA_3_3_70B


# ---------------------------------------------------------------------------
# view_for_generation with model_id
# ---------------------------------------------------------------------------


def test_view_for_generation_model_context_length_as_upper_bound():
    # Mistral 7B has context_length=32768; we have far fewer items so all are returned.
    ctx = ChatContext(model_id=MISTRALAI_MISTRAL_0_3_7B)
    for i in range(10):
        ctx = ctx.add(CBlock(f"item {i}"))
    result = ctx.view_for_generation()
    assert len(result) == 10


def test_view_for_generation_explicit_window_size_beats_model():
    ctx = ChatContext(window_size=2, model_id=IBM_GRANITE_4_1_8B)
    for i in range(10):
        ctx = ctx.add(CBlock(f"item {i}"))
    result = ctx.view_for_generation()
    assert len(result) == 2


def test_view_for_generation_no_model_id_returns_full_history():
    ctx = ChatContext()
    for i in range(8):
        ctx = ctx.add(CBlock(f"item {i}"))
    result = ctx.view_for_generation()
    assert len(result) == 8


def test_view_for_generation_unknown_model_returns_full_history():
    unknown = ModelIdentifier(hf_model_name="org/unknown-model")
    ctx = ChatContext(model_id=unknown)
    for i in range(5):
        ctx = ctx.add(CBlock(f"item {i}"))
    result = ctx.view_for_generation()
    assert len(result) == 5


# ---------------------------------------------------------------------------
# reset_to_new and session.reset()
# ---------------------------------------------------------------------------


def test_reset_to_new_returns_empty_root():
    ctx = ChatContext(model_id=IBM_GRANITE_4_1_8B, window_size=3)
    for i in range(3):
        ctx = ctx.add(CBlock(f"msg {i}"))
    reset_ctx = ctx.reset_to_new()
    assert isinstance(reset_ctx, ChatContext)
    assert reset_ctx.is_root_node
    assert len(reset_ctx.as_list()) == 0


def test_session_reset_rebinds_model_id():
    from mellea.stdlib.session import MelleaSession

    mock_backend = MagicMock()
    mock_backend.model_id = IBM_GRANITE_4_1_8B
    ctx = ChatContext()
    session = MelleaSession(mock_backend, ctx)
    assert session.ctx._model_id is IBM_GRANITE_4_1_8B

    session.ctx = session.ctx.add(CBlock("some history"))
    session.reset()

    assert isinstance(session.ctx, ChatContext)
    assert session.ctx._model_id is IBM_GRANITE_4_1_8B
    assert len(session.ctx.as_list()) == 0


# ---------------------------------------------------------------------------
# MelleaSession auto-binding
# ---------------------------------------------------------------------------


def test_session_binds_model_id_to_chat_context():
    from mellea.stdlib.session import MelleaSession

    mock_backend = MagicMock()
    mock_backend.model_id = IBM_GRANITE_4_1_8B
    ctx = ChatContext()
    session = MelleaSession(mock_backend, ctx)
    assert isinstance(session.ctx, ChatContext)
    assert session.ctx._model_id is IBM_GRANITE_4_1_8B


def test_session_does_not_override_explicit_model_id():
    from mellea.stdlib.session import MelleaSession

    mock_backend = MagicMock()
    mock_backend.model_id = IBM_GRANITE_4_1_8B
    ctx = ChatContext(model_id=META_LLAMA_3_3_70B)
    session = MelleaSession(mock_backend, ctx)
    assert session.ctx._model_id is META_LLAMA_3_3_70B


def test_session_does_not_bind_for_simple_context():
    from mellea.stdlib.session import MelleaSession

    mock_backend = MagicMock()
    mock_backend.model_id = IBM_GRANITE_4_1_8B
    ctx = SimpleContext()
    session = MelleaSession(mock_backend, ctx)
    assert isinstance(session.ctx, SimpleContext)


def test_session_graceful_when_backend_has_no_model_id():
    from mellea.stdlib.session import MelleaSession

    mock_backend = MagicMock(spec=[])  # no attributes at all
    ctx = ChatContext()
    session = MelleaSession(mock_backend, ctx)
    assert session.ctx._model_id is None


if __name__ == "__main__":
    pytest.main([__file__])
