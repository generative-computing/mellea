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
    assert ctx.model_id is IBM_GRANITE_4_1_8B


def test_chat_context_existing_window_size_unchanged():
    ctx = ChatContext(window_size=3)
    for i in range(5):
        ctx = ctx.add(CBlock(f"item {i}"))
    assert len(ctx.view_for_generation()) == 3


def test_chat_context_default_no_model_id():
    ctx = ChatContext()
    assert ctx.model_id is None
    assert ctx._window_size is None


# ---------------------------------------------------------------------------
# _bind_model
# ---------------------------------------------------------------------------


def test__bind_model_returns_new_root_context():
    ctx = ChatContext()
    bound = ctx._bind_model(IBM_GRANITE_4_1_8B)
    assert bound.model_id is IBM_GRANITE_4_1_8B
    assert bound.is_root_node


def test__bind_model_does_not_mutate_original():
    ctx = ChatContext()
    ctx._bind_model(IBM_GRANITE_4_1_8B)
    assert ctx.model_id is None


def test__bind_model_preserves_window_size():
    ctx = ChatContext(window_size=5)
    bound = ctx._bind_model(IBM_GRANITE_4_1_8B)
    assert bound._window_size == 5


def test__bind_model_raises_on_non_root_context():
    ctx = ChatContext()
    ctx = ctx.add(CBlock("some history"))
    with pytest.raises(ValueError, match="_bind_model\\(\\) must be called on a root"):
        ctx._bind_model(IBM_GRANITE_4_1_8B)


# ---------------------------------------------------------------------------
# model_id propagates through add()
# ---------------------------------------------------------------------------


def test_model_id_propagates_through_add():
    ctx = ChatContext(model_id=IBM_GRANITE_4_1_8B)
    ctx = ctx.add(CBlock("hello"))
    ctx = ctx.add(CBlock("world"))
    assert ctx.model_id is IBM_GRANITE_4_1_8B


def test_model_id_propagates_through_bind_then_add():
    ctx = ChatContext()._bind_model(META_LLAMA_3_3_70B)
    for i in range(3):
        ctx = ctx.add(CBlock(f"msg {i}"))
    assert ctx.model_id is META_LLAMA_3_3_70B


# ---------------------------------------------------------------------------
# view_for_generation with model_id
# ---------------------------------------------------------------------------


def test_view_for_generation_model_bound_returns_full_history_when_under_limit():
    # 10 items is far below the 32768-token ceiling — full history returned.
    ctx = ChatContext(model_id=MISTRALAI_MISTRAL_0_3_7B)
    for i in range(10):
        ctx = ctx.add(CBlock(f"item {i}"))
    result = ctx.view_for_generation()
    assert len(result) == 10


def test_view_for_generation_model_bound_used_as_upper_bound():
    # A tiny context_length=10 triggers truncation; most recent items are retained.
    # Each "item N" renders to ~7 chars → cost = max(1, 7 // 4) = 1 token each.
    # Effective budget = int(10 * 0.75) = 7; fits the 7 most-recent items.
    tiny = ModelIdentifier(hf_model_name="org/tiny-model", context_length=10)
    ctx = ChatContext(model_id=tiny)
    for i in range(15):
        ctx = ctx.add(CBlock(f"item {i}"))
    result = ctx.view_for_generation()
    assert len(result) == 7
    assert str(result[-1]) == "item 14"


def test_view_for_generation_token_budget_drops_oldest():
    # Each item is a 100-char string → cost = 101 // 4 = 25 tokens.
    # context_length=130: effective budget = int(130 * 0.75) = 97.
    # Fits 3 items (75 tokens); the 4th would push spent to 100 > 97.
    tiny = ModelIdentifier(hf_model_name="org/tiny-model", context_length=130)
    ctx = ChatContext(model_id=tiny)
    for i in range(7):
        ctx = ctx.add(CBlock("x" * 100 + str(i)))  # 101 chars → cost 25
    result = ctx.view_for_generation()
    assert len(result) == 3
    # Newest items are kept.
    assert str(result[-1]).endswith("6")
    assert str(result[-2]).endswith("5")


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


def test_reset_to_new_classmethod_does_not_preserve_config():
    # reset_to_new() is a classmethod — it returns a bare ChatContext() with
    # no window_size or model_id, regardless of the instance it's called on.
    ctx = ChatContext(model_id=IBM_GRANITE_4_1_8B, window_size=5)
    reset_ctx = ctx.reset_to_new()
    assert reset_ctx.model_id is None
    assert reset_ctx._window_size is None


def test_new_instance_preserves_model_id_and_window_size():
    ctx = ChatContext(model_id=IBM_GRANITE_4_1_8B, window_size=5)
    for i in range(3):
        ctx = ctx.add(CBlock(f"msg {i}"))
    new_ctx = ctx.new_instance()
    assert new_ctx.model_id is IBM_GRANITE_4_1_8B
    assert new_ctx._window_size == 5


def test_session_reset_rebinds_model_id():
    from mellea.stdlib.session import MelleaSession

    mock_backend = MagicMock()
    mock_backend.model_id = IBM_GRANITE_4_1_8B
    ctx = ChatContext()
    session = MelleaSession(mock_backend, ctx)
    assert session.ctx.model_id is IBM_GRANITE_4_1_8B

    session.ctx = session.ctx.add(CBlock("some history"))
    session.reset()

    assert isinstance(session.ctx, ChatContext)
    assert session.ctx.model_id is IBM_GRANITE_4_1_8B
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
    assert session.ctx.model_id is IBM_GRANITE_4_1_8B


def test_session_does_not_override_explicit_model_id():
    from mellea.stdlib.session import MelleaSession

    mock_backend = MagicMock()
    mock_backend.model_id = IBM_GRANITE_4_1_8B
    ctx = ChatContext(model_id=META_LLAMA_3_3_70B)
    session = MelleaSession(mock_backend, ctx)
    assert session.ctx.model_id is META_LLAMA_3_3_70B


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
    assert session.ctx.model_id is None


def test_session_does_not_bind_when_context_has_history():
    from mellea.stdlib.session import MelleaSession

    mock_backend = MagicMock()
    mock_backend.model_id = IBM_GRANITE_4_1_8B
    ctx = ChatContext()
    ctx = ctx.add(CBlock("pre-existing history"))
    session = MelleaSession(mock_backend, ctx)
    # Non-root context must not be auto-bound (would silently discard history).
    assert session.ctx.model_id is None
    assert len(session.ctx.as_list()) == 1


def test_get_context_length_litellm_prefixed_string():
    # LiteLLM prefixes model names with a provider slug, e.g. "ollama_chat/granite4.1:3b".
    # The stripped bare name should resolve to the correct context length.
    assert get_context_length("ollama_chat/granite4.1:3b") == 131072
    assert get_context_length("ollama/granite4.1:8b") == 131072
    # Multi-segment strip still resolves when the remainder is a known HF name.
    assert get_context_length("huggingface/ibm-granite/granite-4.1-3b") == 131072
    # Completely unknown prefix+name returns None.
    assert get_context_length("someprefix/not-a-real-model") is None


def test_clone_does_not_double_bind_replaced_root_context():
    from copy import copy

    from mellea.stdlib.session import MelleaSession

    mock_backend = MagicMock()
    mock_backend.model_id = IBM_GRANITE_4_1_8B
    session = MelleaSession(mock_backend, ChatContext())
    assert session.ctx.model_id is IBM_GRANITE_4_1_8B

    # User replaces ctx with a fresh unbound root — no model_id.
    session.ctx = ChatContext()
    assert session.ctx.model_id is None

    cloned = copy(session)
    # Clone must preserve the replaced (unbound) context, not re-bind from backend.
    assert cloned.ctx.model_id is None
    assert cloned.id != session.id  # fresh session ID


# ---------------------------------------------------------------------------
# _as_list_token_budget boundary and _build_table collision tests
# ---------------------------------------------------------------------------


def test_as_list_token_budget_fits_exactly():
    # Verify > vs >= boundary: an item whose cost equals the remaining budget
    # is INCLUDED (condition is `spent + cost > effective_budget`, so equality passes).
    # context_length=20 → effective = int(20 * 0.75) = 15.
    # Items of 8 chars → cost = max(1, 8 // 4) = 2 tokens each.
    # Seven items = 14 tokens ≤ 15 (all included); 8th = 16 > 15 (excluded).
    tiny = ModelIdentifier(hf_model_name="org/exact-fit-model", context_length=20)
    ctx = ChatContext(model_id=tiny)
    for _ in range(8):
        ctx = ctx.add(CBlock("abcdefgh"))  # 8 chars → cost 2
    result = ctx.view_for_generation()
    assert len(result) == 7


def test_build_table_raises_on_collision(monkeypatch):
    from mellea.backends.context_lengths import _build_table
    from mellea.backends.model_ids import ModelIdentifier

    colliding_a = ModelIdentifier(
        hf_model_name="shared/model-name", context_length=1024
    )
    colliding_b = ModelIdentifier(
        hf_model_name="shared/model-name", context_length=2048
    )

    import mellea.backends.model_ids as _m_module

    monkeypatch.setattr(_m_module, "COLLIDE_A", colliding_a, raising=False)
    monkeypatch.setattr(_m_module, "COLLIDE_B", colliding_b, raising=False)
    with pytest.raises(ValueError, match="context_length collision"):
        _build_table()


if __name__ == "__main__":
    pytest.main([__file__])
