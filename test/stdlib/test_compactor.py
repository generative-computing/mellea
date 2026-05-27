"""Tests for the ``Compactor`` protocol, ``WindowCompactor``, ``ThresholdCompactor``."""

from __future__ import annotations

import pytest

from mellea.core.base import ModelOutputThunk
from mellea.stdlib.components.chat import Message
from mellea.stdlib.context import (
    ChatContext,
    Compactor,
    LLMSummarizeCompactor,
    PinPredicate,
    ThresholdCompactor,
    WindowCompactor,
    pin_nothing,
    pin_system,
    pin_system_and_initial_user,
)
from mellea.stdlib.context.compactor import _last_usage_tokens


def _msg(i: int) -> Message:
    return Message(role="user", content=f"m{i}")


def _thunk(total_tokens: int, value: str = "") -> ModelOutputThunk:
    """Build a ModelOutputThunk with a populated usage dict."""
    mot = ModelOutputThunk(value=value)
    mot.generation.usage = {
        "prompt_tokens": total_tokens,
        "completion_tokens": 0,
        "total_tokens": total_tokens,
    }
    return mot


class TestChatContextDefaults:
    def test_default_has_no_compactor(self):
        # Compaction is opt-in: bare ChatContext() retains full history.
        ctx = ChatContext()
        assert ctx._compactor is None

    def test_default_keeps_full_history(self):
        ctx = ChatContext()
        for i in range(20):
            ctx = ctx.add(_msg(i))
        assert len(ctx.as_list()) == 20

    def test_window_size_arg_constructs_window_compactor(self):
        ctx = ChatContext(window_size=3)
        assert isinstance(ctx._compactor, WindowCompactor)
        assert ctx._compactor.size == 3

    def test_passing_both_args_raises(self):
        with pytest.raises(ValueError):
            ChatContext(compactor=WindowCompactor(size=2), window_size=3)

    def test_explicit_compactor_overrides_default(self):
        comp = WindowCompactor(size=2)
        ctx = ChatContext(compactor=comp)
        assert ctx._compactor is comp


class TestWindowCompactor:
    def test_compact_keeps_last_n(self):
        ctx = ChatContext(window_size=3)
        for i in range(7):
            ctx = ctx.add(_msg(i))
        items = ctx.as_list()
        assert len(items) == 3
        assert [m.content for m in items] == ["m4", "m5", "m6"]

    def test_compact_does_not_mutate_original(self):
        # Build with a permissive window so all 3 items are retained, then
        # apply a tighter compactor manually (Pattern 2).
        ctx = ChatContext(window_size=10_000)
        ctx = ctx.add(_msg(0))
        ctx = ctx.add(_msg(1))
        ctx = ctx.add(_msg(2))
        before_compact = [m.content for m in ctx.as_list()]
        compacted = WindowCompactor(size=2).compact(ctx)
        # original unchanged
        assert [m.content for m in ctx.as_list()] == before_compact
        # compacted is shorter and a different object
        assert compacted is not ctx
        assert len(compacted.as_list()) == 2

    def test_compact_preserves_compactor_on_result(self):
        comp = WindowCompactor(size=2)
        ctx = ChatContext(compactor=comp)
        ctx = ctx.add(_msg(0)).add(_msg(1)).add(_msg(2))
        # subsequent adds keep using the same compactor
        ctx = ctx.add(_msg(3))
        assert ctx._compactor is comp
        assert len(ctx.as_list()) == 2

    def test_view_for_generation_no_double_truncation(self):
        ctx = ChatContext(window_size=3)
        for i in range(7):
            ctx = ctx.add(_msg(i))
        # add() already compacted; view should match the linear history exactly
        view = ctx.view_for_generation()
        assert view is not None
        assert [m.content for m in view] == [m.content for m in ctx.as_list()]

    def test_negative_size_raises(self):
        with pytest.raises(ValueError):
            WindowCompactor(size=-1)

    def test_size_zero_clears_body(self):
        # Regression: `[-0:]` evaluates to `[0:]` in Python, which would keep
        # the entire body instead of nothing. size=0 must keep zero body items.
        ctx = ChatContext(window_size=10_000)
        for i in range(5):
            ctx = ctx.add(_msg(i))
        result = WindowCompactor(size=0).compact(ctx)
        assert result.as_list() == []

    def test_size_zero_keeps_pinned_prefix(self):
        ctx = ChatContext(window_size=10_000)
        ctx = ctx.add(Message(role="system", content="sys"))
        for i in range(3):
            ctx = ctx.add(_msg(i))
        # Default pin_predicate=pin_system → system stays, body cleared.
        result = WindowCompactor(size=0).compact(ctx)
        items = result.as_list()
        assert len(items) == 1
        assert items[0].content == "sys"

    def test_pins_leading_system_message(self):
        ctx = ChatContext(window_size=10_000)
        ctx = ctx.add(Message(role="system", content="You are helpful."))
        for i in range(5):
            ctx = ctx.add(_msg(i))
        # Apply WindowCompactor(size=2) manually — keep system + last 2 body.
        result = WindowCompactor(size=2).compact(ctx)
        items = result.as_list()
        assert len(items) == 3
        assert isinstance(items[0], Message) and items[0].role == "system"
        assert [m.content for m in items[1:]] == ["m3", "m4"]

    def test_pins_multiple_leading_system_messages(self):
        ctx = ChatContext(window_size=10_000)
        ctx = ctx.add(Message(role="system", content="sys1"))
        ctx = ctx.add(Message(role="system", content="sys2"))
        for i in range(5):
            ctx = ctx.add(_msg(i))
        result = WindowCompactor(size=2).compact(ctx)
        items = result.as_list()
        assert [m.content for m in items[:2]] == ["sys1", "sys2"]
        assert [m.content for m in items[2:]] == ["m3", "m4"]

    def test_does_not_pin_non_contiguous_system(self):
        # System message in the middle is NOT pinned — only the contiguous prefix.
        ctx = ChatContext(window_size=10_000)
        ctx = ctx.add(_msg(0))  # body starts here
        ctx = ctx.add(Message(role="system", content="late-sys"))
        for i in range(1, 6):
            ctx = ctx.add(_msg(i))
        result = WindowCompactor(size=2).compact(ctx)
        items = result.as_list()
        assert len(items) == 2
        assert "late-sys" not in [getattr(m, "content", None) for m in items]

    def test_no_system_message_pure_last_n(self):
        # Without any system prefix, behaviour is pure last-N (matches Phase 2 semantics).
        ctx = ChatContext(window_size=10_000)
        for i in range(7):
            ctx = ctx.add(_msg(i))
        result = WindowCompactor(size=3).compact(ctx)
        items = result.as_list()
        assert [m.content for m in items] == ["m4", "m5", "m6"]


class TestCompactorProtocol:
    def test_user_class_satisfies_protocol(self):
        """A plain class with the right method should be a Compactor."""

        class Identity:
            def compact(self, ctx, *, backend=None):
                return ctx

        # structural subtyping check — at runtime this is just isinstance against Protocol
        # which requires `runtime_checkable` to actually work; instead assert duck-typing.
        c = Identity()
        ctx = ChatContext(compactor=c)
        ctx = ctx.add(_msg(0))
        # Identity returns ctx unchanged, so we still see m0
        assert [m.content for m in ctx.as_list()] == ["m0"]

    def test_pattern_2_manual_compaction(self):
        """Pattern 2: caller invokes compactor.compact() directly."""
        comp = WindowCompactor(size=2)
        # context with no auto-compaction would be tricky to construct under the
        # new defaults; instead use a window large enough that auto-compaction
        # never fires, then apply comp manually.
        ctx = ChatContext(window_size=100)
        for i in range(5):
            ctx = ctx.add(_msg(i))
        assert len(ctx.as_list()) == 5
        ctx2 = comp.compact(ctx)
        assert len(ctx2.as_list()) == 2
        # original still untouched
        assert len(ctx.as_list()) == 5


class TestLastUsageTokens:
    def test_no_thunk_returns_none(self):
        ctx = ChatContext(window_size=100).add(_msg(0))
        assert _last_usage_tokens(ctx) is None

    def test_thunk_without_usage_returns_none(self):
        ctx = ChatContext(window_size=100).add(_msg(0)).add(ModelOutputThunk(value="x"))
        assert _last_usage_tokens(ctx) is None

    def test_reads_total_tokens(self):
        ctx = ChatContext(window_size=100).add(_msg(0)).add(_thunk(150))
        assert _last_usage_tokens(ctx) == 150

    def test_falls_back_to_prompt_plus_completion(self):
        mot = ModelOutputThunk(value="x")
        mot.generation.usage = {"prompt_tokens": 40, "completion_tokens": 20}
        ctx = ChatContext(window_size=100).add(_msg(0)).add(mot)
        assert _last_usage_tokens(ctx) == 60

    def test_uses_most_recent_thunk(self):
        ctx = (
            ChatContext(window_size=100).add(_thunk(100)).add(_msg(0)).add(_thunk(500))
        )
        assert _last_usage_tokens(ctx) == 500


class TestThresholdCompactor:
    def test_below_threshold_returns_input(self):
        inner = WindowCompactor(size=2)
        gated = ThresholdCompactor(inner, threshold=1000)
        ctx = ChatContext(window_size=100).add(_msg(0)).add(_thunk(50))
        # 5 components but inner not invoked because token count (50) <= threshold (1000)
        for i in range(1, 6):
            ctx = ctx.add(_msg(i))
        result = gated.compact(ctx)
        assert result is ctx

    def test_above_threshold_runs_inner(self):
        inner = WindowCompactor(size=2)
        gated = ThresholdCompactor(inner, threshold=100)
        # Build a context with the last thunk reporting >threshold tokens.
        ctx = ChatContext(window_size=100)
        for i in range(5):
            ctx = ctx.add(_msg(i))
        ctx = ctx.add(_thunk(500))
        result = gated.compact(ctx)
        # Inner was invoked → only last 2 components retained.
        assert len(result.as_list()) == 2

    def test_no_thunk_no_compaction(self):
        """No thunk means no usage info — gate stays closed."""
        inner = WindowCompactor(size=2)
        gated = ThresholdCompactor(inner, threshold=100)
        ctx = ChatContext(window_size=100)
        for i in range(5):
            ctx = ctx.add(_msg(i))
        result = gated.compact(ctx)
        assert result is ctx

    def test_zero_threshold_disables_gate(self):
        inner = WindowCompactor(size=2)
        gated = ThresholdCompactor(inner, threshold=0)
        ctx = ChatContext(window_size=100).add(_msg(0)).add(_thunk(10_000))
        result = gated.compact(ctx)
        # Threshold 0 means "never trigger" — input passes through.
        assert result is ctx


class TestPinPredicates:
    def test_pin_nothing(self):
        assert pin_nothing([_msg(0), _msg(1)]) == 0
        assert pin_nothing([]) == 0

    def test_pin_system_zero_when_no_system(self):
        assert pin_system([_msg(0), _msg(1)]) == 0

    def test_pin_system_counts_contiguous(self):
        components = [
            Message(role="system", content="s1"),
            Message(role="system", content="s2"),
            _msg(0),
            Message(role="system", content="late-s"),  # not pinned — non-contiguous
        ]
        assert pin_system(components) == 2

    def test_pin_system_and_initial_user_with_both(self):
        components = [
            Message(role="system", content="s1"),
            Message(role="user", content="goal"),
            Message(role="assistant", content="ack"),
        ]
        assert pin_system_and_initial_user(components) == 2

    def test_pin_system_and_initial_user_no_user(self):
        components = [
            Message(role="system", content="s1"),
            Message(role="assistant", content="x"),
        ]
        # First non-system is "assistant", not "user" — not pinned beyond system.
        assert pin_system_and_initial_user(components) == 1

    def test_pin_system_and_initial_user_user_only(self):
        components = [
            Message(role="user", content="goal"),
            Message(role="assistant", content="ok"),
        ]
        assert pin_system_and_initial_user(components) == 1


class TestWindowCompactorPredicate:
    def test_pin_nothing_pure_last_n(self):
        comp = WindowCompactor(size=2, pin_predicate=pin_nothing)
        ctx = ChatContext(window_size=10_000)
        ctx = ctx.add(Message(role="system", content="sys"))
        for i in range(5):
            ctx = ctx.add(_msg(i))
        result = comp.compact(ctx)
        items = result.as_list()
        assert len(items) == 2
        # System is dropped because predicate returned 0.
        assert "sys" not in [getattr(m, "content", None) for m in items]

    def test_pin_system_and_initial_user_protects_first_user(self):
        comp = WindowCompactor(size=2, pin_predicate=pin_system_and_initial_user)
        ctx = ChatContext(window_size=10_000)
        ctx = ctx.add(Message(role="system", content="sys"))
        ctx = ctx.add(Message(role="user", content="goal"))
        for i in range(6):
            ctx = ctx.add(_msg(i))
        result = comp.compact(ctx)
        items = result.as_list()
        # prefix (sys + goal) + last 2 body = 4
        assert len(items) == 4
        assert items[0].content == "sys"
        assert items[1].content == "goal"

    def test_custom_predicate(self):
        # Predicate that pins the first 3 components unconditionally.
        def pin_first_3(components):
            return min(3, len(components))

        comp = WindowCompactor(size=2, pin_predicate=pin_first_3)
        ctx = ChatContext(window_size=10_000)
        for i in range(8):
            ctx = ctx.add(_msg(i))
        result = comp.compact(ctx)
        items = result.as_list()
        # prefix (m0, m1, m2) + last 2 of body (m6, m7) = 5
        assert [m.content for m in items] == ["m0", "m1", "m2", "m6", "m7"]


# --------------------------------------------------------------------------- #
# LLMSummarizeCompactor                                                       #
# --------------------------------------------------------------------------- #


@pytest.fixture
def scripted_summary_backend():
    """Lazy-built fake backend that returns a fixed summary on each generate call."""
    from collections.abc import Sequence

    from mellea.core.backend import Backend, BaseModelSubclass
    from mellea.core.base import C, GenerateLog

    class FakeBackend(Backend):
        def __init__(self, summary: str = "SUMMARY-OF-OLD") -> None:
            self.summary = summary
            self.calls = 0

        async def _generate_from_context(
            self,
            action,
            ctx,
            *,
            format=None,
            model_options=None,
            tool_calls: bool = False,
        ):
            self.calls += 1
            mot = ModelOutputThunk(value=self.summary)
            mot._generate_log = GenerateLog(is_final_result=True)
            return mot, ctx.add(action).add(mot)

        async def generate_from_raw(
            self,
            actions,
            ctx,
            *,
            format=None,
            model_options=None,
            tool_calls: bool = False,
        ):
            raise NotImplementedError

    return FakeBackend()


class TestLLMSummarizeCompactor:
    def test_negative_keep_n_raises(self):
        with pytest.raises(ValueError):
            LLMSummarizeCompactor(keep_n=-1)

    def test_prompt_template_must_have_placeholder(self):
        with pytest.raises(ValueError, match="conversation"):
            LLMSummarizeCompactor(prompt_template="no placeholder here")

    def test_compact_is_sync(self):
        import inspect

        comp = LLMSummarizeCompactor()
        # Sync from the outside even though the implementation calls async backend code.
        assert not inspect.iscoroutinefunction(comp.compact)

    def test_raises_without_backend(self):
        comp = LLMSummarizeCompactor()
        ctx = ChatContext(window_size=10_000)
        for i in range(3):
            ctx = ctx.add(_msg(i))
        with pytest.raises(ValueError, match="backend"):
            comp.compact(ctx)

    def test_short_body_is_noop(self, scripted_summary_backend):
        comp = LLMSummarizeCompactor(keep_n=5)
        ctx = ChatContext(window_size=10_000)
        for i in range(3):
            ctx = ctx.add(_msg(i))
        result = comp.compact(ctx, backend=scripted_summary_backend)
        # body length (3) <= keep_n (5) → no-op, backend not called
        assert result is ctx
        assert scripted_summary_backend.calls == 0

    def test_summarises_old_keeps_recent(self, scripted_summary_backend):
        comp = LLMSummarizeCompactor(keep_n=2)
        ctx = ChatContext(window_size=10_000)
        for i in range(6):
            ctx = ctx.add(_msg(i))
        result = comp.compact(ctx, backend=scripted_summary_backend)
        items = result.as_list()
        # summary (1) + last 2 verbatim = 3
        assert len(items) == 3
        assert "[CONTEXT SUMMARY]" in items[0].content
        assert items[1].content == "m4"
        assert items[2].content == "m5"
        assert scripted_summary_backend.calls == 1

    def test_pin_predicate_preserves_prefix(self, scripted_summary_backend):
        comp = LLMSummarizeCompactor(keep_n=1, pin_predicate=pin_system)
        ctx = ChatContext(window_size=10_000)
        ctx = ctx.add(Message(role="system", content="sys"))
        for i in range(4):
            ctx = ctx.add(_msg(i))
        result = comp.compact(ctx, backend=scripted_summary_backend)
        items = result.as_list()
        # system (pinned) + summary + last 1 verbatim = 3
        assert items[0].role == "system"
        assert items[0].content == "sys"
        assert "[CONTEXT SUMMARY]" in items[1].content
        assert items[2].content == "m3"

    def test_does_not_mutate_original(self, scripted_summary_backend):
        comp = LLMSummarizeCompactor(keep_n=1)
        ctx = ChatContext(window_size=10_000)
        for i in range(4):
            ctx = ctx.add(_msg(i))
        before = [m.content for m in ctx.as_list()]
        comp.compact(ctx, backend=scripted_summary_backend)
        assert [m.content for m in ctx.as_list()] == before

    def test_satisfies_compactor_protocol(self):
        comp: Compactor = LLMSummarizeCompactor()
        # Just a typing-level check that the assignment is accepted.
        assert callable(comp.compact)

    @pytest.mark.asyncio
    async def test_works_inside_running_event_loop(self, scripted_summary_backend):
        """compact() is callable from within an async function — uses worker thread."""
        comp = LLMSummarizeCompactor(keep_n=1)
        ctx = ChatContext(window_size=10_000)
        for i in range(4):
            ctx = ctx.add(_msg(i))
        # No await: this is a sync call from inside an async test.
        result = comp.compact(ctx, backend=scripted_summary_backend)
        items = result.as_list()
        assert "[CONTEXT SUMMARY]" in items[0].content
        assert items[1].content == "m3"
