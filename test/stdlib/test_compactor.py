# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

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


class TestInlineCompactorGuard:
    """ChatContext only accepts InlineCompactor instances."""

    def test_rejects_llm_summarize_compactor_directly(self, scripted_summary_backend):
        # Attaching LLMSummarizeCompactor would invoke the backend on every add().
        comp = LLMSummarizeCompactor(default_backend=scripted_summary_backend)
        with pytest.raises(TypeError, match="requires an InlineCompactor"):
            ChatContext(compactor=comp)

    def test_accepts_threshold_wrapping_window(self):
        # ThresholdCompactor is an InlineCompactor regardless of inner.
        wrapped = ThresholdCompactor(WindowCompactor(size=5), threshold=1000)
        ctx = ChatContext(compactor=wrapped)
        assert ctx._compactor is wrapped

    def test_accepts_threshold_wrapping_llm_summarize(self, scripted_summary_backend):
        # Wrapped is acceptable: ThresholdCompactor gates inner by token usage,
        # so backend isn't called on every add(). Inner's default_backend covers
        # the actual summarisation when the gate trips.
        wrapped = ThresholdCompactor(
            LLMSummarizeCompactor(default_backend=scripted_summary_backend, keep_n=2),
            threshold=1000,
        )
        ctx = ChatContext(compactor=wrapped)
        assert ctx._compactor is wrapped

    def test_accepts_window_compactor(self):
        comp = WindowCompactor(size=5)
        ctx = ChatContext(compactor=comp)
        assert ctx._compactor is comp

    def test_rejects_non_inline_duck_typed_compactor(self):
        class FakeCompactor:
            def compact(self, ctx, *, backend=None):
                return ctx

        with pytest.raises(TypeError, match="requires an InlineCompactor"):
            ChatContext(compactor=FakeCompactor())  # type: ignore[arg-type]


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
    def test_user_class_satisfies_protocol_via_inline_marker(self):
        """A user class structurally matching Compactor and inheriting InlineCompactor
        is accepted by ChatContext."""
        from mellea.stdlib.context import InlineCompactor

        class Identity(InlineCompactor):
            def compact(self, ctx, *, backend=None):
                return ctx

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
            self.last_action_content: str | None = None
            self.last_model_options: dict | None = None

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
            self.last_action_content = getattr(action, "content", str(action))
            self.last_model_options = model_options
            mot = ModelOutputThunk(value=self.summary)
            mot._generate_log = GenerateLog(is_final_result=True)
            return mot, ctx.add(action).add(mot)

        async def _generate_from_raw(
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
    def test_negative_keep_n_raises(self, scripted_summary_backend):
        with pytest.raises(ValueError):
            LLMSummarizeCompactor(default_backend=scripted_summary_backend, keep_n=-1)

    def test_prompt_template_must_have_placeholder(self, scripted_summary_backend):
        with pytest.raises(ValueError, match="conversation"):
            LLMSummarizeCompactor(
                default_backend=scripted_summary_backend,
                prompt_template="no placeholder here",
            )

    def test_default_backend_is_required(self):
        with pytest.raises(TypeError, match="default_backend"):
            LLMSummarizeCompactor()  # type: ignore[call-arg]

    def test_compact_is_sync(self, scripted_summary_backend):
        import inspect

        comp = LLMSummarizeCompactor(default_backend=scripted_summary_backend)
        # Sync from the outside even though the implementation calls async backend code.
        assert not inspect.iscoroutinefunction(comp.compact)

    def test_uses_default_backend_when_call_omits_one(self, scripted_summary_backend):
        comp = LLMSummarizeCompactor(default_backend=scripted_summary_backend, keep_n=1)
        ctx = ChatContext(window_size=10_000)
        for i in range(4):
            ctx = ctx.add(_msg(i))
        # No backend kwarg → falls back to default_backend.
        result = comp.compact(ctx)
        items = result.as_list()
        assert "[CONTEXT SUMMARY]" in items[0].content
        assert scripted_summary_backend.calls == 1

    def test_call_time_backend_overrides_default(self, scripted_summary_backend):
        from mellea.core.backend import Backend
        from mellea.core.base import GenerateLog

        class OtherBackend(Backend):
            def __init__(self) -> None:
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
                mot = ModelOutputThunk(value="OTHER-SUMMARY")
                mot._generate_log = GenerateLog(is_final_result=True)
                return mot, ctx.add(action).add(mot)

            async def _generate_from_raw(self, *a, **kw):
                raise NotImplementedError

        other = OtherBackend()
        comp = LLMSummarizeCompactor(default_backend=scripted_summary_backend, keep_n=1)
        ctx = ChatContext(window_size=10_000)
        for i in range(4):
            ctx = ctx.add(_msg(i))
        result = comp.compact(ctx, backend=other)
        items = result.as_list()
        # Caller-supplied backend wins.
        assert "OTHER-SUMMARY" in items[0].content
        assert other.calls == 1
        assert scripted_summary_backend.calls == 0

    def test_short_body_is_noop(self, scripted_summary_backend):
        comp = LLMSummarizeCompactor(default_backend=scripted_summary_backend, keep_n=5)
        ctx = ChatContext(window_size=10_000)
        for i in range(3):
            ctx = ctx.add(_msg(i))
        result = comp.compact(ctx, backend=scripted_summary_backend)
        # body length (3) <= keep_n (5) → no-op, backend not called
        assert result is ctx
        assert scripted_summary_backend.calls == 0

    def test_backend_failure_returns_ctx_unchanged_and_logs(
        self, scripted_summary_backend, caplog
    ):
        """Compaction is best-effort: backend errors must not propagate."""
        import logging

        from mellea.core.backend import Backend
        from mellea.core.base import GenerateLog

        class BrokenBackend(Backend):
            async def _generate_from_context(
                self,
                action,
                ctx,
                *,
                format=None,
                model_options=None,
                tool_calls: bool = False,
            ):
                raise RuntimeError("simulated rate limit")

            async def _generate_from_raw(self, *a, **kw):
                raise NotImplementedError

        comp = LLMSummarizeCompactor(default_backend=BrokenBackend(), keep_n=1)
        ctx = ChatContext(window_size=10_000)
        for i in range(4):
            ctx = ctx.add(_msg(i))

        with caplog.at_level(logging.WARNING):
            result = comp.compact(ctx)

        # ctx returned unchanged — same object, original history intact.
        assert result is ctx
        assert [m.content for m in result.as_list()] == ["m0", "m1", "m2", "m3"]
        # Warning logged with context for debugging.
        assert any(
            "summarisation backend call failed" in rec.message
            and "RuntimeError" in rec.message
            for rec in caplog.records
        )

    def test_programming_errors_propagate(self):
        """Bugs (TypeError/AttributeError/etc.) must not be swallowed as 'backend failure'."""
        from mellea.core.backend import Backend

        class BuggyBackend(Backend):
            async def _generate_from_context(
                self,
                action,
                ctx,
                *,
                format=None,
                model_options=None,
                tool_calls: bool = False,
            ):
                raise TypeError("simulated programming bug")

            async def _generate_from_raw(self, *a, **kw):
                raise NotImplementedError

        comp = LLMSummarizeCompactor(default_backend=BuggyBackend(), keep_n=1)
        ctx = ChatContext(window_size=10_000)
        for i in range(4):
            ctx = ctx.add(_msg(i))

        with pytest.raises(TypeError, match="simulated programming bug"):
            comp.compact(ctx)

    def test_base_exceptions_propagate(self):
        """KeyboardInterrupt and other BaseExceptions must not be caught.

        The narrow re-raise list and the broad `except Exception` both miss
        BaseException subclasses by design — guards Ctrl-C and async
        cancellation from being silently swallowed across the
        _run_coro_blocking thread bridge.
        """
        from mellea.core.backend import Backend

        class InterruptingBackend(Backend):
            async def _generate_from_context(
                self,
                action,
                ctx,
                *,
                format=None,
                model_options=None,
                tool_calls: bool = False,
            ):
                raise KeyboardInterrupt("simulated Ctrl-C")

            async def _generate_from_raw(self, *a, **kw):
                raise NotImplementedError

        comp = LLMSummarizeCompactor(default_backend=InterruptingBackend(), keep_n=1)
        ctx = ChatContext(window_size=10_000)
        for i in range(4):
            ctx = ctx.add(_msg(i))

        with pytest.raises(KeyboardInterrupt, match="simulated Ctrl-C"):
            comp.compact(ctx)

    def test_renders_thunk_without_value_using_tool_calls(
        self, scripted_summary_backend
    ):
        """Tool-call-only thunks (value=None) render the call name + args, not 'None'."""
        from mellea.core.base import ModelToolCall

        # The compactor's rendering only reads ``name``/``args`` off the
        # ModelToolCall, never invokes ``func`` — pass None to skip
        # AbstractMelleaTool's abstract-method requirements.
        tool_call = ModelToolCall(
            name="search",
            func=None,  # type: ignore[arg-type]
            args={"q": "papers"},
        )
        thunk = ModelOutputThunk(value=None, tool_calls=[tool_call])

        comp = LLMSummarizeCompactor(default_backend=scripted_summary_backend, keep_n=1)
        ctx = ChatContext(window_size=10_000)
        for i in range(2):
            ctx = ctx.add(_msg(i))
        ctx = ctx.add(thunk)
        ctx = ctx.add(_msg(2))  # so the thunk falls into `old`, not `recent`

        comp.compact(ctx)
        rendered = scripted_summary_backend.last_action_content
        assert rendered is not None
        assert "assistant called tools: search" in rendered
        assert "'q': 'papers'" in rendered
        # Old "assistant: None" failure mode must not appear.
        assert "assistant: None" not in rendered

    def test_renders_thunk_with_no_value_and_no_tool_calls(
        self, scripted_summary_backend
    ):
        """A thunk with neither value nor tool_calls is skipped entirely — no
        '<empty>' marker, no 'assistant: None'."""
        thunk = ModelOutputThunk(value=None)

        comp = LLMSummarizeCompactor(default_backend=scripted_summary_backend, keep_n=1)
        ctx = ChatContext(window_size=10_000)
        for i in range(2):
            ctx = ctx.add(_msg(i))
        ctx = ctx.add(thunk)
        ctx = ctx.add(_msg(2))

        comp.compact(ctx)
        rendered = scripted_summary_backend.last_action_content
        assert rendered is not None
        assert "<empty>" not in rendered
        assert "assistant: None" not in rendered
        # The other turns still made it into the prompt.
        assert "user: m0" in rendered
        assert "user: m1" in rendered

    def test_catchall_renders_unknown_component_as_typed_marker(
        self, scripted_summary_backend
    ):
        """Component subclasses that aren't Message/ToolMessage/ModelOutputThunk
        emit a ``<TypeName[: content]>`` marker instead of the default object repr."""
        from mellea.core import Component

        class _CustomMarker(Component):
            """Component without a ``content`` attribute."""

            def parts(self):  # type: ignore[override]
                return []

            def format_for_llm(self):  # type: ignore[override]
                return ""

        comp = LLMSummarizeCompactor(default_backend=scripted_summary_backend, keep_n=1)
        ctx = ChatContext(window_size=10_000)
        ctx = ctx.add(_CustomMarker())  # in `old`
        ctx = ctx.add(_msg(0))
        ctx = ctx.add(_msg(99))  # in `recent`

        comp.compact(ctx)
        rendered = scripted_summary_backend.last_action_content
        assert rendered is not None
        # Type name appears explicitly; raw <object at 0x...> repr does NOT.
        assert "<_CustomMarker>" in rendered
        assert "object at 0x" not in rendered

    def test_renders_message_with_attachments_as_markers(
        self, scripted_summary_backend
    ):
        """Image/document attachments are noted by count; their contents are not reproduced."""
        from mellea.stdlib.components.docs.document import Document

        msg_with_imgs = Message(role="user", content="see these")
        # Bypass the constructor to inject raw lists; the rendering path reads `_images`/`_docs`.
        msg_with_imgs._images = ["IMGDATA1", "IMGDATA2"]  # type: ignore[assignment]
        msg_with_docs = Message(role="user", content="and these")
        msg_with_docs._docs = [Document(text="doc body")]  # type: ignore[assignment]

        comp = LLMSummarizeCompactor(default_backend=scripted_summary_backend, keep_n=1)
        ctx = ChatContext(window_size=10_000)
        ctx = ctx.add(msg_with_imgs)
        ctx = ctx.add(msg_with_docs)
        ctx = ctx.add(_msg(99))  # keeps msg_with_imgs/docs in `old`, this in `recent`

        comp.compact(ctx)
        rendered = scripted_summary_backend.last_action_content
        assert rendered is not None
        assert "[2 image(s) attached]" in rendered
        assert "[1 document(s) attached]" in rendered
        # Image bytes are NOT in the rendered prompt.
        assert "IMGDATA1" not in rendered

    def test_model_options_forwarded_to_backend(self, scripted_summary_backend):
        """model_options set at construction reach the backend's generate call."""
        comp = LLMSummarizeCompactor(
            default_backend=scripted_summary_backend,
            keep_n=1,
            model_options={"max_tokens": 4096, "temperature": 0.0},
        )
        ctx = ChatContext(window_size=10_000)
        for i in range(4):
            ctx = ctx.add(_msg(i))
        comp.compact(ctx)
        assert scripted_summary_backend.last_model_options == {
            "max_tokens": 4096,
            "temperature": 0.0,
        }

    def test_model_options_default_is_empty(self, scripted_summary_backend):
        """When model_options is not set, the backend receives no caller-supplied
        options (falsy: None or {}); upstream defaults govern."""
        comp = LLMSummarizeCompactor(default_backend=scripted_summary_backend, keep_n=1)
        ctx = ChatContext(window_size=10_000)
        for i in range(4):
            ctx = ctx.add(_msg(i))
        comp.compact(ctx)
        assert not scripted_summary_backend.last_model_options

    def test_summarises_old_keeps_recent(self, scripted_summary_backend):
        comp = LLMSummarizeCompactor(default_backend=scripted_summary_backend, keep_n=2)
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
        comp = LLMSummarizeCompactor(
            default_backend=scripted_summary_backend, keep_n=1, pin_predicate=pin_system
        )
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
        comp = LLMSummarizeCompactor(default_backend=scripted_summary_backend, keep_n=1)
        ctx = ChatContext(window_size=10_000)
        for i in range(4):
            ctx = ctx.add(_msg(i))
        before = [m.content for m in ctx.as_list()]
        comp.compact(ctx, backend=scripted_summary_backend)
        assert [m.content for m in ctx.as_list()] == before

    def test_satisfies_compactor_protocol(self, scripted_summary_backend):
        comp: Compactor = LLMSummarizeCompactor(
            default_backend=scripted_summary_backend
        )
        # Just a typing-level check that the assignment is accepted.
        assert callable(comp.compact)

    @pytest.mark.asyncio
    async def test_works_inside_running_event_loop(self, scripted_summary_backend):
        """compact() is callable from within an async function — uses worker thread."""
        comp = LLMSummarizeCompactor(default_backend=scripted_summary_backend, keep_n=1)
        ctx = ChatContext(window_size=10_000)
        for i in range(4):
            ctx = ctx.add(_msg(i))
        # No await: this is a sync call from inside an async test.
        result = comp.compact(ctx, backend=scripted_summary_backend)
        items = result.as_list()
        assert "[CONTEXT SUMMARY]" in items[0].content
        assert items[1].content == "m3"
