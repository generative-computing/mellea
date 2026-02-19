"""Tests documenting memory retention risks from live object references in payloads.

Mellea hook payloads store live Python object references (Backend, Context,
Component, ModelOutputThunk) rather than serialisable copies.  This is
intentional — in-process plugins need direct access.  The risk arises when
a plugin retains a payload beyond the hook call, preventing garbage collection.

These tests document which payload fields hold live references (not copies).
Each assertion that passes confirms that retaining the payload also retains
the referenced object.

See: docs/dev/hook_system_bugs.md §A5
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock
import gc

import pytest

pytest.importorskip("mcpgateway.plugins.framework")


# ---------------------------------------------------------------------------
# Helper: verify same-object identity
# ---------------------------------------------------------------------------


def _same_object(a: Any, b: Any) -> bool:
    """Return True if a and b are the exact same Python object."""
    return a is b


# ---------------------------------------------------------------------------
# Session payloads
# ---------------------------------------------------------------------------


class TestSessionPayloadLiveRefs:
    """Session payloads hold live references to session/context objects."""

    def test_session_post_init_holds_live_session(self):
        """SessionPostInitPayload.session IS the same object passed in."""
        from mellea.plugins.hooks.session import SessionPostInitPayload

        mock_session = MagicMock()
        p = SessionPostInitPayload(session=mock_session)
        assert _same_object(p.session, mock_session), (
            "A5: SessionPostInitPayload.session holds a live reference. "
            "Retaining the payload retains the entire MelleaSession."
        )

    def test_session_reset_holds_live_context(self):
        """SessionResetPayload.previous_context IS the same object passed in."""
        from mellea.plugins.hooks.session import SessionResetPayload

        mock_ctx = MagicMock()
        p = SessionResetPayload(previous_context=mock_ctx)
        assert _same_object(p.previous_context, mock_ctx), (
            "A5: SessionResetPayload.previous_context holds a live reference."
        )

    def test_session_cleanup_holds_live_context(self):
        """SessionCleanupPayload.context IS the same object passed in."""
        from mellea.plugins.hooks.session import SessionCleanupPayload

        mock_ctx = MagicMock()
        p = SessionCleanupPayload(context=mock_ctx)
        assert _same_object(p.context, mock_ctx)


# ---------------------------------------------------------------------------
# Component payloads
# ---------------------------------------------------------------------------


class TestComponentPayloadLiveRefs:
    """Component payloads hold live references to components and contexts."""

    def test_post_success_holds_live_action(self):
        """ComponentPostSuccessPayload.action holds a live reference to the component."""
        from mellea.plugins.hooks.component import ComponentPostSuccessPayload

        mock_action = MagicMock()
        p = ComponentPostSuccessPayload(action=mock_action)
        assert _same_object(p.action, mock_action)

    def test_post_success_holds_live_result(self):
        """ComponentPostSuccessPayload.result holds a live reference to ModelOutputThunk."""
        from mellea.plugins.hooks.component import ComponentPostSuccessPayload

        mock_result = MagicMock()
        p = ComponentPostSuccessPayload(result=mock_result)
        assert _same_object(p.result, mock_result)

    def test_post_success_holds_two_live_contexts(self):
        """ComponentPostSuccessPayload holds BOTH context_before AND context_after live.

        Risk: retaining this payload retains two full conversation history snapshots.
        """
        from mellea.plugins.hooks.component import ComponentPostSuccessPayload

        mock_ctx_before = MagicMock(name="ctx_before")
        mock_ctx_after = MagicMock(name="ctx_after")
        p = ComponentPostSuccessPayload(
            context_before=mock_ctx_before, context_after=mock_ctx_after
        )
        assert _same_object(p.context_before, mock_ctx_before)
        assert _same_object(p.context_after, mock_ctx_after)
        # These are two distinct objects
        assert p.context_before is not p.context_after

    def test_post_success_holds_live_generate_log(self):
        """ComponentPostSuccessPayload.generate_log holds a live reference."""
        from mellea.plugins.hooks.component import ComponentPostSuccessPayload

        mock_log = MagicMock()
        p = ComponentPostSuccessPayload(generate_log=mock_log)
        assert _same_object(p.generate_log, mock_log)

    def test_pre_execute_holds_live_context(self):
        """ComponentPreExecutePayload.context holds a live reference."""
        from mellea.plugins.hooks.component import ComponentPreExecutePayload

        mock_ctx = MagicMock()
        p = ComponentPreExecutePayload(context=mock_ctx)
        assert _same_object(p.context, mock_ctx)


# ---------------------------------------------------------------------------
# Sampling payloads (highest risk — all_results can be many ModelOutputThunks)
# ---------------------------------------------------------------------------


class TestSamplingPayloadLiveRefs:
    """Sampling payloads carry the heaviest object graphs.

    SamplingLoopEndPayload.all_results holds ALL intermediate ModelOutputThunks
    from the entire sampling loop (up to loop_budget entries).  Retaining this
    payload prevents GC of all intermediate generation results.
    """

    def test_loop_end_all_results_holds_live_list(self):
        """SamplingLoopEndPayload.all_results IS the same list passed in (not a copy).

        Risk: the list can contain up to loop_budget ModelOutputThunk instances.
        """
        from mellea.plugins.hooks.sampling import SamplingLoopEndPayload

        mock_result_1 = MagicMock(name="mot_1")
        mock_result_2 = MagicMock(name="mot_2")
        mock_result_3 = MagicMock(name="mot_3")
        results = [mock_result_1, mock_result_2, mock_result_3]

        p = SamplingLoopEndPayload(all_results=results)
        assert len(p.all_results) == 3
        assert _same_object(p.all_results[0], mock_result_1)
        assert _same_object(p.all_results[1], mock_result_2)
        assert _same_object(p.all_results[2], mock_result_3)

    def test_loop_end_final_result_holds_live_thunk(self):
        """SamplingLoopEndPayload.final_result holds a live ModelOutputThunk."""
        from mellea.plugins.hooks.sampling import SamplingLoopEndPayload

        mock_result = MagicMock()
        p = SamplingLoopEndPayload(final_result=mock_result)
        assert _same_object(p.final_result, mock_result)

    def test_loop_end_final_context_holds_live_context(self):
        """SamplingLoopEndPayload.final_context holds a live Context."""
        from mellea.plugins.hooks.sampling import SamplingLoopEndPayload

        mock_ctx = MagicMock()
        p = SamplingLoopEndPayload(final_context=mock_ctx)
        assert _same_object(p.final_context, mock_ctx)

    def test_repair_payload_holds_live_action_and_context(self):
        """SamplingRepairPayload holds live references to both repair_action and repair_context."""
        from mellea.plugins.hooks.sampling import SamplingRepairPayload

        mock_action = MagicMock(name="repair_action")
        mock_ctx = MagicMock(name="repair_context")
        p = SamplingRepairPayload(repair_action=mock_action, repair_context=mock_ctx)
        assert _same_object(p.repair_action, mock_action)
        assert _same_object(p.repair_context, mock_ctx)

    def test_loop_end_all_validations_holds_live_list(self):
        """SamplingLoopEndPayload.all_validations holds a live nested list."""
        from mellea.plugins.hooks.sampling import SamplingLoopEndPayload

        mock_req = MagicMock(name="requirement")
        mock_val = MagicMock(name="validation_result")
        validations = [[(mock_req, mock_val)]]

        p = SamplingLoopEndPayload(all_validations=validations)
        assert len(p.all_validations) == 1
        req, val = p.all_validations[0][0]
        assert _same_object(req, mock_req)
        assert _same_object(val, mock_val)


# ---------------------------------------------------------------------------
# Tool payloads
# ---------------------------------------------------------------------------


class TestToolPayloadLiveRefs:
    """Tool payloads hold live references to callables and tool outputs."""

    def test_pre_invoke_holds_live_callable(self):
        """ToolPreInvokePayload.tool_callable IS the actual function (not a copy).

        Risk: retaining this payload keeps the function (and its closure) alive.
        """
        from mellea.plugins.hooks.tool import ToolPreInvokePayload

        def my_tool(x: int) -> int:
            return x * 2

        p = ToolPreInvokePayload(tool_callable=my_tool)
        assert _same_object(p.tool_callable, my_tool)
        # The callable is still functional via the payload
        assert p.tool_callable(5) == 10

    def test_pre_invoke_holds_live_model_tool_call(self):
        """ToolPreInvokePayload.model_tool_call holds a live ModelToolCall reference."""
        from mellea.plugins.hooks.tool import ToolPreInvokePayload

        mock_call = MagicMock()
        p = ToolPreInvokePayload(model_tool_call=mock_call)
        assert _same_object(p.model_tool_call, mock_call)

    def test_post_invoke_holds_live_output(self):
        """ToolPostInvokePayload.tool_output holds whatever the tool returned."""
        from mellea.plugins.hooks.tool import ToolPostInvokePayload

        # Tool output is arbitrary — could be a large data structure
        large_output = {"data": list(range(1000)), "metadata": {"key": "value"}}
        p = ToolPostInvokePayload(tool_output=large_output)
        assert _same_object(p.tool_output, large_output)

    def test_post_invoke_holds_live_exception(self):
        """ToolPostInvokePayload.error holds a live Exception (with traceback)."""
        from mellea.plugins.hooks.tool import ToolPostInvokePayload

        err = ValueError("tool failed")
        p = ToolPostInvokePayload(error=err, success=False)
        assert _same_object(p.error, err)


# ---------------------------------------------------------------------------
# Memory retention: frozen model preserves references across model_copy
# ---------------------------------------------------------------------------


class TestFrozenModelRetainsReferences:
    """model_copy(update={...}) does NOT deep-copy unchanged fields.

    This confirms that copying a payload to change one field still retains
    live references to all other fields.
    """

    def test_model_copy_does_not_deep_copy_unchanged_fields(self):
        """model_copy preserves identity (not a copy) of unchanged live-reference fields."""
        from mellea.plugins.hooks.component import ComponentPostSuccessPayload

        mock_action = MagicMock(name="action")
        mock_ctx_before = MagicMock(name="ctx_before")
        mock_ctx_after = MagicMock(name="ctx_after")

        original = ComponentPostSuccessPayload(
            action=mock_action,
            context_before=mock_ctx_before,
            context_after=mock_ctx_after,
            latency_ms=100,
        )

        # Copy the payload, changing only latency_ms
        copied = original.model_copy(update={"latency_ms": 200})

        # All live references are preserved (same objects, not copies)
        assert _same_object(copied.action, mock_action)
        assert _same_object(copied.context_before, mock_ctx_before)
        assert _same_object(copied.context_after, mock_ctx_after)
        assert copied.latency_ms == 200  # only this changed
