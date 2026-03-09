"""Tests verifying that WeakProxy fields prevent unintended memory retention (A5 fixed).

Bug A5 is resolved: framework-owned live objects (Context, Component, MelleaSession,
SamplingStrategy) are now stored as ``weakref.proxy()`` in payload fields typed as
``WeakProxy``.  Plugins that cache a payload cannot keep these objects alive beyond
the request lifecycle.

Result/data objects (model_output, result, tool_output, generate_log, etc.) remain
strong references — plugins may legitimately need to retain these.

See: docs/dev/hook_system_bugs.md §A5
"""

from __future__ import annotations

import gc
import weakref
from typing import Any
from unittest.mock import MagicMock

import pytest

pytest.importorskip("cpex.framework")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _is_weak_proxy(obj: Any) -> bool:
    """Return True if *obj* is a weakref.proxy wrapper."""
    return isinstance(obj, weakref.ProxyTypes)


# ---------------------------------------------------------------------------
# Session payloads — WeakProxy fields
# ---------------------------------------------------------------------------


class TestSessionPayloadWeakRefs:
    """Session payloads wrap live system objects in weakref.proxy."""

    def test_session_post_init_session_is_weak_proxy(self):
        from mellea.plugins.hooks.session import SessionPostInitPayload

        mock_session = MagicMock()
        p = SessionPostInitPayload(session=mock_session)
        assert _is_weak_proxy(p.session), "session field must be a weakref.proxy"

    def test_session_post_init_session_is_accessible(self):
        from mellea.plugins.hooks.session import SessionPostInitPayload

        mock_session = MagicMock()
        mock_session.alive = True
        p = SessionPostInitPayload(session=mock_session)
        assert p.session.alive is True

    def test_session_reset_previous_context_is_weak_proxy(self):
        from mellea.plugins.hooks.session import SessionResetPayload

        mock_ctx = MagicMock()
        p = SessionResetPayload(previous_context=mock_ctx)
        assert _is_weak_proxy(p.previous_context)

    def test_session_cleanup_context_is_weak_proxy(self):
        from mellea.plugins.hooks.session import SessionCleanupPayload

        mock_ctx = MagicMock()
        p = SessionCleanupPayload(context=mock_ctx)
        assert _is_weak_proxy(p.context)


# ---------------------------------------------------------------------------
# Component payloads — WeakProxy fields
# ---------------------------------------------------------------------------


class TestComponentPayloadWeakRefs:
    """Component payloads wrap live component and context objects in weakref.proxy."""

    def test_post_success_action_is_weak_proxy(self):
        from mellea.plugins.hooks.component import ComponentPostSuccessPayload

        mock_action = MagicMock()
        p = ComponentPostSuccessPayload(action=mock_action)
        assert _is_weak_proxy(p.action)

    def test_post_success_result_is_strong_ref(self):
        """result (ModelOutputThunk) is a strong reference — plugins may retain results."""
        from mellea.plugins.hooks.component import ComponentPostSuccessPayload

        mock_result = MagicMock()
        p = ComponentPostSuccessPayload(result=mock_result)
        assert p.result is mock_result
        assert not _is_weak_proxy(p.result)

    def test_post_success_contexts_are_weak_proxies(self):
        from mellea.plugins.hooks.component import ComponentPostSuccessPayload

        mock_ctx_before = MagicMock(name="ctx_before")
        mock_ctx_after = MagicMock(name="ctx_after")
        p = ComponentPostSuccessPayload(
            context_before=mock_ctx_before, context_after=mock_ctx_after
        )
        assert _is_weak_proxy(p.context_before)
        assert _is_weak_proxy(p.context_after)
        # Two distinct proxies to two distinct underlying objects
        assert p.context_before is not p.context_after

    def test_post_success_generate_log_is_strong_ref(self):
        """generate_log is a strong reference (log data, not a live system object)."""
        from mellea.plugins.hooks.component import ComponentPostSuccessPayload

        mock_log = MagicMock()
        p = ComponentPostSuccessPayload(generate_log=mock_log)
        assert p.generate_log is mock_log
        assert not _is_weak_proxy(p.generate_log)

    def test_pre_execute_context_is_weak_proxy(self):
        from mellea.plugins.hooks.component import ComponentPreExecutePayload

        mock_ctx = MagicMock()
        p = ComponentPreExecutePayload(context=mock_ctx)
        assert _is_weak_proxy(p.context)


# ---------------------------------------------------------------------------
# Sampling payloads — WeakProxy fields
# ---------------------------------------------------------------------------


class TestSamplingPayloadWeakRefs:
    """Sampling payloads wrap component and context objects in weakref.proxy."""

    def test_loop_end_final_context_is_weak_proxy(self):
        from mellea.plugins.hooks.sampling import SamplingLoopEndPayload

        mock_ctx = MagicMock()
        p = SamplingLoopEndPayload(final_context=mock_ctx)
        assert _is_weak_proxy(p.final_context)

    def test_loop_end_final_result_is_strong_ref(self):
        """final_result (writable ModelOutputThunk) is a strong reference."""
        from mellea.plugins.hooks.sampling import SamplingLoopEndPayload

        mock_result = MagicMock()
        p = SamplingLoopEndPayload(final_result=mock_result)
        assert p.final_result is mock_result
        assert not _is_weak_proxy(p.final_result)

    def test_loop_end_all_results_are_strong_refs(self):
        """all_results holds strong refs — plugins may retain sampled outputs."""
        from mellea.plugins.hooks.sampling import SamplingLoopEndPayload

        mock_r1, mock_r2, mock_r3 = MagicMock(), MagicMock(), MagicMock()
        results = [mock_r1, mock_r2, mock_r3]
        p = SamplingLoopEndPayload(all_results=results)
        assert len(p.all_results) == 3
        assert p.all_results[0] is mock_r1
        assert p.all_results[1] is mock_r2
        assert p.all_results[2] is mock_r3

    def test_repair_payload_action_and_context_are_weak_proxies(self):
        from mellea.plugins.hooks.sampling import SamplingRepairPayload

        mock_action = MagicMock(name="repair_action")
        mock_ctx = MagicMock(name="repair_context")
        p = SamplingRepairPayload(repair_action=mock_action, repair_context=mock_ctx)
        assert _is_weak_proxy(p.repair_action)
        assert _is_weak_proxy(p.repair_context)

    def test_loop_end_all_validations_holds_strong_refs(self):
        """Validation tuples are result data — held as strong references."""
        from mellea.plugins.hooks.sampling import SamplingLoopEndPayload

        mock_req = MagicMock(name="requirement")
        mock_val = MagicMock(name="validation_result")
        validations = [[(mock_req, mock_val)]]
        p = SamplingLoopEndPayload(all_validations=validations)
        req, val = p.all_validations[0][0]
        assert req is mock_req
        assert val is mock_val


# ---------------------------------------------------------------------------
# Tool payloads — strong references (result/data fields)
# ---------------------------------------------------------------------------


class TestToolPayloadStrongRefs:
    """Tool payloads hold strong references to tool call data and outputs."""

    def test_pre_invoke_model_tool_call_is_strong_ref(self):
        from mellea.plugins.hooks.tool import ToolPreInvokePayload

        mock_call = MagicMock()
        p = ToolPreInvokePayload(model_tool_call=mock_call)
        assert p.model_tool_call is mock_call

    def test_post_invoke_tool_output_is_strong_ref(self):
        from mellea.plugins.hooks.tool import ToolPostInvokePayload

        large_output = {"data": list(range(1000))}
        p = ToolPostInvokePayload(tool_output=large_output)
        assert p.tool_output is large_output

    def test_post_invoke_error_is_strong_ref(self):
        from mellea.plugins.hooks.tool import ToolPostInvokePayload

        err = ValueError("tool failed")
        p = ToolPostInvokePayload(error=err, success=False)
        assert p.error is err


# ---------------------------------------------------------------------------
# GC behaviour: WeakProxy fields do not prevent garbage collection
# ---------------------------------------------------------------------------


class TestWeakProxyAllowsGarbageCollection:
    """The primary guarantee of A5 fix: caching a payload does not keep live
    framework objects alive."""

    def test_cached_payload_does_not_prevent_session_gc(self):
        """A cached SessionPostInitPayload does not keep MelleaSession alive."""
        from mellea.plugins.hooks.session import SessionPostInitPayload

        class _FakeSession:
            pass

        session = _FakeSession()
        p = SessionPostInitPayload(session=session)

        # Delete the only strong reference to session
        del session
        gc.collect()

        # The proxy should now be dead — accessing it raises ReferenceError
        with pytest.raises(ReferenceError):
            _ = p.session.some_attr

    def test_cached_payload_does_not_prevent_context_gc(self):
        """A cached SamplingLoopEndPayload does not keep the Context alive."""
        from mellea.plugins.hooks.sampling import SamplingLoopEndPayload

        class _FakeContext:
            pass

        ctx = _FakeContext()
        p = SamplingLoopEndPayload(final_context=ctx)

        del ctx
        gc.collect()

        with pytest.raises(ReferenceError):
            _ = p.final_context.messages


# ---------------------------------------------------------------------------
# model_copy preserves proxy identity for unchanged WeakProxy fields
# ---------------------------------------------------------------------------


class TestModelCopyPreservesProxies:
    """model_copy(update={...}) copies the proxy objects from unchanged fields,
    keeping the weak-reference semantics intact."""

    def test_model_copy_preserves_weak_proxy_in_unchanged_fields(self):
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

        copied = original.model_copy(update={"latency_ms": 200})

        # Proxy fields are preserved (still weakref.proxy, still point to the same mocks)
        assert _is_weak_proxy(copied.action)
        assert _is_weak_proxy(copied.context_before)
        assert _is_weak_proxy(copied.context_after)
        assert copied.latency_ms == 200
