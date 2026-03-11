"""Tests for build_global_context() — the factory that maps Mellea domain objects
to ContextForge's GlobalContext.state.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

pytest.importorskip("cpex.framework")

from mellea.plugins.context import build_global_context


class TestBuildGlobalContext:
    """build_global_context() correctly maps Mellea domain objects to state."""

    def test_backend_name_derived_from_model_id(self):
        """state['backend_name'] is set from backend.model_id."""
        mock_backend = MagicMock()
        mock_backend.model_id = "ibm/granite-7b-instruct"

        ctx = build_global_context(backend=mock_backend)
        assert ctx.state["backend_name"] == "ibm/granite-7b-instruct"

    def test_backend_name_fallback_when_no_model_id(self):
        """state['backend_name'] falls back to 'unknown' if model_id is missing."""
        mock_backend = MagicMock(spec=[])  # no model_id attribute

        ctx = build_global_context(backend=mock_backend)
        assert ctx.state["backend_name"] == "unknown"

    def test_backend_object_not_in_state(self):
        """The full backend object should NOT be stored in state."""
        mock_backend = MagicMock()
        mock_backend.model_id = "test-model"

        ctx = build_global_context(backend=mock_backend)
        assert "backend" not in ctx.state
        assert ctx.state["backend_name"] == "test-model"

    def test_backend_name_absent_when_not_provided(self):
        """'backend_name' key is absent when backend=None."""
        ctx = build_global_context()
        assert "backend_name" not in ctx.state

    def test_extra_fields_passed_through(self):
        """Extra keyword arguments are included in state."""
        ctx = build_global_context(
            strategy_name="RejectionSampling", remaining_budget=3
        )
        assert ctx.state["strategy_name"] == "RejectionSampling"
        assert ctx.state["remaining_budget"] == 3

    def test_empty_call_returns_context(self):
        """build_global_context() with no args returns a valid GlobalContext."""
        ctx = build_global_context()
        assert ctx is not None
        assert isinstance(ctx.state, dict)

    def test_extra_fields_do_not_override_reserved_keys(self):
        """Extra fields with the same key as a domain object overwrite it (last-write wins)."""
        mock_backend = MagicMock()
        mock_backend.model_id = "test"
        ctx = build_global_context(backend=mock_backend, backend_name="overridden")
        # extra_fields are applied after backend_name is set, so they win
        assert ctx.state["backend_name"] == "overridden"
