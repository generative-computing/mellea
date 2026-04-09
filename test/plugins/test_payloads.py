"""Tests for hook payload models."""

import pytest
from pydantic import ValidationError

pytest.importorskip("cpex", reason="cpex not installed — install mellea[hooks]")

from mellea.plugins.base import MelleaBasePayload
from mellea.plugins.hooks.component import ComponentPreExecutePayload
from mellea.plugins.hooks.generation import GenerationPreCallPayload
from mellea.plugins.hooks.session import SessionPreInitPayload
from mellea.plugins.registry import _HAS_PLUGIN_FRAMEWORK


class TestMelleaBasePayload:
    def test_frozen(self):
        payload = MelleaBasePayload(request_id="test-123")
        with pytest.raises(ValidationError):
            payload.request_id = "new-value"

    def test_defaults(self):
        payload = MelleaBasePayload()
        assert payload.session_id is None
        assert payload.request_id == ""
        assert payload.hook == ""
        assert payload.user_metadata == {}
        assert payload.timestamp is not None

    def test_model_copy(self):
        payload = MelleaBasePayload(request_id="test-123", hook="test_hook")
        modified = payload.model_copy(update={"request_id": "new-123"})
        assert modified.request_id == "new-123"
        assert modified.hook == "test_hook"
        # Original unchanged
        assert payload.request_id == "test-123"


class TestSessionPreInitPayload:
    def test_creation(self):
        payload = SessionPreInitPayload(
            backend_name="openai", model_id="gpt-4", model_options={"temperature": 0.7}
        )
        assert payload.backend_name == "openai"
        assert payload.model_id == "gpt-4"
        assert payload.model_options == {"temperature": 0.7}

    def test_frozen(self):
        payload = SessionPreInitPayload(backend_name="openai", model_id="gpt-4")
        with pytest.raises(ValidationError):
            payload.backend_name = "hf"

    def test_model_copy_writable_fields(self):
        payload = SessionPreInitPayload(
            backend_name="openai", model_id="gpt-4", model_options=None
        )
        modified = payload.model_copy(
            update={"model_id": "gpt-3.5", "model_options": {"temperature": 0.5}}
        )
        assert modified.model_id == "gpt-3.5"
        assert modified.model_options == {"temperature": 0.5}


class TestGenerationPreCallPayload:
    def test_creation(self):
        payload = GenerationPreCallPayload(
            model_options={"max_tokens": 100}, format=None
        )
        assert payload.model_options == {"max_tokens": 100}
        assert payload.format is None


# ---------------------------------------------------------------------------
# Payload field tests — all fields are strong references (no WeakProxy)
# ---------------------------------------------------------------------------


class _Dummy:
    """A simple object for testing payload fields."""

    def __init__(self, name: str = "dummy"):
        self.name = name


class TestPayloadFieldsAreStrongRefs:
    """All payload fields store strong references to their values."""

    def test_field_stores_original_object(self):
        original = _Dummy("original")
        payload = ComponentPreExecutePayload(action=original)
        assert payload.action is original

    def test_modified_field_readable_after_model_copy(self):
        original = _Dummy("original")
        replacement = _Dummy("replaced")
        payload = ComponentPreExecutePayload(action=original)
        modified = payload.model_copy(update={"action": replacement})
        assert modified.action.name == "replaced"
        assert payload.action.name == "original"

    @pytest.mark.skipif(not _HAS_PLUGIN_FRAMEWORK, reason="cpex not installed")
    def test_writable_fields_in_policies(self):
        """Writable fields should be listed in their hook policies."""
        from mellea.plugins.policies import MELLEA_HOOK_PAYLOAD_POLICIES

        policy = MELLEA_HOOK_PAYLOAD_POLICIES["component_pre_execute"]
        assert "action" not in policy.writable_fields
        assert "context" not in policy.writable_fields
        assert "context_view" not in policy.writable_fields
        assert "strategy" in policy.writable_fields

        assert "sampling_repair" not in MELLEA_HOOK_PAYLOAD_POLICIES
        assert "sampling_loop_end" not in MELLEA_HOOK_PAYLOAD_POLICIES
        assert "component_post_success" not in MELLEA_HOOK_PAYLOAD_POLICIES
        assert "generation_post_call" not in MELLEA_HOOK_PAYLOAD_POLICIES
