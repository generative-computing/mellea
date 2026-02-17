"""Tests for hook payload models."""

import pytest
from pydantic import ValidationError

from mellea.plugins.base import MelleaBasePayload
from mellea.plugins.hooks.component import ComponentPreCreatePayload
from mellea.plugins.hooks.generation import GenerationPreCallPayload
from mellea.plugins.hooks.session import SessionPreInitPayload


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


class TestComponentPreCreatePayload:
    def test_creation(self):
        payload = ComponentPreCreatePayload(
            component_type="Instruction",
            description="Extract the user's age",
            requirements=["Must be a number"],
        )
        assert payload.component_type == "Instruction"
        assert payload.description == "Extract the user's age"
        assert len(payload.requirements) == 1

    def test_defaults(self):
        payload = ComponentPreCreatePayload()
        assert payload.component_type == ""
        assert payload.description == ""
        assert payload.images is None
        assert payload.requirements == []


class TestGenerationPreCallPayload:
    def test_creation(self):
        payload = GenerationPreCallPayload(
            model_options={"max_tokens": 100}, format=None
        )
        assert payload.model_options == {"max_tokens": 100}
        assert payload.format is None
