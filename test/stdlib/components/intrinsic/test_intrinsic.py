"""Unit tests for the Intrinsic component."""

from mellea.backends.adapters import AdapterType
from mellea.stdlib.components import Intrinsic


class TestAdapterTypesOverride:
    """Verify the adapter_types constructor parameter."""

    def test_default_uses_metadata(self):
        """When adapter_types is not passed, property returns metadata values."""
        intrinsic = Intrinsic("answerability")
        assert intrinsic.adapter_types == intrinsic.metadata.adapter_types

    def test_override_returns_custom_types(self):
        """When adapter_types is passed, property returns the override."""
        override = (AdapterType.LORA,)
        intrinsic = Intrinsic("answerability", adapter_types=override)
        assert intrinsic.adapter_types == override

    def test_explicit_none_uses_metadata(self):
        """Explicit None falls back to metadata."""
        intrinsic = Intrinsic("answerability", adapter_types=None)
        assert intrinsic.adapter_types == intrinsic.metadata.adapter_types

    def test_both_adapter_types(self):
        """Matches what call_intrinsic passes."""
        override = (AdapterType.ALORA, AdapterType.LORA)
        intrinsic = Intrinsic("answerability", adapter_types=override)
        assert intrinsic.adapter_types == override
