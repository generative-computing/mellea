# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the Intrinsic component."""

import pytest

from mellea.backends.adapters import AdapterType
from mellea.stdlib.components import Intrinsic


class TestCustomNonCatalogName:
    """A name outside the intrinsics catalog (Epic #929, issue #1144).

    Replaces the deprecated `CustomIntrinsicAdapter` shim's catalog
    monkey-patch: a custom adapter function can construct an `Intrinsic`
    without any catalog mutation, as long as it supplies `adapter_types`.
    """

    def test_custom_name_without_adapter_types_raises(self):
        with pytest.raises(ValueError, match="Unknown intrinsic name"):
            Intrinsic("totally-made-up-name")

    def test_custom_name_with_adapter_types_constructs(self):
        intrinsic = Intrinsic(
            "totally-made-up-name", adapter_types=(AdapterType.ALORA,)
        )
        assert intrinsic.intrinsic_name == "totally-made-up-name"
        assert intrinsic.adapter_types == (AdapterType.ALORA,)
        assert intrinsic.metadata.name == "totally-made-up-name"

    def test_known_name_ignores_synthetic_fallback(self):
        """A real catalog name never hits the synthetic-entry path, even
        without adapter_types."""
        intrinsic = Intrinsic("answerability")
        assert intrinsic.metadata.repo_id != ""


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
