# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for adapter shim classes (Epic #929 Phase 1, issue #1136).

Verifies that IntrinsicAdapter, EmbeddedIntrinsicAdapter, and CustomIntrinsicAdapter:
  - emit DeprecationWarning on construction
  - are instances of both their own class and the new Adapter dataclass
  - expose a well-formed Identity (name, adapter_type, capability)
  - leave AdapterMixin.resolve_adapter and AdapterMixin.adapter_scope callable
"""

import warnings
from unittest.mock import MagicMock, mock_open, patch

import pytest

from mellea.backends.adapters import Adapter, EmbeddedIntrinsicAdapter, IntrinsicAdapter
from mellea.backends.adapters._core import Identity, LocalFileBinding
from mellea.backends.adapters.adapter import AdapterMixin
from mellea.backends.adapters.catalog import AdapterType, IntrinsicsCatalogEntry

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_MOCK_CATALOG_ENTRY = IntrinsicsCatalogEntry(
    name="answerability",
    repo_id="ibm-granite/granitelib-rag-r1.0",
    revision="abc123deadbeef",
    adapter_types=(AdapterType.ALORA, AdapterType.LORA),
)


def _make_intrinsic_adapter(intrinsic_name: str = "answerability") -> IntrinsicAdapter:
    """Construct IntrinsicAdapter with mocked catalog + config (no HF downloads)."""
    with (
        patch(
            "mellea.backends.adapters.adapter.fetch_intrinsic_metadata",
            return_value=IntrinsicsCatalogEntry(
                name=intrinsic_name,
                repo_id="ibm-granite/granitelib-rag-r1.0",
                revision="abc123",
                adapter_types=(AdapterType.ALORA, AdapterType.LORA),
            ),
        ),
        warnings.catch_warnings(),
    ):
        warnings.simplefilter("ignore", DeprecationWarning)
        return IntrinsicAdapter(
            intrinsic_name,
            adapter_type=AdapterType.ALORA,
            config_dict={"dummy": "config"},
        )


# ---------------------------------------------------------------------------
# EmbeddedIntrinsicAdapter shim tests (no mock needed — no catalog access)
# ---------------------------------------------------------------------------


def test_embedded_emits_deprecation_warning():
    with pytest.warns(
        DeprecationWarning, match="EmbeddedIntrinsicAdapter is deprecated"
    ):
        EmbeddedIntrinsicAdapter("answerability", config={}, technology="alora")


def test_embedded_is_instance_of_new_adapter():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        adapter = EmbeddedIntrinsicAdapter(
            "answerability", config={}, technology="alora"
        )
    assert isinstance(adapter, Adapter), (
        "shim must be instance of new Adapter dataclass"
    )
    assert isinstance(adapter, EmbeddedIntrinsicAdapter), (
        "shim must remain its own type"
    )


def test_embedded_identity_populated():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        adapter = EmbeddedIntrinsicAdapter(
            "answerability", config={}, technology="alora"
        )
    assert isinstance(adapter.identity, Identity)
    assert adapter.identity.name == "answerability"
    assert adapter.identity.capability == "answerability"
    assert adapter.identity.adapter_type == "alora"


def test_embedded_identity_lora_technology():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        adapter = EmbeddedIntrinsicAdapter(
            "answerability", config={}, technology="lora"
        )
    assert adapter.identity.adapter_type == "lora"


def test_embedded_legacy_attributes_preserved():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        adapter = EmbeddedIntrinsicAdapter(
            "answerability", config={"k": 1}, technology="alora"
        )
    assert adapter.intrinsic_name == "answerability"
    assert adapter.config == {"k": 1}
    assert adapter.technology == "alora"
    assert adapter.qualified_name == "answerability_alora"
    assert adapter.backend is None


def test_embedded_backend_mutable():
    """Shim must allow setting backend after construction (frozen bypass)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        adapter = EmbeddedIntrinsicAdapter(
            "answerability", config={}, technology="alora"
        )
    sentinel = object()
    adapter.backend = sentinel  # type: ignore[assignment]
    assert adapter.backend is sentinel


def test_embedded_invalid_technology():
    # Validation runs before the deprecation warning, so no DeprecationWarning fires.
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        with pytest.raises(ValueError, match="technology must be"):
            EmbeddedIntrinsicAdapter("answerability", config={}, technology="qlora")


# ---------------------------------------------------------------------------
# IntrinsicAdapter shim tests (uses patch to avoid catalog / HF access)
# ---------------------------------------------------------------------------


def test_intrinsic_emits_deprecation_warning():
    with (
        patch(
            "mellea.backends.adapters.adapter.fetch_intrinsic_metadata",
            return_value=_MOCK_CATALOG_ENTRY,
        ),
        pytest.warns(DeprecationWarning, match="IntrinsicAdapter is deprecated"),
    ):
        IntrinsicAdapter(
            "answerability",
            adapter_type=AdapterType.ALORA,
            config_dict={"dummy": "config"},
        )


def test_intrinsic_is_instance_of_new_adapter():
    adapter = _make_intrinsic_adapter("answerability")
    assert isinstance(adapter, Adapter), (
        "shim must be instance of new Adapter dataclass"
    )
    assert isinstance(adapter, IntrinsicAdapter), "shim must remain its own type"


def test_intrinsic_identity_populated():
    adapter = _make_intrinsic_adapter("answerability")
    assert isinstance(adapter.identity, Identity)
    assert adapter.identity.name == "answerability"
    assert adapter.identity.capability == "answerability"
    assert adapter.identity.adapter_type == "alora"


def test_intrinsic_identity_lora_adapter_type():
    with (
        patch(
            "mellea.backends.adapters.adapter.fetch_intrinsic_metadata",
            return_value=IntrinsicsCatalogEntry(
                name="answerability",
                repo_id="ibm-granite/granitelib-rag-r1.0",
                revision="abc123",
                adapter_types=(AdapterType.LORA,),
            ),
        ),
        warnings.catch_warnings(),
    ):
        warnings.simplefilter("ignore", DeprecationWarning)
        adapter = IntrinsicAdapter(
            "answerability",
            adapter_type=AdapterType.LORA,
            config_dict={"dummy": "config"},
        )
    assert adapter.identity.adapter_type == "lora"


def test_intrinsic_legacy_attributes_preserved():
    adapter = _make_intrinsic_adapter("answerability")
    assert adapter.intrinsic_name == "answerability"
    assert adapter.config == {"dummy": "config"}
    assert adapter.qualified_name == "answerability_alora"
    assert adapter.backend is None


def test_intrinsic_backend_mutable():
    """Shim must allow setting backend after construction (frozen bypass)."""
    adapter = _make_intrinsic_adapter()
    sentinel = object()
    adapter.backend = sentinel  # type: ignore[assignment]
    assert adapter.backend is sentinel


# ---------------------------------------------------------------------------
# AdapterMixin stub methods
# ---------------------------------------------------------------------------


def test_adapter_mixin_has_resolve_adapter():
    assert callable(getattr(AdapterMixin, "resolve_adapter", None))


def test_adapter_mixin_has_adapter_scope():
    assert callable(getattr(AdapterMixin, "adapter_scope", None))


def test_adapter_scope_is_noop():
    """adapter_scope must work as a no-op context manager via the mixin default."""
    mock_backend = MagicMock(spec=AdapterMixin)
    # Call the real implementation via the class (bypasses mock's own spec)
    with AdapterMixin.adapter_scope(mock_backend, None):
        pass  # must not raise


def test_adapter_scope_raises_for_a_shim_backed_adapter():
    """adapter_scope now activates real weights, so a shim-backed adapter raises.

    Deliberate behaviour change from Phase 1 (issue #1140), where `adapter_scope`
    was `yield` unconditionally regardless of `adapter.weights`. `resolve_adapter()`
    still returns `IntrinsicAdapter`/`LocalHFAdapter` shims carrying
    `_ShimWeightsBinding`, whose `.activate()` raises `NotImplementedError` — so
    `with backend.adapter_scope(backend.resolve_adapter(name)):` goes from a
    no-op to a hard failure for every adapter the public API currently hands
    out. Nothing in the codebase calls `adapter_scope` with a resolved adapter
    yet (#1465 is the tracked cutover), but this pins the change as
    deliberate rather than incidental — if #1465 needs `adapter_scope` to
    tolerate shim/unprepared bindings instead, that decision should update
    this test, not silently contradict it.
    """
    mock_backend = MagicMock(spec=AdapterMixin)
    adapter = _make_intrinsic_adapter("answerability")

    with pytest.raises(NotImplementedError, match="Phase 2"):
        with AdapterMixin.adapter_scope(mock_backend, adapter):
            pytest.fail("body must not run when the shim's activate() raises")


def test_resolve_adapter_returns_existing_by_capability():
    """resolve_adapter must return an already-registered adapter without creating a new one."""
    existing = _make_intrinsic_adapter("answerability")
    mock_backend = MagicMock(spec=AdapterMixin)
    mock_backend._added_adapters = {existing.qualified_name: existing}
    # Route _find_adapter through the real implementation so the _added_adapters search runs.
    mock_backend._find_adapter.side_effect = lambda cap, types=None: (
        AdapterMixin._find_adapter(mock_backend, cap, types)
    )
    result = AdapterMixin.resolve_adapter(mock_backend, "answerability")
    assert result is existing
    mock_backend.add_adapter.assert_not_called()


def test_find_adapter_honours_type_preference_order():
    """_find_adapter must return the highest-priority type, not the insertion-order winner."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        lora = EmbeddedIntrinsicAdapter("answerability", config={}, technology="lora")
        alora = EmbeddedIntrinsicAdapter("answerability", config={}, technology="alora")

    mock_backend = MagicMock(spec=AdapterMixin)
    # Register lora first so insertion order would return it without preference logic.
    mock_backend._added_adapters = {
        lora.qualified_name: lora,
        alora.qualified_name: alora,
    }

    result = AdapterMixin._find_adapter(
        mock_backend, "answerability", ("alora", "lora")
    )
    assert result is alora, "alora must win over lora regardless of insertion order"


def test_resolve_adapter_names_the_conflict_when_a_binding_blocks_registration():
    """resolve_adapter's KeyError should name the collision, not just say "not found".

    Regression guard: a `LocalFileBinding` registered under the same
    qualified-name key space `resolve_adapter` auto-registers into silently
    blocks the new `IntrinsicAdapter` (the backend's duplicate-key guard
    refuses it). `_find_adapter` can't see the `LocalFileBinding` either (not
    an `_AdapterCore`), so without this check the failure surfaced as a bare
    "Adapter 'answerability' not found after registration" with no hint of
    what actually occupied the name.
    """
    binding = LocalFileBinding(name="answerability", adapter_type=AdapterType.LORA)
    mock_backend = MagicMock(spec=AdapterMixin)
    mock_backend.base_model_name = "ibm-granite/granite-4.1-3b"
    mock_backend._uses_embedded_adapters = False
    mock_backend._added_adapters = {binding.qualified_name: binding}
    mock_backend._find_adapter.side_effect = lambda cap, types=None: (
        AdapterMixin._find_adapter(mock_backend, cap, types)
    )
    # Simulates the backend's real duplicate-key guard refusing the new
    # IntrinsicAdapter: registration is attempted but nothing new lands in
    # `_added_adapters`.
    mock_backend.add_adapter.side_effect = lambda a: None

    with (
        patch(
            "mellea.backends.adapters.adapter.fetch_intrinsic_metadata",
            return_value=_MOCK_CATALOG_ENTRY,
        ),
        patch(
            "mellea.backends.adapters.adapter.intrinsics.obtain_io_yaml",
            return_value="/fake/adapter.yaml",
        ),
        patch("builtins.open", mock_open(read_data="key: value")),
    ):
        with pytest.raises(KeyError, match=r"LocalFileBinding.*answerability_lora"):
            AdapterMixin.resolve_adapter(mock_backend, "answerability")


def test_resolve_adapter_raises_without_base_model():
    """resolve_adapter must raise ValueError when the backend has no model ID."""
    mock_backend = MagicMock(spec=AdapterMixin)
    mock_backend._added_adapters = {}
    mock_backend.base_model_name = None
    # _find_adapter returns None so resolve_adapter proceeds to the base-model check.
    mock_backend._find_adapter.return_value = None
    with pytest.raises(ValueError, match="no model ID"):
        AdapterMixin.resolve_adapter(mock_backend, "answerability")


def test_resolve_adapter_lazy_creates_and_returns():
    """resolve_adapter must create an IntrinsicAdapter when none is registered."""
    mock_catalog_entry = IntrinsicsCatalogEntry(
        name="answerability",
        repo_id="ibm-granite/granitelib-rag-r1.0",
        revision="abc123",
        adapter_types=(AdapterType.ALORA, AdapterType.LORA),
    )
    mock_backend = MagicMock(spec=AdapterMixin)
    mock_backend.base_model_name = "ibm-granite/granite-4.1-3b"
    mock_backend._uses_embedded_adapters = False

    created_adapters: list = []

    def fake_add_adapter(a):
        created_adapters.append(a)
        mock_backend._added_adapters[a.qualified_name] = a

    mock_backend._added_adapters = {}
    mock_backend.add_adapter.side_effect = fake_add_adapter
    mock_backend._find_adapter.side_effect = lambda cap, types=None: (
        AdapterMixin._find_adapter(mock_backend, cap, types)
    )

    with (
        patch(
            "mellea.backends.adapters.adapter.fetch_intrinsic_metadata",
            return_value=mock_catalog_entry,
        ),
        patch(
            "mellea.backends.adapters.adapter.intrinsics.obtain_io_yaml",
            return_value="/fake/adapter.yaml",
        ),
        patch("builtins.open", mock_open(read_data="key: value")),
    ):
        result = AdapterMixin.resolve_adapter(mock_backend, "answerability")

    assert mock_backend.add_adapter.called, (
        "add_adapter must be called for a new capability"
    )
    assert len(created_adapters) == 1
    assert isinstance(created_adapters[0], IntrinsicAdapter)
    assert created_adapters[0].adapter_type == AdapterType.LORA
    assert result is created_adapters[0]
