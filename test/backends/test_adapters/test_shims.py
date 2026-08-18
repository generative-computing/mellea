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

from mellea.backends.adapters import (
    Adapter,
    EmbeddedIntrinsicAdapter,
    IntrinsicAdapter,
    get_io_contract,
)
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


def test_embedded_carries_the_declared_io_contract_not_a_shim():
    """resolve_adapter()'s shim path must carry a real contract (issue #1516).

    Regression guard: without this, reverting `get_io_contract(intrinsic_name)`
    back to the Phase 1 `_ShimIOContract()` placeholder would pass every other
    test in this file, since none of them inspect `.io_contract`.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        adapter = EmbeddedIntrinsicAdapter(
            "answerability", config={}, technology="alora"
        )
    assert adapter.io_contract is get_io_contract("answerability")


def test_embedded_identity_lora_technology():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        adapter = EmbeddedIntrinsicAdapter(
            "answerability", config={}, technology="lora"
        )
    assert adapter.identity.adapter_type == "lora"


def test_embedded_catalog_capability_avoids_spurious_registry_warning():
    """Regression (#1563): catalog aliases use their effective capability.

    Before this fix, the hyphenated catalog name emitted a KNOWN_CAPABILITIES
    warning even though the adapter function was registered.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        adapter = EmbeddedIntrinsicAdapter(
            "guardian-core", config={}, technology="alora"
        )

    assert adapter.identity.capability == "guardian_core"
    assert not any(
        warning.category is UserWarning and "KNOWN_CAPABILITIES" in str(warning.message)
        for warning in caught
    )


def test_embedded_custom_capability_still_emits_registry_warning():
    """Custom embedded adapters retain their user-provided capability token."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        adapter = EmbeddedIntrinsicAdapter("custom-adapter", config={})

    assert adapter.identity.capability == "custom-adapter"
    assert any(
        warning.category is UserWarning and "KNOWN_CAPABILITIES" in str(warning.message)
        for warning in caught
    )


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


def test_intrinsic_carries_the_declared_io_contract_not_a_shim():
    """resolve_adapter()'s shim path must carry a real contract (issue #1516).

    Regression guard: without this, reverting `get_io_contract(intrinsic_name)`
    back to the Phase 1 `_ShimIOContract()` placeholder would pass every other
    test in this file, since none of them inspect `.io_contract`.
    """
    adapter = _make_intrinsic_adapter("answerability")
    assert adapter.io_contract is get_io_contract("answerability")


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


def test_intrinsic_catalog_capability_avoids_spurious_registry_warning():
    """Regression (#1563): catalog aliases use their effective capability.

    Before this fix, the hyphenated catalog name emitted a KNOWN_CAPABILITIES
    warning even though the adapter function was registered.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        adapter = IntrinsicAdapter(
            "guardian-core",
            adapter_type=AdapterType.ALORA,
            config_dict={"dummy": "config"},
        )

    assert adapter.identity.capability == "guardian_core"
    assert not any(
        warning.category is UserWarning and "KNOWN_CAPABILITIES" in str(warning.message)
        for warning in caught
    )


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


def test_adapter_scope_rejects_an_embedded_binding():
    """adapter_scope drives the WeightsBinding lifecycle; EmbeddedBinding has none.

    Deliberate behaviour change from #1142: `adapter_scope` used to raise
    `NotImplementedError` from `_ShimWeightsBinding.activate()` for an
    `EmbeddedIntrinsicAdapter` and fire `invocation_complete(outcome="error")`;
    it now raises `TypeError` before entering the try/finally and fires
    nothing. This pins the new guard (`adapter.py`'s `isinstance(adapter.weights,
    WeightsBinding)` check) rather than leaving it to surface as an
    `AttributeError` from `adapter.weights.activate()` the next time someone
    wires embedded generation through this scope.
    """
    mock_backend = MagicMock(spec=AdapterMixin)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        adapter = EmbeddedIntrinsicAdapter(
            "answerability", config={}, technology="alora"
        )

    with pytest.raises(TypeError, match=r"have no activate\(\)/deactivate\(\)"):
        with AdapterMixin.adapter_scope(mock_backend, adapter):
            pytest.fail("body must not run for a binding with no lifecycle")


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


class _MutatingCapability:
    """Capability sentinel whose `__eq__` deletes an entry from the registry
    it is compared against, simulating a concurrent `release()` mutating
    `_added_adapters` mid-iteration.

    `_find_adapter` compares `a.identity.capability == capability` for each
    registered adapter in turn; a real `str.__eq__` against this sentinel
    returns `NotImplemented`, so Python falls back to this class's reflected
    `__eq__` — firing the side effect from inside the loop, deterministically,
    with no actual threading required.
    """

    def __init__(self, registry: dict, key_to_remove: str) -> None:
        self._registry = registry
        self._key_to_remove = key_to_remove
        self.fired = False

    def __eq__(self, other: object) -> bool:
        if not self.fired:
            self.fired = True
            self._registry.pop(self._key_to_remove, None)
        return False

    def __hash__(self) -> int:
        return hash("mutating-capability-sentinel")


class _MutatingName:
    """Capability-name sentinel whose `__str__` pops an entry from the
    registry, simulating a concurrent `release()` mutating
    `_added_adapters` while `resolve_adapter`'s collision scan builds
    `f"{name}_"` on every iteration.

    Companion to `_MutatingCapability` (which fires from a reflected
    `str.__eq__` inside `_find_adapter`): the collision scan never compares
    `name` with `==`, so the side effect hooks `__str__` instead — the
    f-string build fires it from inside the scan, deterministically, with no
    threading. `__add__` answers without firing: `Adapter.__init__` builds
    `qualified_name` as `name + "_" + ...` during the lazy-registration
    step, and the mutation must land in the scan, not in registration.
    """

    def __init__(self, registry: dict, key_to_remove: str, prefix: str) -> None:
        self._registry = registry
        self._key_to_remove = key_to_remove
        self._prefix = prefix
        self.fired = False

    def __add__(self, other: str) -> str:
        return self._prefix + other

    def __str__(self) -> str:
        if not self.fired:
            self.fired = True
            self._registry.pop(self._key_to_remove, None)
        return self._prefix

    def __repr__(self) -> str:
        return f"_MutatingName({self._prefix!r})"


def test_find_adapter_survives_concurrent_removal_during_iteration():
    """`_find_adapter` must not iterate a live view over `_added_adapters`.

    `_added_adapters` was insert-only until `remove_adapter()` (#1528) added
    the first runtime deletion from it. A concurrent `release()` mutating the
    dict while `_find_adapter` holds a live `.values()` view raises
    `RuntimeError: dictionary changed size during iteration`. `_find_adapter`
    must snapshot into a list first.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        first = EmbeddedIntrinsicAdapter("answerability", config={}, technology="lora")
        second = EmbeddedIntrinsicAdapter("uncertainty", config={}, technology="lora")

    mock_backend = MagicMock(spec=AdapterMixin)
    registry = {first.qualified_name: first, second.qualified_name: second}
    mock_backend._added_adapters = registry

    capability = _MutatingCapability(registry, second.qualified_name)
    result = AdapterMixin._find_adapter(mock_backend, capability)  # must not raise

    assert capability.fired, "the mutation must have fired during iteration"
    assert result is None
    assert second.qualified_name not in registry


def test_resolve_adapter_survives_concurrent_removal_during_iteration():
    """`resolve_adapter`'s collision scan must not iterate a live view of `_added_adapters`.

    Guards the `list(...)` snapshot with the same deterministic sentinel
    pattern as
    `test_find_adapter_survives_concurrent_removal_during_iteration`: the
    pop fires from the scan's `f"{name}_"` build on the first (non-matching)
    entry, so a live `.items()` view raises
    `RuntimeError: dictionary changed size during iteration` when the scan
    then advances, while the snapshot iterates on.
    """
    binding = LocalFileBinding(name="answerability", adapter_type=AdapterType.LORA)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        distractor = EmbeddedIntrinsicAdapter(
            "uncertainty", config={}, technology="lora"
        )

    mock_backend = MagicMock(spec=AdapterMixin)
    mock_backend.base_model_name = "ibm-granite/granite-4.1-3b"
    mock_backend._uses_embedded_adapters = False
    registry = {distractor.qualified_name: distractor, binding.qualified_name: binding}
    mock_backend._added_adapters = registry
    mock_backend._find_adapter.side_effect = lambda cap, types=None: (
        AdapterMixin._find_adapter(mock_backend, cap, types)
    )
    # Nothing new lands in the registry: the binding keeps the name, exactly
    # as the real duplicate-key guard would.
    mock_backend.add_adapter.side_effect = lambda a: None

    name = _MutatingName(registry, distractor.qualified_name, "answerability")
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
            AdapterMixin.resolve_adapter(mock_backend, name)

    assert name.fired, "the mutation must have fired during the collision scan"
    assert distractor.qualified_name not in registry
    assert binding.qualified_name in registry


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


def test_resolve_adapter_catalog_alias_returns_registered_adapter():
    """Regression (#1563): lookup accepts the catalog's public alias.

    The shim stores `guardian_core` as its canonical capability, while callers
    resolve the adapter via the catalog name `guardian-core`, including when
    they constrain the adapter type.
    """
    mock_catalog_entry = IntrinsicsCatalogEntry(
        name="guardian-core",
        capability="guardian_core",
        repo_id="ibm-granite/granitelib-guardian-r1.0",
        revision="abc123",
        adapter_types=(AdapterType.LORA,),
    )
    mock_backend = MagicMock(spec=AdapterMixin)
    mock_backend.base_model_name = "ibm-granite/granite-4.1-3b"
    mock_backend._uses_embedded_adapters = False
    mock_backend._added_adapters = {}
    mock_backend.add_adapter.side_effect = lambda adapter: (
        mock_backend._added_adapters.__setitem__(adapter.qualified_name, adapter)
    )
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
        result = AdapterMixin.resolve_adapter(mock_backend, "guardian-core")

    assert result.identity.name == "guardian-core"
    assert result.identity.capability == "guardian_core"
    assert (
        AdapterMixin._find_adapter(mock_backend, "guardian-core", ("lora",)) is result
    )
