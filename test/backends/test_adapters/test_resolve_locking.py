# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Locking, race-safety, and resolution tests for `AdapterMixin` (Epic #929).

These exercise production code in `mellea/backends/adapters/adapter.py` that is
untouched by the shim removal (PR #1700): `resolve_adapter`'s collision scan and
lazy-registration, `_find_adapter`'s snapshot-before-iterate guard, the
`_adapter_resolve_lock()` critical section, and type-preference ordering. They
were ported from the deleted `test_shims.py`, which used the removed
`IntrinsicAdapter`/`EmbeddedIntrinsicAdapter` shims only as convenient fixtures;
here the same paths are driven with composed `Adapter` (`_AdapterCore`) stubs
carrying `LocalFileBinding`/`EmbeddedBinding`, matching the fixture pattern in
`test_huggingface_unit.py`.
"""

import threading
import time
import warnings
from unittest.mock import MagicMock, mock_open, patch

import pytest

from mellea.backends.adapters import Adapter, EmbeddedBinding, get_io_contract
from mellea.backends.adapters._core import (
    Adapter as _AdapterCore,
    Identity,
    LocalFileBinding,
)
from mellea.backends.adapters.adapter import AdapterMixin, _composed_adapter_key
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


def _make_composed_adapter(
    intrinsic_name: str = "answerability",
    technology: str = "alora",
    capability: str | None = None,
) -> _AdapterCore:
    """A composed embedded `Adapter` (EmbeddedBinding) stub.

    Replaces the deleted `EmbeddedIntrinsicAdapter`/`IntrinsicAdapter` shim
    fixtures: carries the same `Identity` (name, type, capability) the resolver
    and `_find_adapter` match on, without any deprecated shim machinery.
    """
    return _AdapterCore(
        identity=Identity(
            name=intrinsic_name,
            adapter_type=technology,  # type: ignore[arg-type]
            capability=capability or intrinsic_name,
        ),
        io_contract=get_io_contract(intrinsic_name),
        weights=EmbeddedBinding(),
    )


# ---------------------------------------------------------------------------
# resolve_adapter / _find_adapter — basic resolution
# ---------------------------------------------------------------------------


def test_resolve_adapter_returns_existing_by_capability():
    """resolve_adapter must return an already-registered adapter without creating a new one."""
    existing = _make_composed_adapter("answerability")
    mock_backend = MagicMock(spec=AdapterMixin)
    mock_backend._added_adapters = {_composed_adapter_key(existing): existing}
    # Route _find_adapter through the real implementation so the _added_adapters search runs.
    mock_backend._find_adapter.side_effect = lambda cap, types=None: (
        AdapterMixin._find_adapter(mock_backend, cap, types)
    )
    result = AdapterMixin.resolve_adapter(mock_backend, "answerability")
    assert result is existing
    mock_backend.add_adapter.assert_not_called()


def test_find_adapter_honours_type_preference_order():
    """_find_adapter must return the highest-priority type, not the insertion-order winner."""
    lora = _make_composed_adapter("answerability", technology="lora")
    alora = _make_composed_adapter("answerability", technology="alora")

    mock_backend = MagicMock(spec=AdapterMixin)
    # Register lora first so insertion order would return it without preference logic.
    mock_backend._added_adapters = {
        _composed_adapter_key(lora): lora,
        _composed_adapter_key(alora): alora,
    }

    result = AdapterMixin._find_adapter(
        mock_backend, "answerability", ("alora", "lora")
    )
    assert result is alora, "alora must win over lora regardless of insertion order"


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
    """resolve_adapter must create a composed Adapter when none is registered.

    Epic #929, issue #1144: resolve_adapter's construction site builds a
    composed `Adapter(identity, io_contract, LocalFileBinding)`, not the
    deprecated `IntrinsicAdapter` shim. The default type is LORA (every catalog
    entry supports it; Phase 2 will select from real catalog availability).
    """
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
        mock_backend._added_adapters[_composed_adapter_key(a)] = a

    mock_backend._added_adapters = {}
    mock_backend.add_adapter.side_effect = fake_add_adapter
    mock_backend._find_adapter.side_effect = lambda cap, types=None: (
        AdapterMixin._find_adapter(mock_backend, cap, types)
    )

    with patch(
        "mellea.backends.adapters.adapter.fetch_intrinsic_metadata",
        return_value=mock_catalog_entry,
    ):
        result = AdapterMixin.resolve_adapter(mock_backend, "answerability")

    assert mock_backend.add_adapter.called, (
        "add_adapter must be called for a new capability"
    )
    assert len(created_adapters) == 1
    assert isinstance(created_adapters[0], Adapter)
    assert isinstance(created_adapters[0].weights, LocalFileBinding)
    assert created_adapters[0].weights.adapter_type == AdapterType.LORA
    assert result is created_adapters[0]


def test_resolve_adapter_catalog_alias_returns_registered_adapter():
    """Regression (#1563): lookup accepts the catalog's public alias.

    The adapter stores `guardian_core` as its canonical capability, while
    callers resolve it via the catalog name `guardian-core`, including when
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
        mock_backend._added_adapters.__setitem__(
            _composed_adapter_key(adapter), adapter
        )
    )
    mock_backend._find_adapter.side_effect = lambda cap, types=None: (
        AdapterMixin._find_adapter(mock_backend, cap, types)
    )

    with patch(
        "mellea.backends.adapters.adapter.fetch_intrinsic_metadata",
        return_value=mock_catalog_entry,
    ):
        result = AdapterMixin.resolve_adapter(mock_backend, "guardian-core")

    assert result.identity.name == "guardian-core"
    assert result.identity.capability == "guardian_core"
    assert (
        AdapterMixin._find_adapter(mock_backend, "guardian-core", ("lora",)) is result
    )


# ---------------------------------------------------------------------------
# Concurrent-mutation-during-iteration guards
# ---------------------------------------------------------------------------


class _MutatingCapability:
    """Capability-name sentinel whose reflected `str.__eq__` pops an entry from
    the registry, simulating a concurrent `release()` mutating `_added_adapters`
    while `_find_adapter` iterates it. A live `.values()` view would raise
    `RuntimeError: dictionary changed size during iteration`; the snapshot
    `_find_adapter` takes must iterate on regardless.
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
    """Capability-name sentinel whose `__str__` pops an entry from the registry,
    simulating a concurrent `release()` mutating `_added_adapters` while
    `resolve_adapter`'s collision scan builds `f"{name}_"` on every iteration.

    `__add__` answers without firing: `Identity`/key construction may build
    `name + "_" + ...`, and the mutation must land in the scan, not in
    registration.
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
    first = _make_composed_adapter("answerability")
    second = _make_composed_adapter("uncertainty")

    mock_backend = MagicMock(spec=AdapterMixin)
    registry = {
        _composed_adapter_key(first): first,
        _composed_adapter_key(second): second,
    }
    mock_backend._added_adapters = registry

    capability = _MutatingCapability(registry, _composed_adapter_key(second))
    result = AdapterMixin._find_adapter(mock_backend, capability)  # must not raise

    assert capability.fired, "the mutation must have fired during iteration"
    assert result is None
    assert _composed_adapter_key(second) not in registry


def test_resolve_adapter_survives_concurrent_removal_during_iteration():
    """`resolve_adapter`'s collision scan must not iterate a live view of `_added_adapters`.

    Guards the `list(...)` snapshot with the same deterministic sentinel
    pattern as `test_find_adapter_survives_concurrent_removal_during_iteration`:
    the pop fires from the scan's `f"{name}_"` build on the first (non-matching)
    entry, so a live `.items()` view raises `RuntimeError: dictionary changed
    size during iteration` when the scan then advances, while the snapshot
    iterates on.
    """
    # resolve_adapter registers under the LORA default, so the colliding
    # binding must claim `answerability_lora` to actually block registration.
    binding = LocalFileBinding(name="answerability", adapter_type=AdapterType.LORA)
    distractor = _make_composed_adapter("uncertainty")

    mock_backend = MagicMock(spec=AdapterMixin)
    mock_backend.base_model_name = "ibm-granite/granite-4.1-3b"
    mock_backend._uses_embedded_adapters = False
    registry = {
        _composed_adapter_key(distractor): distractor,
        binding.qualified_name: binding,
    }
    mock_backend._added_adapters = registry
    mock_backend._find_adapter.side_effect = lambda cap, types=None: (
        AdapterMixin._find_adapter(mock_backend, cap, types)
    )
    # Nothing new lands in the registry: the binding keeps the name, exactly
    # as the real duplicate-key guard would.
    mock_backend.add_adapter.side_effect = lambda a: None

    name = _MutatingName(registry, _composed_adapter_key(distractor), "answerability")
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
    assert _composed_adapter_key(distractor) not in registry
    assert binding.qualified_name in registry


def test_resolve_adapter_names_the_conflict_when_a_binding_blocks_registration():
    """resolve_adapter's KeyError should name the collision, not just say "not found".

    Regression guard: a `LocalFileBinding` registered under the same
    qualified-name key space `resolve_adapter` auto-registers into silently
    blocks the new composed `Adapter` (the backend's duplicate-key guard
    refuses it). `_find_adapter` can't see the `LocalFileBinding` either (not
    an `_AdapterCore`), so without this check the failure surfaced as a bare
    "Adapter 'answerability' not found after registration" with no hint of
    what actually occupied the name. The resolver registers under the LORA
    default, so the colliding key is `answerability_lora`.
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
    # composed Adapter: registration is attempted but nothing new lands in
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


# ---------------------------------------------------------------------------
# _adapter_resolve_lock() critical section
# ---------------------------------------------------------------------------


class _TrackingLock:
    """Real lock that records how many times it was entered/exited.

    Used in place of a bare `MagicMock()` context manager so a test can assert
    the lock was acquired exactly once around the whole registration critical
    section (not once per `add_adapter` call in the embedded loop).
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self.enter_count = 0
        self.exit_count = 0
        self.max_depth = 0

    def __enter__(self):
        self._lock.acquire()
        self.enter_count += 1
        self.max_depth = max(self.max_depth, self.enter_count - self.exit_count)
        return None

    def __exit__(self, *exc_info):
        self.exit_count += 1
        self._lock.release()
        return False


def test_resolve_adapter_survives_reentrant_activation_lock():
    """resolve_adapter() must not deadlock when called while the lock is already held.

    `_adapter_activation_lock()`'s docstring documents a reentrancy contract:
    an override must return a reentrant lock because a caller can already hold
    it on the same thread. Simulate that caller directly: hold the real lock
    this mock backend's `_adapter_activation_lock()` returns, then call
    `resolve_adapter()` from inside that hold.
    """
    mock_catalog_entry = IntrinsicsCatalogEntry(
        name="answerability",
        repo_id="ibm-granite/granitelib-rag-r1.0",
        revision="abc123",
        adapter_types=(AdapterType.ALORA, AdapterType.LORA),
    )
    real_lock = threading.RLock()
    mock_backend = MagicMock(spec=AdapterMixin)
    mock_backend.base_model_name = "ibm-granite/granite-4.1-3b"
    mock_backend._uses_embedded_adapters = False
    mock_backend._added_adapters = {}
    mock_backend._adapter_activation_lock.return_value = real_lock
    mock_backend.add_adapter.side_effect = lambda a: (
        mock_backend._added_adapters.__setitem__(_composed_adapter_key(a), a)
    )
    mock_backend._find_adapter.side_effect = lambda cap, types=None: (
        AdapterMixin._find_adapter(mock_backend, cap, types)
    )

    with (
        patch(
            "mellea.backends.adapters.adapter.fetch_intrinsic_metadata",
            return_value=mock_catalog_entry,
        ),
        real_lock,  # simulate an already-in-progress caller holding the lock
    ):
        result = AdapterMixin.resolve_adapter(mock_backend, "answerability")

    assert result is not None
    assert result.identity.name == "answerability"


def test_resolve_adapter_holds_resolve_lock_during_registration():
    """resolve_adapter's single-adapter path must run inside `_adapter_resolve_lock()`.

    Issue #1562: `add_adapter()` is an unguarded read-then-write on
    `_added_adapters`; every other verb that touches it already holds a lock
    (#1465). Pin the lock's use here so it can't regress silently.

    Not `_adapter_activation_lock()` (Epic #929, issue #1144): that lock is
    taken *inside* `add_adapter()`'s composed-`LocalFileBinding` branch and
    inside every `LocalFileBinding` lifecycle verb, so holding it across this
    whole discover-and-register call would invert the adapter lock order and
    risk deadlocking against a concurrent `prepare()`/`release()`. See
    `_adapter_resolve_lock()`'s docstring for the full lock order.
    """
    mock_catalog_entry = IntrinsicsCatalogEntry(
        name="answerability",
        repo_id="ibm-granite/granitelib-rag-r1.0",
        revision="abc123",
        adapter_types=(AdapterType.ALORA, AdapterType.LORA),
    )
    mock_backend = MagicMock(spec=AdapterMixin)
    mock_backend.base_model_name = "ibm-granite/granite-4.1-3b"
    mock_backend._uses_embedded_adapters = False
    mock_backend._added_adapters = {}

    tracking_lock = _TrackingLock()
    mock_backend._adapter_resolve_lock.return_value = tracking_lock

    def fake_add_adapter(a):
        assert tracking_lock.enter_count == 1 and tracking_lock.exit_count == 0, (
            "add_adapter must run while the resolve lock is held"
        )
        mock_backend._added_adapters[_composed_adapter_key(a)] = a

    mock_backend.add_adapter.side_effect = fake_add_adapter
    mock_backend._find_adapter.side_effect = lambda cap, types=None: (
        AdapterMixin._find_adapter(mock_backend, cap, types)
    )

    with patch(
        "mellea.backends.adapters.adapter.fetch_intrinsic_metadata",
        return_value=mock_catalog_entry,
    ):
        AdapterMixin.resolve_adapter(mock_backend, "answerability")

    assert tracking_lock.enter_count == 1
    assert tracking_lock.exit_count == 1


def test_resolve_adapter_holds_resolve_lock_once_across_embedded_loop():
    """The embedded-adapter loop (#1018) must hold the resolve lock across all iterations.

    `resolve_adapter()` calls `add_adapter()` once per adapter discovered by
    `_discover_embedded_adapters()`. The lock must be acquired once for the
    whole loop, not re-acquired per adapter (that would reopen the race between
    iterations). `_adapter_resolve_lock()`, not `_adapter_activation_lock()` —
    see the lock-order note on the latter.
    """
    embedded_adapters = [
        (_make_composed_adapter("answerability", technology="alora"), {"k": "v"}),
        (_make_composed_adapter("answerability", technology="lora"), {"k": "v"}),
    ]

    mock_backend = MagicMock(spec=AdapterMixin)
    mock_backend.base_model_name = "ibm-granite/granite-4.1-3b"
    mock_backend._uses_embedded_adapters = True
    mock_backend._adapter_source = "ibm-granite/granite-switch-micro"
    mock_backend._added_adapters = {}

    tracking_lock = _TrackingLock()
    mock_backend._adapter_resolve_lock.return_value = tracking_lock

    def fake_add_adapter(a, config=None):
        mock_backend._added_adapters[_composed_adapter_key(a)] = a

    mock_backend.add_adapter.side_effect = fake_add_adapter
    mock_backend._find_adapter.side_effect = lambda cap, types=None: (
        AdapterMixin._find_adapter(mock_backend, cap, types)
    )

    with patch(
        "mellea.backends.adapters.adapter._discover_embedded_adapters",
        return_value=embedded_adapters,
    ):
        AdapterMixin.resolve_adapter(mock_backend, "answerability")

    assert mock_backend.add_adapter.call_count == 2, (
        "both embedded adapters must have been registered"
    )
    assert tracking_lock.enter_count == 1, (
        "the lock must be acquired once for the whole loop, not per adapter"
    )
    assert tracking_lock.exit_count == 1


def test_resolve_adapter_concurrent_first_use_does_not_double_register():
    """Two concurrent first-time resolves for the same name must not race (issue #1562).

    Mirrors the real `add_adapter` duplicate-registration check (`existing =
    registry.get(key); ...; registry[key] = adapter`) with an injected delay
    between the read and the write, which makes the unguarded race
    deterministic: without `_adapter_resolve_lock()` serializing the two
    threads, both would read `existing is None` before either writes, and both
    would overwrite the registry entry independently rather than one reusing
    the other's registration.
    """
    mock_catalog_entry = IntrinsicsCatalogEntry(
        name="answerability",
        repo_id="ibm-granite/granitelib-rag-r1.0",
        revision="abc123",
        adapter_types=(AdapterType.ALORA, AdapterType.LORA),
    )

    registry: dict = {}
    real_lock = threading.RLock()
    registrations: list = []

    mock_backend = MagicMock(spec=AdapterMixin)
    mock_backend.base_model_name = "ibm-granite/granite-4.1-3b"
    mock_backend._uses_embedded_adapters = False
    mock_backend._added_adapters = registry
    mock_backend._adapter_resolve_lock.return_value = real_lock
    mock_backend._find_adapter.side_effect = lambda cap, types=None: (
        AdapterMixin._find_adapter(mock_backend, cap, types)
    )

    def racy_add_adapter(adapter):
        key = _composed_adapter_key(adapter)
        existing = registry.get(key)
        if existing is not None:
            return
        time.sleep(0.02)  # widen the read-then-write window
        registry[key] = adapter
        registrations.append(adapter)

    mock_backend.add_adapter.side_effect = racy_add_adapter

    results: list = [None, None]
    errors: list = []

    def call(index: int) -> None:
        try:
            results[index] = AdapterMixin.resolve_adapter(mock_backend, "answerability")
        except Exception as e:  # pragma: no cover - surfaced via errors list
            errors.append(e)

    # `unittest.mock.patch`'s save/restore of the patched attribute is not
    # thread-safe: two threads independently entering/exiting a `patch(...)` on
    # the same target race on save-then-restore. Install the patch once here,
    # from the main thread, before spawning the workers — only
    # `resolve_adapter`'s own registration path runs concurrently.
    with patch(
        "mellea.backends.adapters.adapter.fetch_intrinsic_metadata",
        return_value=mock_catalog_entry,
    ):
        t1 = threading.Thread(target=call, args=(0,), daemon=True)
        t2 = threading.Thread(target=call, args=(1,), daemon=True)
        t1.start()
        t2.start()
        t1.join(timeout=5)
        t2.join(timeout=5)
        # A still-alive thread here would still be running resolve_adapter()
        # against the patch this `with` block is about to tear down. Assert
        # liveness before leaving the patch context, not after.
        assert not t1.is_alive(), "t1 did not finish within the join timeout"
        assert not t2.is_alive(), "t2 did not finish within the join timeout"

    assert not errors, f"resolve_adapter raised under concurrency: {errors}"
    assert len(registrations) == 1, (
        "exactly one adapter must be registered under the never-before-seen name"
    )
    # `racy_add_adapter`'s own no-op guard (`if existing is not None: return`)
    # would keep the assertion above green even without resolve_adapter's
    # in-lock re-check of `_find_adapter(name)` — the loser would just call
    # `add_adapter` a second time and no-op there instead. Pin the actual
    # optimisation directly: the loser must never call `add_adapter` at all.
    mock_backend.add_adapter.assert_called_once()
    assert results[0] is results[1] is registrations[0], (
        "both concurrent callers must resolve to the single registered adapter"
    )
