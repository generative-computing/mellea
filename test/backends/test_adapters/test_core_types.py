# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the Phase 0 adapter scaffolding types (issue #1134)."""

import dataclasses
import pickle
import warnings

import pytest

from mellea.backends.adapters import (
    KNOWN_CAPABILITIES,
    Adapter,
    AdapterSchemaMismatchError,
    EmbeddedBinding,
    Identity,
    IOContract,
    LocalFileBinding,
    ServerMediatedBinding,
    WeightsBinding,
)
from mellea.backends.adapters.catalog import _INTRINSICS_CATALOG_ENTRIES
from mellea.core import Component


def test_adapter_dataclass_construction():
    class _Contract(IOContract):
        def build_prompt(self, **kwargs: object) -> Component:
            raise NotImplementedError

        def parse(self, raw: str) -> dict[str, object]:
            return {}

    contract = _Contract()
    binding = LocalFileBinding()
    identity = Identity(name="test-adapter", adapter_type="lora")
    adapter = Adapter(identity=identity, io_contract=contract, weights=binding)

    assert adapter.identity is identity
    assert adapter.io_contract is contract
    assert adapter.weights is binding


def test_identity_validation():
    id_lora = Identity(name="my-adapter", adapter_type="lora")
    assert id_lora.adapter_type == "lora"

    id_alora = Identity(name="my-adapter", adapter_type="alora")
    assert id_alora.adapter_type == "alora"

    with pytest.raises(ValueError, match="adapter_type must be"):
        Identity(name="bad", adapter_type="qlora")  # type: ignore[arg-type]


def test_identity_is_frozen_and_hashable():
    identity = Identity(name="x", adapter_type="lora")
    with pytest.raises(dataclasses.FrozenInstanceError):
        identity.adapter_type = "alora"  # type: ignore[misc]
    # Hashable so it can be used as a dict key / set member.
    assert hash(identity) == hash(Identity(name="x", adapter_type="lora"))


def test_adapter_is_frozen():
    class _Contract(IOContract):
        def build_prompt(self, **kwargs: object) -> Component:
            raise NotImplementedError

        def parse(self, raw: str) -> dict[str, object]:
            return {}

    adapter = Adapter(
        identity=Identity(name="x", adapter_type="lora"),
        io_contract=_Contract(),
        weights=LocalFileBinding(),
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        adapter.identity = Identity(name="y", adapter_type="alora")  # type: ignore[misc]


def test_io_contract_abc_enforcement():
    with pytest.raises(TypeError):
        IOContract()  # type: ignore[abstract]

    class MissingParse(IOContract):
        def build_prompt(self, **kwargs: object) -> Component:
            raise NotImplementedError

    with pytest.raises(TypeError):
        MissingParse()  # type: ignore[abstract]

    class MissingBuildPrompt(IOContract):
        def parse(self, raw: str) -> dict[str, object]:
            return {}

    with pytest.raises(TypeError):
        MissingBuildPrompt()  # type: ignore[abstract]


def test_weights_binding_abc_enforcement():
    with pytest.raises(TypeError):
        WeightsBinding()  # type: ignore[abstract]

    class PartialBinding(WeightsBinding):
        def prepare(self) -> None:
            raise NotImplementedError

        def activate(self) -> None:
            raise NotImplementedError

        def deactivate(self) -> None:
            raise NotImplementedError

        # release is missing

    with pytest.raises(TypeError):
        PartialBinding()  # type: ignore[abstract]


@pytest.mark.parametrize(
    "cls", [LocalFileBinding, EmbeddedBinding, ServerMediatedBinding]
)
@pytest.mark.parametrize("verb", ["prepare", "activate", "deactivate", "release"])
def test_stub_binding_subclasses_raise_not_implemented(cls, verb):
    binding = cls()
    with pytest.raises(NotImplementedError, match="Phase 0 stub"):
        getattr(binding, verb)()


def test_adapter_schema_mismatch_error_format():
    observed = frozenset({"key_a", "key_b"})
    expected = frozenset({"key_a", "key_c"})
    err = AdapterSchemaMismatchError(
        name="answerability", observed_keys=observed, expected_keys=expected
    )

    assert err.name == "answerability"
    assert err.observed_keys == observed
    assert err.expected_keys == expected
    msg = str(err)
    assert "answerability" in msg
    assert "Observed keys:" in msg
    assert "expected:" in msg


def test_adapter_schema_mismatch_error_pickles():
    observed = frozenset({"key_a"})
    expected = frozenset({"key_b"})
    err = AdapterSchemaMismatchError(
        name="answerability", observed_keys=observed, expected_keys=expected
    )

    restored = pickle.loads(pickle.dumps(err))

    assert isinstance(restored, AdapterSchemaMismatchError)
    assert restored.name == "answerability"
    assert restored.observed_keys == observed
    assert restored.expected_keys == expected
    assert str(restored) == str(err)


def test_known_capabilities_importable():
    assert isinstance(KNOWN_CAPABILITIES, frozenset)
    assert "answerability" in KNOWN_CAPABILITIES
    # Hyphenated upstream names must NOT be in the capability vocabulary;
    # only the stable underscore forms derived from effective_capability are.
    assert "requirement-check" not in KNOWN_CAPABILITIES
    assert "requirement_check" in KNOWN_CAPABILITIES


def test_identity_known_capability_no_warning():
    # Tight scope: only treat UserWarning from the capability registry as a failure.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        Identity(name="a", adapter_type="lora", capability="answerability")
    capability_warnings = [
        w
        for w in caught
        if issubclass(w.category, UserWarning)
        and "KNOWN_CAPABILITIES" in str(w.message)
    ]
    assert capability_warnings == []


def test_identity_unknown_capability_warns():
    with pytest.warns(UserWarning, match="not in the KNOWN_CAPABILITIES"):
        Identity(name="a", adapter_type="lora", capability="unknown-capability")


def test_identity_requirement_check_underscore_no_warning():
    """requirement_check (underscore) must be a known capability after #1186."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        Identity(
            name="requirement-check",
            adapter_type="lora",
            capability="requirement_check",
        )
    capability_warnings = [
        w
        for w in caught
        if issubclass(w.category, UserWarning)
        and "KNOWN_CAPABILITIES" in str(w.message)
    ]
    assert capability_warnings == []


def test_known_capabilities_contains_no_hyphens():
    # No hyphenated name should leak into KNOWN_CAPABILITIES.  If a future
    # catalog entry uses hyphens in `name` without setting `capability`, this
    # catches it immediately.
    hyphenated = [cap for cap in KNOWN_CAPABILITIES if "-" in cap]
    assert hyphenated == [], f"Hyphenated capabilities found: {hyphenated}"


def test_known_capabilities_count_matches_catalog():
    # Every catalog entry must contribute exactly one distinct effective_capability.
    # If two entries resolve to the same token, the frozenset shrinks and this fails.
    assert len(KNOWN_CAPABILITIES) == len(_INTRINSICS_CATALOG_ENTRIES)
