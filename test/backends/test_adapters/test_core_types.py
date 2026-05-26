"""Unit tests for the Phase 0 adapter scaffolding types (issue #1134)."""

import warnings

import pytest

from mellea.backends.adapters import (
    KNOWN_ROLES,
    Adapter,
    AdapterSchemaMismatchError,
    EmbeddedBinding,
    Identity,
    IOContract,
    LocalFileBinding,
    ServerMediatedBinding,
    WeightsBinding,
)
from mellea.core import Component


def test_adapter_dataclass_construction():
    class _Contract(IOContract):
        def build_prompt(self, **kwargs) -> Component:
            raise NotImplementedError

        def parse(self, raw: str) -> dict:
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


def test_io_contract_abc_enforcement():
    with pytest.raises(TypeError):
        IOContract()  # type: ignore[abstract]

    class MissingParse(IOContract):
        def build_prompt(self, **kwargs) -> Component:
            raise NotImplementedError

    with pytest.raises(TypeError):
        MissingParse()  # type: ignore[abstract]

    class MissingBuildPrompt(IOContract):
        def parse(self, raw: str) -> dict:
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


def test_stub_binding_subclasses_raise_not_implemented():
    for cls in (LocalFileBinding, EmbeddedBinding, ServerMediatedBinding):
        binding = cls()
        with pytest.raises(NotImplementedError):
            binding.prepare()
        with pytest.raises(NotImplementedError):
            binding.activate()
        with pytest.raises(NotImplementedError):
            binding.deactivate()
        with pytest.raises(NotImplementedError):
            binding.release()


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


def test_known_roles_importable():
    assert isinstance(KNOWN_ROLES, frozenset)
    assert "answerability" in KNOWN_ROLES


def test_identity_known_role_no_warning():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        Identity(name="a", adapter_type="lora", role="answerability")


def test_identity_unknown_role_warns():
    with pytest.warns(UserWarning, match="not in the KNOWN_ROLES"):
        Identity(name="a", adapter_type="lora", role="unknown-role")
