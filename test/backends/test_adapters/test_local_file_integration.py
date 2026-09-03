# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Integration test: LocalFileBinding's full lifecycle through a real LocalHFBackend.

A real `LocalHFBackend` instance is used; only the Hugging Face download
(`intrinsics.obtain_lora`) and the underlying PEFT model (`_model`) are mocked,
per test/README.md's definition of `integration` — real framework/library objects
wired together, external network and model weights mocked at the outer boundary.

Telemetry assertions are on the fired **hooks**, not on spans: `adapter_scope`
fires hooks and deliberately opens no spans (#1464 documents the rule, #1466 adds
the spans from a plugin).
"""

from unittest.mock import MagicMock, patch

import pytest

torch = pytest.importorskip("torch", reason="torch not installed — install mellea[hf]")
pytest.importorskip(
    "transformers", reason="transformers not installed — install mellea[hf]"
)
pytest.importorskip(
    "llguidance", reason="llguidance not installed — install mellea[hf]"
)

from mellea.backends.adapters._core import (
    Adapter,
    Identity,
    IOContract,
    LocalFileBinding,
)
from mellea.backends.adapters.catalog import fetch_intrinsic_metadata
from mellea.backends.huggingface import LocalHFBackend
from mellea.core import Component
from test.backends.test_adapters._hook_capture import (
    capture_adapter_hooks,
    hook_payloads,
)

pytestmark = pytest.mark.integration


class _Contract(IOContract):
    def build_prompt(self, **kwargs: object) -> Component:
        raise NotImplementedError

    def parse(self, raw: str) -> dict[str, object]:
        return {}


def _make_backend() -> LocalHFBackend:
    mock_tok = MagicMock(eos_token_id=0, vocab_size=32000)
    mock_tok._tokenizer = MagicMock()
    mock_tok._tokenizer.get_vocab_size.return_value = 32000
    mock_tok.__len__ = MagicMock(return_value=32000)
    mock_model = MagicMock(vocab_size=32000)
    with (
        patch("mellea.backends.huggingface.llguidance") as mock_llg,
        patch("mellea.backends.huggingface.set_seed"),
    ):
        mock_llg.hf.from_tokenizer.return_value = MagicMock(vocab_size=32000)
        return LocalHFBackend(
            model_id="ibm-granite/granite-3.3-8b-instruct",
            custom_config=(mock_tok, mock_model, torch.device("cpu")),
        )


def _make_binding() -> LocalFileBinding:
    metadata = fetch_intrinsic_metadata("answerability")
    return LocalFileBinding(
        name="answerability",
        adapter_type=metadata.adapter_types[0],
        repo_id=metadata.repo_id,
        revision=metadata.revision,
    )


def _make_adapter(binding: LocalFileBinding) -> Adapter:
    # adapter_type must agree with the binding: `from_catalog` takes
    # `metadata.adapter_types[0]`, which for `answerability` is LoRA. Hardcoding
    # "alora" here made the identity contradict the weights actually loaded.
    identity = Identity(
        name="answerability",
        adapter_type=binding.adapter_type.value,
        capability="answerability",
    )
    return Adapter(identity=identity, io_contract=_Contract(), weights=binding)


def test_prepare_activate_deactivate_release_full_lifecycle():
    backend = _make_backend()
    binding = _make_binding()
    adapter = _make_adapter(binding)

    with patch(
        "mellea.formatters.granite.intrinsics.obtain_lora",
        return_value="/fake/local/adapter/path",
    ) as mock_obtain_lora:
        binding.bind_backend(backend)
        binding.prepare()

        assert binding.backend is backend
        assert binding.qualified_name in backend.list_adapters()
        mock_obtain_lora.assert_called_once()
        assert mock_obtain_lora.call_args.kwargs["revision"] == binding.revision

        with capture_adapter_hooks() as mock_invoke:
            with backend.adapter_scope(adapter):
                backend._model.set_adapter.assert_called_with(binding.qualified_name)  # type: ignore[union-attr]

            backend._model.set_adapter.assert_called_with([])  # type: ignore[union-attr]

        binding.release()

    backend._model.delete_adapter.assert_called_once_with(binding.qualified_name)  # type: ignore[union-attr]
    assert binding.backend is None

    recorded = hook_payloads(mock_invoke)
    phases = [p.phase for p in recorded if hasattr(p, "phase")]
    assert phases == ["activate", "deactivate"]

    invocations = [p for p in recorded if hasattr(p, "outcome")]
    assert len(invocations) == 1
    assert invocations[0].outcome == "success"
    assert invocations[0].name == "answerability"
    assert invocations[0].binding_type == "local_file"
    assert invocations[0].adapter_type == binding.adapter_type.value


def test_deactivate_runs_even_when_generation_body_raises():
    backend = _make_backend()
    binding = _make_binding()
    adapter = _make_adapter(binding)

    with patch(
        "mellea.formatters.granite.intrinsics.obtain_lora",
        return_value="/fake/local/adapter/path",
    ):
        binding.bind_backend(backend)
        binding.prepare()

        with capture_adapter_hooks() as mock_invoke:
            with pytest.raises(RuntimeError, match="generation failed"):
                with backend.adapter_scope(adapter):
                    raise RuntimeError("generation failed")

        backend._model.set_adapter.assert_called_with([])  # type: ignore[union-attr]
        binding.release()

    # deactivate still ran, and the invocation is reported as an error carrying
    # the original exception — the behaviour the span status used to assert.
    recorded = hook_payloads(mock_invoke)
    phases = [p.phase for p in recorded if hasattr(p, "phase")]
    assert "deactivate" in phases

    invocations = [p for p in recorded if hasattr(p, "outcome")]
    assert len(invocations) == 1
    assert invocations[0].outcome == "error"
    assert isinstance(invocations[0].error, RuntimeError)


def test_add_adapter_registers_composed_adapter_via_backend():
    """`backend.add_adapter(composed_adapter)` drives the binding lifecycle
    itself (Epic #929, issue #1144) — a caller no longer has to call
    `binding.bind_backend()`/`binding.prepare()` separately before the
    composed `Adapter` becomes resolvable by capability name.
    """
    backend = _make_backend()
    binding = _make_binding()
    adapter = _make_adapter(binding)

    with patch(
        "mellea.formatters.granite.intrinsics.obtain_lora",
        return_value="/fake/local/adapter/path",
    ):
        backend.add_adapter(adapter)

        assert binding.backend is backend
        assert binding.qualified_name in backend.list_adapters()
        # add_adapter also makes the composed Adapter itself discoverable by
        # capability, unlike registering a bare LocalFileBinding directly.
        found = backend._find_adapter("answerability")
        assert found is adapter

        with capture_adapter_hooks() as mock_invoke:
            with backend.adapter_scope(adapter):
                backend._model.set_adapter.assert_called_with(binding.qualified_name)  # type: ignore[union-attr]
            backend._model.set_adapter.assert_called_with([])  # type: ignore[union-attr]

    invocations = [p for p in hook_payloads(mock_invoke) if hasattr(p, "outcome")]
    assert len(invocations) == 1
    assert invocations[0].outcome == "success"


def test_add_adapter_rejects_second_composed_registration_for_same_capability():
    """A second `add_adapter` for the same capability is refused, not silently
    overwritten — mirrors the shim's duplicate-registration guard."""
    backend = _make_backend()
    binding = _make_binding()
    adapter = _make_adapter(binding)
    other_binding = _make_binding()
    other_adapter = _make_adapter(other_binding)

    with patch(
        "mellea.formatters.granite.intrinsics.obtain_lora",
        return_value="/fake/local/adapter/path",
    ):
        backend.add_adapter(adapter)
        backend.add_adapter(other_adapter)

    assert backend._find_adapter("answerability") is adapter
    assert other_binding.backend is None


def test_add_adapter_rejects_binding_already_bound_to_a_different_backend():
    """Registering a composed Adapter whose LocalFileBinding is already bound
    to a *different* backend must raise, not silently misroute.

    Regression: `add_adapter`'s composed-LocalFileBinding branch used to skip
    `bind_backend()` — the only call that raises for this — whenever
    `binding.backend` was already set, then called the now-no-op `prepare()`
    and registered the adapter on the *new* backend anyway. A later
    `adapter_scope()`/`activate()` on the new backend would then activate
    PEFT state on the *original* backend's model instead, with no error at
    any point.
    """
    backend_a = _make_backend()
    backend_b = _make_backend()
    binding = _make_binding()
    adapter = _make_adapter(binding)

    with patch(
        "mellea.formatters.granite.intrinsics.obtain_lora",
        return_value="/fake/local/adapter/path",
    ):
        backend_a.add_adapter(adapter)
        assert binding.backend is backend_a

        with pytest.raises(RuntimeError, match="cannot change the backend"):
            backend_b.add_adapter(adapter)

    # The failed registration attempt must not have touched backend_a's claim.
    assert binding.backend is backend_a
    assert backend_b._find_adapter("answerability") is None


if __name__ == "__main__":
    pytest.main([__file__])
