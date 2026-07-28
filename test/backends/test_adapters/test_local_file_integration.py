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

import contextlib
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

pytestmark = pytest.mark.integration


class _Contract(IOContract):
    def build_prompt(self, **kwargs: object) -> Component:
        raise NotImplementedError

    def parse(self, raw: str) -> dict[str, object]:
        return {}


@contextlib.contextmanager
def capture_adapter_hooks():
    """Record the hook payloads `adapter_scope` fires, without a plugin manager.

    Asserts on hooks rather than spans: `adapter_scope` fires hooks and never
    opens a span (see #1464 for the rule, #1466 for the spans themselves).
    """
    recorded: list[tuple[object, object]] = []

    async def _noop() -> None:
        return None

    def _fake_invoke_hook(hook_type: object, payload: object):
        recorded.append((hook_type, payload))
        return _noop()

    with (
        patch("mellea.backends.adapters.adapter.has_plugins", return_value=True),
        patch(
            "mellea.backends.adapters.adapter.invoke_hook",
            side_effect=_fake_invoke_hook,
        ),
    ):
        yield recorded


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

        with capture_adapter_hooks() as recorded:
            with backend.adapter_scope(adapter):
                backend._model.set_adapter.assert_called_with(binding.qualified_name)  # type: ignore[union-attr]

            backend._model.set_adapter.assert_called_with([])  # type: ignore[union-attr]

        binding.release()

    backend._model.delete_adapter.assert_called_once_with(binding.qualified_name)  # type: ignore[union-attr]
    assert binding.backend is None

    phases = [p.phase for _, p in recorded if hasattr(p, "phase")]
    assert phases == ["activate", "deactivate"]

    invocations = [p for _, p in recorded if hasattr(p, "outcome")]
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

        with capture_adapter_hooks() as recorded:
            with pytest.raises(RuntimeError, match="generation failed"):
                with backend.adapter_scope(adapter):
                    raise RuntimeError("generation failed")

        backend._model.set_adapter.assert_called_with([])  # type: ignore[union-attr]
        binding.release()

    # deactivate still ran, and the invocation is reported as an error carrying
    # the original exception — the behaviour the span status used to assert.
    phases = [p.phase for _, p in recorded if hasattr(p, "phase")]
    assert "deactivate" in phases

    invocations = [p for _, p in recorded if hasattr(p, "outcome")]
    assert len(invocations) == 1
    assert invocations[0].outcome == "error"
    assert isinstance(invocations[0].error, RuntimeError)


if __name__ == "__main__":
    pytest.main([__file__])
