# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Real e2e test: LocalFileBinding's lifecycle against a real PEFT adapter.

Downloads the real "answerability" adapter from Hugging Face and loads it onto
a real Granite base model via LocalHFBackend — no mocking of the HF download,
PEFT machinery, or model. Requires GPU and network/Hub access; not expected to
run in CI or in sandboxes without hardware access (see test/README.md).

`adapter_scope()` is asserted to really flip the real PEFT model's active
adapter set, and to keep it active across a real generate call: the model is
called directly (bypassing `generate_from_context()`'s standard path, which
always deactivates adapters first via `_generate_with_adapter_lock("", ...)`)
so the active-adapter assertion straddling the generate call is a genuine
proof that generation ran with the adapter active, not a smoke test that
generation merely succeeded afterwards. A separate `generate_from_context()`
call after the scope exits is the composition smoke test: mellea's own
generation path must still work cleanly against a backend that has a
scoped-and-released adapter registered — it does not exercise the adapter
itself, since the standard path always deactivates first.

Assertions are structural/functional only (adapter registered, real model
reports it active during and after generation, adapter cleanly released), per
test/README.md's e2e rules — no assertions on generated text content.
"""

import os

import pytest

torch = pytest.importorskip("torch", reason="torch not installed — install mellea[hf]")

from test.predicates import require_gpu

pytestmark = [
    pytest.mark.huggingface,
    pytest.mark.e2e,
    pytest.mark.slow,
    require_gpu(min_vram_gb=20),
    pytest.mark.skipif(
        int(os.environ.get("CICD", 0)) == 1,
        reason="Skipping HuggingFace e2e tests in CI",
    ),
]

from mellea.backends import model_ids
from mellea.backends.adapters._core import (
    Adapter,
    Identity,
    IOContract,
    LocalFileBinding,
)
from mellea.backends.huggingface import LocalHFBackend
from mellea.core import CBlock, Component
from mellea.stdlib.context import SimpleContext
from test.conftest import cleanup_gpu_backend, hf_skip


class _Contract(IOContract):
    def build_prompt(self, **kwargs: object) -> Component:
        raise NotImplementedError

    def parse(self, raw: str) -> dict[str, object]:
        return {}


@pytest.fixture
def backend():
    with hf_skip():
        backend = LocalHFBackend(model_id=model_ids.IBM_GRANITE_4_1_3B)
    yield backend
    cleanup_gpu_backend(backend, backend_name="local_file_e2e")


@pytest.mark.asyncio
async def test_local_file_binding_full_lifecycle_against_real_model(backend):
    binding = LocalFileBinding.from_catalog("answerability")
    # adapter_type must agree with the binding: `from_catalog` takes
    # `metadata.adapter_types[0]`, which for `answerability` is LoRA. Hardcoding
    # "alora" here made the identity contradict the weights actually loaded.
    identity = Identity(
        name="answerability",
        adapter_type=binding.adapter_type.value,
        capability="answerability",
    )
    adapter = Adapter(identity=identity, io_contract=_Contract(), weights=binding)

    with hf_skip():
        binding.bind_backend(backend)
        binding.prepare()

    assert binding.backend is backend
    assert binding.qualified_name in backend.list_adapters()

    with backend.adapter_scope(adapter):
        # Confirms activate() really flipped the real PEFT model's active adapter.
        assert binding.qualified_name in backend._model.active_adapters()  # type: ignore[union-attr]

        # Generate directly against the real model rather than through
        # generate_from_context() — that standard path always deactivates
        # adapters first (_generate_with_adapter_lock("", ...)), which would
        # make this a smoke test that generation merely succeeds afterwards,
        # not a demonstration that generation ran with the adapter active.
        toks = backend._tokenizer("Is the sky blue?", return_tensors="pt").to(  # type: ignore[union-attr]
            backend._device  # type: ignore[union-attr]
        )
        with torch.no_grad():
            out_ids = backend._model.generate(**toks, max_new_tokens=8)  # type: ignore[union-attr]

        # Still inside the scope: the adapter must still be the one active
        # during generation, not just at scope-entry.
        assert binding.qualified_name in backend._model.active_adapters()  # type: ignore[union-attr]
        value = backend._tokenizer.decode(out_ids[0], skip_special_tokens=True)  # type: ignore[union-attr]

    assert binding.qualified_name not in backend._model.active_adapters()  # type: ignore[union-attr]
    assert value.strip()

    # Composition smoke test: generate_from_context() must still work once the
    # scope has exited (see module docstring — it does not exercise the
    # adapter itself, since the standard path always deactivates first).
    ctx = SimpleContext().add(CBlock("Is the sky blue?"))
    mot, _ = await backend.generate_from_context(
        CBlock("Is the sky blue?"), ctx, model_options={}
    )
    composed_value = await mot.avalue()
    assert isinstance(composed_value, str)
    assert len(composed_value) > 0

    binding.release()
    assert binding.backend is None
    # list_adapters() reports everything ever registered via add_adapter,
    # regardless of load state — release() only reverses the load, so check
    # the loaded-adapters bookkeeping directly instead.
    assert binding.qualified_name not in backend._loaded_adapters


if __name__ == "__main__":
    pytest.main([__file__])
