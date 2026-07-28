# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Real e2e test: LocalFileBinding's lifecycle against a real PEFT adapter.

Downloads the real "answerability" adapter from Hugging Face and loads it onto
a real Granite base model via LocalHFBackend — no mocking of the HF download,
PEFT machinery, or model. Requires GPU and network/Hub access; not expected to
run in CI or in sandboxes without hardware access (see test/README.md).

`adapter_scope()` is asserted to really flip the real PEFT model's active
adapter set. `generate_from_context()` on a plain `CBlock` is only a
smoke-test that generation still succeeds afterwards — it does not run
through the activated adapter, since the standard generation path always
deactivates adapters first (`_generate_with_adapter_lock("", ...)`); wiring
that path onto `adapter_scope` is deferred to a follow-up issue.

Assertions are structural/functional only (adapter registered, real model
reports it active, generation succeeds, adapter cleanly released), per
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
    identity = Identity(
        name="answerability", adapter_type="alora", capability="answerability"
    )
    adapter = Adapter(identity=identity, io_contract=_Contract(), weights=binding)

    with hf_skip():
        binding.bind_backend(backend)
        binding.prepare()

    assert binding.backend is backend
    assert binding.qualified_name in backend.list_adapters()

    ctx = SimpleContext().add(CBlock("Is the sky blue?"))
    with backend.adapter_scope(adapter):
        # Confirms activate() really flipped the real PEFT model's active
        # adapter — the generate call below does not run through it (see
        # module docstring), so this is the only in-scope proof of activation.
        assert binding.qualified_name in backend._model.active_adapters()  # type: ignore[union-attr]

        mot, _ = await backend.generate_from_context(
            CBlock("Is the sky blue?"), ctx, model_options={}
        )
        value = await mot.avalue()

    assert binding.qualified_name not in backend._model.active_adapters()  # type: ignore[union-attr]
    assert isinstance(value, str)
    assert len(value) > 0

    binding.release()
    assert binding.backend is None
    # list_adapters() reports everything ever registered via add_adapter,
    # regardless of load state — release() only reverses the load, so check
    # the loaded-adapters bookkeeping directly instead.
    assert binding.qualified_name not in backend._loaded_adapters


if __name__ == "__main__":
    pytest.main([__file__])
