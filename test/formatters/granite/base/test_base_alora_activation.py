# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for aLoRA activation through the bare-model (`load_adapter`) path.

Background: PEFT only computes and injects aLoRA activation masks
(`alora_offsets`) from inside its `PeftModel` wrapper's `generate()`/
`forward()` overrides. Models that load adapters via transformers' native
`PeftAdapterMixin.load_adapter()` — which is what `LocalHFBackend` does —
are never wrapped in `PeftModel`, so the aLoRA weights silently never apply
and generation degrades to base-model behaviour.
`util._alora_activation_context` fixes this by mirroring `PeftModel`'s own
mechanism (offset computation + temporary per-layer pre-forward hooks).

These tests use a tiny randomly-initialised model (no download) and assert
on whether the aLoRA variant layer actually receives `alora_offsets` during
`generate()` — the exact property that was missing before the fix.
"""

# Standard
from unittest.mock import MagicMock

# Third Party
import pytest

torch = pytest.importorskip("torch", reason="torch not installed — install mellea[hf]")
pytest.importorskip(
    "transformers", reason="transformers not installed — install mellea[hf]"
)
peft = pytest.importorskip("peft", reason="peft not installed — install mellea[hf]")
# First Party
import peft.tuners.lora.variants as peft_variants
from peft import LoraConfig
from transformers import LlamaConfig, LlamaForCausalLM

from mellea.backends.huggingface import LocalHFBackend
from mellea.formatters.granite.base.util import (
    _alora_activation_context,
    generate_with_transformers,
)

INVOCATION = [1, 2, 3]


def _tiny_model(adapter_config: LoraConfig | None = None) -> LlamaForCausalLM:
    config = LlamaConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=128,
    )
    model = LlamaForCausalLM(config).eval()
    if adapter_config is not None:
        model.add_adapter(adapter_config, "rc")
        model.set_adapter("rc")
    return model


def _alora_config() -> LoraConfig:
    return LoraConfig(
        r=4,
        lora_alpha=8,
        target_modules=["q_proj", "v_proj"],
        task_type="CAUSAL_LM",
        alora_invocation_tokens=INVOCATION,
    )


def _lora_config() -> LoraConfig:
    return LoraConfig(
        r=4, lora_alpha=8, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM"
    )


class _VariantSpy:
    """Count ALoraLinearVariant.forward calls and record the alora_offsets each received."""

    def __init__(self):
        self.n_calls = 0
        self.offsets_seen: list = []
        self._orig = peft_variants.ALoraLinearVariant.forward

    def __enter__(self):
        def spy(module, active_adapter, x, result, **kwargs):
            self.n_calls += 1
            self.offsets_seen.append(kwargs.get("alora_offsets"))
            return self._orig(module, active_adapter, x, result, **kwargs)

        peft_variants.ALoraLinearVariant.forward = staticmethod(spy)
        return self

    def __exit__(self, *exc):
        # Restore as a staticmethod: class-attribute access to a staticmethod
        # returns the bare function, so a plain reassignment would turn it
        # into an instance method and break every later aLoRA forward.
        peft_variants.ALoraLinearVariant.forward = staticmethod(self._orig)
        return False


class TestAloraActivationContext:
    def test_no_offsets_reach_variant_without_context(self):
        """The regression itself: bare model + aLoRA adapter, no context ->
        the aLoRA variant is called but never receives alora_offsets, so it
        masks to base-model behaviour."""
        model = _tiny_model(_alora_config())
        input_ids = torch.tensor([[10, 1, 2, 3, 40, 50]])
        with _VariantSpy() as spy:
            with torch.no_grad():
                model.generate(input_ids=input_ids, max_new_tokens=2)
        assert spy.n_calls > 0, "aLoRA variant layer should be exercised at all"
        assert spy.offsets_seen == [None] * spy.n_calls

    def test_context_delivers_expected_offsets(self):
        model = _tiny_model(_alora_config())
        # Invocation sequence starts at index 1 of a 6-token prompt; PEFT
        # counts the offset back from the end, so it is 6 - 1 = 5.
        input_ids = torch.tensor([[10, 1, 2, 3, 40, 50]])
        with _VariantSpy() as spy:
            with torch.no_grad(), _alora_activation_context(model, input_ids):
                model.generate(input_ids=input_ids, max_new_tokens=2)
        assert spy.n_calls > 0
        assert spy.offsets_seen[0] == [5]
        assert all(o == [5] for o in spy.offsets_seen)

    def test_context_with_missing_invocation_passes_none_offset(self):
        """Invocation sequence absent -> offsets are [None] (peft semantics:
        adapter inactive for that row), which the variant masks to base."""
        model = _tiny_model(_alora_config())
        input_ids = torch.tensor([[10, 11, 12, 40, 50]])
        with _VariantSpy() as spy:
            with torch.no_grad(), _alora_activation_context(model, input_ids):
                model.generate(input_ids=input_ids, max_new_tokens=2)
        assert spy.offsets_seen and all(o == [None] for o in spy.offsets_seen)

    def test_hooks_removed_after_context_exit(self):
        model = _tiny_model(_alora_config())
        input_ids = torch.tensor([[10, 1, 2, 3, 40, 50]])
        with torch.no_grad(), _alora_activation_context(model, input_ids):
            pass
        n_hooks = sum(
            len(getattr(m, "_forward_pre_hooks", {})) for m in model.modules()
        )
        assert n_hooks == 0, "pre-forward hooks must not outlive the context"

    def test_hooks_removed_on_error(self):
        model = _tiny_model(_alora_config())
        input_ids = torch.tensor([[10, 1, 2, 3, 40, 50]])
        with pytest.raises(RuntimeError, match="boom"):
            with torch.no_grad(), _alora_activation_context(model, input_ids):
                raise RuntimeError("boom")
        n_hooks = sum(
            len(getattr(m, "_forward_pre_hooks", {})) for m in model.modules()
        )
        assert n_hooks == 0

    def test_no_adapter_is_noop(self):
        model = _tiny_model()
        input_ids = torch.tensor([[10, 1, 2, 3, 40, 50]])
        with torch.no_grad(), _alora_activation_context(model, input_ids):
            model.generate(input_ids=input_ids, max_new_tokens=2)  # must not raise

    def test_plain_lora_adapter_is_noop(self):
        model = _tiny_model(_lora_config())
        input_ids = torch.tensor([[10, 1, 2, 3, 40, 50]])
        with torch.no_grad(), _alora_activation_context(model, input_ids):
            model.generate(input_ids=input_ids, max_new_tokens=2)
        n_hooks = sum(
            len(getattr(m, "_forward_pre_hooks", {})) for m in model.modules()
        )
        assert n_hooks == 0

    def test_generate_with_transformers_uses_context(self):
        """Wiring check: the production call path must deliver offsets to the
        variant, not just the standalone context manager."""
        model = _tiny_model(_alora_config())
        input_ids = torch.tensor([[10, 1, 2, 3, 40, 50]])
        tokenizer = MagicMock()
        tokenizer.eos_token_id = 9999  # outside vocab: never in generated tokens
        tokenizer.decode.side_effect = lambda *a, **k: "x"
        tokenizer.batch_decode.side_effect = lambda seqs: ["x"] * len(seqs)
        with _VariantSpy() as spy:
            generate_with_transformers(
                tokenizer,
                model,
                generate_input={
                    "input_tokens": input_ids,
                    "max_new_tokens": 2,
                    "do_sample": False,
                    "return_dict_in_generate": True,
                },
                other_input={},
            )
        assert spy.n_calls > 0
        assert spy.offsets_seen[0] == [5]


@pytest.mark.huggingface
@pytest.mark.e2e
@pytest.mark.qualitative
@pytest.mark.slow
class TestAloraDifferentialEndToEnd:
    """The aLoRA weights must measurably change the score versus no adapter.

    Guards against the silent-degradation failure mode where aLoRA
    generation falls back to base-model behaviour (issue #1679): before the
    fix, adapter-on and adapter-off scores were byte-identical on every item.
    The cardiff probe item is one where granite-4.1-3b's base model is
    confidently wrong (says "yes" to a bullet-points check on a plain
    sentence) and the published requirement-check aLoRA is correct ("no") —
    scores measured 0.047 vs 0.999 in the #1679 diagnostic eval.
    """

    def test_requirement_check_adapter_moves_score(self, gh_run):
        # Skip the expensive model download + inference on CI, following the
        # gh_run pattern used by the other huggingface e2e tests in this suite.
        if gh_run == 1:
            pytest.xfail("Model download + inference not run on CI")

        from mellea.backends.adapters._core import Adapter, Identity, LocalFileBinding
        from mellea.backends.adapters.catalog import (
            AdapterType,
            fetch_intrinsic_metadata,
        )
        from mellea.backends.adapters.io_contracts import get_io_contract
        from mellea.stdlib.components import Message
        from mellea.stdlib.components.intrinsic import core
        from mellea.stdlib.context import ChatContext

        # NOTE: the published requirement-check io.yaml instruction does not
        # tokenise to the adapter's declared invocation sequence (issue
        # #1679). LocalHFBackend repairs it at load time with a warning
        # (`_repair_composed_alora_instruction`) until the publisher
        # republishes a corrected file, so this test runs on the as-published
        # adapter with no local workaround.
        md = fetch_intrinsic_metadata("requirement-check")
        backend = LocalHFBackend(model_id="ibm-granite/granite-4.1-3b")
        backend.add_adapter(
            Adapter(
                identity=Identity(
                    name="requirement-check",
                    adapter_type="alora",
                    capability="requirement_check",
                ),
                io_contract=get_io_contract("requirement-check"),
                weights=LocalFileBinding(
                    name="requirement-check",
                    adapter_type=AdapterType.ALORA,
                    repo_id=md.repo_id,
                    revision=md.revision,
                ),
            )
        )
        ctx = (
            ChatContext()
            .add(Message("user", "Write one sentence about Cardiff."))
            .add(Message("assistant", "Cardiff is the capital city of Wales."))
        )
        requirement = "The response uses bullet points."

        # The model-level PEFT adapter name as registered by load_adapter.
        qualified = next(iter(backend._model.peft_config))

        # Adapter ON: the intrinsic path activates the adapter itself
        # around the generate call; this is the real production flow.
        score_on = core.requirement_check(ctx, backend, requirement)

        # Adapter OFF: the intrinsic scope re-activates the adapter for
        # every call, so the only way to get a genuine no-adapter score
        # is to clear the model's active adapters at the generate level
        # (the same set_adapter([]) override the diagnostic harness used;
        # disable_adapters() is a documented no-op on this model class).
        orig_generate = backend._model.generate

        def off_generate(*a, **kw):
            backend._model.set_adapter([])
            try:
                return orig_generate(*a, **kw)
            finally:
                backend._model.set_adapter(qualified)

        backend._model.generate = off_generate
        try:
            score_off = core.requirement_check(ctx, backend, requirement)
        finally:
            backend._model.generate = orig_generate

        # The adapter must say "no" (low score); the base model says "yes".
        assert score_on < 0.5, (
            f"aLoRA active: expected 'no' (score < 0.5), got {score_on}"
        )
        assert score_off > 0.5, (
            f"base model: expected 'yes' (score > 0.5), got {score_off}"
        )
        assert score_off - score_on > 0.5, (
            f"adapter contribution too small to be a real effect: "
            f"on={score_on}, off={score_off}"
        )


@pytest.mark.huggingface
@pytest.mark.e2e
@pytest.mark.qualitative
@pytest.mark.slow
class TestUncertaintyAloraDifferentialEndToEnd:
    """Second aLoRA capability (uncertainty): confirms the activation fix is
    general, not requirement-check-specific.

    The published uncertainty aLoRA io.yaml has no tokenisation mismatch, so
    this test needs no instruction patching — it exercises the fix on the
    as-published adapter files. Measured on granite-4.1-3b in the #1679
    diagnostic: base model is underconfident (~0.06) on both a right and a
    wrong last response, while the adapter gives ~0.95 (right) and ~0.73
    (wrong) — a wide differential either way.
    """

    def test_check_certainty_adapter_moves_score(self, gh_run):
        if gh_run == 1:
            pytest.xfail("Model download + inference not run on CI")

        from mellea.backends.adapters._core import Adapter, Identity, LocalFileBinding
        from mellea.backends.adapters.catalog import (
            AdapterType,
            fetch_intrinsic_metadata,
        )
        from mellea.backends.adapters.io_contracts import get_io_contract
        from mellea.stdlib.components import Message
        from mellea.stdlib.components.intrinsic import core
        from mellea.stdlib.context import ChatContext

        md = fetch_intrinsic_metadata("uncertainty")
        backend = LocalHFBackend(model_id="ibm-granite/granite-4.1-3b")
        backend.add_adapter(
            Adapter(
                identity=Identity(
                    name="uncertainty", adapter_type="alora", capability="uncertainty"
                ),
                io_contract=get_io_contract("uncertainty"),
                weights=LocalFileBinding(
                    name="uncertainty",
                    adapter_type=AdapterType.ALORA,
                    repo_id=md.repo_id,
                    revision=md.revision,
                ),
            )
        )
        qualified = next(iter(backend._model.peft_config))

        def score_with_adapter_off(user: str, assistant: str) -> float:
            ctx = (
                ChatContext()
                .add(Message("user", user))
                .add(Message("assistant", assistant))
            )
            orig_generate = backend._model.generate

            def off_generate(*a, **kw):
                backend._model.set_adapter([])
                try:
                    return orig_generate(*a, **kw)
                finally:
                    backend._model.set_adapter(qualified)

            backend._model.generate = off_generate
            try:
                return core.check_certainty(ctx, backend)
            finally:
                backend._model.generate = orig_generate

        user = "What is the capital of France?"
        score_on_right = core.check_certainty(
            ChatContext()
            .add(Message("user", user))
            .add(Message("assistant", "The capital of France is Paris.")),
            backend,
        )
        score_off_right = score_with_adapter_off(
            user, "The capital of France is Paris."
        )

        score_on_wrong = core.check_certainty(
            ChatContext()
            .add(Message("user", user))
            .add(Message("assistant", "The capital of France is Madrid.")),
            backend,
        )
        score_off_wrong = score_with_adapter_off(
            user, "The capital of France is Madrid."
        )

        # The adapter must move the certainty score by a wide margin on both
        # cases; the base 3b model sits near 0.06 in both.
        assert score_on_right - score_off_right > 0.5, (
            f"right answer: adapter contribution too small: "
            f"on={score_on_right}, off={score_off_right}"
        )
        assert score_on_wrong - score_off_wrong > 0.5, (
            f"wrong answer: adapter contribution too small: "
            f"on={score_on_wrong}, off={score_off_wrong}"
        )
