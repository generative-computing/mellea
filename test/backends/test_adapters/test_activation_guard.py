# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the aLoRA activation guard (issue #1678).

An aLoRA only applies from the point its declared `alora_invocation_tokens`
sequence appears in the assembled prompt; PEFT gives up silently when it is
absent (`peft.tuners.lora.variants.calculate_alora_offsets`), so a call can
report success while the adapter contributed nothing. These tests cover both
arms called out by the issue's acceptance criteria: a sequence that is present
activates normally, and a sequence that is absent raises before generation
runs rather than being silently accepted.

Two layers are exercised:

* `alora_invocation_sequence_present` — the pure token-subsequence search,
  independent of any backend.
* `mellea.backends.huggingface._check_alora_activation` — the
  generation-time guard wired into `LocalHFBackend._generate_from_intrinsic`,
  using fakes for the model/tokenizer rather than a real Hugging Face model.
  Raises `AloraActivationError` on a mismatch (see that class's docstring for
  who catches it and who lets it propagate); it does not decide recovery
  itself.
"""

import logging
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

torch = pytest.importorskip("torch", reason="torch not installed — install mellea[hf]")

from mellea.backends import huggingface
from mellea.backends.adapters._core import (
    Adapter as _AdapterCore,
    AloraActivationError,
    EmbeddedBinding,
    Identity,
    LocalFileBinding,
)
from mellea.backends.adapters.adapter import (
    EmbeddedIntrinsicAdapter,
    IntrinsicAdapter,
    alora_invocation_sequence_present,
)
from mellea.backends.adapters.catalog import AdapterType

# The requirement-check aLoRA's real declared invocation sequence (#1679):
# decodes to "<requirements>".
_INVOCATION_TOKENS = [27, 71226, 29]


class TestAloraInvocationSequencePresent:
    """Pure subsequence-search tests, independent of any backend."""

    def test_present_in_middle_of_prompt(self):
        prompt = [1, 2, 27, 71226, 29, 3, 4]
        assert alora_invocation_sequence_present(prompt, _INVOCATION_TOKENS)

    def test_present_at_start(self):
        prompt = [27, 71226, 29, 3, 4]
        assert alora_invocation_sequence_present(prompt, _INVOCATION_TOKENS)

    def test_present_at_end(self):
        prompt = [1, 2, 27, 71226, 29]
        assert alora_invocation_sequence_present(prompt, _INVOCATION_TOKENS)

    def test_absent_tokeniser_merge_case(self):
        # The #1679 case: '<requirements>: X' merges '>' and ':' into one
        # token (27916), so the exact 3-token sequence never occurs even
        # though 27 and 71226 both appear.
        prompt = [1, 27, 71226, 27916, 1630]
        assert not alora_invocation_sequence_present(prompt, _INVOCATION_TOKENS)

    def test_absent_when_tokens_not_contiguous(self):
        prompt = [27, 1, 71226, 1, 29]
        assert not alora_invocation_sequence_present(prompt, _INVOCATION_TOKENS)

    def test_absent_from_short_prompt(self):
        assert not alora_invocation_sequence_present([27, 71226], _INVOCATION_TOKENS)

    def test_empty_invocation_tokens_is_vacuously_present(self):
        assert alora_invocation_sequence_present([1, 2, 3], [])

    def test_empty_prompt_with_nonempty_invocation_is_absent(self):
        assert not alora_invocation_sequence_present([], _INVOCATION_TOKENS)

    def test_single_token_sequence(self):
        assert alora_invocation_sequence_present([1, 2, 3], [2])
        assert not alora_invocation_sequence_present([1, 2, 3], [9])


def _identity(adapter_type: str) -> Identity:
    return Identity(name="requirement-check", adapter_type=adapter_type)  # type: ignore[arg-type]


class TestQualifiedNameForActivationCheck:
    """`_alora_qualified_name_for_activation_check` gates which adapter shapes are checked."""

    def test_intrinsic_adapter_alora_returns_qualified_name(self):
        adapter = IntrinsicAdapter.__new__(IntrinsicAdapter)
        adapter.qualified_name = "requirement-check_alora"
        object.__setattr__(adapter, "identity", _identity("alora"))

        result = huggingface._alora_qualified_name_for_activation_check(adapter)
        assert result == "requirement-check_alora"

    def test_intrinsic_adapter_lora_is_skipped(self):
        adapter = IntrinsicAdapter.__new__(IntrinsicAdapter)
        adapter.qualified_name = "requirement-check_lora"
        object.__setattr__(adapter, "identity", _identity("lora"))

        assert huggingface._alora_qualified_name_for_activation_check(adapter) is None

    def test_embedded_adapter_is_always_skipped(self):
        with pytest.warns(DeprecationWarning):
            adapter = EmbeddedIntrinsicAdapter(
                intrinsic_name="requirement-check",
                config={"parameters": {}},
                technology="alora",
            )
        assert huggingface._alora_qualified_name_for_activation_check(adapter) is None

    def test_composed_adapter_alora_with_local_file_binding(self):
        binding = LocalFileBinding(
            name="requirement-check",
            adapter_type=AdapterType.ALORA,
            repo_id="acme/adapters",
            revision="main",
        )
        adapter = _AdapterCore(
            identity=_identity("alora"), io_contract=MagicMock(), weights=binding
        )
        result = huggingface._alora_qualified_name_for_activation_check(adapter)
        assert result == binding.qualified_name

    def test_composed_adapter_lora_is_skipped(self):
        binding = LocalFileBinding(
            name="requirement-check",
            adapter_type=AdapterType.LORA,
            repo_id="acme/adapters",
            revision="main",
        )
        adapter = _AdapterCore(
            identity=_identity("lora"), io_contract=MagicMock(), weights=binding
        )
        assert huggingface._alora_qualified_name_for_activation_check(adapter) is None

    def test_composed_adapter_embedded_binding_is_skipped(self):
        adapter = _AdapterCore(
            identity=_identity("alora"),
            io_contract=MagicMock(),
            weights=EmbeddedBinding(),
        )
        assert huggingface._alora_qualified_name_for_activation_check(adapter) is None


def _make_peft_config_model(qualified_name: str, invocation_tokens: list[int]):
    """A fake `self._model` carrying one loaded PEFT adapter config."""
    cfg = SimpleNamespace(alora_invocation_tokens=invocation_tokens)
    return SimpleNamespace(peft_config={qualified_name: cfg})


def _make_tokenizer():
    tok = MagicMock()
    tok.decode.return_value = "<requirements>"
    return tok


class TestCheckAloraActivation:
    """`_check_alora_activation` — the generation-time guard itself.

    On a mismatch this raises `AloraActivationError` every time (correctness
    cannot be deduplicated away), while the accompanying log line is still
    deduplicated via `warned_about` (noise control only).
    """

    def test_present_sequence_does_not_raise(self, caplog):
        model = _make_peft_config_model("requirement-check_alora", _INVOCATION_TOKENS)
        tokenizer = _make_tokenizer()
        warned_about: set[str] = set()

        prompt = torch.tensor([[1, 2, 27, 71226, 29, 3]])
        with caplog.at_level(logging.WARNING):
            huggingface._check_alora_activation(
                model=model,
                tokenizer=tokenizer,
                base_model_name="granite-4.1-3b",
                warned_about=warned_about,
                capability_name="requirement-check",
                qualified_name="requirement-check_alora",
                prompt_token_ids=prompt,
            )

        assert not any("never activates" in r.message for r in caplog.records)
        assert warned_about == set()

    def test_absent_sequence_raises_every_call_but_warns_once(self, caplog):
        model = _make_peft_config_model("requirement-check_alora", _INVOCATION_TOKENS)
        tokenizer = _make_tokenizer()
        warned_about: set[str] = set()

        # No occurrence of [27, 71226, 29] anywhere in this prompt.
        prompt = torch.tensor([[1, 27, 71226, 27916, 1630]])

        with caplog.at_level(logging.WARNING):
            with pytest.raises(AloraActivationError) as exc_info:
                huggingface._check_alora_activation(
                    model=model,
                    tokenizer=tokenizer,
                    base_model_name="granite-4.1-3b",
                    warned_about=warned_about,
                    capability_name="requirement-check",
                    qualified_name="requirement-check_alora",
                    prompt_token_ids=prompt,
                )

        err = exc_info.value
        assert err.capability_name == "requirement-check"
        assert err.base_model_name == "granite-4.1-3b"
        assert err.qualified_name == "requirement-check_alora"
        assert err.invocation_tokens == tuple(_INVOCATION_TOKENS)
        assert "requirement-check" in str(err)
        assert "granite-4.1-3b" in str(err)

        matching = [r.message for r in caplog.records if "never activates" in r.message]
        assert len(matching) == 1
        assert "requirement-check" in matching[0]
        assert "granite-4.1-3b" in matching[0]
        assert warned_about == {"alora_no_activation_requirement-check_alora"}

        # A second call with the same persisting mismatch must raise again
        # (every call's own output would otherwise be silently wrong) but
        # must not log a second time.
        caplog.clear()
        with caplog.at_level(logging.WARNING):
            with pytest.raises(AloraActivationError):
                huggingface._check_alora_activation(
                    model=model,
                    tokenizer=tokenizer,
                    base_model_name="granite-4.1-3b",
                    warned_about=warned_about,
                    capability_name="requirement-check",
                    qualified_name="requirement-check_alora",
                    prompt_token_ids=prompt,
                )
        assert not any("never activates" in r.message for r in caplog.records)

    def test_plain_list_prompt_token_ids_also_supported(self):
        model = _make_peft_config_model("requirement-check_alora", _INVOCATION_TOKENS)
        with pytest.raises(AloraActivationError):
            huggingface._check_alora_activation(
                model=model,
                tokenizer=_make_tokenizer(),
                base_model_name="granite-4.1-3b",
                warned_about=set(),
                capability_name="requirement-check",
                qualified_name="requirement-check_alora",
                prompt_token_ids=[1, 27, 71226, 27916, 1630],
            )

    def test_no_loaded_peft_config_is_a_no_op(self, caplog):
        model = SimpleNamespace(peft_config={})
        with caplog.at_level(logging.WARNING):
            huggingface._check_alora_activation(
                model=model,
                tokenizer=_make_tokenizer(),
                base_model_name="granite-4.1-3b",
                warned_about=set(),
                capability_name="requirement-check",
                qualified_name="requirement-check_alora",
                prompt_token_ids=[1, 2, 3],
            )
        assert caplog.records == []

    def test_no_declared_invocation_tokens_is_a_no_op(self, caplog):
        model = _make_peft_config_model("requirement-check_alora", [])
        with caplog.at_level(logging.WARNING):
            huggingface._check_alora_activation(
                model=model,
                tokenizer=_make_tokenizer(),
                base_model_name="granite-4.1-3b",
                warned_about=set(),
                capability_name="requirement-check",
                qualified_name="requirement-check_alora",
                prompt_token_ids=[1, 2, 3],
            )
        assert caplog.records == []

    def test_model_without_peft_config_attribute_is_a_no_op(self, caplog):
        with caplog.at_level(logging.WARNING):
            huggingface._check_alora_activation(
                model=object(),
                tokenizer=_make_tokenizer(),
                base_model_name="granite-4.1-3b",
                warned_about=set(),
                capability_name="requirement-check",
                qualified_name="requirement-check_alora",
                prompt_token_ids=[1, 2, 3],
            )
        assert caplog.records == []

    def test_1d_tensor_prompt_token_ids_is_handled(self):
        # PEFT itself accepts a bare 1-D input_ids and unsqueezes it; make
        # sure this guard does not assume the [1, seq_len] batch shape and
        # crash instead of correctly detecting the mismatch.
        model = _make_peft_config_model("requirement-check_alora", _INVOCATION_TOKENS)
        prompt = torch.tensor([1, 27, 71226, 27916, 1630])  # 1-D, no batch dim
        with pytest.raises(AloraActivationError):
            huggingface._check_alora_activation(
                model=model,
                tokenizer=_make_tokenizer(),
                base_model_name="granite-4.1-3b",
                warned_about=set(),
                capability_name="requirement-check",
                qualified_name="requirement-check_alora",
                prompt_token_ids=prompt,
            )

    def test_decode_failure_falls_back_to_repr_in_log_not_exception(self, caplog):
        model = _make_peft_config_model("requirement-check_alora", _INVOCATION_TOKENS)
        tokenizer = MagicMock()
        tokenizer.decode.side_effect = RuntimeError("boom")

        with caplog.at_level(logging.WARNING):
            with pytest.raises(AloraActivationError) as exc_info:
                huggingface._check_alora_activation(
                    model=model,
                    tokenizer=tokenizer,
                    base_model_name="granite-4.1-3b",
                    warned_about=set(),
                    capability_name="requirement-check",
                    qualified_name="requirement-check_alora",
                    prompt_token_ids=[1, 27, 71226, 27916, 1630],
                )

        # The exception itself never calls decode(), so the tokenizer failure
        # cannot affect it.
        assert exc_info.value.invocation_tokens == tuple(_INVOCATION_TOKENS)

        matching = [r.message for r in caplog.records if "never activates" in r.message]
        assert len(matching) == 1
        assert repr(_INVOCATION_TOKENS) in matching[0]
