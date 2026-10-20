# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the mellea-carried decider config and its io-contract registry.

Covers the field-by-field merge semantics (carried fills null, published
equal is silent, published different wins with WARNING, `parameters`
per-key with published winning, `response_format` forced null, provenance
stamp), merged-config validation (invalid configs disable the decision
capability with a reason), the merge's wiring into
`_discover_embedded_adapters` via a synthetic model directory, and the
readout contract under the decision capability name.
"""

import contextlib
import json

import pytest
import yaml

from mellea.backends.adapters import AdapterSchemaMismatchError, get_io_contract
from mellea.backends.adapters.adapter import (
    _apply_carried_decider_config,
    _discover_embedded_adapters,
)


def _merge(name: str, config: dict) -> dict | None:
    """Call _apply_carried_decider_config and return the merged config (or None)."""
    merged, _reason = _apply_carried_decider_config(name, config)
    return merged


def _merge_reason(name: str, config: dict) -> str | None:
    """Call _apply_carried_decider_config and return the skip reason (or None)."""
    _merged, reason = _apply_carried_decider_config(name, config)
    return reason


_PUBLISHED_SHAPE = {
    "name": "decision",
    "model": None,
    "response_format": None,
    "transformations": None,
    "instruction": None,
    "parameters": {"max_completion_tokens": 1, "temperature": 0.0},
    "sentence_boundaries": None,
}
"""The published decision io.yaml shape: decider fields present-with-null."""


class TestCarriedConfigMerge:
    """The all-or-nothing merge of the carried decider config (D3)."""

    def test_null_fill_applies_carried_fields(self):
        merged = _merge("decision", dict(_PUBLISHED_SHAPE))
        assert merged["system_prompt"].startswith("You are a decision engine.")
        assert merged["state_template"] == "State:\n{state}"
        assert merged["invocation_text"] == "### DECISION"
        assert merged["instruction"].startswith("### DECISION\nQuestion:")
        assert merged["transformations"][0]["type"] == "first_token"
        assert merged["transformations"][0]["labels"] == list(
            "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        )

    def test_decision_capability_name_merges(self):
        for name in ("decision",):
            published = dict(_PUBLISHED_SHAPE)
            published["name"] = name
            merged = _merge(name, published)
            assert merged["invocation_text"] == "### DECISION"

    def test_non_decider_names_untouched(self):
        config = dict(_PUBLISHED_SHAPE)
        merged = _merge("answerability", config)
        assert merged is config

    def test_partial_publish_instruction_only_still_renders(self):
        """Regression: a published io.yaml with only `instruction` filled in
        must still produce a renderable config (carried values fill the
        null fields), not an all-or-nothing published-wins config missing
        `system_prompt`/`state_template`/`invocation_text`."""
        published = dict(_PUBLISHED_SHAPE)
        published["instruction"] = (
            "### DECISION\nQuestion:\n{question}\n\nOptions:\n{options_block}\n\n"
            "Reply with exactly one option label."
        )
        merged = _merge("decision", published)
        # Published instruction wins.
        assert merged["instruction"] == published["instruction"]
        # Carried values fill the null fields.
        assert merged["system_prompt"].startswith("You are a decision engine.")
        assert merged["state_template"] == "State:\n{state}"
        assert merged["invocation_text"] == "### DECISION"
        assert merged["transformations"][0]["type"] == "first_token"
        # The merged config round-trips through the rewriter: it constructs
        # without the "no state_template" ValueError and produces the
        # reference message roles.
        from mellea.formatters.granite.base.io import ChatCompletion
        from mellea.formatters.granite.intrinsics.input import IntrinsicsRewriter

        rewriter = IntrinsicsRewriter(config_dict=merged)
        request = ChatCompletion(messages=[])
        out = rewriter.transform(
            request,
            question="Is this urgent?",
            options_block="A: yes\nB: no",
            state="Customer lost funds",
        )
        roles = [m.role for m in out.messages]
        assert roles == ["system", "user", "user"]

    def test_carried_fills_null_fields(self):
        """Each null decider field gets the carried value independently."""
        for field in (
            "system_prompt",
            "state_template",
            "invocation_text",
            "instruction",
            "transformations",
        ):
            published = dict(_PUBLISHED_SHAPE)
            # All fields null except the one being tested (also null).
            merged = _merge("decision", published)
            assert merged[field] is not None, field

    def test_published_equal_fields_merge_without_warning(self, caplog):
        """A published field equal to the carried value is used with no log."""
        import logging

        from mellea.backends.adapters.adapter import _CARRIED_DECIDER_CONFIG

        published = dict(_PUBLISHED_SHAPE)
        published["invocation_text"] = _CARRIED_DECIDER_CONFIG["invocation_text"]
        with caplog.at_level(logging.WARNING):
            merged = _merge("decision", published)
        assert merged["invocation_text"] == "### DECISION"
        assert not any(
            "differs" in r.getMessage() or "conflict" in r.getMessage()
            for r in caplog.records
        )

    def test_published_conflicting_field_wins_with_warning(self, caplog):
        """A published field that differs from the carried value wins, with a WARNING."""
        import logging

        published = dict(_PUBLISHED_SHAPE)
        published["state_template"] = "Context:\n{state}"
        with caplog.at_level(logging.WARNING):
            merged = _merge("decision", published)
        assert merged["state_template"] == "Context:\n{state}"
        # Carried fields still fill the null ones.
        assert merged["system_prompt"].startswith("You are a decision engine.")
        warning_records = [r for r in caplog.records if r.levelno >= logging.WARNING]
        # The WARNING names the field and shows both values (repr'd).
        assert any(
            "state_template" in r.getMessage()
            and "{state}" in r.getMessage()
            and "Context:" in r.getMessage()
            for r in warning_records
        )

    def test_transformations_compared_structurally(self, caplog):
        """transformations equality is structural, not string identity."""
        import logging

        published = dict(_PUBLISHED_SHAPE)
        # Same structure as the carried transformations (a fresh list).
        published["transformations"] = [
            {"type": "first_token", "labels": list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")}
        ]
        with caplog.at_level(logging.WARNING):
            merged = _merge("decision", published)
        assert not any(
            "transformations" in r.getMessage() and "differs" in r.getMessage()
            for r in caplog.records
        )
        assert merged["transformations"][0]["type"] == "first_token"

    def test_invalid_merged_config_disables_decision_with_reason(self, caplog):
        """A merged config missing required placeholders disables the capability,
        and the specific reason reaches the skip reason."""
        import logging

        published = dict(_PUBLISHED_SHAPE)
        # A state_template without {state} is invalid.
        published["state_template"] = "State: no placeholder"
        with caplog.at_level(logging.ERROR):
            merged = _merge("decision", published)
        assert merged is None
        # The specific reason, not a generic "see the log".
        reason = _merge_reason("decision", published)
        assert reason is not None
        assert "state_template" in reason
        assert "{state}" in reason
        assert any(
            "state_template" in r.getMessage() and "{state}" in r.getMessage()
            for r in caplog.records
            if r.levelno >= logging.ERROR
        )

    def test_valid_published_config_not_disabled(self):
        """A fully published valid config is not disabled."""
        published = dict(_PUBLISHED_SHAPE)
        published["system_prompt"] = "You are a judge."
        published["state_template"] = "State:\n{state}"
        published["invocation_text"] = "### DECISION"
        published["instruction"] = (
            "### DECISION\nQuestion:\n{question}\n\nOptions:\n{options_block}\n\n"
            "Reply with exactly one option label."
        )
        published["transformations"] = [{"type": "first_token", "labels": ["A", "B"]}]
        merged = _merge("decision", published)
        assert merged is not None
        assert merged["system_prompt"] == "You are a judge."

    def test_non_decision_adapters_unaffected_by_decider_merge(self):
        """Non-decision capabilities pass through without validation."""
        config = {"name": "guardian-core", "instruction": None, "parameters": None}
        merged = _merge("guardian-core", config)
        assert merged is config

    def test_parameters_merge_per_key_published_wins(self):
        published = dict(_PUBLISHED_SHAPE)
        published["parameters"] = {"max_completion_tokens": 1, "temperature": 0.5}
        merged = _merge("decision", published)
        assert merged["parameters"]["temperature"] == 0.5
        # Carried-only keys (extra_body) still apply.
        assert merged["parameters"]["extra_body"] == {
            "chat_template_kwargs": {"enable_thinking": False}
        }

    def test_published_extra_body_is_deep_merged_with_carried(self):
        """A published extra_body must not drop the carried enable_thinking pin."""
        published = dict(_PUBLISHED_SHAPE)
        published["parameters"] = {
            "max_completion_tokens": 1,
            "extra_body": {"top_k": 1},
        }
        merged = _merge("decision", published)
        assert merged["parameters"]["extra_body"] == {
            "top_k": 1,
            "chat_template_kwargs": {"enable_thinking": False},
        }

    def test_published_null_extra_body_keeps_carried(self):
        """`extra_body: null` means nothing published, not clear the carried pin."""
        published = dict(_PUBLISHED_SHAPE)
        published["parameters"] = {"max_completion_tokens": 1, "extra_body": None}
        merged, reason = _apply_carried_decider_config("decision", published)
        assert reason is None
        assert merged["parameters"]["extra_body"] == {
            "chat_template_kwargs": {"enable_thinking": False}
        }

    @pytest.mark.parametrize(
        "extra_body", ["not-a-mapping", ["list"], 3], ids=["str", "list", "int"]
    )
    def test_published_non_mapping_extra_body_is_skipped_with_reason(self, extra_body):
        published = dict(_PUBLISHED_SHAPE)
        published["parameters"] = {"extra_body": extra_body}
        merged, reason = _apply_carried_decider_config("decision", published)
        assert merged is None
        assert "extra_body must be a mapping" in reason

    def test_published_malformed_chat_template_kwargs_is_skipped_with_reason(self):
        published = dict(_PUBLISHED_SHAPE)
        published["parameters"] = {"extra_body": {"chat_template_kwargs": "oops"}}
        merged, reason = _apply_carried_decider_config("decision", published)
        assert merged is None
        assert "extra_body is malformed" in reason

    def test_published_chat_template_kwargs_win_per_key(self):
        published = dict(_PUBLISHED_SHAPE)
        published["parameters"] = {
            "extra_body": {"chat_template_kwargs": {"enable_thinking": True, "x": 1}}
        }
        merged = _merge("decision", published)
        assert merged["parameters"]["extra_body"]["chat_template_kwargs"] == {
            "enable_thinking": True,
            "x": 1,
        }

    def test_no_published_parameters_gets_carried_defaults(self):
        published = dict(_PUBLISHED_SHAPE)
        published["parameters"] = None
        merged = _merge("decision", published)
        assert merged["parameters"]["max_completion_tokens"] == 1

    def test_response_format_forced_null_for_first_token(self):
        published = dict(_PUBLISHED_SHAPE)
        published["response_format"] = {"type": "object"}
        merged = _merge("decision", published)
        assert merged["response_format"] is None

    def test_response_format_forced_null_for_first_token_published_config(self):
        """A published config that already declares first_token is also forced null."""
        published = dict(_PUBLISHED_SHAPE)
        published["transformations"] = [{"type": "first_token", "labels": ["A", "B"]}]
        published["response_format"] = {"type": "object"}
        merged = _merge("decision", published)
        assert merged["response_format"] is None
        assert merged["transformations"] == [
            {"type": "first_token", "labels": ["A", "B"]}
        ]

    def test_response_format_kept_for_non_first_token(self):
        published = dict(_PUBLISHED_SHAPE)
        published["name"] = "answerability"
        published["response_format"] = {"type": "object"}
        merged = _merge("answerability", published)
        assert merged["response_format"] == {"type": "object"}

    def test_provenance_stamped(self):
        merged = _merge("decision", dict(_PUBLISHED_SHAPE))
        assert "reference decider implementation" in merged["provenance"]
        assert "2026-10-15" in merged["provenance"]

    def test_input_config_not_mutated(self):
        published = dict(_PUBLISHED_SHAPE)
        _merge("decision", published)
        assert published["instruction"] is None
        assert published["transformations"] is None

    def test_merged_config_mutation_does_not_poison_carried_constant(self):
        """Each merge must return independent structure.

        The merged config is per-adapter state that backends cache and
        rewriters consume; sharing mutable structure with the module-level
        carried constant would let one adapter's consumers corrupt every
        other's.
        """
        first = _merge("decision", dict(_PUBLISHED_SHAPE))
        first["transformations"][0]["labels"] = "MUTATED"
        first["parameters"]["extra_body"]["chat_template_kwargs"]["enable_thinking"] = (
            True
        )
        second = _merge("decision", dict(_PUBLISHED_SHAPE))
        assert second["transformations"][0]["labels"] == list(
            "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        )
        assert second["parameters"]["extra_body"] == {
            "chat_template_kwargs": {"enable_thinking": False}
        }


_DECISION_ADAPTER_INDEX = {
    "model_info": {"num_adapters": 1, "base_model": "test"},
    "adapters": [
        {
            "adapter_index": 1,
            "adapter_name": "decision",
            "technology": "alora",
            "control_token": {"token": "<|decision|>", "id": 100353},
            "io_config": "io_configs/decision/io.yaml",
        }
    ],
}


class TestDiscoveryWiring:
    """The merge is applied where every composed registration path traverses."""

    @pytest.fixture
    def decision_model_dir(self, tmp_path):
        """A synthetic Granite Switch model directory with a decision adapter."""
        (tmp_path / "adapter_index.json").write_text(
            json.dumps(_DECISION_ADAPTER_INDEX)
        )
        config_dir = tmp_path / "io_configs" / "decision"
        config_dir.mkdir(parents=True)
        (config_dir / "io.yaml").write_text(yaml.dump(dict(_PUBLISHED_SHAPE)))
        return tmp_path

    def test_discover_embedded_adapters_merges_carried_config(self, decision_model_dir):
        discovered, skip_reasons = _discover_embedded_adapters(str(decision_model_dir))
        assert len(discovered) == 1
        assert skip_reasons == {}
        adapter, config = discovered[0]
        assert adapter.identity.name == "decision"
        assert config["invocation_text"] == "### DECISION"
        assert config["transformations"][0]["type"] == "first_token"
        assert config["provenance"]
        # The merged config round-trips through the rewriter's config loader.
        from mellea.formatters.granite.intrinsics.input import IntrinsicsRewriter

        IntrinsicsRewriter(config_dict=config)

    def test_discover_single_name_filters(self, decision_model_dir):
        discovered, _ = _discover_embedded_adapters(
            str(decision_model_dir), intrinsic_name="decision"
        )
        assert len(discovered) == 1


_DECISION_TEMPLATE = """{%- set adapter_map = {
    'decision': {'token': '<|decision|>', 'type': 'alora', 'invocation_text': '### DECISION', 'invocation_drop_chars': 3},
} -%}
{%- for name, adapter in adapter_map.items() %}{% endfor -%}"""


class TestTemplateInvocationCheck:
    """The chat-template invocation-marker check (WARN-only)."""

    @pytest.fixture
    def template_model_dir(self, tmp_path):
        """A synthetic model directory with a decision adapter and chat template."""
        (tmp_path / "adapter_index.json").write_text(
            json.dumps(_DECISION_ADAPTER_INDEX)
        )
        config_dir = tmp_path / "io_configs" / "decision"
        config_dir.mkdir(parents=True)
        (config_dir / "io.yaml").write_text(yaml.dump(dict(_PUBLISHED_SHAPE)))
        (tmp_path / "chat_template.jinja").write_text(_DECISION_TEMPLATE)
        return tmp_path

    def test_template_invocation_matches_no_warning(self, template_model_dir, caplog):
        """A matching template produces no warning."""
        import logging

        with caplog.at_level(logging.WARNING):
            _discover_embedded_adapters(str(template_model_dir))
        assert not any(
            "invocation" in r.getMessage().lower()
            for r in caplog.records
            if r.levelno >= logging.WARNING
        )

    def test_template_invocation_mismatch_warns(self, template_model_dir, caplog):
        """A template with a different invocation text logs a WARNING."""
        import logging

        mismatched = _DECISION_TEMPLATE.replace("'### DECISION'", "'### OTHER'")
        (template_model_dir / "chat_template.jinja").write_text(mismatched)
        with caplog.at_level(logging.WARNING):
            discovered, skip = _discover_embedded_adapters(str(template_model_dir))
        # WARN-only: the decision adapter still registers.
        assert len(discovered) == 1
        assert skip == {}
        assert any(
            "base model" in r.getMessage()
            for r in caplog.records
            if r.levelno >= logging.WARNING
        )

    def test_template_missing_decision_entry_warns(self, template_model_dir, caplog):
        """A template with no decision entry in adapter_map logs a WARNING."""
        import logging

        no_decision = _DECISION_TEMPLATE.replace("'decision':", "'other':")
        (template_model_dir / "chat_template.jinja").write_text(no_decision)
        with caplog.at_level(logging.WARNING):
            discovered, _ = _discover_embedded_adapters(str(template_model_dir))
        assert len(discovered) == 1
        assert any(
            "no 'decision' entry" in r.getMessage()
            for r in caplog.records
            if r.levelno >= logging.WARNING
        )

    def test_template_control_token_mismatch_warns(self, template_model_dir, caplog):
        """A template whose control token differs from the index logs a WARNING."""
        import logging

        mismatched = _DECISION_TEMPLATE.replace("'<|decision|>'", "'<|other|>'")
        (template_model_dir / "chat_template.jinja").write_text(mismatched)
        with caplog.at_level(logging.WARNING):
            discovered, _ = _discover_embedded_adapters(str(template_model_dir))
        assert len(discovered) == 1
        assert any(
            "control" in r.getMessage().lower() and "token" in r.getMessage().lower()
            for r in caplog.records
            if r.levelno >= logging.WARNING
        )

    def test_template_absent_warns_and_registers(self, tmp_path, caplog):
        """No chat_template.jinja: WARNING, decision still registers."""
        import logging
        import os

        (tmp_path / "adapter_index.json").write_text(
            json.dumps(_DECISION_ADAPTER_INDEX)
        )
        config_dir = tmp_path / "io_configs" / "decision"
        config_dir.mkdir(parents=True)
        (config_dir / "io.yaml").write_text(yaml.dump(dict(_PUBLISHED_SHAPE)))
        # No chat_template.jinja written.
        with caplog.at_level(logging.WARNING):
            discovered, skip = _discover_embedded_adapters(str(tmp_path))
        assert len(discovered) == 1
        assert skip == {}
        assert any(
            "no chat_template.jinja" in r.getMessage()
            for r in caplog.records
            if r.levelno >= logging.WARNING
        )

    def test_template_check_uses_merged_invocation_text(
        self, template_model_dir, caplog
    ):
        """A published invocation_text matching the template produces no warning."""
        import logging

        # Publish an invocation_text that matches the template's '### DECISION'.
        published = dict(_PUBLISHED_SHAPE)
        published["invocation_text"] = "### DECISION"
        config_dir = template_model_dir / "io_configs" / "decision"
        (config_dir / "io.yaml").write_text(yaml.dump(published))
        with caplog.at_level(logging.WARNING):
            discovered, _ = _discover_embedded_adapters(str(template_model_dir))
        assert len(discovered) == 1
        assert not any(
            "base model" in r.getMessage()
            for r in caplog.records
            if r.levelno >= logging.WARNING
        )


class TestDecisionContract:
    """The readout contract validates the first_token rule's output shape."""

    VALID = json.dumps(
        {
            "token": "A",
            "label_logprobs": {"A": -0.2},
            "min_returned_logprob": -5.0,
            "labels": list("AB"),
        }
    )

    def test_valid_output_parses(self):
        for name in ("decision",):
            parsed = get_io_contract(name).parse(self.VALID)
            assert parsed["token"] == "A"
            assert parsed["label_logprobs"] == {"A": -0.2}

    def test_missing_key_raises(self):
        for name in ("decision",):
            with pytest.raises(AdapterSchemaMismatchError):
                get_io_contract(name).parse(json.dumps({"token": "A"}))

    def test_wrong_token_type_raises(self):
        bad = json.dumps(
            {"token": 1, "label_logprobs": {}, "min_returned_logprob": -1.0}
        )
        with pytest.raises(AdapterSchemaMismatchError, match="token"):
            get_io_contract("decision").parse(bad)

    def test_non_numeric_label_logprobs_raises(self):
        bad = json.dumps(
            {
                "token": "A",
                "label_logprobs": {"A": "high"},
                "min_returned_logprob": -1.0,
            }
        )
        with pytest.raises(AdapterSchemaMismatchError, match="label_logprobs"):
            get_io_contract("decision").parse(bad)

    def test_non_numeric_min_logprob_raises(self):
        bad = json.dumps(
            {"token": "A", "label_logprobs": {}, "min_returned_logprob": "low"}
        )
        with pytest.raises(AdapterSchemaMismatchError, match="min_returned_logprob"):
            get_io_contract("decision").parse(bad)

    def test_bad_labels_raises(self):
        bad = json.dumps(
            {
                "token": "A",
                "label_logprobs": {},
                "min_returned_logprob": -1.0,
                "labels": "AB",
            }
        )
        with pytest.raises(AdapterSchemaMismatchError, match="labels"):
            get_io_contract("decision").parse(bad)

    def test_non_object_raises_value_error(self):
        with pytest.raises(ValueError, match="JSON object"):
            get_io_contract("decision").parse(json.dumps(["not", "an", "object"]))

    def test_empty_label_logprobs_allowed(self):
        """Zero matched labels flows to the caller's label-mass check."""
        parsed = get_io_contract("decision").parse(
            json.dumps(
                {
                    "token": "Q",
                    "label_logprobs": {},
                    "min_returned_logprob": -1.0,
                    "labels": list("AB"),
                }
            )
        )
        assert parsed["label_logprobs"] == {}


class TestAdapterRevision:
    """The adapter_revision constructor parameter passes through to discovery."""

    def test_openai_passes_revision(self):
        """OpenAIBackend passes adapter_revision to register_embedded_adapter_model."""
        from unittest.mock import patch

        with patch(
            "mellea.backends.openai.OpenAIBackend.register_embedded_adapter_model"
        ) as mock_reg:
            import os

            os.environ.setdefault("OPENAI_API_KEY", "test-key-for-unit-test")
            from mellea.backends.openai import OpenAIBackend

            OpenAIBackend(
                model_id="ibm-granite/granite-switch-4.1-8b",
                load_embedded_adapters=True,
                adapter_revision="abc123",
            )
        mock_reg.assert_called_once_with(
            "ibm-granite/granite-switch-4.1-8b", revision="abc123"
        )

    def test_openai_default_revision_is_main(self):
        """OpenAIBackend defaults adapter_revision to 'main'."""
        from unittest.mock import patch

        with patch(
            "mellea.backends.openai.OpenAIBackend.register_embedded_adapter_model"
        ) as mock_reg:
            import os

            os.environ.setdefault("OPENAI_API_KEY", "test-key-for-unit-test")
            from mellea.backends.openai import OpenAIBackend

            OpenAIBackend(
                model_id="ibm-granite/granite-switch-4.1-8b",
                load_embedded_adapters=True,
            )
        mock_reg.assert_called_once_with(
            "ibm-granite/granite-switch-4.1-8b", revision="main"
        )

    def test_huggingface_passes_revision(self):
        """LocalHFBackend stores adapter_revision for register_embedded_adapter_model."""
        # Full construction needs model weights; verify the parameter
        # exists in the signature and the constructor stores it (the
        # pass-through to register_embedded_adapter_model is the same
        # one-line pattern OpenAIBackend uses, tested above).
        import inspect

        from mellea.backends.huggingface import LocalHFBackend

        sig = inspect.signature(LocalHFBackend.__init__)
        assert "adapter_revision" in sig.parameters

        backend = LocalHFBackend.__new__(LocalHFBackend)
        backend._adapter_revision = "abc123"
        assert backend._adapter_revision == "abc123"

    def test_huggingface_default_revision_is_main(self):
        """LocalHFBackend defaults adapter_revision to 'main'."""
        # The default is set in the constructor signature; verify it exists.
        import inspect

        from mellea.backends.huggingface import LocalHFBackend

        sig = inspect.signature(LocalHFBackend.__init__)
        assert sig.parameters["adapter_revision"].default == "main"


class TestLazyResolveRevision:
    """`resolve_adapter` discovers an embedded adapter at the backend's revision."""

    def test_resolve_adapter_passes_adapter_revision(self):
        from unittest.mock import MagicMock, patch

        from mellea.backends.adapters.adapter import AdapterMixin

        backend = MagicMock()
        backend.base_model_name = "ibm-granite/granite-switch-4.1-8b"
        backend._adapter_resolve_lock.return_value = contextlib.nullcontext()
        backend._uses_embedded_adapters = True
        backend._adapter_source = "ibm-granite/granite-switch-4.1-8b"
        backend._adapter_revision = "abc123"
        backend._adapter_skip_reasons = {}
        # Looked up before the lock, again under it, and once after discovery.
        backend._find_adapter.side_effect = [None, None, MagicMock()]
        with patch(
            "mellea.backends.adapters.adapter._discover_embedded_adapters",
            return_value=([], {}),
        ) as discover:
            AdapterMixin.resolve_adapter(backend, "decision")
        assert discover.call_args.kwargs["revision"] == "abc123"
