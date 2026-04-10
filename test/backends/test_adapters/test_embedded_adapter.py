"""Tests for EmbeddedIntrinsicAdapter and OpenAI backend integration."""

import json
import os
import pathlib
import tempfile

import pytest
import yaml

from mellea.backends.adapters.adapter import EmbeddedIntrinsicAdapter
from mellea.backends.adapters.catalog import AdapterType


_TEST_DIR = pathlib.Path(__file__).parent
_INTRINSICS_DATA = _TEST_DIR / "intrinsics-data"

# Sample adapter_index.json for testing from_model_directory
_SAMPLE_ADAPTER_INDEX = {
    "model_info": {"num_adapters": 2, "base_model": "granite-4.0-micro"},
    "adapters": [
        {
            "adapter_index": 1,
            "intrinsic_name": "answerability",
            "technology": "alora",
            "io_config": "io_configs/answerability/io.yaml",
            "control_token": {
                "token": "<answerability>",
                "token_visible": "<answerability_visible>",
                "id": 100366,
                "id_visible": 100367,
            },
        },
        {
            "adapter_index": 2,
            "intrinsic_name": "citations",
            "technology": "lora",
            "io_config": "io_configs/citations/io.yaml",
            "control_token": {
                "token": "<citations>",
                "token_visible": "<citations_visible>",
                "id": 100354,
                "id_visible": 100355,
            },
        },
    ],
}

_ANSWERABILITY_CONFIG = yaml.safe_load(
    (_INTRINSICS_DATA / "answerability.yaml").read_text()
)

# Minimal citations config for testing
_CITATIONS_CONFIG = {
    "model": None,
    "response_format": '{"type": "array", "items": {"type": "object"}}',
    "transformations": None,
    "instruction": "Find citations.",
    "parameters": {"max_completion_tokens": 4096},
    "sentence_boundaries": {"last_message": "r", "documents": "c"},
}


class TestEmbeddedIntrinsicAdapterInit:
    def test_basic_init(self):
        adapter = EmbeddedIntrinsicAdapter(
            intrinsic_name="answerability",
            config=_ANSWERABILITY_CONFIG,
            technology="alora",
        )
        assert adapter.intrinsic_name == "answerability"
        assert adapter.name == "answerability"
        assert adapter.technology == "alora"
        assert adapter.adapter_type == AdapterType.ALORA
        assert adapter.qualified_name == "answerability_alora"
        assert adapter.config is not None
        assert adapter.config["parameters"]["max_completion_tokens"] == 6

    def test_lora_technology(self):
        adapter = EmbeddedIntrinsicAdapter(
            intrinsic_name="citations",
            config=_CITATIONS_CONFIG,
            technology="lora",
        )
        assert adapter.adapter_type == AdapterType.LORA
        assert adapter.qualified_name == "citations_lora"

    def test_default_technology_is_lora(self):
        adapter = EmbeddedIntrinsicAdapter(
            intrinsic_name="test", config={"model": None}
        )
        assert adapter.technology == "lora"
        assert adapter.adapter_type == AdapterType.LORA


class TestFromModelDirectory:
    def test_loads_all_adapters(self, tmp_path):
        """Load adapters from a mock model directory."""
        # Write adapter_index.json
        (tmp_path / "adapter_index.json").write_text(
            json.dumps(_SAMPLE_ADAPTER_INDEX)
        )

        # Write io configs
        ans_dir = tmp_path / "io_configs" / "answerability"
        ans_dir.mkdir(parents=True)
        (ans_dir / "io.yaml").write_text(yaml.dump(_ANSWERABILITY_CONFIG))

        cit_dir = tmp_path / "io_configs" / "citations"
        cit_dir.mkdir(parents=True)
        (cit_dir / "io.yaml").write_text(yaml.dump(_CITATIONS_CONFIG))

        adapters = EmbeddedIntrinsicAdapter.from_model_directory(tmp_path)

        assert len(adapters) == 2
        names = {a.intrinsic_name for a in adapters}
        assert names == {"answerability", "citations"}

        ans = next(a for a in adapters if a.intrinsic_name == "answerability")
        assert ans.technology == "alora"
        assert ans.config["parameters"]["max_completion_tokens"] == 6

        cit = next(a for a in adapters if a.intrinsic_name == "citations")
        assert cit.technology == "lora"

    def test_missing_adapter_index(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="adapter_index.json"):
            EmbeddedIntrinsicAdapter.from_model_directory(tmp_path)

    def test_missing_io_yaml(self, tmp_path):
        (tmp_path / "adapter_index.json").write_text(
            json.dumps(_SAMPLE_ADAPTER_INDEX)
        )
        # Don't create io_configs — should raise ValueError
        with pytest.raises(ValueError, match="io.yaml.*not found"):
            EmbeddedIntrinsicAdapter.from_model_directory(tmp_path)

    def test_loads_from_real_granite_switch_model(self):
        """Smoke test against the actual built model if available."""
        model_path = pathlib.Path(
            "/proj/dmfexp/lastrasl/granite-switch/modular-granite"
        )
        if not (model_path / "adapter_index.json").exists():
            pytest.skip("Granite Switch model not available")

        adapters = EmbeddedIntrinsicAdapter.from_model_directory(model_path)
        assert len(adapters) == 8
        names = {a.intrinsic_name for a in adapters}
        assert "answerability" in names
        assert "citations" in names


class TestOpenAIBackendRegistration:
    @pytest.fixture
    def backend(self):
        os.environ.setdefault("OPENAI_API_KEY", "test-key")
        from mellea.backends.openai import OpenAIBackend

        return OpenAIBackend(
            model_id="granite-switch",
            base_url="http://localhost:8000/v1",
        )

    def test_add_embedded_adapter(self, backend):
        adapter = EmbeddedIntrinsicAdapter(
            intrinsic_name="answerability",
            config=_ANSWERABILITY_CONFIG,
            technology="alora",
        )
        backend.add_embedded_adapter(adapter)
        assert "answerability" in backend._embedded_adapters
        assert backend._embedded_adapters["answerability"] is adapter

    def test_register_granite_switch_model(self, backend, tmp_path):
        # Set up mock model directory
        (tmp_path / "adapter_index.json").write_text(
            json.dumps(_SAMPLE_ADAPTER_INDEX)
        )
        ans_dir = tmp_path / "io_configs" / "answerability"
        ans_dir.mkdir(parents=True)
        (ans_dir / "io.yaml").write_text(yaml.dump(_ANSWERABILITY_CONFIG))
        cit_dir = tmp_path / "io_configs" / "citations"
        cit_dir.mkdir(parents=True)
        (cit_dir / "io.yaml").write_text(yaml.dump(_CITATIONS_CONFIG))

        names = backend.register_granite_switch_model(str(tmp_path))

        assert set(names) == {"answerability", "citations"}
        assert len(backend._embedded_adapters) == 2

    def test_register_overwrites_existing(self, backend):
        config1 = {"model": None, "parameters": {"max_completion_tokens": 10}}
        config2 = {"model": None, "parameters": {"max_completion_tokens": 20}}

        backend.add_embedded_adapter(
            EmbeddedIntrinsicAdapter("test", config=config1)
        )
        backend.add_embedded_adapter(
            EmbeddedIntrinsicAdapter("test", config=config2)
        )

        assert (
            backend._embedded_adapters["test"].config["parameters"][
                "max_completion_tokens"
            ]
            == 20
        )


class TestIntrinsicRewriting:
    """Test that IntrinsicsRewriter works correctly with embedded adapter configs."""

    def test_rewriter_with_answerability_config(self):
        from mellea.formatters.granite import IntrinsicsRewriter

        rewriter = IntrinsicsRewriter(config_dict=_ANSWERABILITY_CONFIG)
        request = {
            "messages": [
                {"role": "user", "content": "Can you answer this question?"}
            ],
            "extra_body": {"documents": [{"text": "Some document."}]},
        }
        rewritten = rewriter.transform(request)

        # The rewriter should set max_completion_tokens from config
        assert rewritten is not None
        assert len(rewritten.messages) >= 1

    def test_rewriter_with_citations_config(self):
        pytest.importorskip("nltk", reason="nltk required for sentence boundaries")
        from mellea.formatters.granite import IntrinsicsRewriter

        rewriter = IntrinsicsRewriter(config_dict=_CITATIONS_CONFIG)
        request = {
            "messages": [
                {"role": "user", "content": "What does the doc say?"},
                {"role": "assistant", "content": "The doc says something."},
            ],
            "extra_body": {
                "documents": [{"text": "Some document content.", "doc_id": "0"}]
            },
        }
        rewritten = rewriter.transform(request)

        assert rewritten is not None
        # Citations uses sentence boundaries, so messages should be modified
        assert len(rewritten.messages) >= 2


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
