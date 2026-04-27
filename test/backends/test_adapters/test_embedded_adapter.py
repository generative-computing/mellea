"""Tests for EmbeddedIntrinsicAdapter and OpenAI backend integration."""

import json
import os
import pathlib
from unittest.mock import MagicMock, patch

import pytest
import yaml

from mellea.backends.adapters.adapter import EmbeddedIntrinsicAdapter
from mellea.backends.adapters.catalog import AdapterType

_TEST_DIR = pathlib.Path(__file__).parent
_INTRINSICS_DATA = _TEST_DIR / "intrinsics-data"

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

# Sample adapter_index.json for testing from_model_directory
_SAMPLE_ADAPTER_INDEX = {
    "model_info": {"num_adapters": 2, "base_model": "granite-4.0-micro"},
    "adapters": [
        {
            "adapter_index": 1,
            "adapter_name": "answerability",
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
            "adapter_name": "citations",
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


@pytest.fixture
def model_dir(tmp_path):
    """Create a mock Granite Switch model directory with adapter_index.json and io configs."""
    (tmp_path / "adapter_index.json").write_text(json.dumps(_SAMPLE_ADAPTER_INDEX))

    ans_dir = tmp_path / "io_configs" / "answerability"
    ans_dir.mkdir(parents=True)
    (ans_dir / "io.yaml").write_text(yaml.dump(_ANSWERABILITY_CONFIG))

    cit_dir = tmp_path / "io_configs" / "citations"
    cit_dir.mkdir(parents=True)
    (cit_dir / "io.yaml").write_text(yaml.dump(_CITATIONS_CONFIG))

    return tmp_path


# ---- EmbeddedIntrinsicAdapter.__init__ ----


class TestEmbeddedIntrinsicAdapterInit:
    def test_alora_technology(self):
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
        assert adapter.config is _ANSWERABILITY_CONFIG

    def test_lora_technology(self):
        adapter = EmbeddedIntrinsicAdapter(
            intrinsic_name="citations", config=_CITATIONS_CONFIG, technology="lora"
        )
        assert adapter.adapter_type == AdapterType.LORA
        assert adapter.qualified_name == "citations_lora"

    def test_default_technology_is_lora(self):
        adapter = EmbeddedIntrinsicAdapter(
            intrinsic_name="test", config={"model": None}
        )
        assert adapter.technology == "lora"
        assert adapter.adapter_type == AdapterType.LORA

    def test_invalid_technology_raises(self):
        with pytest.raises(ValueError, match="must be 'lora' or 'alora'"):
            EmbeddedIntrinsicAdapter(
                intrinsic_name="test", config={"model": None}, technology="qlora"
            )

    def test_inherited_adapter_defaults(self):
        adapter = EmbeddedIntrinsicAdapter(
            intrinsic_name="test", config={"model": None}
        )
        assert adapter.backend is None
        assert adapter.path is None


# ---- EmbeddedIntrinsicAdapter.from_model_directory ----


class TestFromModelDirectory:
    def test_loads_all_adapters(self, model_dir):
        adapters = EmbeddedIntrinsicAdapter.from_model_directory(model_dir)

        assert len(adapters) == 2
        names = {a.intrinsic_name for a in adapters}
        assert names == {"answerability", "citations"}

        ans = next(a for a in adapters if a.intrinsic_name == "answerability")
        assert ans.technology == "alora"
        assert ans.config["parameters"]["max_completion_tokens"] == 6

        cit = next(a for a in adapters if a.intrinsic_name == "citations")
        assert cit.technology == "lora"

    def test_accepts_string_path(self, model_dir):
        adapters = EmbeddedIntrinsicAdapter.from_model_directory(str(model_dir))
        assert len(adapters) == 2

    def test_missing_adapter_index(self, tmp_path):
        with pytest.raises(FileNotFoundError, match=r"adapter_index\.json"):
            EmbeddedIntrinsicAdapter.from_model_directory(tmp_path)

    def test_missing_io_yaml(self, tmp_path):
        (tmp_path / "adapter_index.json").write_text(json.dumps(_SAMPLE_ADAPTER_INDEX))
        with pytest.raises(ValueError, match=r"io\.yaml.*not found"):
            EmbeddedIntrinsicAdapter.from_model_directory(tmp_path)

    def test_skips_entry_without_io_config(self, tmp_path):
        """Entries with io_config=None are silently skipped."""
        index = {
            "adapters": [
                {"adapter_name": "no_config", "technology": "lora"},
                {
                    "adapter_name": "has_config",
                    "technology": "lora",
                    "io_config": "io_configs/has_config/io.yaml",
                },
            ]
        }
        (tmp_path / "adapter_index.json").write_text(json.dumps(index))
        cfg_dir = tmp_path / "io_configs" / "has_config"
        cfg_dir.mkdir(parents=True)
        (cfg_dir / "io.yaml").write_text(yaml.dump({"model": None}))

        adapters = EmbeddedIntrinsicAdapter.from_model_directory(tmp_path)
        assert len(adapters) == 1
        assert adapters[0].intrinsic_name == "has_config"

    def test_defaults_technology_to_lora(self, tmp_path):
        """Entries without a 'technology' key default to lora."""
        index = {
            "adapters": [
                {
                    "adapter_name": "test",
                    "io_config": "io_configs/test/io.yaml",
                    # no "technology" key
                }
            ]
        }
        (tmp_path / "adapter_index.json").write_text(json.dumps(index))
        cfg_dir = tmp_path / "io_configs" / "test"
        cfg_dir.mkdir(parents=True)
        (cfg_dir / "io.yaml").write_text(yaml.dump({"model": None}))

        adapters = EmbeddedIntrinsicAdapter.from_model_directory(tmp_path)
        assert len(adapters) == 1
        assert adapters[0].technology == "lora"

    def test_empty_adapters_list(self, tmp_path):
        (tmp_path / "adapter_index.json").write_text(json.dumps({"adapters": []}))
        with pytest.raises(ValueError, match="No adapters found"):
            EmbeddedIntrinsicAdapter.from_model_directory(tmp_path)

    def test_no_adapters_key(self, tmp_path):
        """Index with no 'adapters' key raises ValueError."""
        (tmp_path / "adapter_index.json").write_text(json.dumps({}))
        with pytest.raises(ValueError, match="No adapters found"):
            EmbeddedIntrinsicAdapter.from_model_directory(tmp_path)

    def test_filter_single_intrinsic(self, model_dir):
        adapters = EmbeddedIntrinsicAdapter.from_model_directory(
            model_dir, intrinsic_name="answerability"
        )
        assert len(adapters) == 1
        assert adapters[0].intrinsic_name == "answerability"

    def test_filter_nonexistent_intrinsic(self, model_dir):
        with pytest.raises(
            ValueError, match="No adapter found for intrinsic 'nonexistent'"
        ):
            EmbeddedIntrinsicAdapter.from_model_directory(
                model_dir, intrinsic_name="nonexistent"
            )

    def test_adapter_name_key(self, tmp_path):
        """Index with 'adapter_name' key is read correctly."""
        index = {
            "adapters": [
                {
                    "adapter_name": "answerability",
                    "technology": "alora",
                    "io_config": "io_configs/answerability/io.yaml",
                }
            ]
        }
        (tmp_path / "adapter_index.json").write_text(json.dumps(index))
        cfg_dir = tmp_path / "io_configs" / "answerability"
        cfg_dir.mkdir(parents=True)
        (cfg_dir / "io.yaml").write_text(yaml.dump(_ANSWERABILITY_CONFIG))

        adapters = EmbeddedIntrinsicAdapter.from_model_directory(tmp_path)
        assert len(adapters) == 1
        assert adapters[0].intrinsic_name == "answerability"


# ---- EmbeddedIntrinsicAdapter.from_hub ----


class TestFromHub:
    def test_downloads_and_delegates(self, model_dir):
        """from_hub calls snapshot_download then delegates to from_model_directory."""
        with patch(
            "huggingface_hub.snapshot_download", return_value=str(model_dir)
        ) as mock_dl:
            adapters = EmbeddedIntrinsicAdapter.from_hub(
                "ibm-granite/granite-switch-micro",
                revision="test-rev",
                cache_dir="/tmp/test-cache",
            )

        mock_dl.assert_called_once_with(
            repo_id="ibm-granite/granite-switch-micro",
            allow_patterns=["adapter_index.json", "io_configs/**"],
            cache_dir="/tmp/test-cache",
            revision="test-rev",
        )
        assert len(adapters) == 2

    def test_filter_single_intrinsic(self, model_dir):
        with patch(
            "huggingface_hub.snapshot_download", return_value=str(model_dir)
        ) as mock_dl:
            adapters = EmbeddedIntrinsicAdapter.from_hub(
                "ibm-granite/granite-switch-micro", intrinsic_name="citations"
            )

        mock_dl.assert_called_once_with(
            repo_id="ibm-granite/granite-switch-micro",
            allow_patterns=["adapter_index.json", "io_configs/**"],
            cache_dir=None,
            revision="main",
        )
        assert len(adapters) == 1
        assert adapters[0].intrinsic_name == "citations"

    def test_missing_huggingface_hub_raises(self):
        with patch.dict("sys.modules", {"huggingface_hub": None}):
            with pytest.raises(ImportError, match="huggingface_hub is required"):
                EmbeddedIntrinsicAdapter.from_hub("some/repo")


# ---- EmbeddedIntrinsicAdapter.from_source ----


class TestFromSource:
    def test_local_directory(self, model_dir):
        """Local path routes to from_model_directory."""
        adapters = EmbeddedIntrinsicAdapter.from_source(str(model_dir))
        assert len(adapters) == 2

    def test_local_directory_with_filter(self, model_dir):
        """Local path with intrinsic_name filter."""
        adapters = EmbeddedIntrinsicAdapter.from_source(
            str(model_dir), intrinsic_name="answerability"
        )
        assert len(adapters) == 1
        assert adapters[0].intrinsic_name == "answerability"

    def test_hub_repo_id(self, model_dir):
        """Non-local string routes to from_hub."""
        with patch(
            "huggingface_hub.snapshot_download", return_value=str(model_dir)
        ) as mock_dl:
            adapters = EmbeddedIntrinsicAdapter.from_source(
                "ibm-granite/granite-switch-micro"
            )
        mock_dl.assert_called_once()
        assert len(adapters) == 2

    def test_hub_passes_revision_and_cache(self, model_dir):
        """revision and cache_dir are forwarded to from_hub."""
        with patch(
            "huggingface_hub.snapshot_download", return_value=str(model_dir)
        ) as mock_dl:
            EmbeddedIntrinsicAdapter.from_source(
                "ibm-granite/granite-switch-micro",
                revision="v2",
                cache_dir="/tmp/cache",
            )
        mock_dl.assert_called_once_with(
            repo_id="ibm-granite/granite-switch-micro",
            allow_patterns=["adapter_index.json", "io_configs/**"],
            cache_dir="/tmp/cache",
            revision="v2",
        )


# ---- OpenAIBackend adapter integration ----


class TestOpenAIBackendRegistration:
    @pytest.fixture
    def backend(self):
        os.environ.setdefault("OPENAI_API_KEY", "test-key")
        from mellea.backends.openai import OpenAIBackend

        return OpenAIBackend(
            model_id="granite-switch", base_url="http://localhost:8000/v1"
        )

    def test_add_adapter(self, backend):
        adapter = EmbeddedIntrinsicAdapter(
            intrinsic_name="answerability",
            config=_ANSWERABILITY_CONFIG,
            technology="alora",
        )
        backend.add_adapter(adapter)
        assert "answerability_alora" in backend._added_adapters
        assert backend._added_adapters["answerability_alora"] is adapter
        assert adapter.backend is backend

    def test_add_non_embedded_adapter_raises(self, backend):
        mock_adapter = MagicMock(spec=[])
        with pytest.raises(TypeError, match="only supports EmbeddedIntrinsicAdapter"):
            backend.add_adapter(mock_adapter)

    def test_list_adapters(self, backend):
        backend.add_adapter(
            EmbeddedIntrinsicAdapter(
                "answerability", config=_ANSWERABILITY_CONFIG, technology="alora"
            )
        )
        backend.add_adapter(
            EmbeddedIntrinsicAdapter(
                "citations", config=_CITATIONS_CONFIG, technology="lora"
            )
        )
        assert set(backend.list_adapters()) == {"answerability_alora", "citations_lora"}

    def test_load_unload_are_noops(self, backend):
        """load_adapter and unload_adapter succeed silently for embedded adapters."""
        backend.add_adapter(
            EmbeddedIntrinsicAdapter(
                "answerability", config=_ANSWERABILITY_CONFIG, technology="alora"
            )
        )
        # These should not raise.
        backend.load_adapter("answerability_alora")
        backend.unload_adapter("answerability_alora")
        # Adapter is still registered after load/unload.
        assert "answerability_alora" in backend._added_adapters

    def test_base_model_name(self, backend):
        assert backend.base_model_name == "granite-switch"

    def test_register_embedded_adapter_model(self, backend, model_dir):
        with patch("huggingface_hub.snapshot_download", return_value=str(model_dir)):
            names = backend.register_embedded_adapter_model(
                "ibm-granite/granite-switch-micro"
            )

        assert set(names) == {"answerability", "citations"}
        assert len(backend._added_adapters) == 2

    def test_register_from_local_directory(self, backend, model_dir):
        """register_embedded_adapter_model works with a local directory path."""
        names = backend.register_embedded_adapter_model(str(model_dir))
        assert set(names) == {"answerability", "citations"}
        assert len(backend._added_adapters) == 2

    def test_register_overwrites_existing(self, backend):
        config1 = {"model": None, "parameters": {"max_completion_tokens": 10}}
        config2 = {"model": None, "parameters": {"max_completion_tokens": 20}}

        backend.add_adapter(EmbeddedIntrinsicAdapter("test", config=config1))
        backend.add_adapter(EmbeddedIntrinsicAdapter("test", config=config2))

        assert (
            backend._added_adapters["test_lora"].config["parameters"][
                "max_completion_tokens"
            ]
            == 20
        )

    def test_embedded_adapters_flag_loads_from_model_id(self, model_dir):
        """embedded_adapters=True auto-registers adapters using model_id as source."""
        from mellea.backends.openai import OpenAIBackend

        os.environ.setdefault("OPENAI_API_KEY", "test-key")
        with patch("huggingface_hub.snapshot_download", return_value=str(model_dir)):
            backend = OpenAIBackend(
                model_id="ibm-granite/granite-switch-micro",
                base_url="http://localhost:8000/v1",
                load_embedded_adapters=True,
            )
        assert len(backend._added_adapters) == 2
        assert set(backend.list_adapters()) == {"answerability_alora", "citations_lora"}

    def test_embedded_adapters_flag_defaults_to_false(self, backend):
        """Without the flag, no adapters are loaded."""
        assert len(backend._added_adapters) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
