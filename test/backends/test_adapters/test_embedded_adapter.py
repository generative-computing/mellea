# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for embedded-adapter discovery helpers and OpenAI backend integration.

The `_embedded_adapters_from_*` module-level helpers replace the deprecated
`EmbeddedIntrinsicAdapter.from_*` static methods (Epic #929, issue #1144): they
read a Granite Switch model directory (or Hub snapshot) and return composed
`(Adapter, io_yaml_config)` pairs directly, with no shim intermediary.
"""

import json
import os
import pathlib
from typing import Literal
from unittest.mock import MagicMock, patch

import pytest
import yaml

from mellea.backends.adapters._core import Adapter, EmbeddedBinding
from mellea.backends.adapters.adapter import (
    _embedded_adapters_from_hub,
    _embedded_adapters_from_model_directory,
    _embedded_adapters_from_source,
)
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


@pytest.fixture
def hub_cache_dir(tmp_path, monkeypatch):
    """Redirect the default Hugging Face cache outside the mocked snapshot."""
    cache_dir = tmp_path.parent / f"{tmp_path.name}-hub-cache"
    monkeypatch.setattr("huggingface_hub.constants.HF_HUB_CACHE", str(cache_dir))
    return cache_dir


def _by_name(pairs: list[tuple[Adapter, dict]]) -> dict[str, tuple[Adapter, dict]]:
    """Index discovered `(Adapter, config)` pairs by capability name."""
    return {adapter.identity.name: (adapter, config) for adapter, config in pairs}


# ---- _embedded_adapters_from_model_directory ----


class TestFromModelDirectory:
    def test_loads_all_adapters(self, model_dir):
        pairs = _embedded_adapters_from_model_directory(model_dir)

        assert len(pairs) == 2
        by_name = _by_name(pairs)
        assert set(by_name) == {"answerability", "citations"}

        # Discovered adapters are composed Adapters carrying an EmbeddedBinding.
        ans, ans_config = by_name["answerability"]
        assert isinstance(ans, Adapter)
        assert isinstance(ans.weights, EmbeddedBinding)
        assert ans.identity.adapter_type == "alora"
        assert ans_config["parameters"]["max_completion_tokens"] == 6

        cit, _ = by_name["citations"]
        assert cit.identity.adapter_type == "lora"

    def test_accepts_string_path(self, model_dir):
        pairs = _embedded_adapters_from_model_directory(str(model_dir))
        assert len(pairs) == 2

    def test_missing_adapter_index(self, tmp_path):
        with pytest.raises(FileNotFoundError, match=r"adapter_index\.json"):
            _embedded_adapters_from_model_directory(tmp_path)

    def test_missing_io_yaml(self, tmp_path):
        (tmp_path / "adapter_index.json").write_text(json.dumps(_SAMPLE_ADAPTER_INDEX))
        with pytest.raises(ValueError, match=r"io\.yaml.*not found"):
            _embedded_adapters_from_model_directory(tmp_path)

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

        pairs = _embedded_adapters_from_model_directory(tmp_path)
        assert len(pairs) == 1
        assert pairs[0][0].identity.name == "has_config"

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

        pairs = _embedded_adapters_from_model_directory(tmp_path)
        assert len(pairs) == 1
        assert pairs[0][0].identity.adapter_type == "lora"

    def test_empty_adapters_list(self, tmp_path):
        (tmp_path / "adapter_index.json").write_text(json.dumps({"adapters": []}))
        with pytest.raises(ValueError, match="No adapters found"):
            _embedded_adapters_from_model_directory(tmp_path)

    def test_no_adapters_key(self, tmp_path):
        """Index with no 'adapters' key raises ValueError."""
        (tmp_path / "adapter_index.json").write_text(json.dumps({}))
        with pytest.raises(ValueError, match="No adapters found"):
            _embedded_adapters_from_model_directory(tmp_path)

    def test_filter_single_intrinsic(self, model_dir):
        pairs = _embedded_adapters_from_model_directory(
            model_dir, intrinsic_name="answerability"
        )
        assert len(pairs) == 1
        assert pairs[0][0].identity.name == "answerability"

    def test_filter_nonexistent_intrinsic(self, model_dir):
        with pytest.raises(
            ValueError, match="No adapter found for adapter function 'nonexistent'"
        ):
            _embedded_adapters_from_model_directory(
                model_dir, intrinsic_name="nonexistent"
            )

    def test_invalid_technology_raises(self, tmp_path):
        """An unsupported technology in the index is rejected."""
        index = {
            "adapters": [
                {
                    "adapter_name": "test",
                    "technology": "qlora",
                    "io_config": "io_configs/test/io.yaml",
                }
            ]
        }
        (tmp_path / "adapter_index.json").write_text(json.dumps(index))
        cfg_dir = tmp_path / "io_configs" / "test"
        cfg_dir.mkdir(parents=True)
        (cfg_dir / "io.yaml").write_text(yaml.dump({"model": None}))

        with pytest.raises(ValueError, match="must be 'lora' or 'alora'"):
            _embedded_adapters_from_model_directory(tmp_path)

    def test_path_traversal_in_io_config_raises(self, tmp_path):
        """io_config paths with ../ traversal that escape model_path are rejected."""
        outside = tmp_path / "outside.yaml"
        outside.write_text(yaml.dump({"model": None}))

        index = {
            "adapters": [
                {
                    "adapter_name": "escape",
                    "technology": "lora",
                    "io_config": "../outside.yaml",
                }
            ]
        }
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        (model_dir / "adapter_index.json").write_text(json.dumps(index))

        with pytest.raises(ValueError, match="escapes the model directory"):
            _embedded_adapters_from_model_directory(model_dir)

    def test_symlink_escape_in_io_config_raises(self, tmp_path):
        """io_config paths that resolve via symlink outside model_path are rejected."""
        outside = tmp_path / "secret.yaml"
        outside.write_text(yaml.dump({"model": None}))

        model_dir = tmp_path / "model"
        model_dir.mkdir()
        io_dir = model_dir / "io_configs" / "evil"
        io_dir.mkdir(parents=True)
        (io_dir / "io.yaml").symlink_to(outside)

        index = {
            "adapters": [
                {
                    "adapter_name": "evil",
                    "technology": "lora",
                    "io_config": "io_configs/evil/io.yaml",
                }
            ]
        }
        (model_dir / "adapter_index.json").write_text(json.dumps(index))

        with pytest.raises(ValueError, match="escapes the model directory"):
            _embedded_adapters_from_model_directory(model_dir)

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

        pairs = _embedded_adapters_from_model_directory(tmp_path)
        assert len(pairs) == 1
        assert pairs[0][0].identity.name == "answerability"


# ---- _embedded_adapters_from_hub ----


class TestFromHub:
    def test_downloads_and_delegates(self, model_dir):
        """from_hub calls snapshot_download then delegates to from_model_directory."""
        cache_dir = model_dir.parent / "cache"
        with patch(
            "huggingface_hub.snapshot_download", return_value=str(model_dir)
        ) as mock_dl:
            pairs = _embedded_adapters_from_hub(
                "ibm-granite/granite-switch-micro",
                revision="test-rev",
                cache_dir=str(cache_dir),
            )

        mock_dl.assert_called_once_with(
            repo_id="ibm-granite/granite-switch-micro",
            allow_patterns=["adapter_index.json", "io_configs/**"],
            cache_dir=str(cache_dir),
            revision="test-rev",
        )
        assert len(pairs) == 2

    def test_filter_single_intrinsic(self, model_dir):
        cache_dir = model_dir.parent / "cache"
        with patch(
            "huggingface_hub.snapshot_download", return_value=str(model_dir)
        ) as mock_dl:
            pairs = _embedded_adapters_from_hub(
                "ibm-granite/granite-switch-micro",
                cache_dir=str(cache_dir),
                intrinsic_name="citations",
            )

        mock_dl.assert_called_once_with(
            repo_id="ibm-granite/granite-switch-micro",
            allow_patterns=["adapter_index.json", "io_configs/**"],
            cache_dir=str(cache_dir),
            revision="main",
        )
        assert len(pairs) == 1
        assert pairs[0][0].identity.name == "citations"

    def test_from_hub_materialises_hub_snapshot(self, model_dir, tmp_path):
        """from_hub materialises Hub blob symlinks into its persistent local directory."""
        source_files = [
            model_dir / "adapter_index.json",
            *model_dir.glob("io_configs/*/io.yaml"),
        ]

        def snapshot_download(**_):
            snapshot_dir = tmp_path / "snapshots" / "revision"
            blob_dir = tmp_path / "blobs"
            for source in source_files:
                destination = snapshot_dir / source.relative_to(model_dir)
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(source.read_bytes())

            for io_config in snapshot_dir.glob("io_configs/*/io.yaml"):
                blob = blob_dir / io_config.parent.name
                blob.parent.mkdir(parents=True, exist_ok=True)
                blob.write_bytes(io_config.read_bytes())
                io_config.unlink()
                io_config.symlink_to(os.path.relpath(blob, io_config.parent))
            (snapshot_dir / "model.safetensors").write_bytes(b"weights")

            return str(snapshot_dir)

        with patch("huggingface_hub.snapshot_download", side_effect=snapshot_download):
            pairs = _embedded_adapters_from_hub(
                "ibm-granite/granite-switch-micro", cache_dir=str(tmp_path / "cache")
            )

        assert {adapter.identity.name for adapter, _ in pairs} == {
            "answerability",
            "citations",
        }
        materialized_dir = next(
            (tmp_path / "cache" / "mellea" / "embedded-adapter-configs").iterdir()
        )
        assert not (materialized_dir / "model.safetensors").exists()

    def test_from_hub_materialises_each_snapshot_revision(self, model_dir, tmp_path):
        """from_hub keeps materialised configs isolated by immutable snapshot revision."""
        source_files = [
            model_dir / "adapter_index.json",
            *model_dir.glob("io_configs/*/io.yaml"),
        ]

        def make_snapshot(name):
            snapshot_dir = tmp_path / "snapshots" / name
            for source in source_files:
                destination = snapshot_dir / source.relative_to(model_dir)
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(source.read_bytes())
            return snapshot_dir

        main_snapshot = make_snapshot("main-commit")
        v2_snapshot = make_snapshot("v2-commit")
        cache_dir = tmp_path / "cache"
        with patch(
            "huggingface_hub.snapshot_download",
            side_effect=[str(main_snapshot), str(main_snapshot), str(v2_snapshot)],
        ) as mock_dl:
            _embedded_adapters_from_hub(
                "ibm-granite/granite-switch-micro", cache_dir=str(cache_dir)
            )
            _embedded_adapters_from_hub(
                "ibm-granite/granite-switch-micro", cache_dir=str(cache_dir)
            )
            _embedded_adapters_from_hub(
                "ibm-granite/granite-switch-micro",
                cache_dir=str(cache_dir),
                revision="v2",
            )

        assert mock_dl.call_count == 3
        materialized_dirs = list(
            (cache_dir / "mellea" / "embedded-adapter-configs").iterdir()
        )
        assert len(materialized_dirs) == 2

    def test_missing_huggingface_hub_raises(self):
        with patch.dict("sys.modules", {"huggingface_hub": None}):
            with pytest.raises(ImportError, match="huggingface_hub is required"):
                _embedded_adapters_from_hub("some/repo")

    @pytest.mark.parametrize(
        "error_name", ["GatedRepoError", "RepositoryNotFoundError"]
    )
    def test_auth_error_raises_permission_error(self, error_name):
        """Auth failures from snapshot_download become an actionable PermissionError."""
        from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError

        class _FakeResponse:
            headers: dict = {}
            status_code = 403
            url = "https://huggingface.co"
            request = None

        error_cls = {
            "GatedRepoError": GatedRepoError,
            "RepositoryNotFoundError": RepositoryNotFoundError,
        }[error_name]
        hf_error = error_cls("denied", response=_FakeResponse())

        with patch("huggingface_hub.snapshot_download", side_effect=hf_error):
            with pytest.raises(PermissionError, match="huggingface-cli login") as exc:
                _embedded_adapters_from_hub("ibm-granite/private-switch")

        # Original HF error is chained for debugging.
        assert exc.value.__cause__ is hf_error

    def test_missing_index_after_download_raises_clear_file_not_found(
        self, tmp_path, hub_cache_dir
    ):
        """A snapshot without adapter_index.json raises a repo-scoped FileNotFoundError.

        snapshot_download can return a path that lacks the index (wrong
        repo/revision, not a Switch model, or a stale cache). The message names
        the repo and lists auth as one possible cause, rather than leaking the
        cryptic snapshot-cache path from from_model_directory.
        """
        with patch("huggingface_hub.snapshot_download", return_value=str(tmp_path)):
            with pytest.raises(
                FileNotFoundError, match=r"ibm-granite/some-switch"
            ) as exc:
                _embedded_adapters_from_hub("ibm-granite/some-switch")

        # Not the raw cache-path error, and auth is offered as a possibility.
        assert "huggingface-cli login" in str(exc.value)
        # The original cryptic error is chained for debugging.
        assert isinstance(exc.value.__cause__, FileNotFoundError)


# ---- _embedded_adapters_from_source ----


class TestFromSource:
    def test_local_directory(self, model_dir):
        """Local path routes to from_model_directory."""
        pairs = _embedded_adapters_from_source(str(model_dir))
        assert len(pairs) == 2

    def test_local_directory_with_filter(self, model_dir):
        """Local path with intrinsic_name filter."""
        pairs = _embedded_adapters_from_source(
            str(model_dir), intrinsic_name="answerability"
        )
        assert len(pairs) == 1
        assert pairs[0][0].identity.name == "answerability"

    def test_hub_repo_id(self, model_dir, hub_cache_dir):
        """Non-local string routes to from_hub."""
        with patch(
            "huggingface_hub.snapshot_download", return_value=str(model_dir)
        ) as mock_dl:
            pairs = _embedded_adapters_from_source("ibm-granite/granite-switch-micro")
        mock_dl.assert_called_once()
        assert len(pairs) == 2

    def test_hub_passes_revision_and_cache(self, model_dir):
        """revision and cache_dir are forwarded to from_hub."""
        with patch(
            "huggingface_hub.snapshot_download", return_value=str(model_dir)
        ) as mock_dl:
            _embedded_adapters_from_source(
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


# ---- OpenAIBackend adapter integration (composed Adapter) ----


class TestOpenAIBackendComposedAdapterRegistration:
    """`OpenAIBackend.add_adapter` accepts a composed `Adapter` whose `weights`
    is an `EmbeddedBinding` (Epic #929, issue #1144).
    """

    @pytest.fixture
    def backend(self):
        os.environ.setdefault("OPENAI_API_KEY", "test-key")
        from mellea.backends.openai import OpenAIBackend

        return OpenAIBackend(
            model_id="granite-switch", base_url="http://localhost:8000/v1"
        )

    @staticmethod
    def _make_composed_adapter(
        name: str = "answerability", adapter_type: Literal["lora", "alora"] = "alora"
    ):
        from mellea.backends.adapters._core import Adapter, EmbeddedBinding, Identity
        from mellea.backends.adapters.io_contracts import get_io_contract

        return Adapter(
            identity=Identity(name=name, adapter_type=adapter_type, capability=name),
            io_contract=get_io_contract(name),
            weights=EmbeddedBinding(),
        )

    def test_add_composed_adapter(self, backend):
        adapter = self._make_composed_adapter()
        backend.add_adapter(adapter, config={"parameters": {}})
        assert "answerability_alora" in backend._added_adapters
        assert backend._added_adapters["answerability_alora"] is adapter
        assert adapter.weights.source == backend.base_model_name  # type: ignore[union-attr]
        assert backend._composed_adapter_configs["answerability_alora"] == {
            "parameters": {}
        }

    def test_add_composed_adapter_without_config_raises(self, backend):
        adapter = self._make_composed_adapter()
        with pytest.raises(ValueError, match=r"No io\.yaml config given"):
            backend.add_adapter(adapter)
        assert "answerability_alora" not in backend._added_adapters

    def test_add_composed_adapter_rejects_non_embedded_weights(self, backend):
        from mellea.backends.adapters._core import Adapter, Identity, LocalFileBinding

        adapter = Adapter(
            identity=Identity(name="answerability", adapter_type="lora"),
            io_contract=self._make_composed_adapter().io_contract,
            weights=LocalFileBinding.from_catalog("answerability"),
        )
        with pytest.raises(TypeError, match="Embedded/Granite Switch"):
            backend.add_adapter(adapter)

    def test_add_non_adapter_raises(self, backend):
        mock_adapter = MagicMock(spec=[])
        with pytest.raises(TypeError, match="composed Adapter"):
            backend.add_adapter(mock_adapter)

    def test_add_composed_adapter_refuses_duplicate_name(self, backend):
        first = self._make_composed_adapter()
        second = self._make_composed_adapter()
        backend.add_adapter(first, config={"parameters": {}})
        backend.add_adapter(second, config={"parameters": {}})
        assert backend._added_adapters["answerability_alora"] is first

    def test_composed_adapter_discoverable_by_capability(self, backend):
        adapter = self._make_composed_adapter()
        backend.add_adapter(adapter, config={"parameters": {}})
        assert backend._find_adapter("answerability") is adapter

    def test_list_adapters_includes_composed_adapter(self, backend):
        backend.add_adapter(self._make_composed_adapter(), config={"parameters": {}})
        backend.add_adapter(
            self._make_composed_adapter(name="citations", adapter_type="lora"),
            config=_CITATIONS_CONFIG,
        )
        assert set(backend.list_adapters()) == {"answerability_alora", "citations_lora"}

    def test_base_model_name(self, backend):
        assert backend.base_model_name == "granite-switch"

    def test_register_embedded_adapter_model(self, backend, model_dir, hub_cache_dir):
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

    def test_embedded_adapters_flag_loads_from_model_id(self, model_dir, hub_cache_dir):
        """load_embedded_adapters=True auto-registers adapters using model_id as source."""
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

    def test_adapter_source_used_for_loading(self, model_dir):
        """adapter_source is used instead of model_id for adapter loading."""
        from mellea.backends.openai import OpenAIBackend

        os.environ.setdefault("OPENAI_API_KEY", "test-key")
        backend = OpenAIBackend(
            model_id="granite-switch",
            base_url="http://localhost:8000/v1",
            load_embedded_adapters=True,
            adapter_source=str(model_dir),
        )
        # Adapters loaded from local dir, model_id untouched for API calls
        assert len(backend._added_adapters) == 2
        assert backend._model_id == "granite-switch"

    def test_adapter_source_defaults_to_model_id(self, model_dir, hub_cache_dir):
        """Without adapter_source, model_id is used (existing behavior)."""
        from mellea.backends.openai import OpenAIBackend

        os.environ.setdefault("OPENAI_API_KEY", "test-key")
        with patch("huggingface_hub.snapshot_download", return_value=str(model_dir)):
            backend = OpenAIBackend(
                model_id="ibm-granite/granite-switch-micro",
                base_url="http://localhost:8000/v1",
                load_embedded_adapters=True,
            )
        assert len(backend._added_adapters) == 2


class TestShimsRemoved:
    """The deprecated adapter shims are gone (issue #1621, PR2 of #1144).

    `IntrinsicAdapter`, `EmbeddedIntrinsicAdapter`, and `CustomIntrinsicAdapter`
    were removed once the composed `Adapter` became their working replacement.
    Importing any of them must now fail, not resolve to a lingering alias.
    """

    @pytest.mark.parametrize(
        "name",
        ["IntrinsicAdapter", "EmbeddedIntrinsicAdapter", "CustomIntrinsicAdapter"],
    )
    def test_shim_not_importable_from_package(self, name):
        import mellea.backends.adapters as adapters_pkg

        assert not hasattr(adapters_pkg, name)
        assert name not in adapters_pkg.__all__
        with pytest.raises(ImportError):
            exec(f"from mellea.backends.adapters import {name}")

    @pytest.mark.parametrize(
        "name",
        [
            "IntrinsicAdapter",
            "EmbeddedIntrinsicAdapter",
            "CustomIntrinsicAdapter",
            "_ShimWeightsBinding",
        ],
    )
    def test_shim_not_importable_from_adapter_module(self, name):
        import mellea.backends.adapters.adapter as adapter_mod

        assert not hasattr(adapter_mod, name)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
